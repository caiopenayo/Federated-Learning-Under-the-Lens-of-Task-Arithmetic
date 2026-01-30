import copy
import torch
import numpy as np
from torch import nn, optim
from optim.sparse_sgdm import SparseSGDM
import os

def fed_avg_aggregate(global_model, client_models, client_weights):
    """
    Aggregates client models into the global model using weighted averaging.
    w_global = sum(n_k * w_k) / sum(n_k)
    """
    global_dict = global_model.state_dict()
    new_dict = copy.deepcopy(global_dict)
    
    # Reset accumulators to zero
    for key in new_dict.keys():
        new_dict[key] = torch.zeros_like(new_dict[key], dtype=torch.float32)
        
    total_weight = sum(client_weights)
    
    for client_model, weight in zip(client_models, client_weights):
        client_dict = client_model.state_dict()
        for key in new_dict.keys():
            # Accumulate weighted parameters
            # Use float32 for accumulation to avoid overflow/precision issues
            new_dict[key] += weight * client_dict[key].to(torch.float32)
            
    # Normalize
    for key in new_dict.keys():
        new_dict[key] = (new_dict[key] / total_weight).to(global_dict[key].dtype)
        
    global_model.load_state_dict(new_dict)
    return global_model

def client_update(model, train_loader, steps, lr, device, mask=None):
    """
    Performs J local steps of training on a client.
    """
    model.train()
    if mask is None:
        optimizer = optim.SGD(model.parameters(), lr=lr) # Standard SGD for FedAvg
    else:
        optimizer = SparseSGDM(model.parameters(), lr=lr, mask=mask)
    criterion = nn.CrossEntropyLoss()
    
    iterator = iter(train_loader)
    loss_accum = 0.0
    
    for _ in range(steps):
        try:
            data, target = next(iterator)
        except StopIteration:
            # Restart iterator if we run out of data
            iterator = iter(train_loader)
            data, target = next(iterator)
            
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        loss_accum += loss.item()
        
    return model, loss_accum / steps

def run_fedavg_experiment(
    base_model, 
    client_loaders, 
    test_loader, 
    rounds, 
    C, 
    J, 
    lr=0.01, 
    device="cuda",
    log_every=10,
    mask=None,
    resume=False,
    ckpt_path=None,
    ckpt_every=20,   # NEW: save checkpoint every N rounds
):
    """
    Runs the FedAvg algorithm with optional checkpointing/resume.

    Args:
        C: Client participation rate (0.0 to 1.0)
        J: Number of local steps
        resume: if True, attempt to resume from ckpt_path
        ckpt_path: path to save/load checkpoint
        ckpt_every: save checkpoint every N rounds (in addition to log/eof)
    """
    assert ckpt_path is not None, "Please provide ckpt_path to enable safe resume."

    # Deep copy global model
    global_model = copy.deepcopy(base_model).to(device)

    # Only sample clients that actually have data.
    eligible_clients = []
    for i, loader in enumerate(client_loaders):
        if loader is None:
            continue
        if hasattr(loader, "dataset") and len(loader.dataset) == 0:
            continue
        try:
            if len(loader) == 0:
                continue
        except TypeError:
            pass
        eligible_clients.append(i)

    if len(eligible_clients) == 0:
        raise ValueError("No eligible clients with data. Check your partitioning/sharding settings.")

    m = max(int(C * len(eligible_clients)), 1)
    m = min(m, len(eligible_clients))

    # Initialize RNG once (so we can resume deterministically)
    rng = np.random.default_rng(42)

    # History container
    history = {'rounds': [], 'test_acc': [], 'loss': []}
    start_round = 1

    # -----------------------
    # Resume from checkpoint
    # -----------------------
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)

        # Load model
        global_model.load_state_dict(ckpt["global_model_state"])

        # Load history
        history = ckpt.get("history", history)

        # Load round counter (resume from next round)
        start_round = int(ckpt.get("round", 0)) + 1

        # Restore numpy RNG state if available
        if "np_rng_state" in ckpt:
            try:
                rng = np.random.default_rng()
                rng.bit_generator.state = ckpt["np_rng_state"]
            except Exception:
                # If something goes wrong, fall back to non-deterministic continue
                pass

        print(f"[Resume] Loaded checkpoint from: {ckpt_path}")
        print(f"[Resume] Resuming at round {start_round}/{rounds}")

    print(
        f"  [FedAvg] Start: {rounds} rounds, C={C} "
        f"({m} clients/round from {len(eligible_clients)} eligible), J={J} steps"
    )

    def save_checkpoint(round_idx):
        """
        Save a checkpoint that is safe to resume from after Colab disconnects.
        """
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        payload = {
            "round": round_idx,
            "global_model_state": global_model.state_dict(),
            "history": history,
            "np_rng_state": rng.bit_generator.state,  # ensures same client sampling after resume
            "meta": {
                "C": C, "J": J, "lr": lr, "m": m,
                "eligible_clients": len(eligible_clients),
            }
        }
        torch.save(payload, ckpt_path)

    # -----------------------
    # Main training loop
    # -----------------------
    for r in range(start_round, rounds + 1):
        # 1) Server selects subset of clients
        selected_indices = rng.choice(eligible_clients, m, replace=False)

        local_models = []
        client_weights = []
        loss_sum = 0.0
        trained_clients = 0

        # 2) Clients train
        for client_idx in selected_indices:
            loader = client_loaders[client_idx]

            # Skip empty clients (if any)
            if loader is None:
                continue
            try:
                if len(loader) == 0:
                    continue
            except TypeError:
                pass

            # Create local copy
            local_model = copy.deepcopy(global_model)

            # Local Update (J steps)
            trained_model, loss = client_update(
                local_model,
                loader,
                steps=J,
                lr=lr,
                device=device,
                mask=mask
            )

            local_models.append(trained_model)
            client_weights.append(len(loader.dataset) if hasattr(loader, 'dataset') else 1.0)
            loss_sum += float(loss)
            trained_clients += 1

        # 3) Server Aggregation
        if local_models:
            global_model = fed_avg_aggregate(global_model, local_models, client_weights)

        # 4) Evaluation + logging
        if (r % log_every == 0) or (r == rounds):
            global_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = global_model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            acc = 100 * correct / total
            mean_loss = (loss_sum / trained_clients) if trained_clients > 0 else float('nan')

            history['rounds'].append(r)
            history['test_acc'].append(acc)
            history['loss'].append(mean_loss)

            print(
                f"    Round {r:03d} | Test Acc: {acc:.2f}% | "
                f"Train Loss: {mean_loss:.4f} | Trained clients: {trained_clients}/{m}"
            )

        # 5) Periodic checkpointing
        if (ckpt_every is not None) and (ckpt_every > 0) and (r % ckpt_every == 0):
            save_checkpoint(r)

        # Also checkpoint at the very end
        if r == rounds:
            save_checkpoint(r)

    return history