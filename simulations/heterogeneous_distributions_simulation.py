# ============================================================================
# Heterogeneous Distribution Experiments for FedAvg
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
from tqdm import tqdm
import copy

# Add to your existing imports in Colab
from fl.dataloaders import build_federated_dataloaders
from models.vit_dino import build_dino_vit
from data.datasets import get_cifar100_transforms

# We'll define our own evaluate function to avoid signature issues
def evaluate_model(model, data_loader, device='cuda'):
    """
    Evaluate model on a dataset
    
    Returns:
        loss (float), accuracy (float)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


# ============================================================================
# FedAvg Algorithm Implementation
# ============================================================================

def fedavg_aggregate(global_model, client_models, client_sizes):
    """
    Aggregate client models using FedAvg (weighted averaging by dataset size)
    
    Args:
        global_model: The global model to update
        client_models: List of client model state dicts
        client_sizes: List of dataset sizes for each client
    """
    global_dict = global_model.state_dict()
    
    # Calculate weights (normalize by total samples)
    total_size = sum(client_sizes)
    weights = [size / total_size for size in client_sizes]
    
    # Weighted average of parameters
    for key in global_dict.keys():
        # Skip buffers (like batch norm running stats) if they shouldn't be averaged
        if 'num_batches_tracked' in key:
            continue
            
        global_dict[key] = torch.zeros_like(global_dict[key])
        for client_dict, weight in zip(client_models, weights):
            global_dict[key] += weight * client_dict[key]
    
    global_model.load_state_dict(global_dict)
    return global_model


def client_update(model, train_loader, epochs, lr, device='cuda'):
    """
    Perform local training on a client
    
    Args:
        model: Client model
        train_loader: Client's data loader
        epochs: Number of local epochs (J)
        lr: Learning rate
        device: Device to train on
    
    Returns:
        Updated model state dict and number of samples
    """
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    return model.state_dict(), len(train_loader.dataset)


# ============================================================================
# FedAvg Training Loop
# ============================================================================

def train_fedavg(
    global_model,
    client_loaders,
    val_loader,
    test_loader,
    num_rounds,
    local_epochs,
    client_fraction=0.1,
    lr=0.01,
    device='cuda',
    eval_every=5,
    verbose=True
):
    """
    Main FedAvg training loop
    
    Args:
        global_model: Initial global model
        client_loaders: List of client data loaders
        val_loader: Validation data loader
        test_loader: Test data loader
        num_rounds: Number of communication rounds
        local_epochs: Number of local training epochs (J)
        client_fraction: Fraction of clients to sample each round (C)
        lr: Learning rate
        device: Device to train on
        eval_every: Evaluate every N rounds
        verbose: Print progress
    
    Returns:
        History dictionary with metrics
    """
    K = len(client_loaders)
    num_selected = max(1, int(client_fraction * K))
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_val_acc = 0.0
    
    for round_idx in range(num_rounds):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Round {round_idx + 1}/{num_rounds}")
            print(f"{'='*60}")
        
        # Sample clients
        selected_clients = np.random.choice(K, num_selected, replace=False)
        
        # Client updates
        client_models = []
        client_sizes = []
        
        if verbose:
            pbar = tqdm(selected_clients, desc=f"Round {round_idx+1} - Client Updates")
        else:
            pbar = selected_clients
            
        for client_idx in pbar:
            # Create client model copy
            client_model = copy.deepcopy(global_model)
            
            # Local training
            client_state_dict, dataset_size = client_update(
                client_model,
                client_loaders[client_idx],
                local_epochs,
                lr,
                device
            )
            
            client_models.append(client_state_dict)
            client_sizes.append(dataset_size)
            
            # Clean up
            del client_model
            torch.cuda.empty_cache()
        
        # Aggregate
        global_model = fedavg_aggregate(global_model, client_models, client_sizes)
        
        # Evaluate - adjust based on your evaluate function signature
        if (round_idx + 1) % eval_every == 0 or round_idx == 0:
            global_model.eval()
            
            val_loss, val_acc = evaluate_model(global_model, val_loader, device)
            test_loss, test_acc = evaluate_model(global_model, test_loader, device)
            
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            if verbose:
                print(f"\nValidation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
                print(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if verbose:
                    print(f"✓ New best validation accuracy: {best_val_acc:.2f}%")
    
    return history, best_val_acc


# ============================================================================
# Experiment Runner
# ============================================================================

def run_heterogeneity_experiments(
    K=100,
    C=0.1,
    nc_values=[1, 5, 10, 50],
    J_values=[4, 8, 16],
    base_rounds=50,
    lr=0.01,
    batch_size=64,
    img_size=160,
    seed=42,
    save_dir="./fedavg_heterogeneity_results",
    device='cuda'
):
    """
    Run the complete heterogeneity experiment suite
    
    Args:
        K: Number of clients
        C: Client fraction
        nc_values: List of Nc values (classes per client)
        J_values: List of local step values
        base_rounds: Base number of rounds (for J=4)
        lr: Learning rate
        batch_size: Batch size
        img_size: Image size for ViT
        seed: Random seed
        save_dir: Directory to save results
        device: Device to train on
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get transforms
    transform_train, transform_test = get_cifar100_transforms()
    
    results = []
    
    # First, run IID baseline for each J value
    print("\n" + "="*80)
    print("RUNNING IID BASELINE EXPERIMENTS")
    print("="*80)
    
    for J in J_values:
        # Scale rounds inversely with J (keep total local steps roughly constant)
        num_rounds = int(base_rounds * 4 / J)
        
        print(f"\n{'*'*60}")
        print(f"IID - J={J}, Rounds={num_rounds}")
        print(f"{'*'*60}")
        
        # Build dataloaders
        client_loaders, val_loader, test_loader, *_ = build_federated_dataloaders(
            train_transform=transform_train,
            test_transform=transform_test,
            K=K,
            sharding="iid",
            val_ratio=0.1,
            batch_size=batch_size,
            seed=seed
        )
        
        # Create model
        model = build_dino_vit(img_size=img_size, num_classes=100)
        model = model.to(device)
        
        # Train
        history, best_val_acc = train_fedavg(
            global_model=model,
            client_loaders=client_loaders,
            val_loader=val_loader,
            test_loader=test_loader,
            num_rounds=num_rounds,
            local_epochs=J,
            client_fraction=C,
            lr=lr,
            device=device,
            eval_every=5,
            verbose=True
        )
        
        # Record results
        result = {
            'sharding': 'iid',
            'Nc': 100,  # All classes
            'J': J,
            'rounds': num_rounds,
            'best_val_acc': best_val_acc,
            'final_test_acc': history['test_acc'][-1] if history['test_acc'] else 0.0,
            'history': history
        }
        results.append(result)
        
        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'result': result,
            'history': history
        }, os.path.join(save_dir, f"iid_J{J}.pth"))
        
        print(f"\n✓ IID J={J} completed - Best Val Acc: {best_val_acc:.2f}%")
        
        del model, client_loaders, val_loader, test_loader
        torch.cuda.empty_cache()
    
    # Now run Non-IID experiments
    print("\n" + "="*80)
    print("RUNNING NON-IID EXPERIMENTS")
    print("="*80)
    
    for Nc in nc_values:
        for J in J_values:
            num_rounds = int(base_rounds * 4 / J)
            
            print(f"\n{'*'*60}")
            print(f"Non-IID - Nc={Nc}, J={J}, Rounds={num_rounds}")
            print(f"{'*'*60}")
            
            # Build dataloaders
            client_loaders, val_loader, test_loader, *_ = build_federated_dataloaders(
                train_transform=transform_train,
                test_transform=transform_test,
                K=K,
                sharding="non_iid",
                Nc=Nc,
                val_ratio=0.1,
                batch_size=batch_size,
                seed=seed
            )
            
            # Create model
            model = build_dino_vit(img_size=img_size, num_classes=100)
            model = model.to(device)
            
            # Train
            history, best_val_acc = train_fedavg(
                global_model=model,
                client_loaders=client_loaders,
                val_loader=val_loader,
                test_loader=test_loader,
                num_rounds=num_rounds,
                local_epochs=J,
                client_fraction=C,
                lr=lr,
                device=device,
                eval_every=5,
                verbose=True
            )
            
            # Record results
            result = {
                'sharding': 'non_iid',
                'Nc': Nc,
                'J': J,
                'rounds': num_rounds,
                'best_val_acc': best_val_acc,
                'final_test_acc': history['test_acc'][-1] if history['test_acc'] else 0.0,
                'history': history
            }
            results.append(result)
            
            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'result': result,
                'history': history
            }, os.path.join(save_dir, f"noniid_Nc{Nc}_J{J}.pth"))
            
            print(f"\n✓ Non-IID Nc={Nc} J={J} completed - Best Val Acc: {best_val_acc:.2f}%")
            
            del model, client_loaders, val_loader, test_loader
            torch.cuda.empty_cache()
    
    # Save summary results
    summary_df = pd.DataFrame([{
        'sharding': r['sharding'],
        'Nc': r['Nc'],
        'J': r['J'],
        'rounds': r['rounds'],
        'best_val_acc': r['best_val_acc'],
        'final_test_acc': r['final_test_acc']
    } for r in results])
    
    summary_df.to_csv(os.path.join(save_dir, 'summary_results.csv'), index=False)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    
    return results, summary_df


# ============================================================================
# Visualization
# ============================================================================

def plot_heterogeneity_results(results, save_dir="./fedavg_heterogeneity_results"):
    """
    Create comprehensive plots of the heterogeneity experiments
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data for plotting
    plot_data = []
    for r in results:
        plot_data.append({
            'sharding': r['sharding'],
            'Nc': r['Nc'],
            'J': r['J'],
            'best_val_acc': r['best_val_acc'],
            'final_test_acc': r['final_test_acc']
        })
    df = pd.DataFrame(plot_data)
    
    # Plot 1: Effect of Nc for each J
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    J_values = sorted(df['J'].unique())
    
    for idx, J in enumerate(J_values):
        ax = axes[idx]
        df_j = df[df['J'] == J]
        
        # Get IID baseline
        iid_acc = df_j[df_j['sharding'] == 'iid']['best_val_acc'].values[0]
        
        # Get Non-IID results
        df_noniid = df_j[df_j['sharding'] == 'non_iid'].sort_values('Nc')
        
        ax.plot(df_noniid['Nc'], df_noniid['best_val_acc'], 'o-', label='Non-IID', linewidth=2)
        ax.axhline(y=iid_acc, color='r', linestyle='--', label='IID Baseline', linewidth=2)
        
        ax.set_xlabel('Number of Classes per Client (Nc)', fontsize=12)
        ax.set_ylabel('Best Validation Accuracy (%)', fontsize=12)
        ax.set_title(f'J = {J} local steps', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'effect_of_nc.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Effect of J for each Nc
    nc_values = sorted(df[df['sharding'] == 'non_iid']['Nc'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, Nc in enumerate(nc_values):
        ax = axes[idx]
        
        # Non-IID data
        df_nc = df[(df['Nc'] == Nc) & (df['sharding'] == 'non_iid')].sort_values('J')
        ax.plot(df_nc['J'], df_nc['best_val_acc'], 'o-', label=f'Nc={Nc}', linewidth=2)
        
        # IID baseline
        df_iid = df[df['sharding'] == 'iid'].sort_values('J')
        ax.plot(df_iid['J'], df_iid['best_val_acc'], 's--', label='IID', linewidth=2, color='red')
        
        ax.set_xlabel('Local Steps (J)', fontsize=12)
        ax.set_ylabel('Best Validation Accuracy (%)', fontsize=12)
        ax.set_title(f'Nc = {Nc} classes per client', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([4, 8, 16])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'effect_of_j.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Heatmap of results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, J in enumerate(J_values):
        ax = axes[idx]
        df_j = df[(df['J'] == J) & (df['sharding'] == 'non_iid')].sort_values('Nc')
        
        bars = ax.bar(range(len(df_j)), df_j['best_val_acc'], alpha=0.7)
        
        # Color bars by performance
        iid_acc = df[df['J'] == J][df['sharding'] == 'iid']['best_val_acc'].values[0]
        for bar, acc in zip(bars, df_j['best_val_acc']):
            if acc >= iid_acc * 0.95:
                bar.set_color('green')
            elif acc >= iid_acc * 0.90:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.axhline(y=iid_acc, color='blue', linestyle='--', label='IID Baseline', linewidth=2)
        ax.set_xticks(range(len(df_j)))
        ax.set_xticklabels([f'Nc={nc}' for nc in df_j['Nc']])
        ax.set_ylabel('Best Validation Accuracy (%)', fontsize=12)
        ax.set_title(f'J = {J}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_bars.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Plots saved to {save_dir}")



