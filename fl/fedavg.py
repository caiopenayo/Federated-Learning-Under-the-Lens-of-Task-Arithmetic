import copy
import numpy as np


import torch
from torch import nn, optim




def fed_avg_aggregate(global_model, client_models, client_weights):
   """
   Weighted FedAvg:
   w_global = sum_k (n_k * w_k) / sum_k n_k
   """
   global_dict = global_model.state_dict()


   # float32 accumulators (safer)
   new_dict = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_dict.items()}
   total_weight = float(sum(client_weights))


   if total_weight == 0:
       return global_model  # nothing to aggregate


   for client_model, weight in zip(client_models, client_weights):
       client_dict = client_model.state_dict()
       w = float(weight)
       for k in new_dict:
           new_dict[k] += w * client_dict[k].float()


   for k in new_dict:
       new_dict[k] = (new_dict[k] / total_weight).to(global_dict[k].dtype)


   global_model.load_state_dict(new_dict)
   return global_model




def client_update(model, loader, steps, lr, device, momentum=0.9, weight_decay=0.0):
   """
   Perform J local optimization steps on one client.
   steps = J (mini-batch steps), NOT epochs.
   """
   model.train()
   optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
   criterion = nn.CrossEntropyLoss()


   it = iter(loader)
   loss_sum = 0.0


   for _ in range(steps):
       try:
           x, y = next(it)
       except StopIteration:
           it = iter(loader)
           x, y = next(it)


       x, y = x.to(device), y.to(device)


       optimizer.zero_grad()
       out = model(x)
       loss = criterion(out, y)
       loss.backward()
       optimizer.step()


       loss_sum += loss.item()


   return model, (loss_sum / steps)




@torch.no_grad()
def evaluate_accuracy(model, loader, device):
   """
   Compute accuracy (%) on a dataloader.
   """
   model.eval()
   correct, total = 0, 0
   for x, y in loader:
       x, y = x.to(device), y.to(device)
       out = model(x)
       pred = out.argmax(dim=1)
       correct += (pred == y).sum().item()
       total += y.size(0)
   return 100.0 * correct / max(total, 1)




def run_fedavg_experiment(
   base_model,
   client_loaders,
   test_loader,
   rounds,
   C,
   J,
   lr,
   device="cuda",
   log_every=5,
   seed=0,
   momentum=0.9,
   weight_decay=0.0001,
):
   """
   FedAvg main loop.


   K = len(client_loaders)
   Each round:
     - sample m = max(int(C*K), 1) clients
     - each trains for J local steps
     - server aggregates using weighted average by client dataset size
     - evaluate test accuracy periodically


   Returns:
     history dict with rounds, test_acc, train_loss
   """
   np.random.seed(seed)
   torch.manual_seed(seed)


   global_model = copy.deepcopy(base_model).to(device)


   num_clients = len(client_loaders)
   m = max(int(C * num_clients), 1)


   history = {"rounds": [], "test_acc": [], "train_loss": []}


   print(f"[FedAvg] rounds={rounds}, K={num_clients}, C={C} => clients/round={m}, J={J}, lr={lr}")


   for r in range(1, rounds + 1):
       selected = np.random.choice(num_clients, m, replace=False)


       local_models = []
       weights = []
       losses = []


       for idx in selected:
           loader = client_loaders[idx]
           if loader is None or len(loader) == 0:
               continue


           local_model = copy.deepcopy(global_model)


           trained_model, client_loss = client_update(
               local_model,
               loader,
               steps=J,
               lr=lr,
               device=device,
               momentum=momentum,
               weight_decay=weight_decay,
           )


           local_models.append(trained_model)
           weights.append(len(loader.dataset) if hasattr(loader, "dataset") else 1.0)
           losses.append(client_loss)


       if local_models:
           global_model = fed_avg_aggregate(global_model, local_models, weights)


       # log
       if (r % log_every == 0) or (r == 1) or (r == rounds):
           test_acc = evaluate_accuracy(global_model, test_loader, device)
           avg_loss = float(np.mean(losses)) if len(losses) > 0 else float("nan")


           history["rounds"].append(r)
           history["test_acc"].append(test_acc)
           history["train_loss"].append(avg_loss)


           print(f"Round {r:03d} | Test Acc: {test_acc:6.2f}% | Avg client loss: {avg_loss:.4f}")


   return history, global_model