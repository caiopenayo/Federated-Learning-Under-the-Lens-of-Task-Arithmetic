import torch
import torch.nn as nn

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        loss_sum += loss.item() * bs
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += bs
    return loss_sum / total, correct / total