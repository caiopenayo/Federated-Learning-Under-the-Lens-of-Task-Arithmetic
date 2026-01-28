from __future__ import annotations
import time

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from train.eval import evaluate
from train.utils import train_one_epoch


def train_one_epoch_amp(model, loader, optimizer, criterion, device, scaler, log_every=100):
    model.train()
    total, correct = 0, 0
    loss_sum = 0.0

    t0 = time.time()
    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        loss_sum += loss.item() * bs
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += bs

        if step == 1 or step % log_every == 0:
            elapsed = time.time() - t0
            print(f"  step {step:04d}/{len(loader)} | loss {loss_sum/total:.4f} acc {correct/total:.4f} | {elapsed:.1f}s")

    return loss_sum / total, correct / total

@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 0.03
    wd: float = 5e-4
    momentum: float = 0.9
    nesterov: bool = False

    scheduler_name: str = "cosine"  
    eta_min: float = 0.0  
    warmup_epochs: int = 5  

    amp: bool = True
    grad_clip_norm: Optional[float] = None
    print_every: int = 10
    seed: Optional[int] = None
