from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR

import numpy as np
import torch
import torch.nn as nn

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def make_scheduler(name, optimizer, epochs):
    name = name.lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    if name == "step":
        step_size = max(1, int(0.7 * epochs))
        return StepLR(optimizer, step_size=step_size, gamma=0.1)
    if name == "multistep":
        m1 = max(1, int(0.6 * epochs))
        m2 = max(m1 + 1, int(0.85 * epochs))
        return MultiStepLR(optimizer, milestones=[m1, m2], gamma=0.1)
    if name == "none":
        return None
    raise ValueError(f"Scheduler desconhecido: {name}")








@dataclass
class AverageMeter:
    val: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, v: float, n: int = 1) -> None:
        self.val = float(v)
        self.sum += float(v) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()

def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    *,
    criterion: Optional[nn.Module] = None,
    amp: bool = True,
    grad_clip_norm: Optional[float] = None,
    log_every: int = 0,
) -> Tuple[float, float]:
    """
    Train for a single epoch.

    Returns:
        (avg_loss, avg_acc_top1)
    """
    model.train()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    use_amp = amp and (device.startswith("cuda") and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()

        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))

        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        loss_meter.update(loss.item(), n=bs)
        acc_meter.update(accuracy_top1(logits.detach(), y), n=bs)

        if log_every and (step % log_every == 0):
            print(f"  step {step:04d}/{len(loader)} | loss {loss_meter.avg:.4f} | acc {acc_meter.avg:.4f}")

    return loss_meter.avg, acc_meter.avg


