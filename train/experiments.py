import os
import time
import csv
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR
from train.utils import make_scheduler, set_seed
import torch
import timm

from models.vit_dino import build_dino_vit as create_dino_and_define_train_mode
from data.datasets import get_cifar100, get_cifar100_transforms
from data.partition import make_dataset_loaders
from fl.dataloaders import build_federated_dataloaders
from train.eval import evaluate
from utils.plots import plot_test_curves
from train.trainer import train_one_epoch_amp
from train.utils import make_scheduler, set_seed




import timm


def run_trial(
    scheduler_name: str,
    train_loader,
    val_loader,
    test_loader,
    *,
    model_name="vit_small_patch16_224",
    img_size=160,
    num_classes=100,
    epochs=15,
    lr=0.03,
    momentum=0.9,
    weight_decay=5e-4,
    seed=0,
    device=None,
    out_dir="runs_sched",
    log_every=100,
    resume=False
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"best_{scheduler_name}_seed{seed}.pth")
    last_ckpt_path = os.path.join(out_dir, f"last_{scheduler_name}_seed{seed}.pth")


    model = create_dino_and_define_train_mode(num_classes=num_classes, img_size=img_size, device=device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    scheduler = make_scheduler(scheduler_name, optimizer, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))

    #best_val_acc = -1.0
    #best_epoch = -1
    resume_path = os.path.join(out_dir, f"last_{scheduler_name}_seed{seed}.pth")
    start_epoch = 1
    best_val_acc = -1.0

    if resume and os.path.exists(resume_path):
        start_epoch, best_val_acc = load_last_ckpt(
            resume_path, model, optimizer, scheduler, scaler, device
        )
        print(f"[{scheduler_name}] Resuming from epoch {start_epoch} (best_val_acc={best_val_acc:.4f})")

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": []
    }

    t_run0 = time.time()
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n[{scheduler_name}] Epoch {epoch:03d}/{epochs} ------------------------------")
        tr_loss, tr_acc = train_one_epoch_amp(
            model, train_loader, optimizer, criterion, device, scaler, log_every=log_every
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(cur_lr)

        print(
            f"[{scheduler_name}] epoch {epoch:03d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"lr {cur_lr:.6f}"
        )
        torch.save(
          {
              "model_state": model.state_dict(),
              "epoch": epoch,
              "best_val_acc": best_val_acc,
              "scheduler": scheduler_name,
              "lr": lr,
              "momentum": momentum,
              "weight_decay": weight_decay,
              "img_size": img_size,
              "model_name": model_name,
              "seed": seed,
              "optimizer_state": optimizer.state_dict(),
              "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
              "scaler_state": scaler.state_dict(),
          },
          last_ckpt_path
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "scheduler": scheduler_name,
                    "lr": lr,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                    "img_size": img_size,
                    "model_name": model_name,
                    "seed": seed,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "scaler_state": scaler.state_dict()
                },
                ckpt_path
            )

    run_secs = time.time() - t_run0

    # Test com o melhor checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    result = {
        "scheduler": scheduler_name,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "img_size": img_size,
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "run_minutes": run_secs / 60.0,
        "ckpt_path": ckpt_path,
    }
    return result, history

def save_last_ckpt(path, model, optimizer, scheduler, scaler, epoch, best_val_acc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
    }, path)

def load_last_ckpt(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt["scheduler_state"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt["scaler_state"] is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    start_epoch = ckpt["epoch"] + 1
    best_val_acc = ckpt.get("best_val_acc", -1.0)
    return start_epoch, best_val_acc

# -----------------------------
# FASE 0: sanity check
# -----------------------------
def phase0_sanity(
    train_loader,
    val_loader,
    test_loader,
    *,
    epochs=2,
    lr=0.01,
    weight_decay=5e-4,
    img_size=160,
    seed=0,
    device=None
):
    print("=== PHASE 0 (sanity) ===")
    res, _ = run_trial(
        scheduler_name="cosine",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        img_size=img_size,
        seed=seed,
        device=device,
        out_dir="phase0_sanity",
        log_every=50,
    )
    print("\nPHASE 0 result:")
    for k, v in res.items():
        if k != "ckpt_path":
            print(f"  {k}: {v}")
    print(f"  ckpt_path: {res['ckpt_path']}")
    return res


# -----------------------------
# FASE 1: scheduler sweep
# -----------------------------
def phase1_scheduler_sweep(
    train_loader,
    val_loader,
    test_loader,
    *,
    schedulers=("cosine", "step", "multistep", "none"),
    epochs=15,
    lr=0.03,
    weight_decay=5e-4,
    momentum=0.9,
    img_size=160,
    seed=0,
    device=None,
    out_dir="phase1_schedulers",
    save_csv_path="phase1_results.csv",
):
    print("\n=== PHASE 1 (scheduler sweep) ===")
    results = []
    histories = {}

    for sch in schedulers:
        res, hist = run_trial(
            scheduler_name=sch,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            img_size=img_size,
            seed=seed,
            device=device,
            out_dir=out_dir,
            log_every=100,
        )
        results.append(res)
        histories[sch] = hist

    # ordena pelo melhor val acc
    results_sorted = sorted(results, key=lambda d: d["best_val_acc"], reverse=True)

    print("\nPHASE 1 summary (sorted by best_val_acc):")
    for r in results_sorted:
        print(
            f"  {r['scheduler']:9s} | best_val_acc {r['best_val_acc']:.4f} (epoch {r['best_epoch']})"
            f" | test_acc {r['test_acc']:.4f} | minutes {r['run_minutes']:.1f}"
        )

    # salva CSV
    with open(save_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_sorted[0].keys()))
        writer.writeheader()
        for r in results_sorted:
            writer.writerow(r)

    print(f"\nSaved: {save_csv_path}")
    return results_sorted, histories



def phase2_lr_wd_grid(
    train_loader,
    val_loader,
    test_loader,
    *,
    scheduler_name="cosine",          # vencedor da fase 1
    epochs=15,                        # mantenha igual ao da fase 1 para comparar rápido
    lr_list=(0.003, 0.005, 0.01, 0.02),
    wd_list=(1e-4, 3e-4, 5e-4, 1e-3),
    momentum=0.9,
    img_size=160,
    seed=0,
    device=None,
    out_dir="phase2_grid",
    save_csv_path="phase2_results.csv",
    top_k=5,
    resume=False
):
    """
    Roda grid search em LR x WD mantendo scheduler fixo.
    Retorna lista ordenada (melhor -> pior) por best_val_acc.
    """
    results = []

    for lr in lr_list:
        for wd in wd_list:
            print(f"\n=== PHASE 2 trial: scheduler={scheduler_name} lr={lr} wd={wd} ===")
            trial_dir=os.path.join(out_dir, f"{scheduler_name}_lr{lr}_wd{wd}_seed{seed}")
            res, _ = run_trial(
                scheduler_name=scheduler_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=epochs,
                lr=float(lr),
                momentum=momentum,
                weight_decay=float(wd),
                img_size=img_size,
                seed=seed,
                device=device,
                out_dir=trial_dir,
                log_every=100,
                resume=resume
            )

            # se algum trial der NaN, joga pro fim
            if res["best_val_acc"] != res["best_val_acc"]:  # NaN check
                res["best_val_acc"] = -1.0

            results.append(res)

    results_sorted = sorted(results, key=lambda d: d["best_val_acc"], reverse=True)

    print("\nPHASE 2 summary (Top configs):")
    for r in results_sorted[:top_k]:
        print(
            f"  lr={r['lr']:<7} wd={r['weight_decay']:<8} | "
            f"best_val_acc {r['best_val_acc']:.4f} (epoch {r['best_epoch']}) | "
            f"test_acc {r['test_acc']:.4f}"
        )

    # salva CSV
    with open(save_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_sorted[0].keys()))
        writer.writeheader()
        for r in results_sorted:
            writer.writerow(r)

    print(f"\nSaved: {save_csv_path}")
    return results_sorted