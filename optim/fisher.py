from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Callable, List

import torch
import torch.nn as nn


MaskDict = Dict[str, torch.Tensor]
ScoreDict = Dict[str, torch.Tensor]


@torch.no_grad()
def _init_like_named_params(model: nn.Module, fill_value: float = 0.0) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            out[name] = torch.full_like(p, fill_value, device=p.device)
    return out


def compute_fisher_diag_scores(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
    param_filter: Optional[Callable[[str, torch.Tensor], bool]] = None,
) -> ScoreDict:
    """
    Approx diagonal Fisher via empirical E[grad^2] over batches.

    Returns:
        scores[name] same shape as parameter tensor.
    """
    model.train()  # grads need to flow
    scores = _init_like_named_params(model, fill_value=0.0)

    if param_filter is None:
        def param_filter(name: str, p: torch.Tensor) -> bool:
            return p.requires_grad

    # count batches actually used
    used = 0

    for b_idx, batch in enumerate(dataloader):
        if max_batches is not None and b_idx >= max_batches:
            break

        # Expect either (x, y) or dict-like; adapt if needed
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Batch format not supported. Expected (x, y).")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # accumulate grad^2
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if not param_filter(name, p):
                continue
            if p.grad is None:
                continue
            scores[name].add_(p.grad.detach() ** 2)

        used += 1

    if used == 0:
        raise RuntimeError("No batches were used to compute Fisher scores.")

    # average
    for name in list(scores.keys()):
        scores[name].div_(float(used))

    return scores


@dataclass
class MaskCalibrationConfig:
    target_sparsity: float            # fraction of parameters to UPDATE (mask==1), e.g., 0.1
    rounds: int                       # calibration rounds (multi-round)
    fisher_batches_per_round: int     # how many batches used per round for fisher estimation
    # if True: pick least-sensitive; for guided extension you can swap rule
    rule: str = "least_sensitive"     # {"least_sensitive","most_sensitive","lowest_magnitude","highest_magnitude","random"}


def _flatten_scores(scores: ScoreDict, eligible: MaskDict) -> Tuple[torch.Tensor, List[Tuple[str, torch.Size]]]:
    """
    Flatten eligible scores into one vector.
    Returns (flat_scores, meta) where meta stores (name, shape) in order.
    """
    flats = []
    meta: List[Tuple[str, torch.Size]] = []
    for name, s in scores.items():
        if name not in eligible:
            continue
        # eligible==1 means "still selectable"
        m = eligible[name]
        if m.dtype != torch.bool:
            m_bool = m.bool()
        else:
            m_bool = m
        if m_bool.any():
            flats.append(s[m_bool].reshape(-1))
            meta.append((name, s.shape))
    if len(flats) == 0:
        return torch.empty(0, device=next(iter(scores.values())).device), meta
    return torch.cat(flats, dim=0), meta


def calibrate_gradient_mask_multi_round(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    cfg: MaskCalibrationConfig,
    *,
    param_filter: Optional[Callable[[str, torch.Tensor], bool]] = None,
) -> MaskDict:
    """
    Produces a binary mask per-parameter tensor, where 1 means "allow updates", 0 means "freeze".

    Multi-round strategy:
    - Maintain 'selected' set (mask==1) growing each round
    - Maintain 'eligible' set among remaining parameters to be selected in future rounds
    """
    assert 0.0 < cfg.target_sparsity <= 1.0, "target_sparsity must be in (0,1]."
    assert cfg.rounds >= 1, "rounds must be >= 1."

    model = model.to(device)

    if param_filter is None:
        def param_filter(name: str, p: torch.Tensor) -> bool:
            # Typical default: skip classifier head? (optional)
            return p.requires_grad

    # masks:
    selected: MaskDict = {}
    eligible: MaskDict = {}

    total_params = 0
    for name, p in model.named_parameters():
        if not p.requires_grad or not param_filter(name, p):
            continue
        selected[name] = torch.zeros_like(p, dtype=torch.bool, device=p.device)
        eligible[name] = torch.ones_like(p, dtype=torch.bool, device=p.device)
        total_params += p.numel()

    target_k = int(round(cfg.target_sparsity * total_params))
    target_k = max(1, min(target_k, total_params))

    # how many to add per round (roughly equal split; last round adjusts)
    base_add = max(1, target_k // cfg.rounds)

    for r in range(cfg.rounds):
        already = sum(int(m.sum().item()) for m in selected.values())
        remaining_to_pick = target_k - already
        if remaining_to_pick <= 0:
            break

        add_this_round = base_add if r < cfg.rounds - 1 else remaining_to_pick
        add_this_round = min(add_this_round, remaining_to_pick)

        # Compute Fisher scores for current model
        scores = compute_fisher_diag_scores(
            model,
            dataloader,
            criterion,
            device,
            max_batches=cfg.fisher_batches_per_round,
            param_filter=param_filter,
        )

        # Flatten only eligible entries
        flat, _ = _flatten_scores(scores, eligible)
        if flat.numel() == 0:
            break

        # Decide which entries to select based on rule
        if cfg.rule == "least_sensitive":
            # pick smallest scores
            kth = min(add_this_round, flat.numel())
            thr = torch.kthvalue(flat, kth).values  # threshold among remaining
            # select <= thr; then trim if slight over-selection happens
            pick_small = True

        elif cfg.rule == "most_sensitive":
            kth = min(add_this_round, flat.numel())
            thr = torch.kthvalue(flat, flat.numel() - kth + 1).values
            pick_small = False

        elif cfg.rule in ("lowest_magnitude", "highest_magnitude"):
            # use |w| instead of fisher
            weights_scores: ScoreDict = {}
            for name, p in model.named_parameters():
                if name in eligible:
                    weights_scores[name] = p.detach().abs()
            flat, _ = _flatten_scores(weights_scores, eligible)
            kth = min(add_this_round, flat.numel())
            if cfg.rule == "lowest_magnitude":
                thr = torch.kthvalue(flat, kth).values
                pick_small = True
            else:
                thr = torch.kthvalue(flat, flat.numel() - kth + 1).values
                pick_small = False
            scores = weights_scores  # reuse downstream

        elif cfg.rule == "random":
            # random pick among eligible positions
            # implemented by generating random scores
            rand_scores: ScoreDict = {}
            for name, m in eligible.items():
                rand_scores[name] = torch.rand_like(m.float())
            scores = rand_scores
            flat, _ = _flatten_scores(scores, eligible)
            kth = min(add_this_round, flat.numel())
            thr = torch.kthvalue(flat, kth).values
            pick_small = True

        else:
            raise ValueError(f"Unknown rule: {cfg.rule}")

        # Apply selection back to per-parameter tensors
        picked_total = 0
        for name, s in scores.items():
            if name not in eligible:
                continue
            m = eligible[name]
            if pick_small:
                to_pick = (m & (s <= thr))
            else:
                to_pick = (m & (s >= thr))

            # trim if we overshoot
            if to_pick.any():
                idx = to_pick.view(-1).nonzero(as_tuple=False).view(-1)
                need = add_this_round - picked_total
                if need <= 0:
                    break
                if idx.numel() > need:
                    idx = idx[:need]
                # update masks
                flat_sel = selected[name].view(-1)
                flat_elig = eligible[name].view(-1)
                flat_sel[idx] = True
                flat_elig[idx] = False
                picked_total += idx.numel()

        # If threshold selection undershoots (possible with many equal values), fill randomly
        if picked_total < add_this_round:
            deficit = add_this_round - picked_total
            # collect all remaining eligible positions
            rem = []
            for name, m in eligible.items():
                if m.any():
                    idx = m.view(-1).nonzero(as_tuple=False).view(-1)
                    rem.append((name, idx))
            if rem:
                # sample without replacement across tensors
                # (simple: iterate until deficit is zero)
                for name, idx in rem:
                    if deficit <= 0:
                        break
                    take = min(deficit, idx.numel())
                    chosen = idx[torch.randperm(idx.numel(), device=idx.device)[:take]]
                    selected[name].view(-1)[chosen] = True
                    eligible[name].view(-1)[chosen] = False
                    deficit -= take

    # Convert bool mask to {0,1} float masks if you prefer multiplying grads
    final_mask: MaskDict = {k: v.to(dtype=torch.float32) for k, v in selected.items()}
    return final_mask
