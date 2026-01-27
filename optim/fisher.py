from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Callable, List

import math
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
    """
    model.train()
    scores = _init_like_named_params(model, fill_value=0.0)

    if param_filter is None:
        def param_filter(name: str, p: torch.Tensor) -> bool:
            return bool(p.requires_grad)

    used = 0
    for b_idx, batch in enumerate(dataloader):
        if max_batches is not None and b_idx >= max_batches:
            break

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

    for name in list(scores.keys()):
        scores[name].div_(float(used))
    return scores


@dataclass
class MaskCalibrationConfig:
    trainable_fraction: float
    rounds: int
    fisher_batches_per_round: int
    rule: str = "least_sensitive"  # {"least_sensitive","most_sensitive","lowest_magnitude","highest_magnitude","random"}


def _flatten_scores(
    scores: ScoreDict,
    eligible: MaskDict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    flat_scores_parts: List[torch.Tensor] = []
    flat_param_parts: List[torch.Tensor] = []
    flat_local_parts: List[torch.Tensor] = []
    param_names: List[str] = []
    device: Optional[torch.device] = None

    for name, s in scores.items():
        if name not in eligible:
            continue
        if device is None:
            device = s.device

        m = eligible[name]
        m_bool = m if m.dtype == torch.bool else m.bool()
        if not m_bool.any():
            continue

        local_idx = m_bool.view(-1).nonzero(as_tuple=False).view(-1)
        flat_s = s.view(-1).index_select(0, local_idx)

        pid = len(param_names)
        param_names.append(name)
        flat_scores_parts.append(flat_s)
        flat_local_parts.append(local_idx)
        flat_param_parts.append(
            torch.full((local_idx.numel(),), pid, device=local_idx.device, dtype=torch.long)
        )

    if len(flat_scores_parts) == 0:
        empty_device = device if device is not None else torch.device("cpu")
        empty = torch.empty(0, device=empty_device)
        return empty, empty.to(dtype=torch.long), empty.to(dtype=torch.long), []

    return (
        torch.cat(flat_scores_parts, dim=0),
        torch.cat(flat_param_parts, dim=0),
        torch.cat(flat_local_parts, dim=0),
        param_names,
    )


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
    TaLoS-like multi-round calibration:
      - selected grows by union across rounds
      - each round recomputes scores on CURRENT model
      - selection uses a quantile/threshold over eligible scores, then trims to exact k
      - per-round budget is ceil(remaining / remaining_rounds)
    """
    assert 0.0 < cfg.trainable_fraction <= 1.0
    assert cfg.rounds >= 1

    model = model.to(device)

    if param_filter is None:
        def param_filter(name: str, p: torch.Tensor) -> bool:
            return bool(p.requires_grad)

    # init masks
    selected: MaskDict = {}
    eligible: MaskDict = {}
    total_params = 0

    for name, p in model.named_parameters():
        if not p.requires_grad or not param_filter(name, p):
            continue
        selected[name] = torch.zeros_like(p, dtype=torch.bool, device=p.device)
        eligible[name] = torch.ones_like(p, dtype=torch.bool, device=p.device)
        total_params += p.numel()

    k_trainable = int(round(cfg.trainable_fraction * total_params))
    k_trainable = max(1, min(k_trainable, total_params))

    for r in range(cfg.rounds):
        already = sum(int(m.sum().item()) for m in selected.values())
        remaining = k_trainable - already
        if remaining <= 0:
            break

        remaining_rounds = cfg.rounds - r
        add_this_round = int(math.ceil(remaining / float(remaining_rounds)))
        add_this_round = max(1, min(add_this_round, remaining))

        # scores depending on rule
        if cfg.rule in ("least_sensitive", "most_sensitive"):
            scores = compute_fisher_diag_scores(
                model, dataloader, criterion, device,
                max_batches=cfg.fisher_batches_per_round,
                param_filter=param_filter,
            )
            pick_small = (cfg.rule == "least_sensitive")

        elif cfg.rule in ("lowest_magnitude", "highest_magnitude"):
            scores = {}
            for name, p in model.named_parameters():
                if name in eligible:
                    scores[name] = p.detach().abs()
            pick_small = (cfg.rule == "lowest_magnitude")

        elif cfg.rule == "random":
            scores = {name: torch.rand_like(m.float()) for name, m in eligible.items()}
            pick_small = True

        else:
            raise ValueError(f"Unknown rule: {cfg.rule}")

        flat, flat_param_ids, flat_local_ids, param_names = _flatten_scores(scores, eligible)
        if flat.numel() == 0:
            break

        # ---- TaLoS-like: thresholding via quantile, then trim to exact budget ----
        # We want approx "add_this_round" elements selected among eligible in this round.
        k = min(add_this_round, flat.numel())

        if pick_small:
            # select those <= q where q is k-th smallest (quantile)
            # Use kthvalue for determinism.
            kth = torch.kthvalue(flat, k).values
            cand = (flat <= kth)
        else:
            # select those >= kth largest
            kth = torch.kthvalue(flat, flat.numel() - k + 1).values
            cand = (flat >= kth)

        cand_idx = cand.nonzero(as_tuple=False).view(-1)

        # If threshold yields too many (ties), trim deterministically by sorting key
        if cand_idx.numel() > k:
            cand_scores = flat.index_select(0, cand_idx)
            # stable-ish tie-break: add tiny index-based perturbation
            if cand_scores.dtype.is_floating_point:
                eps = torch.finfo(cand_scores.dtype).eps
                max_abs = cand_scores.detach().abs().max()
                tiny = (eps * (max_abs + 1.0)) / float(cand_scores.numel() + 1)
                idx = cand_idx.to(dtype=cand_scores.dtype)
                key = cand_scores + idx * tiny if pick_small else cand_scores - idx * tiny
            else:
                key = cand_scores

            # pick best k among candidates
            # For pick_small: smallest keys; else: largest keys
            sel_local = torch.topk(key, k=k, largest=not pick_small).indices
            sel_global = cand_idx.index_select(0, sel_local)
        else:
            # If too few due to weird distribution (rare), fall back to topk on all
            if cand_idx.numel() < k:
                # direct topk (same as your original)
                largest = not pick_small
                sel_global = torch.topk(flat, k=k, largest=largest).indices
            else:
                sel_global = cand_idx  # exact match

        # map back
        sel_param_ids = flat_param_ids.index_select(0, sel_global)
        sel_local_ids = flat_local_ids.index_select(0, sel_global)

        # update masks (union)
        for pid in sel_param_ids.unique():
            pid_int = int(pid.item())
            name = param_names[pid_int]
            mask = (sel_param_ids == pid)
            local_idx = sel_local_ids[mask]
            selected[name].view(-1)[local_idx] = True
            eligible[name].view(-1)[local_idx] = False

    # final float mask for grad-multiplication
    final_mask: MaskDict = {k: v.to(dtype=torch.float32) for k, v in selected.items()}
    return final_mask
