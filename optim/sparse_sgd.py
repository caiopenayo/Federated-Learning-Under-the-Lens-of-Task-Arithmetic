# optim/sparse_sgd.py
from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Union

import torch
from torch.optim.optimizer import Optimizer, required


MaskType = Union[
    torch.Tensor,
    Dict[torch.nn.Parameter, torch.Tensor],
    Dict[str, torch.Tensor],
    Iterable[torch.Tensor],
    Callable[[torch.nn.Parameter], torch.Tensor],
]


class SparseSGDM(Optimizer):
    """
    SGD with momentum (SGDM) + optional Nesterov + weight decay,
    extended to support a gradient mask.

    The mask is applied elementwise to the *gradient* before momentum/update:
        grad = grad * mask

    Accepted mask formats:
      1) dict[param] -> mask_tensor (same shape as param)
      2) dict[name]  -> mask_tensor (if you pass param names in param_groups via {'name': ...})
      3) list/tuple of mask tensors aligned with params order
      4) a single mask tensor (broadcastable) applied to all params (rare)
      5) callable: mask_fn(param) -> mask tensor

    Usage:
      opt = SparseSGDM(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
      opt.step(mask=mask_dict)

    Notes:
      - If mask entry is missing for a param, that param is updated normally.
      - If you want to freeze a param entirely, provide a zero mask for it.
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening = 0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    def _get_mask_for_param(self, p, group, mask: Optional[MaskType], mask_iter_state: dict) -> Optional[torch.Tensor]:
        if mask is None:
            return None

        # callable
        if callable(mask):
            m = mask(p)
            return m

        # dict by param
        if isinstance(mask, dict) and p in mask:
            return mask[p]

        # dict by name if provided
        if isinstance(mask, dict):
            name = group.get("name", None)
            if isinstance(name, str) and name in mask:
                return mask[name]

        # list/tuple aligned with params iteration
        if isinstance(mask, (list, tuple)):
            idx = mask_iter_state.get("idx", 0)
            mask_iter_state["idx"] = idx + 1
            if idx < len(mask):
                return mask[idx]
            return None

        # single tensor
        if torch.is_tensor(mask):
            return mask

        return None

    @torch.no_grad()
    def step(self, closure=None, *, mask: Optional[MaskType] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        mask_iter_state: dict = {"idx": 0}

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad

                if d_p.is_sparse:
                    raise RuntimeError("SparseSGDM does not support sparse gradients.")

                # weight decay (L2)
                if weight_decay != 0.0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # apply mask on gradient
                m = self._get_mask_for_param(p, group, mask, mask_iter_state)
                if m is not None:
                    if not torch.is_tensor(m):
                        raise TypeError("Mask must be a torch.Tensor (or dict/list/callable returning it).")
                    if m.device != d_p.device:
                        m = m.to(d_p.device)
                    d_p = d_p.mul(m)

                # momentum
                if momentum != 0.0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=(1.0 - dampening))

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # update
                p.add_(d_p, alpha=-lr)

        return loss
