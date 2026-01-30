from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer

MaskDict = Dict[str, torch.Tensor]


class SparseSGDM(Optimizer):
    """
    SGD with momentum / weight decay + gradient mask.
    mask: dict name->tensor or list aligned to params is possible; here we support:
      - mask as dict keyed by parameter id (id(p)) OR
      - mask as dict keyed by parameter name via set_named_mask(...)
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
        *,
        mask: Optional[Dict[int, torch.Tensor]] = None,  # keyed by id(param)
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError("Nesterov requires momentum > 0 and dampening == 0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )
        super().__init__(params, defaults)

        self._mask_by_id: Dict[int, torch.Tensor] = mask or {}

    def set_mask_by_param(self, param: torch.Tensor, mask_tensor: torch.Tensor) -> None:
        self._mask_by_id[id(param)] = mask_tensor

    def set_mask_from_named_params(self, named_params: Iterable[Tuple[str, torch.Tensor]], named_mask: MaskDict) -> None:
        """
        Helper if you have mask keyed by parameter name from calibrate_gradient_mask_multi_round.
        """
        for name, p in named_params:
            if name in named_mask:
                self._mask_by_id[id(p)] = named_mask[name].to(device=p.device, dtype=p.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                if maximize:
                    d_p = -d_p

                # weight decay (L2) as in SGD pseudo-code (add wd * p to grad) :contentReference[oaicite:3]{index=3}
                if weight_decay != 0.0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # apply mask: zero out entries with mask==0
                m = self._mask_by_id.get(id(p), None)
                if m is not None:
                    # ensure broadcastable and same dtype
                    d_p = d_p.mul(m)

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

                p.add_(d_p, alpha=-lr)

        return loss
