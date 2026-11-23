"""One update. completed increments only after AdamW.step."""

from __future__ import annotations

from .objective import load_criterion
from .samples import assert_no_target_leak


def optimizer_step(model, items: list[dict], opt, clip: float) -> int:
    import torch

    criterion = load_criterion()
    opt.zero_grad(set_to_none=True)
    total = None
    for item in items:
        assert_no_target_leak(item)
        loss = criterion(model.forward(**item["model_kwargs"]), item["supervision"])["total"]
        total = loss if total is None else total + loss
    (total / len(items)).backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if not bool(torch.isfinite(grad)):
        raise ValueError("nonfinite gradient")
    opt.step()
    return 1
