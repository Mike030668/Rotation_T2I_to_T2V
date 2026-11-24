"""One update with C3/C4 pre-forward receipt-bound digest and supervision verification."""

from __future__ import annotations

import math

from .forward_binding import prepare_verified_forward_call
from .objective import load_criterion
from .samples import assert_no_target_leak


def optimizer_step(model, items: list[dict], opt, clip: float, *, expected_forward_digests: list[str]) -> int:
    import torch

    if type(items) is not list or not items:
        raise ValueError("optimizer_step requires a nonempty batch list")
    if type(expected_forward_digests) is not list or len(expected_forward_digests) != len(items):
        raise ValueError("expected_forward_digests length does not match batch")
    if type(clip) not in (int, float) or isinstance(clip, bool) or not math.isfinite(float(clip)) or float(clip) <= 0.0:
        raise ValueError("clip must be a finite positive number")

    verified_calls: list[tuple[dict, dict]] = []
    for item, exp_digest in zip(items, expected_forward_digests):
        frozen_kwargs = prepare_verified_forward_call(
            item,
            expected_digest=exp_digest,
            context="train",
            verify_canonical_supervision=True,
        )
        assert_no_target_leak(item, verify_canonical_oracle=True)
        verified_calls.append((frozen_kwargs, item["supervision"]))

    criterion = load_criterion()
    opt.zero_grad(set_to_none=True)
    total = None
    for frozen_kwargs, supervision in verified_calls:
        loss = criterion(model.forward(**frozen_kwargs), supervision)["total"]
        if not bool(torch.isfinite(loss).all().item()):
            raise ValueError("nonfinite training loss")
        total = loss if total is None else total + loss
    (total / len(verified_calls)).backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
    if not bool(torch.isfinite(grad).all().item()):
        raise ValueError("nonfinite gradient")
    opt.step()
    return 1
