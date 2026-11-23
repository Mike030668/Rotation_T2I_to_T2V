"""Pinned prepared loss. Training uses total. Selector does not replace it."""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import LOSS_SHA256


def load_criterion():
    path = Path(__file__).with_name("pinned_losses_v3.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != LOSS_SHA256:
        raise ValueError("prepared loss SHA mismatch")
    from . import pinned_losses_v3 as module

    return module.SplitterV3Loss(module.LossConfig())
