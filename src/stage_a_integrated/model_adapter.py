"""Pinned model loader. The copied file must keep the frozen SHA256."""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import CANDIDATE_SHA256


def load_class():
    path = Path(__file__).with_name("pinned_splitter_v3.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != CANDIDATE_SHA256:
        raise ValueError("candidate SHA mismatch")
    from . import pinned_splitter_v3 as module

    return module.SplitterV3, module.SplitterV3Config
