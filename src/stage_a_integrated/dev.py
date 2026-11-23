"""Frozen 147-key DEV selector proposal. Not DS-frozen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import HORIZONS_SHA256

BETA = 0.05


def load_rows(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HORIZONS_SHA256:
        raise ValueError("horizons SHA mismatch")
    rows = json.loads(raw.decode("utf-8"))["rows"]
    if len(rows) != 50:
        raise ValueError("dev manifest is not 50 videos")
    return rows


def load_keys(path: Path) -> set[tuple]:
    rows = load_rows(path)
    keys = []
    for row in rows:
        for horizon in ("near", "mid", "far"):
            if row.get(horizon) is None:
                continue
            keys.append((str(row["id_movi"]), int(row["current"]), int(row[horizon]), horizon))
    if len(keys) != 147 or len(set(keys)) != 147 or len({key[0] for key in keys}) != 50:
        raise ValueError("dev support is not 147 pairs over 50 videos")
    return set(keys)


def evaluate(model, items: list[dict], expected: set[tuple]) -> float:
    import torch

    seen = set()
    grouped: dict[str, list[float]] = {}
    model.eval()
    with torch.no_grad():
        for item in items:
            key = (item["id_movi"], item["current"], item["target"], item["horizon"])
            if key not in expected or key in seen:
                raise ValueError("dev pair is not in the frozen 147")
            seen.add(key)
            output = model.forward(**item["model_kwargs"])
            local = torch.nn.functional.smooth_l1_loss(output["delta_local"], item["supervision"]["target_delta_local"], beta=BETA)
            anchor = torch.nn.functional.smooth_l1_loss(output["delta_anchor"], item["supervision"]["target_delta_anchor"], beta=BETA)
            if not bool(torch.isfinite(local)) or not bool(torch.isfinite(anchor)):
                raise ValueError("nonfinite dev")
            grouped.setdefault(item["id_movi"], []).append(float(0.5 * (local + anchor)))
    if seen != expected or len(grouped) != 50:
        raise ValueError("dev coverage mismatch")
    means = [sum(values) / len(values) for values in grouped.values()]
    value = sum(means) / 50.0
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("nonfinite dev macro")
    return value
