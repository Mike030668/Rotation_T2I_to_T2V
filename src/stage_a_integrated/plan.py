"""SYNTHETIC_APPARATUS. Not a frozen historical replay. No corpus read."""

from __future__ import annotations

from .samples import synthetic_item
from .stage_a_sampler_v2 import sample


def train_ids() -> list[str]:
    return [f"syn-{index:04d}" for index in range(400)]


def next_batch(seed: int, epoch: int, cursor: int, batch: int) -> tuple[list[dict], int, list[str]]:
    if batch < 1:
        raise ValueError("batch must be positive")
    ids = train_ids()
    chosen = [ids[(seed + epoch + cursor + offset) % len(ids)] for offset in range(batch)]
    items = []
    for video in chosen:
        labels = [0, 2, 9, 10, 40]
        ordinals, history = sample(labels, video, epoch, seed)
        current_ord, target_ord = ordinals[-2], ordinals[-1]
        current = labels[current_ord]
        target = labels[target_ord]
        selected = [labels[index] for index in history[current_ord]]
        items.append(synthetic_item(video, labels, selected, current, target, "near"))
    return items, cursor + batch, chosen


def dev_items(rows: list[dict]) -> list[dict]:
    items = []
    for row in rows:
        labels = list(range(int(row["n"])))
        history = [int(frame) for frame in row["history"]]
        current = int(row["current"])
        for horizon in ("near", "mid", "far"):
            if row.get(horizon) is None:
                continue
            items.append(synthetic_item(str(row["id_movi"]), labels, history, current, int(row[horizon]), horizon))
    if len(items) != 147:
        raise ValueError("synthetic dev provider is not 147 pairs")
    return items
