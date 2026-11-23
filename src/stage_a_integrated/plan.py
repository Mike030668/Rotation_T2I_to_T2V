"""Synthetic 400-id exposure. Does not read the historical corpus."""

from __future__ import annotations

from .samples import synthetic_item


def train_ids() -> list[str]:
    return [str(index) for index in range(400)]


def next_batch(seed: int, epoch: int, cursor: int, batch: int) -> tuple[list[dict], int]:
    ids = train_ids()
    start = (seed + epoch + cursor) % len(ids)
    chosen = [ids[(start + offset) % len(ids)] for offset in range(batch)]
    items = [synthetic_item(video, [0, 2, 9, 10, 40], [0, 2], 9, 10, "near") for video in chosen]
    return items, cursor + batch


def dev_items(keys: set[tuple]) -> list[dict]:
    items = []
    for id_movi, current, target, horizon in sorted(keys):
        labels = sorted({0, int(current), int(target)})
        history = [label for label in labels if label <= int(current)]
        items.append(synthetic_item(str(id_movi), labels, history, int(current), int(target), horizon))
    if len(items) != 147:
        raise ValueError("synthetic dev provider is not 147 pairs")
    return items
