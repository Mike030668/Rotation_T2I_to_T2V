"""SYNTHETIC_APPARATUS. Not a frozen historical replay. No corpus read."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import SAMPLER_SHA256
from .samples import synthetic_item

SYNTHETIC_SOURCE_ID = "SYNTHETIC_APPARATUS_400_V1"
SYNTHETIC_TRAIN_SIZE = 400
SYNTHETIC_TRAIN_IDS_SHA256 = "76f17184b06cb65936befc25f2661fd2907f4d464c2f88b2a67782452ae68684"
REMAINDER_POLICY = "rollover_before_overflow"
PINNED_SYNTHETIC_LABELS = (0, 2, 9, 10, 40)
DEV_TEMPORAL_AXIS_SEMANTICS = "SYNTHETIC_ORDINAL_0_TO_N_MINUS_1_NOT_HISTORICAL_IDS_FRAMES"


def require_exact_int(value, name: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def verify_sampler_pin() -> str:
    path = Path(__file__).with_name("stage_a_sampler_v2.py")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SAMPLER_SHA256:
        raise ValueError("frozen sampler SHA mismatch")
    return digest


def verify_train_ids_pin(ids: list[str]) -> str:
    if type(ids) is not list or len(ids) != SYNTHETIC_TRAIN_SIZE:
        raise ValueError("synthetic train_ids must be a 400-element list")
    if any(type(item) is not str or not item for item in ids):
        raise ValueError("synthetic train_ids entries must be nonempty strings")
    encoded = json.dumps(ids, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != SYNTHETIC_TRAIN_IDS_SHA256:
        raise ValueError("synthetic train_ids SHA mismatch")
    return digest


def train_ids() -> list[str]:
    ids = [f"syn-{index:04d}" for index in range(SYNTHETIC_TRAIN_SIZE)]
    verify_train_ids_pin(ids)
    return ids


def advance_window(
    epoch: int,
    cursor: int,
    batch: int,
    train_size: int = SYNTHETIC_TRAIN_SIZE,
) -> tuple[int, int, int]:
    epoch_val = require_exact_int(epoch, "epoch", minimum=0)
    size_val = require_exact_int(
        train_size,
        "train_size",
        minimum=SYNTHETIC_TRAIN_SIZE,
        maximum=SYNTHETIC_TRAIN_SIZE,
    )
    batch_val = require_exact_int(batch, "batch", minimum=1, maximum=size_val)
    cursor_val = require_exact_int(cursor, "cursor", minimum=0, maximum=size_val)
    if cursor_val + batch_val > size_val:
        next_epoch = epoch_val + 1
        cursor_before = 0
    else:
        next_epoch = epoch_val
        cursor_before = cursor_val
    cursor_after = cursor_before + batch_val
    return next_epoch, cursor_before, cursor_after


def select_video_ids(seed: int, epoch: int, cursor_before: int, batch: int) -> list[str]:
    seed_val = require_exact_int(seed, "seed", minimum=0)
    epoch_val = require_exact_int(epoch, "epoch", minimum=0)
    batch_val = require_exact_int(batch, "batch", minimum=1, maximum=SYNTHETIC_TRAIN_SIZE)
    before_val = require_exact_int(cursor_before, "cursor_before", minimum=0, maximum=SYNTHETIC_TRAIN_SIZE - 1)
    if before_val + batch_val > SYNTHETIC_TRAIN_SIZE:
        raise ValueError("cursor window exceeds synthetic train size")
    ids = train_ids()
    return [ids[(seed_val + epoch_val + before_val + offset) % len(ids)] for offset in range(batch_val)]


def next_batch(seed: int, epoch: int, cursor_before: int, batch: int) -> tuple[list[dict], int, list[str]]:
    chosen = select_video_ids(seed, epoch, cursor_before, batch)
    sampler_digest = verify_sampler_pin()
    from .stage_a_sampler_v2 import sample

    labels = list(PINNED_SYNTHETIC_LABELS)
    items = []
    for video in chosen:
        ordinals, history = sample(labels, video, epoch, seed)
        current_ord, target_ord = ordinals[-2], ordinals[-1]
        current = labels[current_ord]
        target = labels[target_ord]
        selected_ordinals = list(history[current_ord])
        selected = [labels[index] for index in selected_ordinals]
        item = synthetic_item(video, labels, selected, current, target, "near")
        item["selected_ordinals"] = selected_ordinals
        item["selected_labels"] = selected
        item["current_ordinal"] = current_ord
        item["target_ordinal"] = target_ord
        item["epoch"] = epoch
        item["seed"] = seed
        item["provider"] = "SYNTHETIC_APPARATUS"
        item["source_id"] = SYNTHETIC_SOURCE_ID
        item["source_sha256"] = SYNTHETIC_TRAIN_IDS_SHA256
        item["sampler_sha256"] = sampler_digest
        item["temporal_axis_semantics"] = "SYNTHETIC_PINNED_LABELS_0_2_9_10_40"
        items.append(item)
    return items, cursor_before + batch, chosen


def dev_items(rows: list[dict]) -> list[dict]:
    if type(rows) is not list or len(rows) != 50:
        raise ValueError("dev_items requires exact 50 manifest rows")
    items = []
    for row in rows:
        if type(row) is not dict:
            raise ValueError("dev_items row must be a dict")
        id_movi = row.get("id_movi")
        if type(id_movi) is not str or not id_movi:
            raise ValueError("dev_items row.id_movi must be a nonempty str")
        n_frames = require_exact_int(row.get("n"), "row.n", minimum=2)
        labels = list(range(n_frames))
        raw_history = row.get("history")
        if type(raw_history) is not list or not raw_history:
            raise ValueError("dev_items row.history must be a nonempty list")
        history = [require_exact_int(frame, "row.history", minimum=0, maximum=n_frames - 1) for frame in raw_history]
        current = require_exact_int(row.get("current"), "row.current", minimum=0, maximum=n_frames - 2)
        for horizon in ("near", "mid", "far"):
            if row.get(horizon) is None:
                continue
            target = require_exact_int(row[horizon], f"row.{horizon}", minimum=current + 1, maximum=n_frames - 1)
            item = synthetic_item(id_movi, labels, history, current, target, horizon)
            item["provider"] = "SYNTHETIC_APPARATUS_DEV"
            item["temporal_axis_semantics"] = DEV_TEMPORAL_AXIS_SEMANTICS
            items.append(item)
    if len(items) != 147:
        raise ValueError("synthetic dev provider is not 147 pairs")
    return items
