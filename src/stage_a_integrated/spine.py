"""One future-authorized synthetic path. Default CLI still raises."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from . import SAMPLER_SHA256
from .authorization import ExecutionNotAuthorized, load
from .checkpoints import pack, publish, unpack
from .dev import evaluate, load_keys, load_rows
from .exposure import (
    FORWARD_BINDING,
    MAX_SEED,
    SAMPLER_SCHEMA_VERSION,
    SCHEMA_VERSION,
    build_receipt_bytes,
    build_sampler_section_bytes,
    reconcile_checkpoint_state,
    require_record,
)
from .model_adapter import load_class
from .plan import (
    REMAINDER_POLICY,
    SYNTHETIC_SOURCE_ID,
    SYNTHETIC_TRAIN_IDS_SHA256,
    SYNTHETIC_TRAIN_SIZE,
    advance_window,
    dev_items,
    next_batch,
    require_exact_int,
    train_ids,
    verify_sampler_pin,
    verify_train_ids_pin,
)
from .trainer import optimizer_step


def _validate_run_config(config: dict) -> tuple[int, int, int, list[int], float]:
    if type(config) is not dict:
        raise ValueError("config must be a dict")
    max_updates = require_exact_int(config.get("max_updates"), "config.max_updates", minimum=1)
    batch_size = require_exact_int(config.get("batch"), "config.batch", minimum=1, maximum=SYNTHETIC_TRAIN_SIZE)
    seed = require_exact_int(config.get("seed"), "config.seed", minimum=0, maximum=MAX_SEED)
    raw_events = config.get("dev_events")
    if type(raw_events) is not list or any(type(event) is not int or event < 1 for event in raw_events):
        raise ValueError("config.dev_events must be a list of positive exact ints")
    dev_events = list(raw_events)
    raw_clip = config.get("clip")
    if type(raw_clip) not in (int, float) or not math.isfinite(float(raw_clip)) or float(raw_clip) <= 0.0:
        raise ValueError("config.clip must be a finite positive number")
    clip = float(raw_clip)
    if type(config.get("horizons_path")) is not str or not config["horizons_path"]:
        raise ValueError("config.horizons_path must be a nonempty string")
    pin_checks = {
        "sampler_sha256": SAMPLER_SHA256,
        "exposure_schema": SCHEMA_VERSION,
        "sampler_schema": SAMPLER_SCHEMA_VERSION,
        "synthetic_source_id": SYNTHETIC_SOURCE_ID,
        "synthetic_train_size": SYNTHETIC_TRAIN_SIZE,
        "synthetic_train_ids_sha256": SYNTHETIC_TRAIN_IDS_SHA256,
        "synthetic_remainder_policy": REMAINDER_POLICY,
        "forward_binding": FORWARD_BINDING,
    }
    for key, expected_val in pin_checks.items():
        if key in config and config[key] != expected_val:
            raise ValueError(f"config.{key} does not match pinned C1 contract")
    return max_updates, batch_size, seed, dev_events, clip


def run_synthetic(release_path: str, trust_anchor_path: str, expected: dict, config: dict) -> dict:
    release = load(release_path, trust_anchor_path, expected)
    if release.get("scope") != "stage_a_synthetic_preflight":
        raise ExecutionNotAuthorized("scope is not the synthetic preflight")
    max_updates, batch_size, seed, dev_events, clip = _validate_run_config(config)
    if max_updates > int(release.get("max_updates")):
        raise ExecutionNotAuthorized("config exceeds the released budget")
    sampler_digest = verify_sampler_pin()
    source_digest = verify_train_ids_pin(train_ids())
    import torch

    model_cls, config_cls = load_class()
    model = model_cls(config_cls())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    rows = load_rows(Path(config["horizons_path"]))
    keys = load_keys(Path(config["horizons_path"]))
    completed = 0
    epoch = 0
    cursor = 0
    last_cursor_before: int | None = None
    last_receipt_sha256: str | None = None
    best = None
    macros = {}
    events = []
    exposure_frames: list[bytes] = []
    if config.get("resume_bytes"):
        restored = resume(
            config["resume_bytes"],
            model,
            opt,
            {
                "seed": seed,
                "batch": batch_size,
                "source_id": SYNTHETIC_SOURCE_ID,
                "source_sha256": source_digest,
                "sampler_sha256": sampler_digest,
            },
        )
        completed = int(restored["completed"])
        epoch = int(restored["epoch"])
        cursor = int(restored["cursor"])
        last_cursor_before = restored["last_cursor_before"]
        last_receipt_sha256 = restored["last_receipt_sha256"]
        best = restored["best"]
        events = [int(event) for event in restored["events"]]
        macros = {int(step): float(value) for step, value in restored["macros"].items()}
        exposure_frames = list(restored["exposure_frames"])
    root = Path(release.get("output_root"))
    items = dev_items(rows)
    while completed < max_updates:
        step_epoch, cursor_before, expected_cursor_after = advance_window(epoch, cursor, batch_size)
        batch, cursor_after, chosen = next_batch(seed, step_epoch, cursor_before, batch_size)
        if cursor_after != expected_cursor_after:
            raise ValueError("next_batch cursor_after mismatch")
        completed_next = completed + 1
        proposed_bytes = build_receipt_bytes(
            batch,
            completed_next,
            step_epoch,
            cursor_before,
            cursor_after,
            seed,
            batch_size,
        )
        record, frozen_bytes = require_record(
            proposed_bytes,
            completed_next,
            expected_seed=seed,
            expected_batch=batch_size,
            expected_epoch=step_epoch,
            expected_cursor_before=cursor_before,
            expected_cursor_after=cursor_after,
            expected_source_id=SYNTHETIC_SOURCE_ID,
            expected_source_sha256=source_digest,
            expected_sampler_sha256=sampler_digest,
        )
        if record["videos"] != chosen:
            raise ValueError("receipt videos do not match next_batch selection")
        step_inc = optimizer_step(model, batch, opt, clip)
        if step_inc != 1:
            raise ValueError("optimizer_step did not complete one update")
        completed += step_inc
        epoch = step_epoch
        cursor = cursor_after
        last_cursor_before = cursor_before
        last_receipt_sha256 = record["receipt_sha256"]
        exposure_frames.append(frozen_bytes)
        if completed in dev_events:
            value = evaluate(model, items, keys)
            model.train()
            macros[completed] = value
            if best is None or value < best[1]:
                best = (completed, value)
            events.append(completed)
            history = [] if best is None else [{"step": best[0], "macro": best[1]}]
            exposure_journal_bytes = b"[" + b",".join(exposure_frames) + b"]"
            exposure_journal_sha256 = hashlib.sha256(exposure_journal_bytes).hexdigest()
            sampler_bytes = build_sampler_section_bytes(
                completed=completed,
                epoch=epoch,
                cursor=cursor,
                last_cursor_before=last_cursor_before,
                seed=seed,
                batch_size=batch_size,
                exposure_journal_bytes=exposure_journal_bytes,
                last_receipt_sha256=last_receipt_sha256,
            )
            payload = pack(
                {
                    "completed": completed,
                    "epoch": epoch,
                    "cursor": cursor,
                    "seed": seed,
                    "batch_size": batch_size,
                    "source_id": SYNTHETIC_SOURCE_ID,
                    "source_sha256": source_digest,
                    "sampler_sha256": sampler_digest,
                    "exposure_schema": SCHEMA_VERSION,
                    "exposure_journal_sha256": exposure_journal_sha256,
                    "last_receipt_sha256": last_receipt_sha256,
                    "macro": value,
                    "best_step": None if best is None else best[0],
                    "best_macro": None if best is None else best[1],
                    "best_history": history,
                    "events": events,
                    "macros": macros,
                },
                _sections(model, opt, sampler_bytes, exposure_journal_bytes),
            )
            publish(root, f"u{completed}", "LAST", payload)
            if best is not None and best[0] == completed:
                publish(root, f"u{completed}", "BEST", payload)
    return {
        "completed_updates": completed,
        "best_step": None if best is None else best[0],
        "last_step": completed,
        "macros": macros,
        "events": events,
        "observed": False,
    }


def _admit(decoded: dict) -> dict:
    if type(decoded) is not dict or type(decoded.get("header")) is not dict:
        raise ValueError("checkpoint payload missing header")
    header = decoded["header"]
    events = header.get("events")
    macros = header.get("macros")
    if type(events) is not list or type(macros) is not dict:
        raise ValueError("checkpoint header missing events or macros")
    if events != sorted(set(events)) or any(type(event) is not int or event < 1 for event in events):
        raise ValueError("event journal is not a strictly increasing positive int sequence")
    if any(type(step) is not str or not step.isdigit() or str(int(step)) != step for step in macros):
        raise ValueError("macro keys are not canonical decimal strings")
    if set(events) != {int(step) for step in macros}:
        raise ValueError("macro keys do not match event ids")
    for value in macros.values():
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError("nonfinite restored macro")
    numeric = {int(step): float(value) for step, value in macros.items()}
    completed = header.get("completed")
    if type(completed) is not int or completed < 0 or any(event > completed for event in events):
        raise ValueError("event is after completed step")
    best_step = header.get("best_step")
    best_macro = header.get("best_macro")
    if events and best_step is None:
        raise ValueError("best is missing while events exist")
    if not events and (best_step is not None or best_macro is not None):
        raise ValueError("best exists while events are empty")
    if best_step is not None:
        if type(best_step) is not int or type(best_macro) not in (int, float) or not math.isfinite(float(best_macro)):
            raise ValueError("best step or macro has invalid type")
        earliest = min(events, key=lambda step: (numeric[step], step))
        if best_step != earliest or float(best_macro) != numeric[earliest]:
            raise ValueError("best is not the earliest finite minimum")
    return header


def _numpy_state() -> list:
    import numpy

    algo, keys, pos, has_gauss, cached = numpy.random.get_state()
    return [algo, keys.tolist(), pos, has_gauss, cached]


def resume(payload: bytes, model, opt, expected_config: dict) -> dict:
    import io
    import json
    import random

    import numpy
    import torch

    if type(expected_config) is not dict:
        raise ValueError("expected_config must be a dict")
    expected_seed = require_exact_int(expected_config.get("seed"), "expected_config.seed", minimum=0, maximum=MAX_SEED)
    expected_batch = require_exact_int(
        expected_config.get("batch"),
        "expected_config.batch",
        minimum=1,
        maximum=SYNTHETIC_TRAIN_SIZE,
    )
    expected_source_id = expected_config.get("source_id", SYNTHETIC_SOURCE_ID)
    expected_source_sha256 = expected_config.get("source_sha256", SYNTHETIC_TRAIN_IDS_SHA256)
    expected_sampler_sha256 = expected_config.get("sampler_sha256", SAMPLER_SHA256)

    decoded = unpack(payload)
    header = _admit(decoded)
    reconciled = reconcile_checkpoint_state(
        header,
        decoded["sampler"],
        decoded["exposure"],
        expected_seed=expected_seed,
        expected_batch=expected_batch,
        expected_source_id=expected_source_id,
        expected_source_sha256=expected_source_sha256,
        expected_sampler_sha256=expected_sampler_sha256,
    )

    model.load_state_dict(torch.load(io.BytesIO(decoded["model"]), map_location="cpu", weights_only=True))
    opt.load_state_dict(torch.load(io.BytesIO(decoded["adamw"]), map_location="cpu", weights_only=True))
    version, state, gauss = json.loads(decoded["python_rng"].decode("utf-8"))
    random.setstate((version, tuple(state), gauss))
    algo, keys, pos, has_gauss, cached = json.loads(decoded["numpy_rng"].decode("utf-8"))
    numpy.random.set_state((algo, numpy.array(keys, dtype="uint32"), pos, has_gauss, cached))
    torch.set_rng_state(torch.tensor(list(decoded["torch_rng"]), dtype=torch.uint8))

    final_state = reconciled["final_state"]
    return {
        "completed": final_state["completed"],
        "epoch": final_state["epoch"],
        "cursor": final_state["cursor"],
        "last_cursor_before": final_state["last_cursor_before"],
        "last_receipt_sha256": final_state["last_receipt_sha256"],
        "best": None if header["best_step"] is None else (header["best_step"], header["best_macro"]),
        "events": header["events"],
        "macros": header["macros"],
        "exposure": reconciled["records"],
        "exposure_frames": reconciled["frames"],
        "sampler": reconciled["sampler"],
    }


def _sections(model, opt, sampler_bytes: bytes, exposure_journal_bytes: bytes) -> dict[str, bytes]:
    import io
    import json
    import random

    import torch

    model_buf = io.BytesIO()
    torch.save(model.state_dict(), model_buf)
    opt_buf = io.BytesIO()
    torch.save(opt.state_dict(), opt_buf)
    return {
        "model": model_buf.getvalue(),
        "adamw": opt_buf.getvalue(),
        "python_rng": json.dumps(random.getstate()).encode("utf-8"),
        "numpy_rng": json.dumps(_numpy_state()).encode("utf-8"),
        "torch_rng": bytes(torch.get_rng_state().tolist()),
        "sampler": sampler_bytes,
        "exposure": exposure_journal_bytes,
    }


def main() -> None:
    raise ExecutionNotAuthorized("NOT_RUN")
