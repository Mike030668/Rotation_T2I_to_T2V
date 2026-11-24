"""One future-authorized synthetic path. Default CLI still raises."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from . import CANDIDATE_SHA256, HORIZONS_SHA256, LOSS_SHA256, SAMPLER_SHA256
from .authorization import ExecutionNotAuthorized, load
from .checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    MAX_HORIZONS_FILE_BYTES,
    MAX_RNG_SECTION_BYTES,
    OWNER_TRUST_ROOT_STATUS,
    POINTER_SCHEMA_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    build_expected_provenance,
    clone_optimizer_state_dict,
    file_sha256,
    pack,
    publish,
    resolve_pointer,
    stage_adamw_state,
    stage_model_state,
    stage_numpy_rng,
    stage_python_rng,
    stage_torch_rng,
    unpack,
    validate_numpy_rng_bytes,
    validate_python_rng_bytes,
    validate_torch_rng_bytes,
    verify_baseline_provenance,
    verify_checkpoint_provenance,
    write_failure_receipt,
)
from .dev import evaluate, load_keys, load_rows
from .exposure import (
    FORWARD_BINDING,
    MAX_SEED,
    SAMPLER_SCHEMA_VERSION,
    SCHEMA_VERSION,
    build_receipt_bytes,
    build_sampler_section_bytes,
    canonical_json_bytes,
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
    if "release_created" in config and config["release_created"] is not False:
        raise ValueError("config.release_created must be false")
    pin_checks = {
        "candidate_sha256": CANDIDATE_SHA256,
        "loss_sha256": LOSS_SHA256,
        "horizons_sha256": HORIZONS_SHA256,
        "sampler_sha256": SAMPLER_SHA256,
        "exposure_schema": SCHEMA_VERSION,
        "sampler_schema": SAMPLER_SCHEMA_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "provenance_schema": PROVENANCE_SCHEMA_VERSION,
        "pointer_schema": POINTER_SCHEMA_VERSION,
        "owner_trust_root_status": OWNER_TRUST_ROOT_STATUS,
        "synthetic_source_id": SYNTHETIC_SOURCE_ID,
        "synthetic_train_size": SYNTHETIC_TRAIN_SIZE,
        "synthetic_train_ids_sha256": SYNTHETIC_TRAIN_IDS_SHA256,
        "synthetic_remainder_policy": REMAINDER_POLICY,
        "forward_binding": FORWARD_BINDING,
    }
    for key, expected_val in pin_checks.items():
        if key in config and config[key] != expected_val:
            raise ValueError(f"config.{key} does not match pinned C2 contract")
    return max_updates, batch_size, seed, dev_events, clip


def _canonical_effective_config_bytes(
    *,
    max_updates: int,
    batch_size: int,
    seed: int,
    dev_events: list[int],
    clip: float,
    horizons_file_sha256: str,
) -> bytes:
    effective = {
        "max_updates": max_updates,
        "batch": batch_size,
        "seed": seed,
        "dev_events": list(dev_events),
        "clip": clip,
        "candidate_sha256": CANDIDATE_SHA256,
        "loss_sha256": LOSS_SHA256,
        "horizons_sha256": HORIZONS_SHA256,
        "horizons_file_sha256": horizons_file_sha256,
        "sampler_sha256": SAMPLER_SHA256,
        "exposure_schema": SCHEMA_VERSION,
        "sampler_schema": SAMPLER_SCHEMA_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "provenance_schema": PROVENANCE_SCHEMA_VERSION,
        "pointer_schema": POINTER_SCHEMA_VERSION,
        "owner_trust_root_status": OWNER_TRUST_ROOT_STATUS,
        "synthetic_source_id": SYNTHETIC_SOURCE_ID,
        "synthetic_train_size": SYNTHETIC_TRAIN_SIZE,
        "synthetic_train_ids_sha256": SYNTHETIC_TRAIN_IDS_SHA256,
        "synthetic_remainder_policy": REMAINDER_POLICY,
        "forward_binding": FORWARD_BINDING,
        "release_created": False,
    }
    return canonical_json_bytes(effective, max_bytes=65536)


def _preflight_and_reconcile(payload: bytes, expected_config: dict) -> dict:
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
    expected_prov = expected_config.get("expected_provenance")
    provenance = (
        verify_checkpoint_provenance(header, expected_prov)
        if expected_prov is not None
        else verify_baseline_provenance(header)
    )
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
    py_rng = validate_python_rng_bytes(decoded["python_rng"])
    np_rng = validate_numpy_rng_bytes(decoded["numpy_rng"])
    torch_rng_bytes = validate_torch_rng_bytes(decoded["torch_rng"])
    return {
        "decoded": decoded,
        "header": header,
        "provenance": provenance,
        "reconciled": reconciled,
        "py_rng": py_rng,
        "np_rng": np_rng,
        "torch_rng_bytes": torch_rng_bytes,
    }


def run_synthetic(release_path: str, trust_anchor_path: str, expected: dict, config: dict) -> dict:
    release = load(release_path, trust_anchor_path, expected)
    if release.get("scope") != "stage_a_synthetic_preflight":
        raise ExecutionNotAuthorized("scope is not the synthetic preflight")
    max_updates, batch_size, seed, dev_events, clip = _validate_run_config(config)
    if max_updates > int(release.get("max_updates")):
        raise ExecutionNotAuthorized("config exceeds the released budget")
    sampler_digest = verify_sampler_pin()
    source_digest = verify_train_ids_pin(train_ids())
    root = Path(release.get("output_root"))
    horizons_path = Path(config["horizons_path"])
    horizons_file_sha256 = file_sha256(horizons_path, max_bytes=MAX_HORIZONS_FILE_BYTES)
    effective_config_bytes = _canonical_effective_config_bytes(
        max_updates=max_updates,
        batch_size=batch_size,
        seed=seed,
        dev_events=dev_events,
        clip=clip,
        horizons_file_sha256=horizons_file_sha256,
    )
    base_provenance = build_expected_provenance(
        effective_config_bytes,
        horizons_path,
        generation_id=None,
        horizons_file_sha256=horizons_file_sha256,
    )

    has_resume_bytes = config.get("resume_bytes") is not None
    has_resume_role = config.get("resume_role") is not None
    if has_resume_bytes and has_resume_role:
        raise ValueError("config cannot specify both resume_bytes and resume_role")

    resume_payload: bytes | None = None
    expected_resume_cfg = {
        "seed": seed,
        "batch": batch_size,
        "source_id": SYNTHETIC_SOURCE_ID,
        "source_sha256": source_digest,
        "sampler_sha256": sampler_digest,
        "expected_provenance": base_provenance,
    }
    if has_resume_bytes or has_resume_role:
        try:
            if has_resume_role:
                role = config["resume_role"]
                resolved = resolve_pointer(root, role)
                resume_payload = resolved["payload"]
            else:
                raw_resume = config["resume_bytes"]
                if type(raw_resume) is not bytes:
                    raise ValueError("config.resume_bytes must be bytes")
                resume_payload = raw_resume
            preflight = _preflight_and_reconcile(resume_payload, expected_resume_cfg)
            if int(preflight["reconciled"]["final_state"]["completed"]) > max_updates:
                raise ValueError("checkpoint completed step exceeds config.max_updates")
        except Exception as exc:
            raw_role = config.get("resume_role")
            write_failure_receipt(
                root,
                "resume_preflight",
                str(exc),
                details={"resume_role": raw_role if type(raw_role) in (str, type(None)) else repr(raw_role)},
            )
            raise

    import torch

    model_cls, config_cls = load_class()
    model = model_cls(config_cls())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    rows = load_rows(horizons_path)
    keys = load_keys(horizons_path)
    completed = 0
    epoch = 0
    cursor = 0
    last_cursor_before: int | None = None
    last_receipt_sha256: str | None = None
    best = None
    macros = {}
    events = []
    exposure_frames: list[bytes] = []
    if resume_payload is not None:
        restored = resume(
            resume_payload,
            model,
            opt,
            expected_resume_cfg,
            failure_root=root,
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
            generation_id = f"u{completed}"
            try:
                value = evaluate(model, items, keys)
            except Exception as exc:
                write_failure_receipt(
                    root,
                    "dev_eval_failure",
                    str(exc),
                    generation_id=generation_id,
                    details={"completed": completed},
                )
                raise
            finally:
                model.train()
            if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)):
                write_failure_receipt(
                    root,
                    "dev_nonfinite",
                    f"nonfinite DEV metric at {generation_id}",
                    generation_id=generation_id,
                    details={"completed": completed, "value": repr(value)},
                )
                raise ValueError(f"nonfinite DEV metric at step {completed}")
            value = float(value)
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
            step_provenance = build_expected_provenance(
                effective_config_bytes,
                horizons_path,
                generation_id=generation_id,
                horizons_file_sha256=horizons_file_sha256,
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
                    "provenance": step_provenance,
                    "macro": value,
                    "best_step": None if best is None else best[0],
                    "best_macro": None if best is None else best[1],
                    "best_history": history,
                    "events": events,
                    "macros": {str(step): metric for step, metric in macros.items()},
                },
                _sections(model, opt, sampler_bytes, exposure_journal_bytes),
            )
            try:
                publish(root, generation_id, "LAST", payload)
                if best is not None and best[0] == completed:
                    publish(root, generation_id, "BEST", payload)
            except Exception as exc:
                write_failure_receipt(
                    root,
                    "checkpoint_publish",
                    str(exc),
                    generation_id=generation_id,
                    details={"completed": completed},
                )
                raise
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


def _python_state() -> list:
    import random

    version, state_tuple, gauss = random.getstate()
    return [int(version), [int(word) for word in state_tuple], None if gauss is None else float(gauss)]


def _numpy_state() -> list:
    import numpy

    algo, keys, pos, has_gauss, cached = numpy.random.get_state()
    return [str(algo), [int(word) for word in keys.tolist()], int(pos), int(has_gauss), float(cached)]


def _rollback_live_state(
    model,
    opt,
    snap_model: dict,
    snap_adamw: dict,
    snap_py_rng: tuple,
    snap_np_rng: tuple,
    snap_torch_rng,
) -> None:
    import random

    import numpy
    import torch

    errors = []
    for name, action in (
        ("model", lambda: model.load_state_dict(snap_model, strict=True)),
        ("adamw", lambda: opt.load_state_dict(snap_adamw)),
        ("python_rng", lambda: random.setstate(snap_py_rng)),
        ("numpy_rng", lambda: numpy.random.set_state(snap_np_rng)),
        ("torch_rng", lambda: torch.set_rng_state(snap_torch_rng)),
    ):
        try:
            action()
        except Exception as rb_exc:
            errors.append(f"{name}: {rb_exc}")
    if errors:
        raise RuntimeError("resume rollback failure: " + "; ".join(errors))


def resume(
    payload: bytes,
    model,
    opt,
    expected_config: dict,
    *,
    failure_root: Path | None = None,
) -> dict:
    import random

    import numpy
    import torch

    generation_id: str | None = None
    try:
        checked = _preflight_and_reconcile(payload, expected_config)
        decoded = checked["decoded"]
        header = checked["header"]
        reconciled = checked["reconciled"]
        final_state = reconciled["final_state"]
        generation_id = checked["provenance"]["generation_id"]

        staged_model = stage_model_state(decoded["model"], model)
        staged_adamw = stage_adamw_state(decoded["adamw"], opt, completed=int(final_state["completed"]))
        staged_py_rng = stage_python_rng(decoded["python_rng"])
        staged_np_rng = stage_numpy_rng(decoded["numpy_rng"])
        staged_torch_rng = stage_torch_rng(decoded["torch_rng"])
    except Exception as exc:
        if failure_root is not None:
            write_failure_receipt(
                failure_root,
                "resume_stage",
                str(exc),
                generation_id=generation_id,
            )
        raise

    snap_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
    snap_adamw = clone_optimizer_state_dict(opt)
    snap_py_rng = random.getstate()
    np_algo, np_keys, np_pos, np_has_gauss, np_cached = numpy.random.get_state()
    snap_np_rng = (np_algo, np_keys.copy(), np_pos, np_has_gauss, np_cached)
    snap_torch_rng = torch.get_rng_state().clone()

    try:
        model.load_state_dict(staged_model, strict=True)
        opt.load_state_dict(staged_adamw)
        random.setstate(staged_py_rng)
        numpy.random.set_state(staged_np_rng)
        torch.set_rng_state(staged_torch_rng)
    except Exception as exc:
        try:
            _rollback_live_state(
                model,
                opt,
                snap_model,
                snap_adamw,
                snap_py_rng,
                snap_np_rng,
                snap_torch_rng,
            )
        finally:
            if failure_root is not None:
                write_failure_receipt(
                    failure_root,
                    "resume_apply",
                    str(exc),
                    generation_id=generation_id,
                )
        raise

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
        "provenance": checked["provenance"],
    }


def _sections(model, opt, sampler_bytes: bytes, exposure_journal_bytes: bytes) -> dict[str, bytes]:
    import io

    import torch

    model_buf = io.BytesIO()
    torch.save(model.state_dict(), model_buf)
    opt_buf = io.BytesIO()
    torch.save(opt.state_dict(), opt_buf)
    return {
        "model": model_buf.getvalue(),
        "adamw": opt_buf.getvalue(),
        "python_rng": canonical_json_bytes(_python_state(), max_bytes=MAX_RNG_SECTION_BYTES),
        "numpy_rng": canonical_json_bytes(_numpy_state(), max_bytes=MAX_RNG_SECTION_BYTES),
        "torch_rng": bytes(torch.get_rng_state().tolist()),
        "sampler": sampler_bytes,
        "exposure": exposure_journal_bytes,
    }


def main() -> None:
    raise ExecutionNotAuthorized("NOT_RUN")
