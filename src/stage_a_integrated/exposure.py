"""Canonical exposure-v2 and sampler-v2 admission with C3 forward-input binding."""

from __future__ import annotations

import hashlib
import json

from . import SAMPLER_SHA256
from .forward_binding import (
    FORWARD_BINDING_VERSION,
    REPRESENTATION_FIREWALL,
    expected_synthetic_forward_digest,
)
from .plan import (
    PINNED_SYNTHETIC_LABELS,
    REMAINDER_POLICY,
    SYNTHETIC_SOURCE_ID,
    SYNTHETIC_TRAIN_IDS_SHA256,
    SYNTHETIC_TRAIN_SIZE,
    advance_window,
    require_exact_int,
    select_video_ids,
    train_ids,
    verify_sampler_pin,
    verify_train_ids_pin,
)

SCHEMA_VERSION = "exposure-v2"
LEGACY_REJECTED_SCHEMAS = frozenset({"exposure-v1"})
SAMPLER_SCHEMA_VERSION = "sampler-v2"
PROVIDER_ID = "SYNTHETIC_APPARATUS"
FORWARD_BINDING = FORWARD_BINDING_VERSION
SYNTHETIC_TRAIN_HORIZON = "near"

MAX_JOURNAL_RECORDS = 100_000
MAX_RECEIPT_BYTES = 65_536
MAX_JOURNAL_BYTES = 16 * 1024 * 1024
MAX_SAMPLER_BYTES = 4_096
MAX_VIDEO_ID_LEN = 64
MAX_SEED = (1 << 63) - 1
MAX_EPOCH = 1_000_000

PINNED_LABELS_LIST = list(PINNED_SYNTHETIC_LABELS)
HEX_CHARS = frozenset("0123456789abcdef")

ITEM_REQUIRED_KEYS = (
    "id_movi",
    "ids_frames",
    "anchor_ordinal",
    "anchor",
    "selected_ordinals",
    "selected_labels",
    "current_ordinal",
    "target_ordinal",
    "current",
    "target",
    "horizon",
    "epoch",
    "seed",
    "provider",
    "source_id",
    "source_sha256",
    "sampler_sha256",
    "forward_binding",
    "representation_firewall",
    "forward_input_sha256",
)
COLUMN_KEYS = (
    "videos",
    "anchor_ordinal",
    "anchor",
    "selected_ordinals",
    "selected_labels",
    "current_ordinal",
    "target_ordinal",
    "current",
    "target",
    "ids_frames",
    "horizon",
    "forward_input_sha256",
)
EXPOSURE_V2_BODY_KEYS = (
    "schema",
    "completed",
    "provider",
    "source_id",
    "source_sha256",
    "sampler_sha256",
    "remainder_policy",
    "forward_binding",
    "seed",
    "epoch",
    "batch_size",
    "cursor_before",
    "cursor_after",
    *COLUMN_KEYS,
)
EXPOSURE_V2_KEYS = frozenset((*EXPOSURE_V2_BODY_KEYS, "receipt_sha256"))
SAMPLER_V2_KEYS = frozenset(
    (
        "schema",
        "exposure_schema",
        "completed",
        "epoch",
        "cursor",
        "last_cursor_before",
        "seed",
        "batch_size",
        "source_id",
        "source_sha256",
        "sampler_sha256",
        "remainder_policy",
        "exposure_journal_sha256",
        "last_receipt_sha256",
    )
)


def _require_exact_str(value, name: str, *, min_len: int = 1, max_len: int = 256) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a str")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} is not valid UTF-8") from exc
    if len(value) < min_len or len(value) > max_len:
        raise ValueError(f"{name} length out of bounds")
    return value


def _require_sha256_hex(value, name: str) -> str:
    digest = _require_exact_str(value, name, min_len=64, max_len=64)
    if any(ch not in HEX_CHARS for ch in digest):
        raise ValueError(f"{name} is not lowercase sha256 hex")
    return digest


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(token: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {token}")


def strict_json_loads(raw: bytes | str, *, max_bytes: int = MAX_JOURNAL_BYTES):
    if type(raw) is bytes:
        if len(raw) > max_bytes:
            raise ValueError("JSON payload exceeds size bound")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("JSON payload is not valid UTF-8") from exc
    elif type(raw) is str:
        try:
            raw_bytes = raw.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("JSON text is not valid UTF-8") from exc
        if len(raw_bytes) > max_bytes:
            raise ValueError("JSON payload exceeds size bound")
        text = raw
    else:
        raise ValueError("JSON payload must be bytes or str")
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("malformed JSON payload") from exc


def canonical_json_bytes(obj, *, max_bytes: int = MAX_RECEIPT_BYTES) -> bytes:
    try:
        encoded = json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("object cannot be canonically JSON-encoded") from exc
    if len(encoded) > max_bytes:
        raise ValueError("canonical JSON payload exceeds size bound")
    return encoded


def receipt_body_digest(body: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(body, max_bytes=MAX_RECEIPT_BYTES)).hexdigest()


def _validate_row_semantics(
    *,
    video,
    frame_labels,
    anchor_ordinal,
    anchor,
    ordinals,
    labels,
    current_ord,
    target_ord,
    current,
    target,
    horizon,
    forward_input_sha256,
) -> None:
    _require_exact_str(video, "video", min_len=1, max_len=MAX_VIDEO_ID_LEN)
    if type(frame_labels) is not list or any(type(val) is not int for val in frame_labels):
        raise ValueError("ids_frames must be a list of exact ints")
    if frame_labels != PINNED_LABELS_LIST:
        raise ValueError("ids_frames does not match the pinned synthetic source")
    anchor_ord_val = require_exact_int(anchor_ordinal, "anchor_ordinal", minimum=0, maximum=0)
    anchor_val = require_exact_int(anchor, "anchor")
    if frame_labels[anchor_ord_val] != anchor_val:
        raise ValueError("anchor label does not match ids_frames[0]")
    if type(ordinals) is not list or not ordinals or len(ordinals) > len(frame_labels):
        raise ValueError("observed prefix is empty or oversized")
    if any(type(val) is not int or val < 0 or val >= len(frame_labels) for val in ordinals):
        raise ValueError("selected_ordinals entry is invalid or out of bounds")
    if type(labels) is not list or len(labels) != len(ordinals):
        raise ValueError("selected_labels length does not match selected_ordinals")
    if any(type(val) is not int for val in labels):
        raise ValueError("selected_labels entry is not an exact int")
    cur_ord_val = require_exact_int(current_ord, "current_ordinal", minimum=0, maximum=len(frame_labels) - 1)
    tgt_ord_val = require_exact_int(target_ord, "target_ordinal", minimum=0, maximum=len(frame_labels) - 1)
    cur_val = require_exact_int(current, "current")
    tgt_val = require_exact_int(target, "target")
    if _require_exact_str(horizon, "horizon", min_len=1, max_len=16) != SYNTHETIC_TRAIN_HORIZON:
        raise ValueError("unexpected synthetic train horizon")
    if ordinals != sorted(set(ordinals)) or ordinals[-1] != cur_ord_val or tgt_ord_val <= cur_ord_val:
        raise ValueError("history is not a strict prefix ending at current")
    if cur_ord_val not in ordinals or tgt_ord_val in ordinals:
        raise ValueError("observed prefix must contain current and exclude target")
    if any(frame_labels[ordinal] != label for ordinal, label in zip(ordinals, labels)):
        raise ValueError("selected ordinal does not match the pinned source label")
    if frame_labels[cur_ord_val] != cur_val or frame_labels[tgt_ord_val] != tgt_val:
        raise ValueError("current or target label does not match the pinned source")

    fwd_hex = _require_sha256_hex(forward_input_sha256, "forward_input_sha256")
    expected_fwd_hex = expected_synthetic_forward_digest(
        video,
        frame_labels,
        labels,
        cur_val,
        tgt_val,
        horizon,
    )
    if fwd_hex != expected_fwd_hex:
        raise ValueError("exposure forward_input_sha256 disagrees with independent causal oracle")


def require_item(item: dict) -> None:
    if type(item) is not dict:
        raise ValueError("batch item must be an object")
    if any(key not in item for key in ITEM_REQUIRED_KEYS):
        raise ValueError("batch item is not exposure-v2")
    if _require_exact_str(item["provider"], "provider") != PROVIDER_ID:
        raise ValueError("batch item provider mismatch")
    if _require_exact_str(item["source_id"], "source_id") != SYNTHETIC_SOURCE_ID:
        raise ValueError("batch item source_id mismatch")
    if _require_sha256_hex(item["source_sha256"], "source_sha256") != SYNTHETIC_TRAIN_IDS_SHA256:
        raise ValueError("batch item source_sha256 mismatch")
    if _require_sha256_hex(item["sampler_sha256"], "sampler_sha256") != verify_sampler_pin():
        raise ValueError("batch item sampler_sha256 mismatch")
    if _require_exact_str(item["forward_binding"], "forward_binding") != FORWARD_BINDING:
        raise ValueError("batch item forward_binding mismatch")
    if _require_exact_str(item["representation_firewall"], "representation_firewall") != REPRESENTATION_FIREWALL:
        raise ValueError("batch item representation_firewall mismatch")
    require_exact_int(item["epoch"], "epoch", minimum=0, maximum=MAX_EPOCH)
    require_exact_int(item["seed"], "seed", minimum=0, maximum=MAX_SEED)
    _validate_row_semantics(
        video=item["id_movi"],
        frame_labels=item["ids_frames"],
        anchor_ordinal=item["anchor_ordinal"],
        anchor=item["anchor"],
        ordinals=item["selected_ordinals"],
        labels=item["selected_labels"],
        current_ord=item["current_ordinal"],
        target_ord=item["target_ordinal"],
        current=item["current"],
        target=item["target"],
        horizon=item["horizon"],
        forward_input_sha256=item["forward_input_sha256"],
    )
    if item["id_movi"] not in set(train_ids()):
        raise ValueError("batch item video is outside pinned synthetic train_ids")


def receipt(
    batch: list[dict],
    completed: int,
    epoch: int,
    cursor_before: int,
    cursor_after: int,
    seed: int,
    batch_size: int,
) -> dict:
    if type(batch) is not list:
        raise ValueError("batch must be a list")
    completed_val = require_exact_int(completed, "completed", minimum=1, maximum=MAX_JOURNAL_RECORDS)
    epoch_val = require_exact_int(epoch, "epoch", minimum=0, maximum=MAX_EPOCH)
    batch_val = require_exact_int(batch_size, "batch_size", minimum=1, maximum=SYNTHETIC_TRAIN_SIZE)
    before_val = require_exact_int(cursor_before, "cursor_before", minimum=0, maximum=SYNTHETIC_TRAIN_SIZE - 1)
    after_val = require_exact_int(cursor_after, "cursor_after", minimum=1, maximum=SYNTHETIC_TRAIN_SIZE)
    seed_val = require_exact_int(seed, "seed", minimum=0, maximum=MAX_SEED)
    if len(batch) != batch_val:
        raise ValueError("batch length does not match batch_size")
    if after_val != before_val + batch_val:
        raise ValueError("cursor step does not match batch_size")
    sampler_digest = verify_sampler_pin()
    for item in batch:
        require_item(item)
        if item["epoch"] != epoch_val or item["seed"] != seed_val:
            raise ValueError("batch item epoch or seed does not match receipt")
    body = {
        "schema": SCHEMA_VERSION,
        "completed": completed_val,
        "provider": PROVIDER_ID,
        "source_id": SYNTHETIC_SOURCE_ID,
        "source_sha256": SYNTHETIC_TRAIN_IDS_SHA256,
        "sampler_sha256": sampler_digest,
        "remainder_policy": REMAINDER_POLICY,
        "forward_binding": FORWARD_BINDING,
        "seed": seed_val,
        "epoch": epoch_val,
        "batch_size": batch_val,
        "cursor_before": before_val,
        "cursor_after": after_val,
        "videos": [item["id_movi"] for item in batch],
        "anchor_ordinal": [item["anchor_ordinal"] for item in batch],
        "anchor": [item["anchor"] for item in batch],
        "selected_ordinals": [item["selected_ordinals"] for item in batch],
        "selected_labels": [item["selected_labels"] for item in batch],
        "current_ordinal": [item["current_ordinal"] for item in batch],
        "target_ordinal": [item["target_ordinal"] for item in batch],
        "current": [item["current"] for item in batch],
        "target": [item["target"] for item in batch],
        "ids_frames": [item["ids_frames"] for item in batch],
        "horizon": [item["horizon"] for item in batch],
        "forward_input_sha256": [item["forward_input_sha256"] for item in batch],
    }
    body_bytes = canonical_json_bytes(body, max_bytes=MAX_RECEIPT_BYTES)
    digest = hashlib.sha256(body_bytes).hexdigest()
    detached = strict_json_loads(body_bytes, max_bytes=MAX_RECEIPT_BYTES)
    detached["receipt_sha256"] = digest
    return detached


def build_receipt_bytes(
    batch: list[dict],
    completed: int,
    epoch: int,
    cursor_before: int,
    cursor_after: int,
    seed: int,
    batch_size: int,
) -> bytes:
    record = receipt(batch, completed, epoch, cursor_before, cursor_after, seed, batch_size)
    return canonical_json_bytes(record, max_bytes=MAX_RECEIPT_BYTES)


def replay_ordinals(record: dict) -> None:
    verify_sampler_pin()
    from .stage_a_sampler_v2 import sample

    for video, ordinals, labels, frame_labels, current_ord, target_ord in zip(
        record["videos"],
        record["selected_ordinals"],
        record["selected_labels"],
        record["ids_frames"],
        record["current_ordinal"],
        record["target_ordinal"],
    ):
        sampled, history = sample(frame_labels, video, record["epoch"], record["seed"])
        if (
            len(sampled) < 2
            or sampled[-2] != current_ord
            or sampled[-1] != target_ord
            or list(history[sampled[-2]]) != ordinals
            or [frame_labels[index] for index in history[sampled[-2]]] != labels
        ):
            raise ValueError("exposure selection does not match the frozen sampler")


def _validate_expected_pins(
    *,
    expected_seed: int,
    expected_batch: int,
    expected_source_id: str,
    expected_source_sha256: str,
    expected_sampler_sha256: str,
) -> tuple[int, int, str, str, str]:
    seed_val = require_exact_int(expected_seed, "expected_seed", minimum=0, maximum=MAX_SEED)
    batch_val = require_exact_int(expected_batch, "expected_batch", minimum=1, maximum=SYNTHETIC_TRAIN_SIZE)
    source_id_val = _require_exact_str(expected_source_id, "expected_source_id", min_len=1, max_len=128)
    source_sha_val = _require_sha256_hex(expected_source_sha256, "expected_source_sha256")
    sampler_sha_val = _require_sha256_hex(expected_sampler_sha256, "expected_sampler_sha256")
    if source_id_val != SYNTHETIC_SOURCE_ID or source_sha_val != SYNTHETIC_TRAIN_IDS_SHA256:
        raise ValueError("expected synthetic source mismatch")
    if sampler_sha_val != verify_sampler_pin():
        raise ValueError("expected sampler pin mismatch")
    verify_train_ids_pin(train_ids())
    return seed_val, batch_val, source_id_val, source_sha_val, sampler_sha_val


def validate_record_structure(
    record: dict,
    *,
    expected_step: int,
    expected_seed: int,
    expected_batch: int,
    expected_epoch: int,
    expected_cursor_before: int,
    expected_cursor_after: int,
    expected_source_id: str = SYNTHETIC_SOURCE_ID,
    expected_source_sha256: str = SYNTHETIC_TRAIN_IDS_SHA256,
    expected_sampler_sha256: str = SAMPLER_SHA256,
) -> dict:
    step_val = require_exact_int(expected_step, "expected_step", minimum=1, maximum=MAX_JOURNAL_RECORDS)
    seed_val, batch_val, source_id_val, source_sha_val, sampler_sha_val = _validate_expected_pins(
        expected_seed=expected_seed,
        expected_batch=expected_batch,
        expected_source_id=expected_source_id,
        expected_source_sha256=expected_source_sha256,
        expected_sampler_sha256=expected_sampler_sha256,
    )
    epoch_val = require_exact_int(expected_epoch, "expected_epoch", minimum=0, maximum=MAX_EPOCH)
    before_val = require_exact_int(expected_cursor_before, "expected_cursor_before", minimum=0, maximum=SYNTHETIC_TRAIN_SIZE - 1)
    after_val = require_exact_int(expected_cursor_after, "expected_cursor_after", minimum=1, maximum=SYNTHETIC_TRAIN_SIZE)
    if after_val != before_val + batch_val:
        raise ValueError("expected cursor step does not match expected batch")

    if type(record) is not dict:
        raise ValueError("exposure record is not an object")
    schema = record.get("schema")
    if type(schema) is not str:
        raise ValueError("exposure schema must be a string")
    if schema in LEGACY_REJECTED_SCHEMAS:
        raise ValueError("legacy schema exposure-v1 is not supported for resume; re-run under exposure-v2")
    if schema != SCHEMA_VERSION:
        raise ValueError("unsupported exposure schema")
    if set(record.keys()) != EXPOSURE_V2_KEYS:
        raise ValueError("exposure record keys do not match exposure-v2")

    completed = require_exact_int(record["completed"], "completed", minimum=1, maximum=MAX_JOURNAL_RECORDS)
    if completed != step_val:
        raise ValueError("proposed exposure step is not the expected update")
    if _require_exact_str(record["provider"], "provider") != PROVIDER_ID:
        raise ValueError("exposure provider mismatch")
    if _require_exact_str(record["source_id"], "source_id") != source_id_val:
        raise ValueError("exposure source_id mismatch")
    if _require_sha256_hex(record["source_sha256"], "source_sha256") != source_sha_val:
        raise ValueError("exposure source_sha256 mismatch")
    if _require_sha256_hex(record["sampler_sha256"], "sampler_sha256") != sampler_sha_val:
        raise ValueError("exposure sampler_sha256 mismatch")
    if _require_exact_str(record["remainder_policy"], "remainder_policy") != REMAINDER_POLICY:
        raise ValueError("exposure remainder_policy mismatch")
    if _require_exact_str(record["forward_binding"], "forward_binding") != FORWARD_BINDING:
        raise ValueError("exposure forward_binding mismatch")

    rec_seed = require_exact_int(record["seed"], "seed", minimum=0, maximum=MAX_SEED)
    if rec_seed != seed_val:
        raise ValueError("exposure seed does not match config")
    rec_epoch = require_exact_int(record["epoch"], "epoch", minimum=0, maximum=MAX_EPOCH)
    if rec_epoch != epoch_val:
        raise ValueError("exposure epoch does not match global replay")
    rec_batch = require_exact_int(record["batch_size"], "batch_size", minimum=1, maximum=SYNTHETIC_TRAIN_SIZE)
    if rec_batch != batch_val:
        raise ValueError("exposure batch_size does not match config")
    rec_before = require_exact_int(record["cursor_before"], "cursor_before", minimum=0, maximum=SYNTHETIC_TRAIN_SIZE - 1)
    rec_after = require_exact_int(record["cursor_after"], "cursor_after", minimum=1, maximum=SYNTHETIC_TRAIN_SIZE)
    if rec_after != rec_before + rec_batch:
        raise ValueError("cursor step does not match batch_size")
    if rec_before != before_val or rec_after != after_val:
        raise ValueError("exposure cursor does not match global replay")
    rec_digest = _require_sha256_hex(record["receipt_sha256"], "receipt_sha256")

    for key in COLUMN_KEYS:
        column = record[key]
        if type(column) is not list:
            raise ValueError(f"exposure column {key} must be a list")
        if len(column) != batch_val:
            raise ValueError(f"exposure column {key} length does not match batch_size")

    for index in range(batch_val):
        _validate_row_semantics(
            video=record["videos"][index],
            frame_labels=record["ids_frames"][index],
            anchor_ordinal=record["anchor_ordinal"][index],
            anchor=record["anchor"][index],
            ordinals=record["selected_ordinals"][index],
            labels=record["selected_labels"][index],
            current_ord=record["current_ordinal"][index],
            target_ord=record["target_ordinal"][index],
            current=record["current"][index],
            target=record["target"][index],
            horizon=record["horizon"][index],
            forward_input_sha256=record["forward_input_sha256"][index],
        )

    body = {key: record[key] for key in EXPOSURE_V2_BODY_KEYS}
    if receipt_body_digest(body) != rec_digest:
        raise ValueError("exposure receipt_sha256 mismatch")

    expected_videos = select_video_ids(seed_val, epoch_val, before_val, batch_val)
    if record["videos"] != expected_videos:
        raise ValueError("exposure videos do not match global sampler order")

    replay_ordinals(record)
    return record


def require_record(
    record_or_bytes: dict | bytes,
    expected_step: int,
    *,
    expected_seed: int,
    expected_batch: int,
    expected_epoch: int,
    expected_cursor_before: int,
    expected_cursor_after: int,
    expected_source_id: str = SYNTHETIC_SOURCE_ID,
    expected_source_sha256: str = SYNTHETIC_TRAIN_IDS_SHA256,
    expected_sampler_sha256: str = SAMPLER_SHA256,
) -> tuple[dict, bytes]:
    if type(record_or_bytes) is bytes:
        parsed = strict_json_loads(record_or_bytes, max_bytes=MAX_RECEIPT_BYTES)
        if type(parsed) is not dict:
            raise ValueError("exposure receipt bytes did not decode to an object")
        validated = validate_record_structure(
            parsed,
            expected_step=expected_step,
            expected_seed=expected_seed,
            expected_batch=expected_batch,
            expected_epoch=expected_epoch,
            expected_cursor_before=expected_cursor_before,
            expected_cursor_after=expected_cursor_after,
            expected_source_id=expected_source_id,
            expected_source_sha256=expected_source_sha256,
            expected_sampler_sha256=expected_sampler_sha256,
        )
        canonical = canonical_json_bytes(validated, max_bytes=MAX_RECEIPT_BYTES)
        if canonical != record_or_bytes:
            raise ValueError("exposure receipt bytes are not canonical")
        return validated, canonical
    if type(record_or_bytes) is dict:
        canonical = canonical_json_bytes(record_or_bytes, max_bytes=MAX_RECEIPT_BYTES)
        parsed = strict_json_loads(canonical, max_bytes=MAX_RECEIPT_BYTES)
        if type(parsed) is not dict:
            raise ValueError("exposure record did not roundtrip to an object")
        validated = validate_record_structure(
            parsed,
            expected_step=expected_step,
            expected_seed=expected_seed,
            expected_batch=expected_batch,
            expected_epoch=expected_epoch,
            expected_cursor_before=expected_cursor_before,
            expected_cursor_after=expected_cursor_after,
            expected_source_id=expected_source_id,
            expected_source_sha256=expected_source_sha256,
            expected_sampler_sha256=expected_sampler_sha256,
        )
        return validated, canonical
    raise ValueError("exposure record must be canonical bytes or dict")


def require_records(
    records_or_bytes: list | bytes,
    completed: int,
    *,
    expected_seed: int,
    expected_batch: int,
    expected_source_id: str = SYNTHETIC_SOURCE_ID,
    expected_source_sha256: str = SYNTHETIC_TRAIN_IDS_SHA256,
    expected_sampler_sha256: str = SAMPLER_SHA256,
) -> tuple[list[dict], list[bytes], dict]:
    completed_val = require_exact_int(completed, "completed", minimum=0, maximum=MAX_JOURNAL_RECORDS)
    seed_val, batch_val, source_id_val, source_sha_val, sampler_sha_val = _validate_expected_pins(
        expected_seed=expected_seed,
        expected_batch=expected_batch,
        expected_source_id=expected_source_id,
        expected_source_sha256=expected_source_sha256,
        expected_sampler_sha256=expected_sampler_sha256,
    )

    raw_bytes: bytes | None = None
    if type(records_or_bytes) is bytes:
        raw_bytes = records_or_bytes
        parsed = strict_json_loads(records_or_bytes, max_bytes=MAX_JOURNAL_BYTES)
    elif type(records_or_bytes) is list:
        parsed = records_or_bytes
    else:
        raise ValueError("exposure records must be canonical bytes or list")

    if type(parsed) is not list:
        raise ValueError("exposure journal must be a list")
    if len(parsed) != completed_val:
        raise ValueError("exposure length does not match completed")

    validated_records: list[dict] = []
    canonical_frames: list[bytes] = []
    cur_epoch = 0
    cur_cursor = 0
    for step_index, record in enumerate(parsed, start=1):
        exp_epoch, exp_before, exp_after = advance_window(cur_epoch, cur_cursor, batch_val)
        validated = validate_record_structure(
            record,
            expected_step=step_index,
            expected_seed=seed_val,
            expected_batch=batch_val,
            expected_epoch=exp_epoch,
            expected_cursor_before=exp_before,
            expected_cursor_after=exp_after,
            expected_source_id=source_id_val,
            expected_source_sha256=source_sha_val,
            expected_sampler_sha256=sampler_sha_val,
        )
        frame_bytes = canonical_json_bytes(validated, max_bytes=MAX_RECEIPT_BYTES)
        validated_records.append(validated)
        canonical_frames.append(frame_bytes)
        cur_epoch = exp_epoch
        cur_cursor = exp_after

    canonical_journal = b"[" + b",".join(canonical_frames) + b"]"
    if len(canonical_journal) > MAX_JOURNAL_BYTES:
        raise ValueError("canonical exposure journal exceeds size bound")
    if raw_bytes is not None and canonical_journal != raw_bytes:
        raise ValueError("exposure section bytes are not canonical")

    journal_sha256 = hashlib.sha256(canonical_journal).hexdigest()
    last_cursor_before = None if completed_val == 0 else validated_records[-1]["cursor_before"]
    last_receipt_sha256 = None if completed_val == 0 else validated_records[-1]["receipt_sha256"]
    final_state = {
        "completed": completed_val,
        "epoch": cur_epoch,
        "cursor": cur_cursor,
        "last_cursor_before": last_cursor_before,
        "seed": seed_val,
        "batch_size": batch_val,
        "source_id": source_id_val,
        "source_sha256": source_sha_val,
        "sampler_sha256": sampler_sha_val,
        "exposure_schema": SCHEMA_VERSION,
        "remainder_policy": REMAINDER_POLICY,
        "exposure_journal_sha256": journal_sha256,
        "last_receipt_sha256": last_receipt_sha256,
    }
    return validated_records, canonical_frames, final_state


def build_sampler_section_bytes(
    *,
    completed: int,
    epoch: int,
    cursor: int,
    last_cursor_before: int | None,
    seed: int,
    batch_size: int,
    exposure_journal_bytes: bytes,
    last_receipt_sha256: str | None,
) -> bytes:
    completed_val = require_exact_int(completed, "completed", minimum=0, maximum=MAX_JOURNAL_RECORDS)
    epoch_val = require_exact_int(epoch, "epoch", minimum=0, maximum=MAX_EPOCH)
    cursor_val = require_exact_int(cursor, "cursor", minimum=0, maximum=SYNTHETIC_TRAIN_SIZE)
    seed_val = require_exact_int(seed, "seed", minimum=0, maximum=MAX_SEED)
    batch_val = require_exact_int(batch_size, "batch_size", minimum=1, maximum=SYNTHETIC_TRAIN_SIZE)
    if type(exposure_journal_bytes) is not bytes or len(exposure_journal_bytes) > MAX_JOURNAL_BYTES:
        raise ValueError("exposure_journal_bytes must be bounded bytes")
    if completed_val == 0:
        if last_cursor_before is not None or last_receipt_sha256 is not None:
            raise ValueError("zero-step sampler state cannot carry last receipt fields")
    else:
        require_exact_int(last_cursor_before, "last_cursor_before", minimum=0, maximum=SYNTHETIC_TRAIN_SIZE - 1)
        _require_sha256_hex(last_receipt_sha256, "last_receipt_sha256")
    sampler_digest = verify_sampler_pin()
    payload = {
        "schema": SAMPLER_SCHEMA_VERSION,
        "exposure_schema": SCHEMA_VERSION,
        "completed": completed_val,
        "epoch": epoch_val,
        "cursor": cursor_val,
        "last_cursor_before": last_cursor_before,
        "seed": seed_val,
        "batch_size": batch_val,
        "source_id": SYNTHETIC_SOURCE_ID,
        "source_sha256": SYNTHETIC_TRAIN_IDS_SHA256,
        "sampler_sha256": sampler_digest,
        "remainder_policy": REMAINDER_POLICY,
        "exposure_journal_sha256": hashlib.sha256(exposure_journal_bytes).hexdigest(),
        "last_receipt_sha256": last_receipt_sha256,
    }
    return canonical_json_bytes(payload, max_bytes=MAX_SAMPLER_BYTES)


def reconcile_checkpoint_state(
    header: dict,
    sampler_raw: bytes,
    exposure_raw: bytes,
    *,
    expected_seed: int,
    expected_batch: int,
    expected_source_id: str = SYNTHETIC_SOURCE_ID,
    expected_source_sha256: str = SYNTHETIC_TRAIN_IDS_SHA256,
    expected_sampler_sha256: str = SAMPLER_SHA256,
) -> dict:
    seed_val, batch_val, source_id_val, source_sha_val, sampler_sha_val = _validate_expected_pins(
        expected_seed=expected_seed,
        expected_batch=expected_batch,
        expected_source_id=expected_source_id,
        expected_source_sha256=expected_source_sha256,
        expected_sampler_sha256=expected_sampler_sha256,
    )

    if type(header) is not dict:
        raise ValueError("checkpoint header must be an object")
    if "exposure_schema" not in header:
        raise ValueError("legacy checkpoint without exposure-v2 header metadata is not supported; re-run under exposure-v2")
    header_exposure_schema = header["exposure_schema"]
    if type(header_exposure_schema) is not str:
        raise ValueError("header exposure_schema must be a string")
    if header_exposure_schema in LEGACY_REJECTED_SCHEMAS:
        raise ValueError("legacy schema exposure-v1 is not supported for resume; re-run under exposure-v2")
    if header_exposure_schema != SCHEMA_VERSION:
        raise ValueError("unsupported header exposure_schema")

    header_required = (
        "completed",
        "epoch",
        "cursor",
        "seed",
        "batch_size",
        "source_id",
        "source_sha256",
        "sampler_sha256",
        "exposure_schema",
        "exposure_journal_sha256",
        "last_receipt_sha256",
    )
    if any(key not in header for key in header_required):
        raise ValueError("checkpoint header is missing C1 provenance fields")

    h_completed = require_exact_int(header["completed"], "header.completed", minimum=0, maximum=MAX_JOURNAL_RECORDS)
    h_epoch = require_exact_int(header["epoch"], "header.epoch", minimum=0, maximum=MAX_EPOCH)
    h_cursor = require_exact_int(header["cursor"], "header.cursor", minimum=0, maximum=SYNTHETIC_TRAIN_SIZE)
    h_seed = require_exact_int(header["seed"], "header.seed", minimum=0, maximum=MAX_SEED)
    h_batch = require_exact_int(header["batch_size"], "header.batch_size", minimum=1, maximum=SYNTHETIC_TRAIN_SIZE)
    h_source_id = _require_exact_str(header["source_id"], "header.source_id")
    h_source_sha = _require_sha256_hex(header["source_sha256"], "header.source_sha256")
    h_sampler_sha = _require_sha256_hex(header["sampler_sha256"], "header.sampler_sha256")
    h_journal_sha = _require_sha256_hex(header["exposure_journal_sha256"], "header.exposure_journal_sha256")
    if h_completed == 0:
        if header["last_receipt_sha256"] is not None:
            raise ValueError("zero-step header cannot carry last_receipt_sha256")
        h_last_receipt = None
    else:
        h_last_receipt = _require_sha256_hex(header["last_receipt_sha256"], "header.last_receipt_sha256")

    if h_seed != seed_val or h_batch != batch_val:
        raise ValueError("header seed or batch_size does not match expected config")
    if h_source_id != source_id_val or h_source_sha != source_sha_val or h_sampler_sha != sampler_sha_val:
        raise ValueError("header source or sampler pin mismatch")

    if type(sampler_raw) is not bytes:
        raise ValueError("sampler section must be bytes")
    sampler = strict_json_loads(sampler_raw, max_bytes=MAX_SAMPLER_BYTES)
    if type(sampler) is not dict:
        raise ValueError("sampler section must decode to an object")
    if "schema" not in sampler:
        raise ValueError("legacy sampler section without schema is not supported; re-run under sampler-v2")
    if type(sampler["schema"]) is not str or sampler["schema"] != SAMPLER_SCHEMA_VERSION:
        raise ValueError("unsupported sampler schema")
    if set(sampler.keys()) != SAMPLER_V2_KEYS:
        raise ValueError("sampler section keys do not match sampler-v2")
    if canonical_json_bytes(sampler, max_bytes=MAX_SAMPLER_BYTES) != sampler_raw:
        raise ValueError("sampler section bytes are not canonical")

    s_completed = require_exact_int(sampler["completed"], "sampler.completed", minimum=0, maximum=MAX_JOURNAL_RECORDS)
    s_epoch = require_exact_int(sampler["epoch"], "sampler.epoch", minimum=0, maximum=MAX_EPOCH)
    s_cursor = require_exact_int(sampler["cursor"], "sampler.cursor", minimum=0, maximum=SYNTHETIC_TRAIN_SIZE)
    s_seed = require_exact_int(sampler["seed"], "sampler.seed", minimum=0, maximum=MAX_SEED)
    s_batch = require_exact_int(sampler["batch_size"], "sampler.batch_size", minimum=1, maximum=SYNTHETIC_TRAIN_SIZE)
    s_exp_schema = _require_exact_str(sampler["exposure_schema"], "sampler.exposure_schema")
    s_source_id = _require_exact_str(sampler["source_id"], "sampler.source_id")
    s_source_sha = _require_sha256_hex(sampler["source_sha256"], "sampler.source_sha256")
    s_sampler_sha = _require_sha256_hex(sampler["sampler_sha256"], "sampler.sampler_sha256")
    s_policy = _require_exact_str(sampler["remainder_policy"], "sampler.remainder_policy")
    s_journal_sha = _require_sha256_hex(sampler["exposure_journal_sha256"], "sampler.exposure_journal_sha256")
    if s_completed == 0:
        if sampler["last_cursor_before"] is not None or sampler["last_receipt_sha256"] is not None:
            raise ValueError("zero-step sampler cannot carry last receipt fields")
        s_last_before = None
        s_last_receipt = None
    else:
        s_last_before = require_exact_int(
            sampler["last_cursor_before"],
            "sampler.last_cursor_before",
            minimum=0,
            maximum=SYNTHETIC_TRAIN_SIZE - 1,
        )
        s_last_receipt = _require_sha256_hex(sampler["last_receipt_sha256"], "sampler.last_receipt_sha256")

    if (
        s_completed != h_completed
        or s_epoch != h_epoch
        or s_cursor != h_cursor
        or s_seed != seed_val
        or s_batch != batch_val
        or s_exp_schema != SCHEMA_VERSION
        or s_source_id != source_id_val
        or s_source_sha != source_sha_val
        or s_sampler_sha != sampler_sha_val
        or s_policy != REMAINDER_POLICY
        or s_journal_sha != h_journal_sha
        or s_last_receipt != h_last_receipt
    ):
        raise ValueError("sampler section does not match header or expected config")

    if type(exposure_raw) is not bytes:
        raise ValueError("exposure section must be bytes")
    records, frames, final_state = require_records(
        exposure_raw,
        h_completed,
        expected_seed=seed_val,
        expected_batch=batch_val,
        expected_source_id=source_id_val,
        expected_source_sha256=source_sha_val,
        expected_sampler_sha256=sampler_sha_val,
    )

    if (
        final_state["completed"] != h_completed
        or final_state["epoch"] != h_epoch
        or final_state["cursor"] != h_cursor
        or final_state["last_cursor_before"] != s_last_before
        or final_state["exposure_journal_sha256"] != h_journal_sha
        or final_state["last_receipt_sha256"] != h_last_receipt
    ):
        raise ValueError("exposure journal final state does not match header and sampler")

    return {
        "records": records,
        "frames": frames,
        "final_state": final_state,
        "sampler": sampler,
    }
