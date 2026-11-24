"""Canonical C3 forward-input binding (`fwd-input-v1`) and representation firewall.

This module intentionally avoids importing ``torch`` or ``numpy`` at import time so
checkpoint preflight and ``exposure-v2`` structural reconciliation remain Torch-free
before model construction.
"""

from __future__ import annotations

import hashlib
import math
import struct

FORWARD_BINDING_VERSION = "fwd-input-v1"
FORWARD_DIGEST_MAGIC = b"STCA-FWD-INPUT-V1\x00"
REPRESENTATION_FIREWALL = "PROXY_UNVALIDATED_E_TO_Z0"

WIDTH = 1280
TRANSITION_DIM = 5

ALLOWED_FORWARD_KWARGS = (
    "anchor_latent",
    "current_latent",
    "history_latents",
    "history_progress",
    "history_valid_mask",
    "mode",
    "text_hidden_states",
    "text_valid_mask",
    "transition",
)
ALLOWED_MODES = ("hybrid",)
REQUIRED_SUPERVISION_KEYS = (
    "target_delta_local",
    "target_delta_anchor",
    "target_latent_proxy",
)
ALLOWED_HORIZONS = ("near", "mid", "far")
HEX_CHARS = frozenset("0123456789abcdef")

FLOAT_KWARG_KEYS = frozenset(
    {
        "anchor_latent",
        "current_latent",
        "history_latents",
        "history_progress",
        "text_hidden_states",
        "transition",
    }
)
MASK_KWARG_KEYS = frozenset({"history_valid_mask", "text_valid_mask"})


def _f32(value: float) -> float:
    val = float(value)
    if not math.isfinite(val):
        raise ValueError("nonfinite binary32 scalar")
    out = struct.unpack("<f", struct.pack("<f", val))[0]
    return 0.0 if out == 0.0 else out


def _pack_f32_sequence(values: list[float]) -> bytes:
    normalized = [_f32(val) for val in values]
    return struct.pack(f"<{len(normalized)}f", *normalized)


def _require_valid_utf8_str(value: object, name: str, *, max_len: int = 256) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty str")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} is not valid UTF-8") from exc
    if len(value) > max_len:
        raise ValueError(f"{name} exceeds max length {max_len}")
    return value


def _require_sha256_hex(value: object, name: str) -> str:
    digest = _require_valid_utf8_str(value, name, max_len=64)
    if len(digest) != 64 or any(ch not in HEX_CHARS for ch in digest):
        raise ValueError(f"{name} must be a 64-char lowercase hex sha256")
    return digest


def _require_exact_int(value: object, name: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def validate_causal_transition_meta(
    id_movi: str,
    ids_frames: list[int],
    history: list[int],
    current: int,
    target: int,
    horizon: str,
    *,
    anchor_ordinal: int = 0,
    anchor: int | None = None,
) -> tuple[dict[int, int], int, int]:
    """Validate strictly causal frame ordering, anchor grounding, and horizon."""
    _require_valid_utf8_str(id_movi, "id_movi", max_len=64)
    if type(ids_frames) is not list or len(ids_frames) < 2:
        raise ValueError("ids_frames must be a list of at least 2 frame labels")
    if any(type(frame) is not int for frame in ids_frames):
        raise ValueError("ids_frames entries must be exact ints")
    if len(set(ids_frames)) != len(ids_frames) or ids_frames != sorted(ids_frames):
        raise ValueError("ids_frames must be strictly increasing")
    span = ids_frames[-1] - ids_frames[0]
    if span <= 0:
        raise ValueError("ids_frames do not span a positive interval")

    anchor_ord = _require_exact_int(anchor_ordinal, "anchor_ordinal", minimum=0, maximum=0)
    if anchor is not None:
        anchor_val = _require_exact_int(anchor, "anchor")
        if anchor_val != ids_frames[anchor_ord]:
            raise ValueError("anchor label must match ids_frames[0]")

    cur_val = _require_exact_int(current, "current")
    tgt_val = _require_exact_int(target, "target")
    if _require_valid_utf8_str(horizon, "horizon", max_len=16) not in ALLOWED_HORIZONS:
        raise ValueError("horizon must be one of near/mid/far")

    if type(history) is not list or not history:
        raise ValueError("history must be a nonempty list of frame labels")
    if any(type(frame) is not int for frame in history):
        raise ValueError("history entries must be exact ints")

    ordinal = {label: index for index, label in enumerate(ids_frames)}
    if cur_val not in ordinal or tgt_val not in ordinal:
        raise ValueError("current or target label missing from ids_frames")
    if any(frame not in ordinal for frame in history):
        raise ValueError("history contains frame label missing from ids_frames")
    if len(set(history)) != len(history) or history != sorted(history):
        raise ValueError("history must be strictly increasing unique frame labels")
    if any(frame > cur_val for frame in history):
        raise ValueError("future frame leaked into history")
    if history[-1] != cur_val:
        raise ValueError("history must terminate at current frame label")
    if tgt_val in history or tgt_val <= cur_val:
        raise ValueError("target frame must be strictly in the future relative to current and history")

    return ordinal, ordinal[cur_val], ordinal[tgt_val]


def synthetic_schedule_f32(ids_frames: list[int], current_label: int, target_label: int) -> list[float]:
    if type(ids_frames) is not list or len(ids_frames) < 2 or any(type(f) is not int for f in ids_frames):
        raise ValueError("ids_frames must be an int list of length >= 2")
    if len(set(ids_frames)) != len(ids_frames) or ids_frames != sorted(ids_frames):
        raise ValueError("ids_frames must be strictly increasing")
    cur_val = _require_exact_int(current_label, "current_label")
    tgt_val = _require_exact_int(target_label, "target_label")
    if cur_val not in ids_frames or tgt_val not in ids_frames or tgt_val <= cur_val:
        raise ValueError("transition requires valid causal current < target in ids_frames")
    span = float(ids_frames[-1] - ids_frames[0])
    if span <= 0.0:
        raise ValueError("frame labels do not span a positive interval")
    origin = float(ids_frames[0])
    local = _f32(float(tgt_val - cur_val) / span)
    anchor_step = _f32(float(tgt_val - origin) / span)
    cur_prog = _f32(float(cur_val - origin) / span)
    tgt_prog = anchor_step
    strength = _f32(abs(local))
    return [local, anchor_step, cur_prog, tgt_prog, strength]


def synthetic_progress_f32(ids_frames: list[int], history: list[int]) -> list[float]:
    origin = float(ids_frames[0])
    span = float(ids_frames[-1] - ids_frames[0])
    if span <= 0.0:
        raise ValueError("frame labels do not span a positive interval")
    return [_f32(float(frame - origin) / span) for frame in history]


def synthetic_image_row_f32(id_movi: str, frame: int) -> list[float]:
    """Synthetic CLIP image feature e_t for one video frame (IEEE-754 float32)."""
    _require_valid_utf8_str(id_movi, "id_movi", max_len=64)
    _require_exact_int(frame, "frame")
    video = int(hashlib.sha256(id_movi.encode("utf-8")).hexdigest()[:8], 16)
    phase = float((video + frame * 13) % 997)
    frame_f = float(frame)
    return [_f32((float(k) * phase + frame_f) / 996.0) for k in range(WIDTH)]


def synthetic_prior_anchor_f32(id_movi: str) -> list[float]:
    """Distinct synthetic caption-derived unCLIP prior anchor z_0 != e_0."""
    _require_valid_utf8_str(id_movi, "id_movi", max_len=64)
    seed_val = int(hashlib.sha256(f"prior-anchor-v1:{id_movi}".encode("utf-8")).hexdigest()[:8], 16)
    anchor_phase = (seed_val % 991) + 7
    anchor_bias = ((seed_val >> 8) % 97) + 1
    return [_f32(float(((k * anchor_phase + anchor_bias) % 997) + 1) / 997.0) for k in range(WIDTH)]


def synthetic_proxy_row_f32(z0: list[float], e_t: list[float], e0: list[float]) -> list[float]:
    """Unvalidated cross-space proxy assumption: z_proxy(t) = z_0 + (e_t - e_0)."""
    if len(z0) != WIDTH or len(e_t) != WIDTH or len(e0) != WIDTH:
        raise ValueError("proxy vectors must have width 1280")
    out: list[float] = []
    for z0_val, et_val, e0_val in zip(z0, e_t, e0):
        delta = _f32(et_val - e0_val)
        out.append(_f32(z0_val + delta))
    return out


def synthetic_text_hidden_f32() -> list[float]:
    return [_f32(float(k) / 996.0) for k in range(WIDTH)]


def _hash_framed_entries(entries: dict[str, tuple]) -> str:
    if tuple(sorted(entries.keys())) != ALLOWED_FORWARD_KWARGS:
        raise ValueError("canonical forward digest requires exact 9 allowed kwargs")
    hasher = hashlib.sha256(FORWARD_DIGEST_MAGIC)
    hasher.update(struct.pack("<I", len(ALLOWED_FORWARD_KWARGS)))
    for key in ALLOWED_FORWARD_KWARGS:
        key_bytes = key.encode("utf-8")
        hasher.update(struct.pack("<I", len(key_bytes)))
        hasher.update(key_bytes)
        entry = entries[key]
        kind = entry[0]
        if kind == "S":
            if len(entry) != 2 or type(entry[1]) is not str:
                raise ValueError(f"invalid string frame entry for {key}")
            val_bytes = entry[1].encode("utf-8")
            hasher.update(b"S")
            hasher.update(struct.pack("<I", len(val_bytes)))
            hasher.update(val_bytes)
        elif kind == "T":
            if len(entry) != 4:
                raise ValueError(f"invalid tensor frame entry for {key}")
            _, dtype_name, shape, raw_bytes = entry
            if type(dtype_name) is not str or type(shape) is not tuple or type(raw_bytes) is not bytes:
                raise ValueError(f"malformed tensor frame metadata for {key}")
            dtype_bytes = dtype_name.encode("utf-8")
            hasher.update(b"T")
            hasher.update(struct.pack("<I", len(dtype_bytes)))
            hasher.update(dtype_bytes)
            hasher.update(struct.pack("<I", len(shape)))
            for dim in shape:
                if type(dim) is not int or dim < 0:
                    raise ValueError(f"invalid tensor dimension in {key}")
                hasher.update(struct.pack("<Q", dim))
            hasher.update(struct.pack("<Q", len(raw_bytes)))
            hasher.update(raw_bytes)
        else:
            raise ValueError(f"unsupported frame entry kind for {key}")
    return hasher.hexdigest()


def expected_synthetic_forward_digest(
    id_movi: str,
    ids_frames: list[int],
    history: list[int],
    current: int,
    target: int,
    horizon: str,
) -> str:
    """Independent Torch-free pure-Python oracle for synthetic causal forward kwargs."""
    validate_causal_transition_meta(
        id_movi,
        ids_frames,
        history,
        current,
        target,
        horizon,
        anchor_ordinal=0,
        anchor=ids_frames[0],
    )
    e0 = synthetic_image_row_f32(id_movi, ids_frames[0])
    z0 = synthetic_prior_anchor_f32(id_movi)
    hist_flat: list[float] = []
    cur_row: list[float] = []
    for frame in history:
        e_t = e0 if frame == ids_frames[0] else synthetic_image_row_f32(id_movi, frame)
        row = synthetic_proxy_row_f32(z0, e_t, e0)
        hist_flat.extend(row)
        if frame == current:
            cur_row = row
    if len(cur_row) != WIDTH:
        raise ValueError("current causal latent row missing")
    hist_len = len(history)
    progress = synthetic_progress_f32(ids_frames, history)
    text_row = synthetic_text_hidden_f32()
    transition = synthetic_schedule_f32(ids_frames, current, target)

    entries = {
        "anchor_latent": ("T", "float32", (1, WIDTH), _pack_f32_sequence(z0)),
        "current_latent": ("T", "float32", (1, WIDTH), _pack_f32_sequence(cur_row)),
        "history_latents": ("T", "float32", (1, hist_len, WIDTH), _pack_f32_sequence(hist_flat)),
        "history_progress": ("T", "float32", (1, hist_len), _pack_f32_sequence(progress)),
        "history_valid_mask": ("T", "bool", (1, hist_len), b"\x01" * hist_len),
        "mode": ("S", "hybrid"),
        "text_hidden_states": ("T", "float32", (1, 1, WIDTH), _pack_f32_sequence(text_row)),
        "text_valid_mask": ("T", "bool", (1, 1), b"\x01"),
        "transition": ("T", "float32", (1, TRANSITION_DIM), _pack_f32_sequence(transition)),
    }
    return _hash_framed_entries(entries)


def _storage_ptrs(container: dict | None) -> set[int]:
    if not isinstance(container, dict):
        return set()
    ptrs: set[int] = set()
    for val in container.values():
        if hasattr(val, "data_ptr") and hasattr(val, "untyped_storage"):
            ptrs.add(int(val.data_ptr()))
            ptrs.add(int(val.untyped_storage().data_ptr()))
    return ptrs


def canonical_forward_kwargs_digest(kwargs: dict) -> tuple[str, dict]:
    """Validate, detach-clone (aliasing/mutation guard), and canonically hash forward kwargs."""
    import torch

    if type(kwargs) is not dict:
        raise ValueError("model_kwargs must be a dict")
    if set(kwargs.keys()) != set(ALLOWED_FORWARD_KWARGS):
        extra = sorted(set(kwargs.keys()) - set(ALLOWED_FORWARD_KWARGS))
        missing = sorted(set(ALLOWED_FORWARD_KWARGS) - set(kwargs.keys()))
        raise ValueError(f"forbidden or missing forward kwargs (extra={extra}, missing={missing})")

    mode_val = kwargs["mode"]
    if type(mode_val) is not str or mode_val not in ALLOWED_MODES:
        raise ValueError("forward kwarg mode must be 'hybrid'")

    frozen_kwargs: dict = {"mode": mode_val}
    for key in ALLOWED_FORWARD_KWARGS:
        if key == "mode":
            continue
        tensor = kwargs[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"forward kwarg {key} must be a torch.Tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"forward kwarg {key} must be on cpu")
        if key in FLOAT_KWARG_KEYS:
            if tensor.dtype != torch.float32:
                raise ValueError(f"forward kwarg {key} must have dtype torch.float32")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"forward kwarg {key} contains nonfinite values")
        elif key in MASK_KWARG_KEYS:
            if tensor.dtype != torch.bool:
                raise ValueError(f"forward kwarg {key} must have dtype torch.bool")
        frozen_kwargs[key] = tensor.detach().contiguous().clone()

    hist_lat = frozen_kwargs["history_latents"]
    hist_prog = frozen_kwargs["history_progress"]
    hist_mask = frozen_kwargs["history_valid_mask"]
    text_hid = frozen_kwargs["text_hidden_states"]
    text_mask = frozen_kwargs["text_valid_mask"]
    cur_lat = frozen_kwargs["current_latent"]
    anc_lat = frozen_kwargs["anchor_latent"]
    trans = frozen_kwargs["transition"]

    if hist_lat.ndim != 3 or hist_lat.size(0) != 1 or hist_lat.size(1) < 1 or hist_lat.size(2) != WIDTH:
        raise ValueError(f"history_latents must have shape [1, L>=1, {WIDTH}]")
    hist_len = hist_lat.size(1)
    if tuple(hist_prog.shape) != (1, hist_len):
        raise ValueError("history_progress shape does not match history_latents")
    if tuple(hist_mask.shape) != (1, hist_len):
        raise ValueError("history_valid_mask shape does not match history_latents")
    if not bool(hist_mask.any(dim=1).all().item()):
        raise ValueError("history_valid_mask must have at least one True position")

    if text_hid.ndim != 3 or text_hid.size(0) != 1 or text_hid.size(1) < 1 or text_hid.size(2) != WIDTH:
        raise ValueError(f"text_hidden_states must have shape [1, T>=1, {WIDTH}]")
    if tuple(text_mask.shape) != (1, text_hid.size(1)):
        raise ValueError("text_valid_mask shape does not match text_hidden_states")
    if not bool(text_mask.any(dim=1).all().item()):
        raise ValueError("text_valid_mask must have at least one True position")

    if tuple(cur_lat.shape) != (1, WIDTH) or tuple(anc_lat.shape) != (1, WIDTH):
        raise ValueError(f"current_latent and anchor_latent must have shape [1, {WIDTH}]")
    if tuple(trans.shape) != (1, TRANSITION_DIM):
        raise ValueError(f"transition must have shape [1, {TRANSITION_DIM}]")

    prog_vals = hist_prog[0].tolist()
    mask_vals = hist_mask[0].tolist()
    valid_prog = [float(p) for p, m in zip(prog_vals, mask_vals) if m]
    if any(p < 0.0 or p > 1.0 for p in valid_prog):
        raise ValueError("history_progress values out of [0, 1] causal bounds")
    if any(b <= a for a, b in zip(valid_prog, valid_prog[1:])):
        raise ValueError("history_progress must be strictly increasing across valid history positions")

    trans_vals = [float(v) for v in trans[0].tolist()]
    if trans_vals[0] <= 0.0 or trans_vals[1] <= 0.0:
        raise ValueError("transition steps must be strictly positive causal deltas")
    if trans_vals[2] < 0.0 or trans_vals[2] >= 1.0 or trans_vals[3] <= trans_vals[2] or trans_vals[3] > 1.0:
        raise ValueError("transition progress endpoints are not causal in [0, 1]")
    if abs(trans_vals[4] - abs(trans_vals[0])) > 1e-6:
        raise ValueError("transition column 4 disagrees with known schedule strength")

    entries: dict[str, tuple] = {}
    for key in ALLOWED_FORWARD_KWARGS:
        if key == "mode":
            entries[key] = ("S", mode_val)
            continue
        ftensor = frozen_kwargs[key]
        shape = tuple(int(dim) for dim in ftensor.shape)
        if key in FLOAT_KWARG_KEYS:
            raw_bytes = _pack_f32_sequence(ftensor.reshape(-1).tolist())
            dtype_name = "float32"
        else:
            raw_bytes = bytes(1 if val else 0 for val in ftensor.reshape(-1).tolist())
            dtype_name = "bool"
        entries[key] = ("T", dtype_name, shape, raw_bytes)

    return _hash_framed_entries(entries), frozen_kwargs


def validate_supervision(
    supervision: dict,
    *,
    forbidden_kwargs: dict | None = None,
    frozen_kwargs: dict | None = None,
) -> None:
    """Validate supervision tensor shapes, finiteness, and strict storage isolation."""
    import torch

    if type(supervision) is not dict:
        raise ValueError("supervision must be a dict")
    if set(supervision.keys()) != set(REQUIRED_SUPERVISION_KEYS):
        raise ValueError("supervision keys do not match pinned SplitterV3 supervision contract")

    forbidden_ptrs = _storage_ptrs(forbidden_kwargs) | _storage_ptrs(frozen_kwargs)
    for key in REQUIRED_SUPERVISION_KEYS:
        tensor = supervision[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"supervision {key} must be a torch.Tensor")
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError(f"supervision {key} must be CPU torch.float32")
        if tuple(tensor.shape) != (1, WIDTH):
            raise ValueError(f"supervision {key} must have shape [1, {WIDTH}]")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"supervision {key} contains nonfinite values")
        if int(tensor.data_ptr()) in forbidden_ptrs or int(tensor.untyped_storage().data_ptr()) in forbidden_ptrs:
            raise ValueError(f"supervision {key} aliases forward kwargs storage")


def prepare_verified_forward_call(item: dict, *, expected_digest: str, context: str) -> dict:
    """Verify receipt/manifest forward digest against oracle and freeze clean kwargs pre-call."""
    if type(item) is not dict:
        raise ValueError(f"{context} item must be a dict")
    expected_hex = _require_sha256_hex(expected_digest, f"{context}.expected_digest")
    if item.get("forward_binding") != FORWARD_BINDING_VERSION:
        raise ValueError(f"{context} item forward_binding mismatch")
    if item.get("representation_firewall") != REPRESENTATION_FIREWALL:
        raise ValueError(f"{context} item representation_firewall mismatch")
    if item.get("anchor_ordinal") is None or item.get("anchor") is None:
        raise ValueError(f"{context} item missing non-null anchor grounding fields")
    if "forward_input_sha256" not in item:
        raise ValueError(f"{context} item missing forward_input_sha256")

    validate_causal_transition_meta(
        item.get("id_movi"),
        item.get("ids_frames"),
        item.get("history"),
        item.get("current"),
        item.get("target"),
        item.get("horizon"),
        anchor_ordinal=item["anchor_ordinal"],
        anchor=item["anchor"],
    )
    oracle_digest = expected_synthetic_forward_digest(
        item["id_movi"],
        item["ids_frames"],
        item["history"],
        item["current"],
        item["target"],
        item["horizon"],
    )
    if oracle_digest != expected_hex:
        raise ValueError(f"{context} expected_digest disagrees with independent causal oracle")
    if item["forward_input_sha256"] != expected_hex:
        raise ValueError(f"{context} item forward_input_sha256 disagrees with expected_digest")

    actual_digest, frozen_kwargs = canonical_forward_kwargs_digest(item.get("model_kwargs"))
    if actual_digest != expected_hex:
        raise ValueError(f"{context} actual forward kwargs digest mismatch prior to model.forward")
    validate_supervision(
        item.get("supervision"),
        forbidden_kwargs=item.get("model_kwargs"),
        frozen_kwargs=frozen_kwargs,
    )
    return frozen_kwargs


def verify_forward_target_invariance(model, item: dict) -> dict:
    """Prospective C3-02 runtime oracle (PREPARED_NOT_RUN under SOURCE_ONLY)."""
    import torch

    from .checkpoints import capture_cpu_rng_snapshot, restore_cpu_rng_snapshot
    from .objective import load_criterion
    from .samples import _image, synthetic_item

    oracle_digest = _require_sha256_hex(item.get("forward_input_sha256"), "item.forward_input_sha256")
    prepare_verified_forward_call(item, expected_digest=oracle_digest, context="invariance_entry")
    image = _image(item["id_movi"], item["ids_frames"])
    mutated_image = image.clone()
    mutated_image[item["ids_frames"].index(item["target"])] += 1.0

    clean = synthetic_item(
        item["id_movi"],
        item["ids_frames"],
        item["history"],
        item["current"],
        item["target"],
        item["horizon"],
        image=image,
    )
    dirty = synthetic_item(
        item["id_movi"],
        item["ids_frames"],
        item["history"],
        item["current"],
        item["target"],
        item["horizon"],
        image=mutated_image,
    )
    clean_kwargs = prepare_verified_forward_call(clean, expected_digest=oracle_digest, context="invariance_clean")
    dirty_kwargs = prepare_verified_forward_call(dirty, expected_digest=oracle_digest, context="invariance_dirty")

    criterion = load_criterion()
    was_training = bool(model.training)
    had_last_sem = hasattr(model, "_last_semantic_context")
    prev_last_sem = getattr(model, "_last_semantic_context", None)
    if isinstance(prev_last_sem, torch.Tensor):
        prev_last_sem = prev_last_sem.detach().clone()
    rng_snap = capture_cpu_rng_snapshot()
    try:
        model.eval()
        with torch.no_grad():
            out_clean = model.forward(**clean_kwargs)
            out_dirty = model.forward(**dirty_kwargs)
            loss_clean = float(criterion(out_clean, clean["supervision"])["total"].item())
            loss_dirty = float(criterion(out_dirty, dirty["supervision"])["total"].item())
    finally:
        model.train(was_training)
        if had_last_sem:
            model._last_semantic_context = prev_last_sem
        elif hasattr(model, "_last_semantic_context"):
            delattr(model, "_last_semantic_context")
        rb_errors = restore_cpu_rng_snapshot(rng_snap)
        if rb_errors:
            raise RuntimeError(f"RNG restore failed in invariance check: {rb_errors}")

    for key in ("pred_hybrid", "delta_local", "delta_anchor"):
        if not bool(torch.equal(out_clean[key], out_dirty[key])):
            raise ValueError(f"target-only mutation altered forward prediction {key}")
    if loss_clean == loss_dirty:
        raise ValueError("target-only mutation did not alter supervision loss")
    return {
        "forward_input_sha256": oracle_digest,
        "loss_clean": loss_clean,
        "loss_dirty": loss_dirty,
    }
