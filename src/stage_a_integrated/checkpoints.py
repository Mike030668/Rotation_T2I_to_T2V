"""Bounded stca-v2 checkpoints, staged resume validation, prov-v1, and crash-safe BEST/LAST."""

from __future__ import annotations

import hashlib
import io
import math
import os
from pathlib import Path

from . import CANDIDATE_SHA256, HORIZONS_SHA256, LOSS_SHA256, SAMPLER_SHA256
from .exposure import (
    FORWARD_BINDING,
    HEX_CHARS,
    MAX_JOURNAL_BYTES,
    MAX_SAMPLER_BYTES,
    SAMPLER_SCHEMA_VERSION,
    SCHEMA_VERSION,
    canonical_json_bytes,
    strict_json_loads,
)
from .plan import REMAINDER_POLICY, SYNTHETIC_SOURCE_ID, SYNTHETIC_TRAIN_IDS_SHA256, require_exact_int

MAGIC = b"STCA"
CHECKPOINT_SCHEMA_VERSION = "stca-v2"
MANIFEST_SCHEMA_VERSION = "ckpt-manifest-v1"
PROVENANCE_SCHEMA_VERSION = "prov-v1"
POINTER_SCHEMA_VERSION = "ptr-v1"
FAILURE_RECEIPT_SCHEMA_VERSION = "fail-receipt-v1"
OWNER_TRUST_ROOT_STATUS = "PENDING_OWNER_C4"

SECTIONS = ("model", "adamw", "python_rng", "numpy_rng", "torch_rng", "sampler", "exposure")

MAX_HEADER_BYTES = 1 * 1024 * 1024
MAX_MODEL_SECTION_BYTES = 128 * 1024 * 1024
MAX_ADAMW_SECTION_BYTES = 128 * 1024 * 1024
MAX_RNG_SECTION_BYTES = 65536
MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024
MAX_POINTER_BYTES = 16384
MAX_MANIFEST_BYTES = 65536
MAX_HORIZONS_FILE_BYTES = 16 * 1024 * 1024
MAX_FAILURE_RECEIPTS = 1024

SECTION_MIN_BYTES = {
    "model": 1,
    "adamw": 1,
    "python_rng": 2,
    "numpy_rng": 2,
    "torch_rng": 1,
    "sampler": 2,
    "exposure": 2,
}
SECTION_MAX_BYTES = {
    "model": MAX_MODEL_SECTION_BYTES,
    "adamw": MAX_ADAMW_SECTION_BYTES,
    "python_rng": MAX_RNG_SECTION_BYTES,
    "numpy_rng": MAX_RNG_SECTION_BYTES,
    "torch_rng": MAX_RNG_SECTION_BYTES,
    "sampler": MAX_SAMPLER_BYTES,
    "exposure": MAX_JOURNAL_BYTES,
}


def _is_hex64(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(ch in HEX_CHARS for ch in value)


def _require_hex64(value: object, name: str) -> str:
    if not _is_hex64(value):
        raise ValueError(f"{name} must be a 64-char lowercase hex sha256")
    return str(value)


def parse_generation_id(generation_id: object) -> int:
    if type(generation_id) is not str or not generation_id.startswith("u"):
        raise ValueError("generation_id must be a canonical u<step> string")
    step_text = generation_id[1:]
    if not step_text.isascii() or not step_text.isdigit():
        raise ValueError("generation_id must be a canonical u<step> string")
    step = int(step_text)
    if generation_id != f"u{step}":
        raise ValueError("generation_id must not contain leading zeros")
    return step


def file_sha256(path: Path, *, max_bytes: int = MAX_CHECKPOINT_BYTES) -> str:
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_file():
        raise ValueError(f"expected regular file: {path.name}")
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"file exceeds bounded limit ({size} > {max_bytes})")
    digest = hashlib.sha256()
    seen = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            seen += len(chunk)
            if seen > max_bytes:
                raise ValueError("file grew past bounded limit while hashing")
            digest.update(chunk)
    return digest.hexdigest()


def _read_bounded_bytes(path: Path, *, max_bytes: int, label: str) -> bytes:
    if not path.is_file():
        raise ValueError(f"{label} file missing")
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"{label} exceeds bounded limit ({size} > {max_bytes})")
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ValueError(f"{label} read exceeded bounded limit")
    return raw


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def build_expected_provenance(
    effective_config_bytes: bytes,
    horizons_path: Path,
    *,
    generation_id: str | None = None,
    horizons_file_sha256: str | None = None,
) -> dict:
    if type(effective_config_bytes) is not bytes or not effective_config_bytes:
        raise ValueError("effective_config_bytes must be non-empty bytes")
    if generation_id is not None:
        parse_generation_id(generation_id)
    horizons_digest = (
        _require_hex64(horizons_file_sha256, "horizons_file_sha256")
        if horizons_file_sha256 is not None
        else file_sha256(horizons_path, max_bytes=MAX_HORIZONS_FILE_BYTES)
    )
    return {
        "provenance_schema": PROVENANCE_SCHEMA_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "manifest_schema": MANIFEST_SCHEMA_VERSION,
        "pointer_schema": POINTER_SCHEMA_VERSION,
        "failure_receipt_schema": FAILURE_RECEIPT_SCHEMA_VERSION,
        "source_id": SYNTHETIC_SOURCE_ID,
        "source_sha256": SYNTHETIC_TRAIN_IDS_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "loss_sha256": LOSS_SHA256,
        "horizons_sha256": HORIZONS_SHA256,
        "horizons_file_sha256": horizons_digest,
        "sampler_sha256": SAMPLER_SHA256,
        "exposure_schema": SCHEMA_VERSION,
        "sampler_schema": SAMPLER_SCHEMA_VERSION,
        "remainder_policy": REMAINDER_POLICY,
        "forward_binding": FORWARD_BINDING,
        "effective_config_sha256": hashlib.sha256(effective_config_bytes).hexdigest(),
        "generation_id": generation_id,
        "owner_trust_root_status": OWNER_TRUST_ROOT_STATUS,
        "release_created": False,
    }


def verify_baseline_provenance(header: dict) -> dict:
    if type(header) is not dict:
        raise ValueError("header must be a dict")
    completed = require_exact_int(header.get("completed"), "header.completed", minimum=0)
    prov = header.get("provenance")
    if type(prov) is not dict:
        raise ValueError("checkpoint header missing prov-v1 provenance")
    required_keys = {
        "provenance_schema",
        "checkpoint_schema",
        "manifest_schema",
        "pointer_schema",
        "failure_receipt_schema",
        "source_id",
        "source_sha256",
        "candidate_sha256",
        "loss_sha256",
        "horizons_sha256",
        "horizons_file_sha256",
        "sampler_sha256",
        "exposure_schema",
        "sampler_schema",
        "remainder_policy",
        "forward_binding",
        "effective_config_sha256",
        "generation_id",
        "owner_trust_root_status",
        "release_created",
    }
    if set(prov) != required_keys:
        raise ValueError("checkpoint provenance field set invalid")
    expected_pins = {
        "provenance_schema": PROVENANCE_SCHEMA_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "manifest_schema": MANIFEST_SCHEMA_VERSION,
        "pointer_schema": POINTER_SCHEMA_VERSION,
        "failure_receipt_schema": FAILURE_RECEIPT_SCHEMA_VERSION,
        "source_id": SYNTHETIC_SOURCE_ID,
        "source_sha256": SYNTHETIC_TRAIN_IDS_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "loss_sha256": LOSS_SHA256,
        "horizons_sha256": HORIZONS_SHA256,
        "sampler_sha256": SAMPLER_SHA256,
        "exposure_schema": SCHEMA_VERSION,
        "sampler_schema": SAMPLER_SCHEMA_VERSION,
        "remainder_policy": REMAINDER_POLICY,
        "forward_binding": FORWARD_BINDING,
        "owner_trust_root_status": OWNER_TRUST_ROOT_STATUS,
    }
    for key, expected_val in expected_pins.items():
        if prov.get(key) != expected_val or type(prov.get(key)) is not str:
            raise ValueError(f"checkpoint provenance.{key} mismatch")
    if prov.get("release_created") is not False:
        raise ValueError("checkpoint provenance.release_created must be false")
    _require_hex64(prov.get("horizons_file_sha256"), "provenance.horizons_file_sha256")
    _require_hex64(prov.get("effective_config_sha256"), "provenance.effective_config_sha256")
    gen_step = parse_generation_id(prov.get("generation_id"))
    if gen_step != completed:
        raise ValueError("checkpoint provenance.generation_id does not match completed")
    if header.get("source_id") != prov["source_id"]:
        raise ValueError("header.source_id provenance mismatch")
    if header.get("source_sha256") != prov["source_sha256"]:
        raise ValueError("header.source_sha256 provenance mismatch")
    if header.get("sampler_sha256") != prov["sampler_sha256"]:
        raise ValueError("header.sampler_sha256 provenance mismatch")
    if header.get("exposure_schema") != prov["exposure_schema"]:
        raise ValueError("header.exposure_schema provenance mismatch")
    return prov


def verify_checkpoint_provenance(header: dict, expected_provenance: dict) -> dict:
    if type(expected_provenance) is not dict:
        raise ValueError("expected_provenance must be a dict")
    prov = verify_baseline_provenance(header)
    if set(prov) != set(expected_provenance):
        raise ValueError("checkpoint provenance field set mismatch")
    for key, expected_val in expected_provenance.items():
        if key == "generation_id":
            if expected_val is not None and prov[key] != expected_val:
                raise ValueError("checkpoint provenance.generation_id mismatch")
        elif prov[key] != expected_val or type(prov[key]) is not type(expected_val):
            raise ValueError(f"checkpoint provenance.{key} mismatch")
    return prov


def pack(header: dict, sections: dict[str, bytes]) -> bytes:
    if type(header) is not dict or "completed" not in header:
        raise ValueError("checkpoint header is incomplete")
    require_exact_int(header.get("completed"), "header.completed", minimum=0)
    if "provenance" in header and type(header["provenance"]) is not dict:
        raise ValueError("checkpoint provenance must be a dict")
    if type(sections) is not dict or set(sections) != set(SECTIONS):
        raise ValueError("checkpoint section set mismatch")

    body = bytearray()
    table = []
    for name in SECTIONS:
        raw = sections[name]
        if type(raw) is not bytes:
            raise ValueError(f"checkpoint section {name} must be bytes")
        if len(raw) < SECTION_MIN_BYTES[name] or len(raw) > SECTION_MAX_BYTES[name]:
            raise ValueError(f"checkpoint section {name} length out of bounds")
        table.append(
            {
                "name": name,
                "offset": len(body),
                "length": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        body += raw

    stored = dict(header)
    stored["checkpoint_schema"] = CHECKPOINT_SCHEMA_VERSION
    stored["sections"] = table
    stored["body_bytes"] = len(body)
    stored["body_sha256"] = hashlib.sha256(body).hexdigest()
    encoded = canonical_json_bytes(stored, max_bytes=MAX_HEADER_BYTES)
    payload = MAGIC + len(encoded).to_bytes(4, "little") + encoded + bytes(body)
    if len(payload) > MAX_CHECKPOINT_BYTES:
        raise ValueError("packed checkpoint exceeds MAX_CHECKPOINT_BYTES")
    return payload


def unpack(payload: bytes) -> dict:
    if type(payload) is not bytes:
        raise ValueError("checkpoint payload must be bytes")
    if len(payload) < 8 or len(payload) > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint payload size out of bounds")
    if not payload.startswith(MAGIC):
        raise ValueError("checkpoint magic mismatch")

    header_len = int.from_bytes(payload[4:8], "little")
    if header_len < 2 or header_len > MAX_HEADER_BYTES or 8 + header_len > len(payload):
        raise ValueError("checkpoint header length invalid")

    header_bytes = payload[8 : 8 + header_len]
    header = strict_json_loads(header_bytes, max_bytes=MAX_HEADER_BYTES)
    if type(header) is not dict:
        raise ValueError("checkpoint header must be a JSON object")
    if header.get("checkpoint_schema") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema must be stca-v2")
    require_exact_int(header.get("completed"), "header.completed", minimum=0)

    body = payload[8 + header_len :]
    body_bytes = require_exact_int(
        header.get("body_bytes"),
        "header.body_bytes",
        minimum=1,
        maximum=MAX_CHECKPOINT_BYTES,
    )
    if body_bytes != len(body):
        raise ValueError("checkpoint body_bytes length mismatch")
    body_sha256 = _require_hex64(header.get("body_sha256"), "header.body_sha256")
    if hashlib.sha256(body).hexdigest() != body_sha256:
        raise ValueError("checkpoint body sha256 mismatch")

    table = header.get("sections")
    if type(table) is not list or len(table) != len(SECTIONS):
        raise ValueError("checkpoint section set mismatch")

    decoded = {"header": header}
    cursor = 0
    for expected_name, item in zip(SECTIONS, table):
        if type(item) is not dict or set(item) != {"name", "offset", "length", "sha256"}:
            raise ValueError("checkpoint section entry malformed")
        if item.get("name") != expected_name:
            raise ValueError("checkpoint section set mismatch")
        offset = require_exact_int(
            item.get("offset"),
            f"sections.{expected_name}.offset",
            minimum=0,
            maximum=len(body),
        )
        length = require_exact_int(
            item.get("length"),
            f"sections.{expected_name}.length",
            minimum=SECTION_MIN_BYTES[expected_name],
            maximum=SECTION_MAX_BYTES[expected_name],
        )
        sha256 = _require_hex64(item.get("sha256"), f"sections.{expected_name}.sha256")
        if offset != cursor:
            raise ValueError(f"checkpoint section {expected_name} offset discontinuity")
        end = offset + length
        if end > len(body):
            raise ValueError(f"checkpoint section {expected_name} truncated")
        raw = body[offset:end]
        if hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError(f"checkpoint section {expected_name} mismatch")
        decoded[expected_name] = raw
        cursor = end

    if cursor != len(body) or cursor != body_bytes:
        raise ValueError("checkpoint has trailing bytes")
    return decoded


def validate_python_rng_bytes(raw: bytes) -> tuple[int, tuple[int, ...], float | None]:
    if type(raw) is not bytes or len(raw) > MAX_RNG_SECTION_BYTES:
        raise ValueError("python_rng section must be bounded bytes")
    obj = strict_json_loads(raw, max_bytes=MAX_RNG_SECTION_BYTES)
    if type(obj) is not list or len(obj) != 3:
        raise ValueError("python_rng must be a 3-element JSON list")
    version = require_exact_int(obj[0], "python_rng.version", minimum=3, maximum=3)
    raw_state = obj[1]
    if type(raw_state) is not list or len(raw_state) != 625:
        raise ValueError("python_rng.state must have 625 MT19937 words")
    words = [
        require_exact_int(word, "python_rng.word", minimum=0, maximum=0xFFFFFFFF)
        for word in raw_state[:624]
    ]
    index = require_exact_int(raw_state[624], "python_rng.index", minimum=0, maximum=624)
    gauss = obj[2]
    if gauss is not None:
        if type(gauss) not in (int, float) or isinstance(gauss, bool) or not math.isfinite(float(gauss)):
            raise ValueError("python_rng.gauss must be null or finite float")
        gauss = float(gauss)
    return (version, tuple(words + [index]), gauss)


def validate_numpy_rng_bytes(raw: bytes) -> tuple[str, list[int], int, int, float]:
    if type(raw) is not bytes or len(raw) > MAX_RNG_SECTION_BYTES:
        raise ValueError("numpy_rng section must be bounded bytes")
    obj = strict_json_loads(raw, max_bytes=MAX_RNG_SECTION_BYTES)
    if type(obj) is not list or len(obj) != 5:
        raise ValueError("numpy_rng must be a 5-element JSON list")
    algo, keys, pos, has_gauss, cached = obj
    if type(algo) is not str or algo != "MT19937":
        raise ValueError("numpy_rng.algo must be MT19937")
    if type(keys) is not list or len(keys) != 624:
        raise ValueError("numpy_rng.keys must have 624 MT19937 words")
    validated_keys = [
        require_exact_int(word, "numpy_rng.key", minimum=0, maximum=0xFFFFFFFF)
        for word in keys
    ]
    validated_pos = require_exact_int(pos, "numpy_rng.pos", minimum=0, maximum=624)
    validated_has_gauss = require_exact_int(has_gauss, "numpy_rng.has_gauss", minimum=0, maximum=1)
    if type(cached) not in (int, float) or isinstance(cached, bool) or not math.isfinite(float(cached)):
        raise ValueError("numpy_rng.cached must be finite numeric")
    return (algo, validated_keys, validated_pos, validated_has_gauss, float(cached))


def validate_torch_rng_bytes(raw: bytes, *, expected_len: int | None = None) -> bytes:
    if type(raw) is not bytes or len(raw) < 1 or len(raw) > MAX_RNG_SECTION_BYTES:
        raise ValueError("torch_rng section must be bounded non-empty bytes")
    if expected_len is not None and len(raw) != expected_len:
        raise ValueError(f"torch_rng byte length mismatch ({len(raw)} != {expected_len})")
    return raw


def stage_python_rng(raw: bytes) -> tuple[int, tuple[int, ...], float | None]:
    return validate_python_rng_bytes(raw)


def stage_numpy_rng(raw: bytes) -> tuple:
    import numpy

    algo, keys, pos, has_gauss, cached = validate_numpy_rng_bytes(raw)
    return (algo, numpy.array(keys, dtype="uint32"), pos, has_gauss, cached)


def stage_torch_rng(raw: bytes):
    import torch

    expected_len = int(torch.get_rng_state().numel())
    validate_torch_rng_bytes(raw, expected_len=expected_len)
    candidate = torch.tensor(list(raw), dtype=torch.uint8)
    probe = torch.Generator(device="cpu")
    probe.set_state(candidate)
    return candidate


def stage_model_state(raw: bytes, model) -> dict:
    import torch

    if type(raw) is not bytes or len(raw) < 1 or len(raw) > MAX_MODEL_SECTION_BYTES:
        raise ValueError("model section size out of bounds")
    loaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if type(loaded) is not dict:
        raise ValueError("model checkpoint state must be a dict")
    expected_state = model.state_dict()
    if set(loaded.keys()) != set(expected_state.keys()):
        raise ValueError("model state_dict key set mismatch")
    staged = {}
    for key, ref_tensor in expected_state.items():
        tensor = loaded[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"model state {key} is not a Tensor")
        if tensor.shape != ref_tensor.shape or tensor.dtype != ref_tensor.dtype:
            raise ValueError(f"model state {key} shape/dtype mismatch")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"model state {key} contains nonfinite values")
        staged[key] = tensor.detach().clone()
    return staged


def _clone_state_value(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if type(value) is dict:
        return {k: _clone_state_value(v) for k, v in value.items()}
    if type(value) is list:
        return [_clone_state_value(v) for v in value]
    if type(value) is tuple:
        return tuple(_clone_state_value(v) for v in value)
    return value


def _validate_adamw_slot(slot: object, expected_shape, *, completed: int) -> dict:
    import torch

    if type(slot) is not dict:
        raise ValueError("adamw state entry must be a dict")
    if not slot:
        if completed == 0:
            return {}
        raise ValueError("adamw state entry unexpectedly empty after updates")
    validated = {}
    for field, val in slot.items():
        if isinstance(val, torch.Tensor):
            if val.is_floating_point() or val.is_complex():
                if not bool(torch.isfinite(val).all().item()):
                    raise ValueError(f"adamw state {field} contains nonfinite values")
            if field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") and val.shape != expected_shape:
                raise ValueError(f"adamw state {field} tensor shape mismatch")
            if field == "step":
                if val.ndim != 0 or float(val.item()) < 0.0:
                    raise ValueError("adamw state step tensor must be non-negative scalar")
            validated[field] = val.detach().clone()
        elif field == "step":
            step_val = require_exact_int(val, "adamw.step", minimum=0)
            validated[field] = step_val
        elif type(val) in (int, float, bool, str) or val is None:
            if type(val) is float and not math.isfinite(val):
                raise ValueError(f"adamw state {field} is nonfinite")
            validated[field] = val
        else:
            raise ValueError(f"adamw state {field} has unsupported type")
    return validated


def stage_adamw_state(raw: bytes, opt, *, completed: int = 1) -> dict:
    import torch

    if type(raw) is not bytes or len(raw) < 1 or len(raw) > MAX_ADAMW_SECTION_BYTES:
        raise ValueError("adamw section size out of bounds")
    loaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if type(loaded) is not dict or set(loaded.keys()) != {"state", "param_groups"}:
        raise ValueError("adamw checkpoint state structure invalid")
    raw_groups = loaded.get("param_groups")
    raw_state = loaded.get("state")
    if type(raw_groups) is not list or type(raw_state) is not dict:
        raise ValueError("adamw param_groups/state type invalid")
    if len(raw_groups) != len(opt.param_groups) or not raw_groups:
        raise ValueError("adamw param_groups count mismatch")

    saved_param_ids: list[int] = []
    live_params = []
    staged_groups = []
    for grp, live_grp in zip(raw_groups, opt.param_groups):
        if type(grp) is not dict or type(grp.get("params")) is not list:
            raise ValueError("adamw param_group malformed")
        if len(grp["params"]) != len(live_grp["params"]):
            raise ValueError("adamw param_group parameter count mismatch")
        pids = [require_exact_int(pid, "adamw.param_id", minimum=0) for pid in grp["params"]]
        saved_param_ids.extend(pids)
        live_params.extend(live_grp["params"])
        staged_groups.append(_clone_state_value(grp))

    if len(set(saved_param_ids)) != len(saved_param_ids):
        raise ValueError("adamw duplicate parameter id in param_groups")
    id_to_shape = dict(zip(saved_param_ids, (p.shape for p in live_params)))
    expected_state_pids = {
        pid for pid, param in zip(saved_param_ids, live_params) if getattr(param, "requires_grad", True)
    }
    if completed > 0 and set(raw_state.keys()) != expected_state_pids:
        raise ValueError("adamw state entry set mismatch after completed updates")

    staged_state = {}
    for pid, slot in raw_state.items():
        if type(pid) is not int or isinstance(pid, bool) or pid not in id_to_shape:
            raise ValueError("adamw state entry references unknown parameter id")
        staged_state[pid] = _validate_adamw_slot(slot, id_to_shape[pid], completed=completed)
    return {"state": staged_state, "param_groups": staged_groups}


def clone_optimizer_state_dict(opt) -> dict:
    return _clone_state_value(opt.state_dict())


def _safe_generation_relpath(root: Path, relpath: object, *, label: str) -> Path:
    if type(relpath) is not str or not relpath or relpath.startswith("/") or "\\" in relpath:
        raise ValueError(f"pointer {label} must be a canonical relative POSIX path")
    parts = relpath.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"pointer {label} contains invalid path segment")
    generations_root = (root / "generations").resolve()
    resolved = (root / relpath).resolve()
    try:
        resolved.relative_to(generations_root)
    except ValueError as exc:
        raise ValueError(f"pointer {label} escapes generations root") from exc
    return resolved


def _parse_pointer_bytes(root: Path, raw: bytes, expected_role: str | None = None) -> dict:
    ptr = strict_json_loads(raw, max_bytes=MAX_POINTER_BYTES)
    required = {
        "pointer_schema",
        "role",
        "generation_id",
        "generation_dir",
        "checkpoint_relpath",
        "manifest_relpath",
        "checkpoint_sha256",
        "manifest_sha256",
        "completed",
        "macro",
        "previous_generation_id",
        "previous_checkpoint_sha256",
    }
    if type(ptr) is not dict or set(ptr) != required:
        raise ValueError("pointer field set invalid")
    if ptr.get("pointer_schema") != POINTER_SCHEMA_VERSION:
        raise ValueError("pointer schema mismatch")
    role = ptr.get("role")
    if role not in {"BEST", "LAST"}:
        raise ValueError("pointer role invalid")
    if expected_role is not None and role != expected_role:
        raise ValueError("pointer role mismatch")
    gen_id = ptr.get("generation_id")
    gen_step = parse_generation_id(gen_id)
    completed = require_exact_int(ptr.get("completed"), "pointer.completed", minimum=0)
    if completed != gen_step:
        raise ValueError("pointer generation_id/completed mismatch")
    _require_hex64(ptr.get("checkpoint_sha256"), "pointer.checkpoint_sha256")
    _require_hex64(ptr.get("manifest_sha256"), "pointer.manifest_sha256")
    if ptr.get("generation_dir") != f"generations/{gen_id}-{role}":
        raise ValueError("pointer.generation_dir mismatch")
    if ptr.get("checkpoint_relpath") != f"generations/{gen_id}-{role}/checkpoint.bin":
        raise ValueError("pointer.checkpoint_relpath mismatch")
    if ptr.get("manifest_relpath") != f"generations/{gen_id}-{role}/manifest.json":
        raise ValueError("pointer.manifest_relpath mismatch")
    _safe_generation_relpath(root, ptr["checkpoint_relpath"], label="checkpoint_relpath")
    _safe_generation_relpath(root, ptr["manifest_relpath"], label="manifest_relpath")
    prev_gen = ptr.get("previous_generation_id")
    prev_sha = ptr.get("previous_checkpoint_sha256")
    if (prev_gen is None) != (prev_sha is None):
        raise ValueError("pointer previous generation pairing incomplete")
    if prev_gen is not None:
        parse_generation_id(prev_gen)
        _require_hex64(prev_sha, "pointer.previous_checkpoint_sha256")
    macro = ptr.get("macro")
    if macro is not None and (type(macro) not in (int, float) or isinstance(macro, bool) or not math.isfinite(float(macro))):
        raise ValueError("pointer.macro must be null or finite float")
    return ptr


def resolve_pointer(root: Path, role: str) -> dict:
    if not isinstance(root, Path):
        root = Path(root)
    if role not in {"BEST", "LAST"} or "_scored_event_pkg" in str(root):
        raise ValueError("pointer role or root rejected")
    pointer_path = root / f"CURRENT_{role}"
    ptr_bytes = _read_bounded_bytes(pointer_path, max_bytes=MAX_POINTER_BYTES, label=f"CURRENT_{role}")
    ptr = _parse_pointer_bytes(root, ptr_bytes, expected_role=role)

    man_path = _safe_generation_relpath(root, ptr["manifest_relpath"], label="manifest_relpath")
    man_bytes = _read_bounded_bytes(man_path, max_bytes=MAX_MANIFEST_BYTES, label="manifest.json")
    if hashlib.sha256(man_bytes).hexdigest() != ptr["manifest_sha256"]:
        raise ValueError("checkpoint manifest sha256 mismatch")
    manifest = strict_json_loads(man_bytes, max_bytes=MAX_MANIFEST_BYTES)
    required_man = {
        "manifest_schema",
        "checkpoint_schema",
        "provenance_schema",
        "generation_id",
        "role",
        "completed",
        "checkpoint_bytes",
        "checkpoint_sha256",
        "header_sha256",
        "provenance_sha256",
        "provenance",
    }
    if type(manifest) is not dict or set(manifest) != required_man:
        raise ValueError("checkpoint manifest field set invalid")
    if manifest.get("manifest_schema") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("checkpoint manifest schema mismatch")
    if manifest.get("checkpoint_schema") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint manifest checkpoint_schema mismatch")
    if manifest.get("provenance_schema") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("checkpoint manifest provenance_schema mismatch")
    if manifest.get("generation_id") != ptr["generation_id"] or manifest.get("role") != role:
        raise ValueError("checkpoint manifest identity mismatch")
    if manifest.get("completed") != ptr["completed"] or manifest.get("checkpoint_sha256") != ptr["checkpoint_sha256"]:
        raise ValueError("checkpoint manifest digest/step mismatch")

    ckpt_path = _safe_generation_relpath(root, ptr["checkpoint_relpath"], label="checkpoint_relpath")
    payload = _read_bounded_bytes(ckpt_path, max_bytes=MAX_CHECKPOINT_BYTES, label="checkpoint.bin")
    if len(payload) != manifest.get("checkpoint_bytes"):
        raise ValueError("checkpoint byte count mismatch")
    if hashlib.sha256(payload).hexdigest() != ptr["checkpoint_sha256"]:
        raise ValueError("checkpoint sha256 mismatch")

    decoded = unpack(payload)
    header = decoded["header"]
    if header.get("completed") != ptr["completed"]:
        raise ValueError("checkpoint header.completed disagrees with pointer")
    header_sha = hashlib.sha256(canonical_json_bytes(header, max_bytes=MAX_HEADER_BYTES)).hexdigest()
    if header_sha != manifest.get("header_sha256"):
        raise ValueError("checkpoint header sha256 disagrees with manifest")
    prov = verify_baseline_provenance(header)
    if prov.get("generation_id") != ptr["generation_id"]:
        raise ValueError("checkpoint header provenance disagrees with pointer generation_id")
    if manifest.get("provenance") != prov:
        raise ValueError("checkpoint manifest provenance disagrees with header provenance")
    prov_sha = hashlib.sha256(canonical_json_bytes(prov, max_bytes=MAX_HEADER_BYTES)).hexdigest()
    if prov_sha != manifest.get("provenance_sha256"):
        raise ValueError("checkpoint provenance sha256 mismatch")

    return {"pointer": ptr, "manifest": manifest, "payload": payload, "decoded": decoded}


def write_failure_receipt(
    root: Path,
    stage: str,
    reason: str,
    *,
    generation_id: str | None = None,
    details: dict | None = None,
) -> dict:
    if not isinstance(root, Path):
        root = Path(root)
    if "_scored_event_pkg" in str(root):
        raise ValueError("failure receipt root rejected")
    if type(stage) is not str or not stage:
        raise ValueError("failure stage must be a non-empty string")
    if type(reason) is not str or not reason:
        reason = "unspecified failure"
    if generation_id is not None:
        parse_generation_id(generation_id)
    if details is not None and type(details) is not dict:
        raise ValueError("failure details must be a dict or None")

    failures_dir = root / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    safe_stage = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stage)[:48] or "failure"
    receipt = {
        "failure_receipt_schema": FAILURE_RECEIPT_SCHEMA_VERSION,
        "stage": stage,
        "generation_id": generation_id,
        "reason": reason[:512],
        "details": {} if details is None else details,
        "owner_trust_root_status": OWNER_TRUST_ROOT_STATUS,
        "release_created": False,
    }
    receipt_bytes = canonical_json_bytes(receipt, max_bytes=MAX_MANIFEST_BYTES)
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()

    slot_path: Path | None = None
    for seq in range(1, MAX_FAILURE_RECEIPTS + 1):
        candidate = failures_dir / f"{seq:04d}-{safe_stage}.json"
        if not candidate.exists():
            slot_path = candidate
            break
    if slot_path is None:
        raise ValueError("failure receipt directory reached MAX_FAILURE_RECEIPTS")

    partial = failures_dir / f".{slot_path.name}.partial"
    partial.write_bytes(receipt_bytes)
    _fsync_file(partial)
    if partial.read_bytes() != receipt_bytes:
        raise ValueError("failure receipt staged readback mismatch")
    os.replace(partial, slot_path)
    _fsync_dir(failures_dir)
    if slot_path.read_bytes() != receipt_bytes:
        raise ValueError("failure receipt readback mismatch")

    return {
        "path": slot_path.name,
        "receipt_sha256": receipt_sha256,
        "receipt": receipt,
    }


def publish(root: Path, generation_id: str, role: str, payload: bytes) -> dict:
    if not isinstance(root, Path):
        root = Path(root)
    if role not in {"BEST", "LAST"} or "_scored_event_pkg" in str(root):
        raise ValueError("checkpoint path rejected")

    gen_step = parse_generation_id(generation_id)
    decoded = unpack(payload)
    header = decoded["header"]
    if header.get("completed") != gen_step:
        raise ValueError("publish generation_id does not match header.completed")
    prov = verify_baseline_provenance(header)
    if prov.get("generation_id") != generation_id:
        raise ValueError("publish provenance.generation_id mismatch")

    macro_val = header.get("macro")
    if macro_val is not None and (
        type(macro_val) not in (int, float) or isinstance(macro_val, bool) or not math.isfinite(float(macro_val))
    ):
        raise ValueError("publish header.macro must be null or finite numeric")

    generations = root / "generations"
    stage_root = root / "stage"
    generations.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)
    if generations.stat().st_dev != stage_root.stat().st_dev:
        raise ValueError("stage and generations must be on the same filesystem")

    final = generations / f"{generation_id}-{role}"
    if final.exists():
        raise ValueError("generation already exists; immutable BEST/LAST refuses overwrite")

    pointer_name = f"CURRENT_{role}"
    pointer_path = root / pointer_name
    prev_gen_id: str | None = None
    prev_ckpt_sha256: str | None = None
    if pointer_path.exists():
        prev_bytes = _read_bounded_bytes(pointer_path, max_bytes=MAX_POINTER_BYTES, label=pointer_name)
        prev_ptr = _parse_pointer_bytes(root, prev_bytes, expected_role=role)
        prev_gen_id = prev_ptr["generation_id"]
        prev_ckpt_sha256 = prev_ptr["checkpoint_sha256"]

    checkpoint_sha256 = hashlib.sha256(payload).hexdigest()
    header_sha256 = hashlib.sha256(canonical_json_bytes(header, max_bytes=MAX_HEADER_BYTES)).hexdigest()
    provenance_sha256 = hashlib.sha256(canonical_json_bytes(prov, max_bytes=MAX_HEADER_BYTES)).hexdigest()
    manifest = {
        "manifest_schema": MANIFEST_SCHEMA_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "provenance_schema": PROVENANCE_SCHEMA_VERSION,
        "generation_id": generation_id,
        "role": role,
        "completed": gen_step,
        "checkpoint_bytes": len(payload),
        "checkpoint_sha256": checkpoint_sha256,
        "header_sha256": header_sha256,
        "provenance_sha256": provenance_sha256,
        "provenance": prov,
    }
    manifest_bytes = canonical_json_bytes(manifest, max_bytes=MAX_MANIFEST_BYTES)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    stage = stage_root / f"{generation_id}-{role}"
    stage.mkdir(parents=False, exist_ok=False)
    target = stage / "checkpoint.bin"
    target.write_bytes(payload)
    _fsync_file(target)
    man_stage = stage / "manifest.json"
    man_stage.write_bytes(manifest_bytes)
    _fsync_file(man_stage)
    _fsync_dir(stage)

    staged_readback = target.read_bytes()
    if staged_readback != payload or hashlib.sha256(staged_readback).hexdigest() != checkpoint_sha256:
        raise ValueError("staged readback mismatch")
    if man_stage.read_bytes() != manifest_bytes:
        raise ValueError("staged manifest readback mismatch")
    unpack(staged_readback)

    os.rename(stage, final)
    _fsync_dir(generations)
    _fsync_dir(stage_root)

    if (final / "checkpoint.bin").read_bytes() != payload:
        raise ValueError("checkpoint readback mismatch")
    if (final / "manifest.json").read_bytes() != manifest_bytes:
        raise ValueError("manifest readback mismatch")

    ptr = {
        "pointer_schema": POINTER_SCHEMA_VERSION,
        "role": role,
        "generation_id": generation_id,
        "generation_dir": f"generations/{generation_id}-{role}",
        "checkpoint_relpath": f"generations/{generation_id}-{role}/checkpoint.bin",
        "manifest_relpath": f"generations/{generation_id}-{role}/manifest.json",
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "completed": gen_step,
        "macro": macro_val,
        "previous_generation_id": prev_gen_id,
        "previous_checkpoint_sha256": prev_ckpt_sha256,
    }
    ptr_bytes = canonical_json_bytes(ptr, max_bytes=MAX_POINTER_BYTES)
    partial = root / f"{pointer_name}.partial"
    partial.write_bytes(ptr_bytes)
    _fsync_file(partial)
    if partial.read_bytes() != ptr_bytes:
        raise ValueError("pointer staged readback mismatch")
    os.replace(partial, pointer_path)
    _fsync_dir(root)
    if pointer_path.read_bytes() != ptr_bytes:
        raise ValueError("pointer readback mismatch")

    return {
        "generation_id": generation_id,
        "role": role,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "pointer_sha256": hashlib.sha256(ptr_bytes).hexdigest(),
        "previous_generation_id": prev_gen_id,
    }
