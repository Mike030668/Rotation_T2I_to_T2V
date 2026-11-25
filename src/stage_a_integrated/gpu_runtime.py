"""GPU Device Contract, FP32 determinism, CPU+CUDA RNG, portable stca-v3-gpu checkpoints, GPU train step & frozen DEV eval."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import fcntl
import hashlib
import inspect
import io
import math
import os
from pathlib import Path

from . import CANDIDATE_SHA256, HORIZONS_SHA256, LOSS_SHA256, SAMPLER_SHA256
from .authorization import OWNER_TRUST_STATUS_SYNTHETIC_TEST, OWNER_TRUST_STATUS_VERIFIED
from .checkpoints import (
    COMMIT_STATE_DURABLE,
    DEV_EXPECTED_HORIZON_COUNTS,
    DEV_METRIC_ID,
    DEV_METRIC_STATUS,
    DEV_METRIC_VERSION,
    DEV_REPORT_RECOMPUTE_TOL,
    DEV_REPORT_SCHEMA_VERSION,
    EXPECTED_ADAMW_AMSGRAD,
    EXPECTED_ADAMW_BETAS,
    EXPECTED_ADAMW_EPS,
    EXPECTED_ADAMW_LR,
    EXPECTED_ADAMW_WEIGHT_DECAY,
    FAILURE_RECEIPT_SCHEMA_VERSION,
    MAGIC,
    MANIFEST_SCHEMA_VERSION,
    MAX_ADAMW_SECTION_BYTES,
    MAX_BEST_HISTORY,
    MAX_CHECKPOINT_BYTES,
    MAX_COMMIT_BYTES,
    MAX_DEV_REPORT_BYTES,
    MAX_FAILURE_RECEIPTS,
    MAX_HEADER_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_MODEL_SECTION_BYTES,
    MAX_POINTER_BYTES,
    MAX_RNG_SECTION_BYTES,
    POINTER_SCHEMA_VERSION,
    CommitDurabilityUnknownError,
    PostCommitError,
    parse_generation_id,
    validate_event_and_best_history,
)
from .dev import (
    ANCHOR_WEIGHT,
    BETA,
    DIAG_EPS,
    DIAGNOSTICS_POLICY,
    EXPECTED_DEV_PAIRS,
    EXPECTED_DEV_VIDEOS,
    H2_POLICY,
    HORIZON_NAMES,
    LOCAL_WEIGHT,
    METRIC_AGGREGATION,
    METRIC_DIRECTION,
    METRIC_EPSILON,
    METRIC_NAME,
    METRIC_STATUS,
    METRIC_TIE_POLICY,
    METRIC_VERSION,
    PREDECLARED_ABSENT_MID_VIDEOS,
    EvaluatorMutationError,
)
from .exposure import (
    FORWARD_BINDING,
    HEX_CHARS,
    MAX_JOURNAL_BYTES,
    MAX_SAMPLER_BYTES,
    MAX_SEED,
    SAMPLER_SCHEMA_VERSION,
    SCHEMA_VERSION,
    canonical_json_bytes,
    strict_json_loads,
)
from .forward_binding import (
    ALLOWED_FORWARD_KWARGS,
    FLOAT_KWARG_KEYS,
    MASK_KWARG_KEYS,
    REQUIRED_SUPERVISION_KEYS,
    WIDTH,
    canonical_forward_kwargs_digest,
    validate_causal_transition_meta,
    validate_supervision,
)
from .historical_corpus import (
    HISTORICAL_SOURCE_ID,
    HISTORICAL_TEMPORAL_AXIS_SEMANTICS,
    PORTABLE_MANIFEST_SHA256,
    PORTABLE_SHARD_ATTESTED_SHA256,
    TEXT_MASK_POLICY,
)
from .objective import load_criterion
from .plan import REMAINDER_POLICY, require_exact_int

PRECISION_POLICY = "FP32_STRICT_NO_AMP_NO_TF32"
GPU_RNG_SCOPE = "cpu_and_cuda_device"
CUDA_RNG_SCHEMA_VERSION = "cuda-rng-v1"
GPU_CHECKPOINT_SCHEMA_VERSION = "stca-v3-gpu"
GPU_PROVENANCE_SCHEMA_VERSION = "prov-v3-gpu"
GPU_COMMIT_SCHEMA_VERSION = "commit-v3-gpu"

GPU_SECTIONS = (
    "model",
    "adamw",
    "python_rng",
    "numpy_rng",
    "torch_rng",
    "cuda_rng",
    "sampler",
    "exposure",
)
GPU_SECTION_MIN_BYTES = {
    "model": 1,
    "adamw": 1,
    "python_rng": 2,
    "numpy_rng": 2,
    "torch_rng": 1,
    "cuda_rng": 2,
    "sampler": 2,
    "exposure": 2,
}
GPU_SECTION_MAX_BYTES = {
    "model": MAX_MODEL_SECTION_BYTES,
    "adamw": MAX_ADAMW_SECTION_BYTES,
    "python_rng": MAX_RNG_SECTION_BYTES,
    "numpy_rng": MAX_RNG_SECTION_BYTES,
    "torch_rng": MAX_RNG_SECTION_BYTES,
    "cuda_rng": MAX_RNG_SECTION_BYTES,
    "sampler": MAX_SAMPLER_BYTES,
    "exposure": MAX_JOURNAL_BYTES,
}

CROSS_DEVICE_BYTE_EXACT_CLAIMED = False
SAME_GPU_PARITY_ATOL = 1e-5
SAME_GPU_PARITY_RTOL = 1e-4
CROSS_DEVICE_DIAGNOSTIC_ATOL = 1e-4
CROSS_DEVICE_DIAGNOSTIC_RTOL = 1e-3
ADAMW_STEP_DEVICE_POLICY = "CPU_SCALAR_UNCAPTURED_OR_PARAM_DEVICE_CAPTURED"
GPU_DETERMINISM_POLICY = "gpu_fp32_tf32_off_cudnn_deterministic_no_cross_device_byte_exact_claim"
ALLOWED_ACTIVE_TRUST = frozenset({OWNER_TRUST_STATUS_VERIFIED, OWNER_TRUST_STATUS_SYNTHETIC_TEST})

GPU_PROVENANCE_KEYS = frozenset(
    {
        "provenance_schema",
        "checkpoint_schema",
        "manifest_schema",
        "pointer_schema",
        "commit_schema",
        "dev_report_schema",
        "dev_metric_id",
        "dev_metric_version",
        "dev_metric_status",
        "failure_receipt_schema",
        "precision_policy",
        "rng_scope",
        "device_type",
        "cross_device_byte_exact_claimed",
        "source_id",
        "corpus_manifest_sha256",
        "corpus_shard_attested_sha256",
        "train_exclusion_manifest_sha256",
        "excluded_train_count",
        "eligible_train_ids_sha256",
        "eligible_train_count",
        "preflight_receipt_sha256",
        "candidate_sha256",
        "loss_sha256",
        "horizons_sha256",
        "sampler_sha256",
        "exposure_schema",
        "sampler_schema",
        "remainder_policy",
        "forward_binding",
        "temporal_axis_semantics",
        "text_mask_policy",
        "effective_config_sha256",
        "generation_id",
        "owner_trust_anchor_sha256",
        "owner_trust_root_status",
        "release_created",
    }
)


def _is_hex64(val: object) -> bool:
    return type(val) is str and len(val) == 64 and all(c in HEX_CHARS for c in val)


def _req_hex64(val: object, name: str) -> str:
    if not _is_hex64(val):
        raise ValueError(f"{name} must be a 64-char lowercase hex sha256")
    return str(val)


def _req_finite_num(val: object, name: str) -> float:
    if type(val) not in (int, float) or isinstance(val, bool) or not math.isfinite(float(val)):
        raise ValueError(f"{name} must be a finite int or float")
    return float(val)


def _read_bounded(path: Path, *, max_bytes: int, label: str) -> bytes:
    if not path.is_file():
        raise ValueError(f"{label} file missing: {path.name}")
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


@contextmanager
def _gpu_writer_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".writer.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        raise RuntimeError("GPU checkpoint root is locked by another writer") from exc
    try:
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def resolve_runtime_device(device_spec: str):
    """Explicit CUDA device resolution; fail closed if CUDA is unavailable or index invalid (never falls back to CPU)."""
    import torch

    if type(device_spec) is not str or not device_spec:
        raise ValueError("device_spec must be a nonempty str")
    if device_spec == "cuda":
        idx = 0
    elif device_spec.startswith("cuda:"):
        raw_idx = device_spec[5:]
        if not raw_idx.isascii() or not raw_idx.isdigit():
            raise ValueError(f"invalid CUDA device specification: {device_spec!r}")
        idx = int(raw_idx)
    else:
        raise ValueError(f"GPU runtime requires explicit 'cuda' or 'cuda:<index>', got {device_spec!r}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing CPU fallback under GPU device contract")
    count = int(torch.cuda.device_count())
    if idx < 0 or idx >= count:
        raise RuntimeError(f"CUDA device index {idx} out of range (device_count={count})")
    torch.cuda.set_device(idx)
    return torch.device(f"cuda:{idx}")


def configure_fp32_precision_and_determinism(device) -> dict:
    """Enforce strict FP32 (TF32 disabled, benchmark disabled, deterministic cuDNN) and record GPU telemetry."""
    import platform
    import sys

    import numpy
    import torch

    if device.type != "cuda":
        raise ValueError("configure_fp32_precision_and_determinism requires a CUDA device")

    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn is not None:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    idx = int(device.index if device.index is not None else torch.cuda.current_device())
    props = torch.cuda.get_device_properties(idx)
    cudnn_ver = torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available() else None

    return {
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "numpy_version": str(numpy.__version__),
        "platform": platform.platform(),
        "device": f"cuda:{idx}",
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": cudnn_ver,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": str(props.name),
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": int(props.total_memory),
        "precision_policy": PRECISION_POLICY,
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32) if hasattr(torch.backends, "cudnn") else None,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic) if hasattr(torch.backends, "cudnn") else None,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark) if hasattr(torch.backends, "cudnn") else None,
        "cross_device_byte_exact_claimed": CROSS_DEVICE_BYTE_EXACT_CLAIMED,
        "same_gpu_parity_tol": {"atol": SAME_GPU_PARITY_ATOL, "rtol": SAME_GPU_PARITY_RTOL},
        "cross_device_diagnostic_tol": {"atol": CROSS_DEVICE_DIAGNOSTIC_ATOL, "rtol": CROSS_DEVICE_DIAGNOSTIC_RTOL},
        "determinism_policy": GPU_DETERMINISM_POLICY,
    }


def verify_model_device_and_dtype(model, device) -> None:
    """Verify every parameter AND buffer (including non-persistent buffers like ScalarFourierEmbedding.frequencies) matches device, FP32, and finiteness."""
    import torch

    for name, param in model.named_parameters():
        if param.device != device:
            raise ValueError(f"parameter {name} on {param.device}, expected {device}")
        if param.dtype != torch.float32:
            raise ValueError(f"parameter {name} dtype {param.dtype} != torch.float32")
        if not bool(torch.isfinite(param).all().item()):
            raise ValueError(f"parameter {name} contains nonfinite values")

    for name, buf in model.named_buffers():
        if buf.device != device:
            raise ValueError(f"buffer {name} on {buf.device}, expected {device}")
        if buf.is_floating_point() and buf.dtype != torch.float32:
            raise ValueError(f"buffer {name} dtype {buf.dtype} != torch.float32")
        if (buf.is_floating_point() or buf.is_complex()) and not bool(torch.isfinite(buf).all().item()):
            raise ValueError(f"buffer {name} contains nonfinite values")


def create_pinned_adamw(model):
    """Instantiate Stage A pinned AdamW (1e-4/1e-4) with explicit capturable=False, fused=False when supported."""
    import torch

    kwargs: dict = {
        "lr": EXPECTED_ADAMW_LR,
        "weight_decay": EXPECTED_ADAMW_WEIGHT_DECAY,
        "betas": EXPECTED_ADAMW_BETAS,
        "eps": EXPECTED_ADAMW_EPS,
        "amsgrad": EXPECTED_ADAMW_AMSGRAD,
    }
    sig_params = inspect.signature(torch.optim.AdamW.__init__).parameters
    if "capturable" in sig_params:
        kwargs["capturable"] = False
    if "fused" in sig_params:
        kwargs["fused"] = False
    return torch.optim.AdamW(model.parameters(), **kwargs)


def adamw_step_expected_device(group: dict, live_param) -> str:
    """Under PyTorch AdamW semantics, step is a param-device scalar tensor iff capturable or fused, else CPU."""
    import torch

    is_capturable = bool(group.get("capturable", False))
    is_fused = bool(group.get("fused", False))
    if is_capturable or is_fused:
        return str(live_param.device)
    return str(torch.device("cpu"))


def move_forward_kwargs_to_device(frozen_cpu_kwargs: dict, device) -> dict:
    """Explicitly transfer CPU validated forward kwargs onto target device in FP32 / bool without AMP or implicit casts."""
    import torch

    if type(frozen_cpu_kwargs) is not dict or set(frozen_cpu_kwargs.keys()) != set(ALLOWED_FORWARD_KWARGS):
        raise ValueError("forward kwargs key mismatch before GPU transfer")
    out: dict = {"mode": frozen_cpu_kwargs["mode"]}
    for key in ALLOWED_FORWARD_KWARGS:
        if key == "mode":
            continue
        t = frozen_cpu_kwargs[key]
        if key in FLOAT_KWARG_KEYS:
            if t.dtype != torch.float32:
                raise ValueError(f"forward kwarg {key} must start as float32")
            moved = t.to(device=device, dtype=torch.float32, non_blocking=False).contiguous()
            if moved.device != device or moved.dtype != torch.float32 or not bool(torch.isfinite(moved).all().item()):
                raise ValueError(f"GPU transfer validation failed for {key}")
        elif key in MASK_KWARG_KEYS:
            if t.dtype != torch.bool:
                raise ValueError(f"forward mask {key} must start as bool")
            moved = t.to(device=device, dtype=torch.bool, non_blocking=False).contiguous()
            if moved.device != device or moved.dtype != torch.bool:
                raise ValueError(f"GPU mask transfer validation failed for {key}")
        else:
            raise ValueError(f"unclassified forward kwarg key: {key}")
        out[key] = moved
    return out


def move_supervision_to_device(cpu_supervision: dict, device) -> dict:
    """Explicitly transfer validated CPU float32 supervision targets onto target device."""
    import torch

    if type(cpu_supervision) is not dict or set(cpu_supervision.keys()) != set(REQUIRED_SUPERVISION_KEYS):
        raise ValueError("supervision key mismatch before GPU transfer")
    out = {}
    for key in REQUIRED_SUPERVISION_KEYS:
        t = cpu_supervision[key]
        if t.dtype != torch.float32:
            raise ValueError(f"supervision {key} must start as float32")
        moved = t.to(device=device, dtype=torch.float32, non_blocking=False).contiguous()
        if moved.device != device or moved.dtype != torch.float32 or not bool(torch.isfinite(moved).all().item()):
            raise ValueError(f"GPU supervision validation failed for {key}")
        out[key] = moved
    return out


def seed_fresh_runtime_rng(seed: int, device) -> None:
    """Seed Python, NumPy, Torch CPU, and active CUDA device RNG on fresh runs ONLY."""
    import random

    import numpy
    import torch

    seed_val = require_exact_int(seed, "seed", minimum=0, maximum=MAX_SEED)
    random.seed(seed_val)
    numpy.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if device.type == "cuda":
        idx = int(device.index if device.index is not None else torch.cuda.current_device())
        with torch.cuda.device(idx):
            torch.cuda.manual_seed(seed_val)


def capture_runtime_rng_snapshot(device) -> tuple:
    import random

    import numpy
    import torch

    py_state = random.getstate()
    np_algo, np_keys, np_pos, np_has_gauss, np_cached = numpy.random.get_state()
    np_state = (np_algo, np_keys.copy(), np_pos, np_has_gauss, np_cached)
    torch_cpu = torch.get_rng_state().clone()
    idx = int(device.index if device.index is not None else torch.cuda.current_device())
    cuda_state = torch.cuda.get_rng_state(idx).clone()
    return (py_state, np_state, torch_cpu, idx, cuda_state)


def restore_runtime_rng_snapshot(snapshot: tuple) -> list[str]:
    import random

    import numpy
    import torch

    py_state, np_state, torch_cpu, idx, cuda_state = snapshot
    errors: list[str] = []
    for name, action in (
        ("python_rng", lambda: random.setstate(py_state)),
        ("numpy_rng", lambda: numpy.random.set_state(np_state)),
        ("torch_rng", lambda: torch.set_rng_state(torch_cpu)),
        ("cuda_rng", lambda: torch.cuda.set_rng_state(cuda_state, device=idx)),
    ):
        try:
            action()
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    return errors


def rng_snapshots_equal(left: tuple, right: tuple) -> bool:
    import numpy
    import torch

    py_l, np_l, tc_l, idx_l, cu_l = left
    py_r, np_r, tc_r, idx_r, cu_r = right
    if py_l != py_r or idx_l != idx_r:
        return False
    if np_l[0] != np_r[0] or not bool(numpy.array_equal(np_l[1], np_r[1])) or np_l[2:] != np_r[2:]:
        return False
    return bool(torch.equal(tc_l, tc_r)) and bool(torch.equal(cu_l, cu_r))


def serialize_cuda_rng_bytes(device) -> bytes:
    import torch

    idx = int(device.index if device.index is not None else torch.cuda.current_device())
    raw_bytes = bytes(torch.cuda.get_rng_state(idx).cpu().tolist())
    doc = {
        "schema": CUDA_RNG_SCHEMA_VERSION,
        "device_type": "cuda",
        "device_index": idx,
        "num_bytes": len(raw_bytes),
        "state_hex": raw_bytes.hex(),
    }
    return canonical_json_bytes(doc, max_bytes=MAX_RNG_SECTION_BYTES)


def validate_cuda_rng_bytes(raw: bytes, *, expected_device_index: int | None = None):
    import torch

    if type(raw) is not bytes or len(raw) < 2 or len(raw) > MAX_RNG_SECTION_BYTES:
        raise ValueError("cuda_rng section must be bounded bytes")
    doc = strict_json_loads(raw, max_bytes=MAX_RNG_SECTION_BYTES)
    if type(doc) is not dict or set(doc.keys()) != {"schema", "device_type", "device_index", "num_bytes", "state_hex"}:
        raise ValueError("cuda_rng section does not match cuda-rng-v1 schema")
    if doc["schema"] != CUDA_RNG_SCHEMA_VERSION or doc["device_type"] != "cuda":
        raise ValueError("cuda_rng schema or device_type mismatch")
    dev_idx = require_exact_int(doc["device_index"], "cuda_rng.device_index", minimum=0)
    if expected_device_index is not None and dev_idx != expected_device_index:
        raise ValueError(f"cuda_rng device_index {dev_idx} != expected {expected_device_index}")
    nbytes = require_exact_int(doc["num_bytes"], "cuda_rng.num_bytes", minimum=1, maximum=MAX_RNG_SECTION_BYTES)
    hex_str = doc["state_hex"]
    if type(hex_str) is not str or len(hex_str) != nbytes * 2 or any(c not in HEX_CHARS for c in hex_str):
        raise ValueError("cuda_rng state_hex invalid or length mismatch")
    raw_state = bytes.fromhex(hex_str)
    candidate = torch.tensor(list(raw_state), dtype=torch.uint8, device="cpu")
    return candidate, dev_idx


def stage_cuda_rng(raw: bytes, device):
    import torch

    idx = int(device.index if device.index is not None else torch.cuda.current_device())
    expected_len = int(torch.cuda.get_rng_state(idx).numel())
    candidate, _ = validate_cuda_rng_bytes(raw, expected_device_index=idx)
    if int(candidate.numel()) != expected_len:
        raise ValueError(f"cuda_rng byte length mismatch ({candidate.numel()} != {expected_len})")
    return candidate


def serialize_portable_model_bytes(model) -> bytes:
    """Serialize model state_dict cleanly on CPU in canonical OrderedDict format for portable cross-host resume."""
    import torch

    cpu_sd = OrderedDict()
    for key, val in model.state_dict().items():
        if not isinstance(val, torch.Tensor):
            raise ValueError(f"non-tensor entry in model.state_dict(): {key}")
        if (val.is_floating_point() or val.is_complex()) and not bool(torch.isfinite(val).all().item()):
            raise ValueError(f"refusing to serialize nonfinite model state tensor {key}")
        cpu_sd[key] = val.detach().to(device="cpu").contiguous().clone()
    buf = io.BytesIO()
    torch.save(cpu_sd, buf)
    return buf.getvalue()


def serialize_portable_adamw_bytes(opt) -> bytes:
    """Serialize AdamW state_dict on CPU preserving canonical ('step', 'exp_avg', 'exp_avg_sq') key order."""
    import torch

    live_sd = opt.state_dict()
    cpu_state: dict = {}
    for pid, slot in live_sd["state"].items():
        if set(slot.keys()) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"unexpected AdamW slot keys for param {pid}: {sorted(slot.keys())}")
        step_val = slot["step"]
        if isinstance(step_val, torch.Tensor):
            if not bool(torch.isfinite(step_val).all().item()):
                raise ValueError(f"nonfinite AdamW step tensor for param {pid}")
            cpu_step = step_val.detach().to(device="cpu").clone()
        else:
            cpu_step = step_val
        ordered_slot = {"step": cpu_step}
        for field in ("exp_avg", "exp_avg_sq"):
            t = slot[field]
            if not isinstance(t, torch.Tensor) or not bool(torch.isfinite(t).all().item()):
                raise ValueError(f"nonfinite AdamW {field} tensor for param {pid}")
            ordered_slot[field] = t.detach().to(device="cpu").contiguous().clone()
        cpu_state[pid] = ordered_slot

    cpu_sd = {
        "state": cpu_state,
        "param_groups": [dict(grp) for grp in live_sd["param_groups"]],
    }
    buf = io.BytesIO()
    torch.save(cpu_sd, buf)
    return buf.getvalue()


def stage_portable_model_state(raw: bytes, model, device) -> dict:
    """Load CPU-serialized model checkpoint via weights_only=True, admit dict/OrderedDict, and map to target device."""
    import torch

    if type(raw) is not bytes or len(raw) < 1 or len(raw) > MAX_MODEL_SECTION_BYTES:
        raise ValueError("model section size out of bounds")
    loaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if type(loaded) not in (dict, OrderedDict):
        raise ValueError("model checkpoint state must be a dict or OrderedDict")
    expected_state = model.state_dict()
    if set(loaded.keys()) != set(expected_state.keys()):
        raise ValueError("model state_dict key set mismatch")
    staged = {}
    for key, ref_tensor in expected_state.items():
        if ref_tensor.device != device:
            raise ValueError(f"live model tensor {key} on {ref_tensor.device}, expected {device}")
        tensor = loaded[key]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise ValueError(f"loaded model state {key} must be a CPU Tensor prior to mapping")
        if tensor.shape != ref_tensor.shape or tensor.dtype != ref_tensor.dtype:
            raise ValueError(f"model state {key} shape/dtype mismatch")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"model state {key} contains nonfinite values")
        staged[key] = tensor.detach().to(device=device, non_blocking=False).contiguous().clone()
    return staged


def _clone_opt_val(val):
    import torch

    if isinstance(val, torch.Tensor):
        return val.detach().clone()
    if type(val) is dict:
        return {k: _clone_opt_val(v) for k, v in val.items()}
    if type(val) is list:
        return [_clone_opt_val(v) for v in val]
    if type(val) is tuple:
        return tuple(_clone_opt_val(v) for v in val)
    return val


def clone_runtime_optimizer_state_dict(opt) -> dict:
    return _clone_opt_val(opt.state_dict())


def _validate_portable_adamw_slot(slot: object, live_param, *, group: dict, completed: int, target_device) -> dict:
    import torch

    if type(slot) is not dict or set(slot.keys()) != {"step", "exp_avg", "exp_avg_sq"}:
        raise ValueError("adamw state entry must be a dict with exact keys {step, exp_avg, exp_avg_sq}")
    if live_param.device != target_device or not live_param.is_floating_point() or live_param.dtype != torch.float32:
        raise ValueError("live parameter must be float32 on target device")

    expected_step_dev_str = adamw_step_expected_device(group, live_param)
    expected_step_device = torch.device(expected_step_dev_str)

    step_val = slot["step"]
    if isinstance(step_val, torch.Tensor):
        if step_val.device.type != "cpu" or step_val.ndim != 0 or step_val.dtype == torch.bool or step_val.is_complex():
            raise ValueError("portable adamw step tensor must be non-bool real CPU scalar prior to mapping")
        if not bool(torch.isfinite(step_val).item()):
            raise ValueError("adamw step tensor is nonfinite")
        step_num = float(step_val.item())
        if not step_num.is_integer() or int(step_num) != completed:
            raise ValueError("adamw step tensor must be integral and equal completed")
        validated_step = step_val.detach().to(device=expected_step_device, non_blocking=False).clone()
    elif type(step_val) is int:
        if step_val != completed:
            raise ValueError("adamw step int must equal completed")
        validated_step = step_val
    else:
        raise ValueError("adamw step must be an integral scalar Tensor or exact int")

    # Preserve canonical PyTorch AdamW slot insertion order ("step", "exp_avg", "exp_avg_sq").
    validated = {"step": validated_step}
    for field in ("exp_avg", "exp_avg_sq"):
        tensor = slot[field]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise ValueError(f"portable adamw {field} must be a CPU Tensor")
        if tensor.shape != live_param.shape or tensor.dtype != torch.float32:
            raise ValueError(f"adamw {field} shape/dtype mismatch")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"adamw {field} contains nonfinite values")
        if field == "exp_avg_sq" and not bool((tensor >= 0).all().item()):
            raise ValueError("adamw exp_avg_sq contains negative entries")
        validated[field] = tensor.detach().to(device=target_device, non_blocking=False).contiguous().clone()

    return validated


def stage_portable_adamw_state(
    raw: bytes,
    opt,
    device,
    *,
    completed: int,
    expected_updated_param_ids: set[int] | None = None,
) -> dict:
    """Load portable CPU AdamW state via weights_only=True, enforce hyperparameters/shapes/orders, and map slots cleanly."""
    import torch

    completed_val = require_exact_int(completed, "completed", minimum=0)
    if type(raw) is not bytes or len(raw) < 1 or len(raw) > MAX_ADAMW_SECTION_BYTES:
        raise ValueError("adamw section size out of bounds")
    loaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if type(loaded) is not dict or set(loaded.keys()) != {"state", "param_groups"}:
        raise ValueError("adamw checkpoint state structure invalid")
    raw_groups = loaded.get("param_groups")
    raw_state = loaded.get("state")
    if type(raw_groups) is not list or type(raw_state) is not dict:
        raise ValueError("adamw param_groups/state type invalid")

    live_sd = opt.state_dict()
    live_sd_groups = live_sd["param_groups"]
    if len(raw_groups) != len(opt.param_groups) or len(raw_groups) != len(live_sd_groups) or not raw_groups:
        raise ValueError("adamw param_groups count mismatch")

    saved_param_ids: list[int] = []
    live_params = []
    pid_to_group: dict[int, dict] = {}
    for grp, live_grp, live_sd_grp in zip(raw_groups, opt.param_groups, live_sd_groups):
        if type(grp) is not dict or set(grp.keys()) != set(live_sd_grp.keys()):
            raise ValueError("adamw param_group keys mismatch")
        if _req_finite_num(grp["lr"], "adamw.lr") != EXPECTED_ADAMW_LR:
            raise ValueError("adamw lr mismatch")
        if _req_finite_num(grp["weight_decay"], "adamw.weight_decay") != EXPECTED_ADAMW_WEIGHT_DECAY:
            raise ValueError("adamw weight_decay mismatch")
        if _req_finite_num(grp["eps"], "adamw.eps") != EXPECTED_ADAMW_EPS:
            raise ValueError("adamw eps mismatch")
        betas = grp["betas"]
        if type(betas) not in (tuple, list) or len(betas) != 2:
            raise ValueError("adamw betas invalid")
        if (_req_finite_num(betas[0], "betas[0]"), _req_finite_num(betas[1], "betas[1]")) != EXPECTED_ADAMW_BETAS:
            raise ValueError("adamw betas mismatch")
        if grp["amsgrad"] is not EXPECTED_ADAMW_AMSGRAD:
            raise ValueError("adamw amsgrad mismatch")
        for opt_key, live_opt_val in live_sd_grp.items():
            if opt_key in ("params", "lr", "weight_decay", "betas", "eps", "amsgrad"):
                continue
            ckpt_val = grp[opt_key]
            if type(ckpt_val) is not type(live_opt_val) or ckpt_val != live_opt_val:
                raise ValueError(f"adamw param_group option {opt_key} disagrees with live optimizer")
        pids = [require_exact_int(pid, "adamw.param_id", minimum=0) for pid in grp["params"]]
        if pids != [int(pid) for pid in live_sd_grp["params"]]:
            raise ValueError("adamw parameter mapping/order mismatch")
        for pid in pids:
            pid_to_group[pid] = live_sd_grp
        saved_param_ids.extend(pids)
        live_params.extend(live_grp["params"])

    staged_groups = _clone_opt_val(live_sd_groups)
    if completed_val == 0:
        if raw_state:
            raise ValueError("adamw state must be empty when completed == 0")
        return {"state": {}, "param_groups": staged_groups}

    id_to_param = dict(zip(saved_param_ids, live_params))
    expected_pids = (
        set(expected_updated_param_ids)
        if expected_updated_param_ids is not None
        else {pid for pid, p in id_to_param.items() if getattr(p, "requires_grad", True)}
    )
    if any(type(pid) is not int for pid in raw_state.keys()) or set(raw_state.keys()) != expected_pids:
        raise ValueError("adamw state entry set mismatch")

    staged_state = {}
    for pid in sorted(expected_pids):
        staged_state[pid] = _validate_portable_adamw_slot(
            raw_state[pid],
            id_to_param[pid],
            group=pid_to_group[pid],
            completed=completed_val,
            target_device=device,
        )
    return {"state": staged_state, "param_groups": staged_groups}


def pack_gpu_checkpoint(header: dict, sections: dict[str, bytes]) -> bytes:
    """Pack bounded 8-section stca-v3-gpu container with per-section and whole-body SHA-256."""
    if type(header) is not dict or "completed" not in header:
        raise ValueError("GPU checkpoint header incomplete")
    require_exact_int(header.get("completed"), "header.completed", minimum=0)
    if type(sections) is not dict or set(sections) != set(GPU_SECTIONS):
        raise ValueError("GPU checkpoint section set mismatch")

    body = bytearray()
    table = []
    for name in GPU_SECTIONS:
        raw = sections[name]
        if type(raw) is not bytes:
            raise ValueError(f"GPU checkpoint section {name} must be bytes")
        if len(raw) < GPU_SECTION_MIN_BYTES[name] or len(raw) > GPU_SECTION_MAX_BYTES[name]:
            raise ValueError(f"GPU checkpoint section {name} length out of bounds")
        table.append({"name": name, "offset": len(body), "length": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        body += raw

    stored = dict(header)
    stored["checkpoint_schema"] = GPU_CHECKPOINT_SCHEMA_VERSION
    stored["sections"] = table
    stored["body_bytes"] = len(body)
    stored["body_sha256"] = hashlib.sha256(body).hexdigest()
    encoded = canonical_json_bytes(stored, max_bytes=MAX_HEADER_BYTES)
    payload = MAGIC + len(encoded).to_bytes(4, "little") + encoded + bytes(body)
    if len(payload) > MAX_CHECKPOINT_BYTES:
        raise ValueError("packed GPU checkpoint exceeds MAX_CHECKPOINT_BYTES")
    return payload


def unpack_gpu_checkpoint(payload: bytes) -> dict:
    """Unpack and verify stca-v3-gpu container framing, SHA-256s, and section bounds without unpickling."""
    if type(payload) is not bytes or len(payload) < 8 or len(payload) > MAX_CHECKPOINT_BYTES:
        raise ValueError("GPU checkpoint payload size out of bounds")
    if not payload.startswith(MAGIC):
        raise ValueError("GPU checkpoint magic mismatch")
    header_len = int.from_bytes(payload[4:8], "little")
    if header_len < 2 or header_len > MAX_HEADER_BYTES or 8 + header_len > len(payload):
        raise ValueError("GPU checkpoint header length invalid")
    header = strict_json_loads(payload[8 : 8 + header_len], max_bytes=MAX_HEADER_BYTES)
    if type(header) is not dict or header.get("checkpoint_schema") != GPU_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint_schema must be stca-v3-gpu")
    require_exact_int(header.get("completed"), "header.completed", minimum=0)
    body = payload[8 + header_len :]
    body_bytes = require_exact_int(header.get("body_bytes"), "header.body_bytes", minimum=1, maximum=MAX_CHECKPOINT_BYTES)
    if body_bytes != len(body) or hashlib.sha256(body).hexdigest() != _req_hex64(header.get("body_sha256"), "body_sha256"):
        raise ValueError("GPU checkpoint body integrity mismatch")

    table = header.get("sections")
    if type(table) is not list or len(table) != len(GPU_SECTIONS):
        raise ValueError("GPU checkpoint section set mismatch")
    decoded = {"header": header}
    cursor = 0
    for expected_name, item in zip(GPU_SECTIONS, table):
        if type(item) is not dict or set(item) != {"name", "offset", "length", "sha256"} or item.get("name") != expected_name:
            raise ValueError("GPU checkpoint section entry malformed")
        offset = require_exact_int(item.get("offset"), f"{expected_name}.offset", minimum=0, maximum=len(body))
        length = require_exact_int(
            item.get("length"),
            f"{expected_name}.length",
            minimum=GPU_SECTION_MIN_BYTES[expected_name],
            maximum=GPU_SECTION_MAX_BYTES[expected_name],
        )
        sha = _req_hex64(item.get("sha256"), f"{expected_name}.sha256")
        if offset != cursor or offset + length > len(body):
            raise ValueError(f"GPU checkpoint section {expected_name} offset error")
        raw = body[offset : offset + length]
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError(f"GPU checkpoint section {expected_name} sha256 mismatch")
        decoded[expected_name] = raw
        cursor = offset + length
    if cursor != len(body):
        raise ValueError("GPU checkpoint trailing bytes")
    return decoded


def build_gpu_provenance(
    effective_config_sha256: str,
    *,
    train_exclusion_manifest_sha256: str,
    excluded_train_count: int,
    eligible_train_ids_sha256: str,
    eligible_train_count: int,
    preflight_receipt_sha256: str,
    owner_trust_anchor_sha256: str,
    owner_trust_root_status: str,
    generation_id: str | None = None,
) -> dict:
    _req_hex64(effective_config_sha256, "effective_config_sha256")
    _req_hex64(train_exclusion_manifest_sha256, "train_exclusion_manifest_sha256")
    excl_cnt = require_exact_int(excluded_train_count, "excluded_train_count", minimum=0, maximum=396)
    _req_hex64(eligible_train_ids_sha256, "eligible_train_ids_sha256")
    _req_hex64(preflight_receipt_sha256, "preflight_receipt_sha256")
    _req_hex64(owner_trust_anchor_sha256, "owner_trust_anchor_sha256")
    elig_cnt = require_exact_int(eligible_train_count, "eligible_train_count", minimum=1, maximum=400)
    if owner_trust_root_status not in ALLOWED_ACTIVE_TRUST:
        raise ValueError("owner_trust_root_status invalid")
    if generation_id is not None:
        parse_generation_id(generation_id)
    return {
        "provenance_schema": GPU_PROVENANCE_SCHEMA_VERSION,
        "checkpoint_schema": GPU_CHECKPOINT_SCHEMA_VERSION,
        "manifest_schema": MANIFEST_SCHEMA_VERSION,
        "pointer_schema": POINTER_SCHEMA_VERSION,
        "commit_schema": GPU_COMMIT_SCHEMA_VERSION,
        "dev_report_schema": DEV_REPORT_SCHEMA_VERSION,
        "dev_metric_id": DEV_METRIC_ID,
        "dev_metric_version": DEV_METRIC_VERSION,
        "dev_metric_status": DEV_METRIC_STATUS,
        "failure_receipt_schema": FAILURE_RECEIPT_SCHEMA_VERSION,
        "precision_policy": PRECISION_POLICY,
        "rng_scope": GPU_RNG_SCOPE,
        "device_type": "cuda",
        "cross_device_byte_exact_claimed": CROSS_DEVICE_BYTE_EXACT_CLAIMED,
        "source_id": HISTORICAL_SOURCE_ID,
        "corpus_manifest_sha256": PORTABLE_MANIFEST_SHA256,
        "corpus_shard_attested_sha256": PORTABLE_SHARD_ATTESTED_SHA256,
        "train_exclusion_manifest_sha256": train_exclusion_manifest_sha256,
        "excluded_train_count": excl_cnt,
        "eligible_train_ids_sha256": eligible_train_ids_sha256,
        "eligible_train_count": elig_cnt,
        "preflight_receipt_sha256": preflight_receipt_sha256,
        "candidate_sha256": CANDIDATE_SHA256,
        "loss_sha256": LOSS_SHA256,
        "horizons_sha256": HORIZONS_SHA256,
        "sampler_sha256": SAMPLER_SHA256,
        "exposure_schema": SCHEMA_VERSION,
        "sampler_schema": SAMPLER_SCHEMA_VERSION,
        "remainder_policy": REMAINDER_POLICY,
        "forward_binding": FORWARD_BINDING,
        "temporal_axis_semantics": HISTORICAL_TEMPORAL_AXIS_SEMANTICS,
        "text_mask_policy": TEXT_MASK_POLICY,
        "effective_config_sha256": effective_config_sha256,
        "generation_id": generation_id,
        "owner_trust_anchor_sha256": owner_trust_anchor_sha256,
        "owner_trust_root_status": owner_trust_root_status,
        "release_created": False,
    }


def verify_gpu_provenance(header: dict, expected_provenance: dict, *, expected_generation_id: str | None = None) -> dict:
    if type(header) is not dict or type(expected_provenance) is not dict:
        raise ValueError("header and expected_provenance must be dicts")
    if set(expected_provenance) != GPU_PROVENANCE_KEYS:
        raise ValueError("expected_provenance key set invalid")
    prov = header.get("provenance")
    if type(prov) is not dict or set(prov) != GPU_PROVENANCE_KEYS:
        raise ValueError("header missing prov-v3-gpu provenance")
    completed = require_exact_int(header.get("completed"), "header.completed", minimum=0)
    gen_step = parse_generation_id(prov.get("generation_id"))
    if gen_step != completed:
        raise ValueError("GPU provenance generation_id disagrees with completed")
    for k, exp_v in expected_provenance.items():
        if k == "generation_id":
            if exp_v is not None:
                parse_generation_id(exp_v)
                if prov[k] != exp_v:
                    raise ValueError("GPU provenance generation_id mismatch")
        elif prov[k] != exp_v or type(prov[k]) is not type(exp_v):
            raise ValueError(f"GPU provenance.{k} mismatch")
    if expected_generation_id is not None and prov["generation_id"] != expected_generation_id:
        raise ValueError("GPU provenance generation_id disagrees with expected_generation_id")
    return prov


def verify_historical_dev_report_consistency(
    report: object,
    *,
    expected_macro: float | None = None,
) -> dict:
    """Verify frozen stage-a-dev-v1 DEV report bound to HISTORICAL_TEMPORAL_AXIS_SEMANTICS."""
    if type(report) is not dict:
        raise ValueError("dev_report must be a dict")
    if report.get("control_plane_fixture") is not False or report.get("scientific_metric_valid") is not True:
        raise ValueError("historical GPU pilot dev_report must have control_plane_fixture=False, scientific_metric_valid=True")
    if report.get("dev_report_schema") != DEV_REPORT_SCHEMA_VERSION:
        raise ValueError("dev_report_schema mismatch")
    if report.get("metric_name") != DEV_METRIC_ID or report.get("metric_version") != DEV_METRIC_VERSION or report.get("metric_status") != DEV_METRIC_STATUS:
        raise ValueError("dev_report metric identity mismatch")
    if _req_finite_num(report.get("beta"), "beta") != BETA:
        raise ValueError("dev_report beta mismatch")
    if _req_finite_num(report.get("local_weight"), "local_weight") != LOCAL_WEIGHT or _req_finite_num(report.get("anchor_weight"), "anchor_weight") != ANCHOR_WEIGHT:
        raise ValueError("dev_report head weights mismatch")
    if report.get("temporal_axis_semantics") != HISTORICAL_TEMPORAL_AXIS_SEMANTICS:
        raise ValueError("historical dev_report temporal_axis_semantics mismatch")
    if report.get("horizons_sha256") != HORIZONS_SHA256:
        raise ValueError("dev_report horizons_sha256 mismatch")
    _req_hex64(report.get("dev_forward_digests_sha256"), "dev_forward_digests_sha256")
    if report.get("video_count") != EXPECTED_DEV_VIDEOS or report.get("pair_count") != EXPECTED_DEV_PAIRS:
        raise ValueError("dev_report counts mismatch")
    if report.get("predeclared_absent_mid_videos") != sorted(PREDECLARED_ABSENT_MID_VIDEOS):
        raise ValueError("dev_report predeclared_absent_mid_videos mismatch")

    pvm = report.get("per_video_macro")
    pvnm = report.get("per_video_no_motion_macro")
    pvd = report.get("per_video_delta_vs_no_motion")
    pvc = report.get("per_video_counts")
    if any(type(x) is not dict or len(x) != EXPECTED_DEV_VIDEOS for x in (pvm, pvnm, pvd, pvc)):
        raise ValueError("dev_report per_video maps incomplete")
    if not (set(pvm) == set(pvnm) == set(pvd) == set(pvc)):
        raise ValueError("dev_report per_video keys mismatch")

    total_pairs = 0
    vsum = 0.0
    vnm_sum = 0.0
    for vid in sorted(pvm):
        exp_c = 2 if vid in PREDECLARED_ABSENT_MID_VIDEOS else 3
        vc = require_exact_int(pvc[vid], f"per_video_counts.{vid}", minimum=exp_c, maximum=exp_c)
        vm = _req_finite_num(pvm[vid], f"per_video_macro.{vid}")
        vnm = _req_finite_num(pvnm[vid], f"per_video_no_motion_macro.{vid}")
        vd = _req_finite_num(pvd[vid], f"per_video_delta.{vid}")
        if vm < 0.0 or vnm < 0.0 or abs((vm - vnm) - vd) > DEV_REPORT_RECOMPUTE_TOL:
            raise ValueError(f"dev_report per_video inconsistency for {vid}")
        total_pairs += vc
        vsum += vm
        vnm_sum += vnm
    if total_pairs != EXPECTED_DEV_PAIRS:
        raise ValueError("dev_report total pair count != 147")

    macro = _req_finite_num(report.get("macro"), "macro")
    nm = _req_finite_num(report.get("no_motion_macro"), "no_motion_macro")
    delta = _req_finite_num(report.get("delta_vs_no_motion"), "delta_vs_no_motion")
    rm = vsum / float(EXPECTED_DEV_VIDEOS)
    rnm = vnm_sum / float(EXPECTED_DEV_VIDEOS)
    if abs(macro - rm) > DEV_REPORT_RECOMPUTE_TOL or abs(nm - rnm) > DEV_REPORT_RECOMPUTE_TOL or abs(delta - (macro - nm)) > DEV_REPORT_RECOMPUTE_TOL:
        raise ValueError("dev_report aggregate mismatch")
    if expected_macro is not None and macro != float(expected_macro):
        raise ValueError("dev_report.macro disagrees with checkpoint header")
    if report.get("beats_no_motion") is not (macro < nm):
        raise ValueError("beats_no_motion disagrees with comparator")
    if report.get("signal_status") != ("STRICTLY_BELOW_NO_MOTION" if macro < nm else "NO_POSITIVE_DELTA_SIGNAL"):
        raise ValueError("signal_status disagrees with comparator")
    phc = report.get("per_horizon_counts")
    if phc != DEV_EXPECTED_HORIZON_COUNTS:
        raise ValueError("per_horizon_counts mismatch")
    for name in ("per_horizon_macro", "per_horizon_no_motion_macro", "per_horizon_delta_vs_no_motion"):
        blk = report.get(name)
        if type(blk) is not dict or set(blk.keys()) != set(HORIZON_NAMES):
            raise ValueError(f"dev_report.{name} keys mismatch")
    for h in HORIZON_NAMES:
        hm = _req_finite_num(report["per_horizon_macro"][h], f"per_horizon_macro.{h}")
        hnm = _req_finite_num(report["per_horizon_no_motion_macro"][h], f"per_horizon_no_motion_macro.{h}")
        hd = _req_finite_num(report["per_horizon_delta_vs_no_motion"][h], f"per_horizon_delta.{h}")
        if hm < 0.0 or hnm < 0.0 or abs((hm - hnm) - hd) > DEV_REPORT_RECOMPUTE_TOL:
            raise ValueError(f"per_horizon consistency failed for {h}")
    for sec in ("directional_cosine_loss", "raw_cosine", "magnitude_log_ratio_loss", "norms_and_ratios"):
        if type(report.get(sec)) is not dict:
            raise ValueError(f"dev_report.{sec} must be a dict")
    return {"recomputed_macro": rm, "recomputed_no_motion_macro": rnm}


def _model_state_and_grad_digest_any_device(model) -> str:
    """Order-stable CPU byte digest of model parameters, buffers, and gradients across devices."""
    import torch

    hasher = hashlib.sha256(b"STCA-DEV-STATE-V1\x00")
    for name, param in model.named_parameters():
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\x00P")
        hasher.update(b"1" if param.requires_grad else b"0")
        p_cpu = param.detach().to(device="cpu").contiguous()
        hasher.update(str(p_cpu.dtype).encode("utf-8"))
        hasher.update(str(tuple(p_cpu.shape)).encode("utf-8"))
        p_u8 = p_cpu.to(torch.uint8) if p_cpu.dtype == torch.bool else p_cpu.view(torch.uint8)
        hasher.update(p_u8.numpy().tobytes())
        if param.grad is None:
            hasher.update(b"\x00G0")
        else:
            hasher.update(b"\x00G1")
            g_cpu = param.grad.detach().to(device="cpu").contiguous()
            g_u8 = g_cpu.to(torch.uint8) if g_cpu.dtype == torch.bool else g_cpu.view(torch.uint8)
            hasher.update(g_u8.numpy().tobytes())
    for name, buf in model.named_buffers():
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\x00B")
        b_cpu = buf.detach().to(device="cpu").contiguous()
        hasher.update(str(b_cpu.dtype).encode("utf-8"))
        hasher.update(str(tuple(b_cpu.shape)).encode("utf-8"))
        b_u8 = b_cpu.to(torch.uint8) if b_cpu.dtype == torch.bool else b_cpu.view(torch.uint8)
        hasher.update(b_u8.numpy().tobytes())
    return hasher.hexdigest()


def _assert_uniform_mode(model) -> bool:
    modes = {bool(m.training) for _, m in model.named_modules()}
    if not modes:
        return bool(model.training)
    if len(modes) != 1:
        raise ValueError("mixed module training/eval modes rejected")
    return next(iter(modes))


def gpu_optimizer_step(
    model,
    items: list[dict],
    opt,
    clip: float,
    *,
    device,
    expected_forward_digests: list[str],
) -> dict:
    """Execute one FP32 CUDA optimizer update with pre-forward CPU digest verification and pre-clip finite grad checks."""
    import torch

    if type(items) is not list or not items or len(items) != len(expected_forward_digests):
        raise ValueError("gpu_optimizer_step batch/digest mismatch")
    clip_val = _req_finite_num(clip, "clip")
    if clip_val <= 0.0:
        raise ValueError("clip must be strictly positive")

    verify_model_device_and_dtype(model, device)
    model.train()

    gpu_calls: list[tuple[dict, dict]] = []
    for item, exp_digest in zip(items, expected_forward_digests):
        actual_digest, frozen_cpu_kwargs = canonical_forward_kwargs_digest(item["model_kwargs"])
        if actual_digest != exp_digest or item.get("forward_input_sha256") != exp_digest:
            raise ValueError("train item forward digest mismatch prior to GPU move")
        validate_supervision(item["supervision"], forbidden_kwargs=item["model_kwargs"], frozen_kwargs=frozen_cpu_kwargs)
        validate_causal_transition_meta(
            item["id_movi"],
            item["ids_frames"],
            item["history"],
            item["current"],
            item["target"],
            item["horizon"],
            anchor_ordinal=item["anchor_ordinal"],
            anchor=item["anchor"],
        )
        gpu_kwargs = move_forward_kwargs_to_device(frozen_cpu_kwargs, device)
        gpu_sup = move_supervision_to_device(item["supervision"], device)
        gpu_calls.append((gpu_kwargs, gpu_sup))

    criterion = load_criterion()
    opt.zero_grad(set_to_none=True)
    total_loss = None
    comp_sums = {"total": 0.0, "delta_huber": 0.0, "direction": 0.0, "magnitude": 0.0, "hybrid_state": 0.0, "dual_consistency": 0.0}

    for gpu_kwargs, gpu_sup in gpu_calls:
        out = model.forward(**gpu_kwargs)
        loss_dict = criterion(out, gpu_sup)
        for k, v in loss_dict.items():
            if not bool(torch.isfinite(v).all().item()):
                raise ValueError(f"nonfinite training loss component {k}")
        for ck in comp_sums:
            comp_sums[ck] += float(loss_dict[ck].detach().item())
        total_loss = loss_dict["total"] if total_loss is None else total_loss + loss_dict["total"]

    (total_loss / float(len(gpu_calls))).backward()

    # Check every gradient for finiteness BEFORE clipping (never step on nonfinite gradients).
    for name, p in model.named_parameters():
        if p.grad is not None and not bool(torch.isfinite(p.grad).all().item()):
            opt.zero_grad(set_to_none=True)
            raise ValueError(f"nonfinite gradient detected before clipping in {name}")

    pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val, error_if_nonfinite=True).item())
    if not math.isfinite(pre_clip_norm):
        opt.zero_grad(set_to_none=True)
        raise ValueError("nonfinite pre-clip gradient norm")

    sq_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq_sum += float(torch.sum(p.grad.detach() * p.grad.detach()).item())
    post_clip_norm = math.sqrt(sq_sum)
    if not math.isfinite(post_clip_norm):
        opt.zero_grad(set_to_none=True)
        raise ValueError("nonfinite post-clip gradient norm")

    opt.step()
    torch.cuda.synchronize(device)
    verify_model_device_and_dtype(model, device)

    inv_n = 1.0 / float(len(gpu_calls))
    return {
        "step_inc": 1,
        "loss_means": {k: v * inv_n for k, v in comp_sums.items()},
        "pre_clip_grad_norm": pre_clip_norm,
        "post_clip_grad_norm": post_clip_norm,
    }


def evaluate_historical_dev_with_diagnostics(
    model,
    items: list[dict],
    expected_keys: set[tuple],
    *,
    device,
    expected_digests: dict[tuple, str],
    forbidden_video_ids: set[str] | frozenset[str],
) -> dict:
    """Evaluate exact 147 historical DEV pairs on GPU under frozen stage-a-dev-v1 with CPU+CUDA RNG/mode/state guards."""
    import torch

    if type(expected_keys) is not set or len(expected_keys) != EXPECTED_DEV_PAIRS:
        raise ValueError("expected_keys must contain exact 147 DEV pairs")
    expected_videos = {k[0] for k in expected_keys}
    if len(expected_videos) != EXPECTED_DEV_VIDEOS:
        raise ValueError("expected DEV key set must cover exact 50 videos")
    if not expected_videos.isdisjoint(forbidden_video_ids):
        raise PermissionError("DEV evaluation contains train/heldout video IDs")
    if type(items) is not list or len(items) != EXPECTED_DEV_PAIRS:
        raise ValueError("dev items length must equal 147")
    if set(expected_digests.keys()) != expected_keys:
        raise ValueError("expected_digests keys disagree with frozen 147 DEV pairs")

    verify_model_device_and_dtype(model, device)

    seen: set[tuple] = set()
    prepared_entries: list[tuple[tuple, dict, dict, str]] = []
    for item in items:
        key = (item.get("id_movi"), item.get("current_ordinal"), item.get("target_ordinal"), item.get("horizon"))
        if key not in expected_keys or key in seen or key[0] in forbidden_video_ids:
            raise ValueError(f"invalid, duplicated, or forbidden DEV key: {key}")
        seen.add(key)
        actual_digest, frozen_cpu = canonical_forward_kwargs_digest(item["model_kwargs"])
        exp_digest = expected_digests[key]
        if actual_digest != exp_digest or item.get("forward_input_sha256") != exp_digest:
            raise ValueError(f"DEV item forward digest mismatch for {key}")
        validate_supervision(item["supervision"], forbidden_kwargs=item["model_kwargs"], frozen_kwargs=frozen_cpu)
        gpu_kwargs = move_forward_kwargs_to_device(frozen_cpu, device)
        gpu_sup = move_supervision_to_device(item["supervision"], device)
        prepared_entries.append((key, gpu_kwargs, gpu_sup, exp_digest))
    if seen != expected_keys:
        raise ValueError("DEV pair coverage mismatch")

    loss_path = Path(__file__).with_name("pinned_losses_v3.py")
    if hashlib.sha256(loss_path.read_bytes()).hexdigest() != LOSS_SHA256:
        raise ValueError("prepared loss SHA mismatch")
    from . import pinned_losses_v3 as loss_mod

    eps = loss_mod.LossConfig().eps
    dir_cosine_fn = loss_mod.directional_cosine_per_sample
    mag_ratio_fn = loss_mod.magnitude_log_ratio_per_sample

    was_training = _assert_uniform_mode(model)
    had_last_sem = hasattr(model, "_last_semantic_context")
    prev_last_sem = getattr(model, "_last_semantic_context", None)
    if isinstance(prev_last_sem, torch.Tensor):
        prev_last_sem = prev_last_sem.detach().clone()
    pre_state_digest = _model_state_and_grad_digest_any_device(model)
    pre_rng = capture_runtime_rng_snapshot(device)

    grouped: dict[str, list[float]] = {}
    grouped_no_motion: dict[str, list[float]] = {}
    horizon_values: dict[str, list[float]] = {h: [] for h in HORIZON_NAMES}
    horizon_no_motion: dict[str, list[float]] = {h: [] for h in HORIZON_NAMES}

    dir_loss_local, dir_loss_anchor = [], []
    dir_valid_tgt_local, dir_valid_tgt_anchor = 0, 0
    raw_cos_local, raw_cos_anchor = [], []
    raw_cos_undef_local, raw_cos_undef_anchor = 0, 0
    mag_loss_local, mag_loss_anchor = [], []
    pred_norms_local, pred_norms_anchor = [], []
    tgt_norms_local, tgt_norms_anchor = [], []
    norm_ratios_local, norm_ratios_anchor = [], []
    zero_pred_local, zero_pred_anchor = 0, 0
    zero_tgt_local, zero_tgt_anchor = 0, 0

    digest_chain = hashlib.sha256()
    for _, _, _, exp_digest in sorted(prepared_entries, key=lambda e: e[0]):
        digest_chain.update(exp_digest.encode("ascii"))

    eval_exc: Exception | None = None
    try:
        model.eval()
        with torch.no_grad():
            for key, gpu_kwargs, gpu_sup, _ in prepared_entries:
                out = model.forward(**gpu_kwargs)
                pred_local = out["delta_local"]
                pred_anchor = out["delta_anchor"]
                for pname, pt in (("delta_local", pred_local), ("delta_anchor", pred_anchor)):
                    if (
                        not isinstance(pt, torch.Tensor)
                        or pt.device != device
                        or pt.dtype != torch.float32
                        or tuple(pt.shape) != (1, WIDTH)
                        or not bool(torch.isfinite(pt).all().item())
                    ):
                        raise ValueError(f"nonfinite or malformed GPU DEV output {pname}")
                tgt_local = gpu_sup["target_delta_local"]
                tgt_anchor = gpu_sup["target_delta_anchor"]

                local = torch.nn.functional.smooth_l1_loss(pred_local, tgt_local, beta=BETA)
                anchor = torch.nn.functional.smooth_l1_loss(pred_anchor, tgt_anchor, beta=BETA)
                pair_loss = float(LOCAL_WEIGHT * float(local.item()) + ANCHOR_WEIGHT * float(anchor.item()))

                zeros = torch.zeros_like(tgt_local)
                nm_local = float(torch.nn.functional.smooth_l1_loss(zeros, tgt_local, beta=BETA).item())
                nm_anchor = float(torch.nn.functional.smooth_l1_loss(zeros, tgt_anchor, beta=BETA).item())
                pair_nm = LOCAL_WEIGHT * nm_local + ANCHOR_WEIGHT * nm_anchor

                d_loc = float(dir_cosine_fn(pred_local, tgt_local, eps=eps).item())
                d_anc = float(dir_cosine_fn(pred_anchor, tgt_anchor, eps=eps).item())
                m_loc = float(mag_ratio_fn(pred_local, tgt_local, eps=eps).item())
                m_anc = float(mag_ratio_fn(pred_anchor, tgt_anchor, eps=eps).item())

                p_norm_loc = float(torch.linalg.vector_norm(pred_local, dim=-1).item())
                p_norm_anc = float(torch.linalg.vector_norm(pred_anchor, dim=-1).item())
                t_norm_loc = float(torch.linalg.vector_norm(tgt_local, dim=-1).item())
                t_norm_anc = float(torch.linalg.vector_norm(tgt_anchor, dim=-1).item())

                if not all(math.isfinite(x) for x in (pair_loss, pair_nm, d_loc, d_anc, m_loc, m_anc, p_norm_loc, p_norm_anc, t_norm_loc, t_norm_anc)):
                    raise ValueError("nonfinite GPU DEV scalar")

                id_movi, _, _, horizon = key
                grouped.setdefault(id_movi, []).append(pair_loss)
                grouped_no_motion.setdefault(id_movi, []).append(pair_nm)
                horizon_values[horizon].append(pair_loss)
                horizon_no_motion[horizon].append(pair_nm)

                dir_loss_local.append(d_loc)
                dir_loss_anchor.append(d_anc)
                mag_loss_local.append(m_loc)
                mag_loss_anchor.append(m_anc)
                pred_norms_local.append(p_norm_loc)
                pred_norms_anchor.append(p_norm_anc)
                tgt_norms_local.append(t_norm_loc)
                tgt_norms_anchor.append(t_norm_anc)

                if p_norm_loc <= DIAG_EPS:
                    zero_pred_local += 1
                if p_norm_anc <= DIAG_EPS:
                    zero_pred_anchor += 1

                if t_norm_loc > DIAG_EPS:
                    dir_valid_tgt_local += 1
                    norm_ratios_local.append(p_norm_loc / t_norm_loc)
                    if p_norm_loc > DIAG_EPS:
                        dot_loc = float(torch.dot(pred_local[0], tgt_local[0]).item())
                        cos_loc = max(-1.0, min(1.0, dot_loc / (p_norm_loc * t_norm_loc)))
                        raw_cos_local.append(cos_loc)
                    else:
                        raw_cos_undef_local += 1
                else:
                    zero_tgt_local += 1
                    raw_cos_undef_local += 1

                if t_norm_anc > DIAG_EPS:
                    dir_valid_tgt_anchor += 1
                    norm_ratios_anchor.append(p_norm_anc / t_norm_anc)
                    if p_norm_anc > DIAG_EPS:
                        dot_anc = float(torch.dot(pred_anchor[0], tgt_anchor[0]).item())
                        cos_anc = max(-1.0, min(1.0, dot_anc / (p_norm_anc * t_norm_anc)))
                        raw_cos_anchor.append(cos_anc)
                    else:
                        raw_cos_undef_anchor += 1
                else:
                    zero_tgt_anchor += 1
                    raw_cos_undef_anchor += 1
    except Exception as exc:
        eval_exc = exc
    finally:
        model.train(was_training)
        if had_last_sem:
            model._last_semantic_context = prev_last_sem
        elif hasattr(model, "_last_semantic_context"):
            delattr(model, "_last_semantic_context")
        post_modes = {bool(m.training) for _, m in model.named_modules()}
        mode_ok = (post_modes == {was_training}) if post_modes else (bool(model.training) == was_training)
        state_ok = _model_state_and_grad_digest_any_device(model) == pre_state_digest
        rng_ok = rng_snapshots_equal(pre_rng, capture_runtime_rng_snapshot(device))
        if not (mode_ok and state_ok and rng_ok):
            raise EvaluatorMutationError(
                "GPU DEV evaluation mutated model mode, parameters, buffers, gradients, or CPU+CUDA RNG state",
                mutation_details={"mode_restored": mode_ok, "state_unchanged": state_ok, "rng_unchanged": rng_ok},
                original_error=None if eval_exc is None else str(eval_exc),
            ) from eval_exc

    if eval_exc is not None:
        raise eval_exc

    ordered_vids = sorted(grouped.keys())
    per_video_macro = {v: sum(grouped[v]) / len(grouped[v]) for v in ordered_vids}
    per_video_nm = {v: sum(grouped_no_motion[v]) / len(grouped_no_motion[v]) for v in ordered_vids}
    per_video_delta = {v: per_video_macro[v] - per_video_nm[v] for v in ordered_vids}
    per_video_counts = {v: len(grouped[v]) for v in ordered_vids}

    macro = sum(per_video_macro.values()) / float(EXPECTED_DEV_VIDEOS)
    no_motion_macro = sum(per_video_nm.values()) / float(EXPECTED_DEV_VIDEOS)
    if not math.isfinite(macro) or not math.isfinite(no_motion_macro):
        raise ValueError("nonfinite DEV aggregate")

    raw_cos_loc_mean = (sum(raw_cos_local) / len(raw_cos_local)) if raw_cos_local else None
    raw_cos_anc_mean = (sum(raw_cos_anchor) / len(raw_cos_anchor)) if raw_cos_anchor else None
    raw_cos_mean = 0.5 * (raw_cos_loc_mean + raw_cos_anc_mean) if (raw_cos_loc_mean is not None and raw_cos_anc_mean is not None) else None
    per_horizon_macro = {h: sum(horizon_values[h]) / len(horizon_values[h]) for h in HORIZON_NAMES}
    per_horizon_nm = {h: sum(horizon_no_motion[h]) / len(horizon_no_motion[h]) for h in HORIZON_NAMES}
    per_horizon_delta = {h: per_horizon_macro[h] - per_horizon_nm[h] for h in HORIZON_NAMES}

    dir_loc_mean = sum(dir_loss_local) / len(dir_loss_local)
    dir_anc_mean = sum(dir_loss_anchor) / len(dir_loss_anchor)
    mag_loc_mean = sum(mag_loss_local) / len(mag_loss_local)
    mag_anc_mean = sum(mag_loss_anchor) / len(mag_loss_anchor)

    return {
        "dev_report_schema": DEV_REPORT_SCHEMA_VERSION,
        "metric_name": METRIC_NAME,
        "metric_version": METRIC_VERSION,
        "metric_status": METRIC_STATUS,
        "beta": BETA,
        "local_weight": LOCAL_WEIGHT,
        "anchor_weight": ANCHOR_WEIGHT,
        "aggregation": METRIC_AGGREGATION,
        "direction": METRIC_DIRECTION,
        "tie_policy": METRIC_TIE_POLICY,
        "epsilon": METRIC_EPSILON,
        "diagnostics_policy": DIAGNOSTICS_POLICY,
        "h2_policy": H2_POLICY,
        "temporal_axis_semantics": HISTORICAL_TEMPORAL_AXIS_SEMANTICS,
        "horizons_sha256": HORIZONS_SHA256,
        "macro": float(macro),
        "no_motion_macro": float(no_motion_macro),
        "delta_vs_no_motion": float(macro - no_motion_macro),
        "beats_no_motion": bool(macro < no_motion_macro),
        "signal_status": "STRICTLY_BELOW_NO_MOTION" if macro < no_motion_macro else "NO_POSITIVE_DELTA_SIGNAL",
        "video_count": len(ordered_vids),
        "pair_count": EXPECTED_DEV_PAIRS,
        "predeclared_absent_mid_videos": sorted(PREDECLARED_ABSENT_MID_VIDEOS),
        "per_video_macro": per_video_macro,
        "per_video_no_motion_macro": per_video_nm,
        "per_video_delta_vs_no_motion": per_video_delta,
        "per_video_counts": per_video_counts,
        "per_horizon_macro": per_horizon_macro,
        "per_horizon_no_motion_macro": per_horizon_nm,
        "per_horizon_delta_vs_no_motion": per_horizon_delta,
        "per_horizon_counts": {h: len(horizon_values[h]) for h in HORIZON_NAMES},
        "directional_cosine_loss": {
            "definition": "pinned_losses_v3.directional_cosine_per_sample: 1 - cos(pred, tgt) when ||tgt|| > 1e-8 else 0",
            "local_mean": float(dir_loc_mean),
            "anchor_mean": float(dir_anc_mean),
            "mean": float(0.5 * (dir_loc_mean + dir_anc_mean)),
            "valid_target_count_local": dir_valid_tgt_local,
            "valid_target_count_anchor": dir_valid_tgt_anchor,
        },
        "raw_cosine": {
            "definition": "cos(pred, tgt) over nonzero pairs with ||pred|| > 1e-8 AND ||tgt|| > 1e-8; None if undefined",
            "local_mean": raw_cos_loc_mean,
            "anchor_mean": raw_cos_anc_mean,
            "mean": raw_cos_mean,
            "valid_both_count_local": len(raw_cos_local),
            "valid_both_count_anchor": len(raw_cos_anchor),
            "undefined_count_local": raw_cos_undef_local,
            "undefined_count_anchor": raw_cos_undef_anchor,
        },
        "magnitude_log_ratio_loss": {
            "definition": "pinned_losses_v3.magnitude_log_ratio_per_sample: |log(max(||pred||, 1e-8)) - log(max(||tgt||, 1e-8))|",
            "local_mean": float(mag_loc_mean),
            "anchor_mean": float(mag_anc_mean),
            "mean": float(0.5 * (mag_loc_mean + mag_anc_mean)),
        },
        "norms_and_ratios": {
            "pred_norm_mean_local": float(sum(pred_norms_local) / len(pred_norms_local)),
            "pred_norm_mean_anchor": float(sum(pred_norms_anchor) / len(pred_norms_anchor)),
            "target_norm_mean_local": float(sum(tgt_norms_local) / len(tgt_norms_local)),
            "target_norm_mean_anchor": float(sum(tgt_norms_anchor) / len(tgt_norms_anchor)),
            "raw_norm_ratio_mean_local": (sum(norm_ratios_local) / len(norm_ratios_local)) if norm_ratios_local else None,
            "raw_norm_ratio_mean_anchor": (sum(norm_ratios_anchor) / len(norm_ratios_anchor)) if norm_ratios_anchor else None,
            "zero_pred_count_local": zero_pred_local,
            "zero_pred_count_anchor": zero_pred_anchor,
            "zero_target_count_local": zero_tgt_local,
            "zero_target_count_anchor": zero_tgt_anchor,
            "undefined_ratio_count_local": zero_tgt_local,
            "undefined_ratio_count_anchor": zero_tgt_anchor,
        },
        "dev_forward_digests_sha256": digest_chain.hexdigest(),
        "control_plane_fixture": False,
        "scientific_metric_valid": True,
    }


def _safe_rel_gen_path(root: Path, relpath: object, label: str) -> Path:
    if type(relpath) is not str or not relpath or relpath.startswith("/") or "\\" in relpath:
        raise ValueError(f"{label} invalid relative POSIX path")
    if any(seg in ("", ".", "..") for seg in relpath.split("/")):
        raise ValueError(f"{label} invalid path segment")
    gens = (root / "generations").resolve()
    res = (root / relpath).resolve()
    res.relative_to(gens)
    return res


def _validate_gen_rec(root: Path, entry: object, expected_role: str) -> dict:
    req = {
        "generation_id",
        "role",
        "generation_dir",
        "checkpoint_relpath",
        "dev_report_relpath",
        "manifest_relpath",
        "checkpoint_sha256",
        "dev_report_sha256",
        "manifest_sha256",
        "provenance_sha256",
        "completed",
        "macro",
    }
    if type(entry) is not dict or set(entry) != req or entry.get("role") != expected_role:
        raise ValueError(f"invalid GPU generation record ({expected_role})")
    gen_id = entry["generation_id"]
    step = parse_generation_id(gen_id)
    if require_exact_int(entry.get("completed"), "completed", minimum=0) != step:
        raise ValueError("generation_id/completed mismatch")
    for k in ("checkpoint_sha256", "dev_report_sha256", "manifest_sha256", "provenance_sha256"):
        _req_hex64(entry.get(k), k)
    _req_finite_num(entry.get("macro"), "macro")
    for k in ("generation_dir", "checkpoint_relpath", "dev_report_relpath", "manifest_relpath"):
        _safe_rel_gen_path(root, entry[k], k)
    return entry


def _parse_gpu_commit(root: Path, raw: bytes) -> dict:
    doc = strict_json_loads(raw, max_bytes=MAX_COMMIT_BYTES)
    req = {
        "commit_schema",
        "manifest_schema",
        "pointer_schema",
        "provenance_schema",
        "dev_report_schema",
        "commit_seq",
        "effective_config_sha256",
        "last",
        "best",
        "best_history",
        "owner_trust_anchor_sha256",
        "owner_trust_root_status",
        "release_created",
    }
    if type(doc) is not dict or set(doc) != req:
        raise ValueError("CURRENT_COMMIT field set invalid")
    if doc.get("commit_schema") != GPU_COMMIT_SCHEMA_VERSION or doc.get("provenance_schema") != GPU_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("CURRENT_COMMIT GPU schema mismatch")
    last_r = _validate_gen_rec(root, doc.get("last"), "LAST")
    best_r = _validate_gen_rec(root, doc.get("best"), "BEST")
    if best_r["completed"] > last_r["completed"]:
        raise ValueError("best step exceeds last step")
    if doc.get("release_created") is not False:
        raise ValueError("release_created must be false")
    raw_hist = doc.get("best_history")
    if type(raw_hist) is not list or not raw_hist or len(raw_hist) > MAX_BEST_HISTORY:
        raise ValueError("CURRENT_COMMIT best_history invalid")
    prev_step = -1
    prev_macro = None
    for h_item in raw_hist:
        rec = _validate_gen_rec(root, h_item, "BEST")
        if rec["completed"] <= prev_step or (prev_macro is not None and not (float(rec["macro"]) < prev_macro)):
            raise ValueError("CURRENT_COMMIT best_history violates strict progression")
        prev_step = rec["completed"]
        prev_macro = float(rec["macro"])
    if raw_hist[-1] != best_r:
        raise ValueError("CURRENT_COMMIT best_history[-1] disagrees with best")
    return doc


def _materialize_gpu_gen(
    root: Path,
    generation_id: str,
    role: str,
    completed: int,
    payload: bytes,
    dev_report_bytes: bytes,
    header: dict,
    prov: dict,
    release_audit: dict,
) -> dict:
    gens = root / "generations"
    stage_root = root / "stage"
    gens.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)
    _fsync_dir(root)

    ckpt_sha = hashlib.sha256(payload).hexdigest()
    dev_sha = hashlib.sha256(dev_report_bytes).hexdigest()
    hdr_sha = hashlib.sha256(canonical_json_bytes(header, max_bytes=MAX_HEADER_BYTES)).hexdigest()
    prov_sha = hashlib.sha256(canonical_json_bytes(prov, max_bytes=MAX_HEADER_BYTES)).hexdigest()
    manifest = {
        "manifest_schema": MANIFEST_SCHEMA_VERSION,
        "checkpoint_schema": GPU_CHECKPOINT_SCHEMA_VERSION,
        "provenance_schema": GPU_PROVENANCE_SCHEMA_VERSION,
        "dev_report_schema": DEV_REPORT_SCHEMA_VERSION,
        "generation_id": generation_id,
        "role": role,
        "completed": completed,
        "checkpoint_bytes": len(payload),
        "checkpoint_sha256": ckpt_sha,
        "dev_report_bytes": len(dev_report_bytes),
        "dev_report_sha256": dev_sha,
        "header_sha256": hdr_sha,
        "provenance_sha256": prov_sha,
        "provenance": prov,
        "release_audit": release_audit,
    }
    man_bytes = canonical_json_bytes(manifest, max_bytes=MAX_MANIFEST_BYTES)
    man_sha = hashlib.sha256(man_bytes).hexdigest()
    final = gens / f"{generation_id}-{role}"
    if not final.exists():
        stage = None
        for seq in range(1, MAX_FAILURE_RECEIPTS + 1):
            cand = stage_root / f"{generation_id}-{role}.tmp-{seq:04d}"
            if not cand.exists():
                stage = cand
                break
        if stage is None:
            raise ValueError("stage directory full")
        stage.mkdir()
        (stage / "checkpoint.bin").write_bytes(payload)
        _fsync_file(stage / "checkpoint.bin")
        (stage / "dev_report.json").write_bytes(dev_report_bytes)
        _fsync_file(stage / "dev_report.json")
        (stage / "manifest.json").write_bytes(man_bytes)
        _fsync_file(stage / "manifest.json")
        _fsync_dir(stage)
        os.rename(stage, final)
        _fsync_dir(gens)
    else:
        if _read_bounded(final / "checkpoint.bin", max_bytes=MAX_CHECKPOINT_BYTES, label="existing ckpt") != payload:
            raise ValueError("conflicting existing GPU generation")
    return {
        "generation_id": generation_id,
        "role": role,
        "generation_dir": f"generations/{generation_id}-{role}",
        "checkpoint_relpath": f"generations/{generation_id}-{role}/checkpoint.bin",
        "dev_report_relpath": f"generations/{generation_id}-{role}/dev_report.json",
        "manifest_relpath": f"generations/{generation_id}-{role}/manifest.json",
        "checkpoint_sha256": ckpt_sha,
        "dev_report_sha256": dev_sha,
        "manifest_sha256": man_sha,
        "provenance_sha256": prov_sha,
        "completed": completed,
        "macro": float(header["macro"]),
    }


def _sync_gpu_mirrors(root: Path, commit: dict, commit_sha256: str) -> None:
    history = commit["best_history"]
    prev_best_gen = history[-2]["generation_id"] if len(history) >= 2 else None
    prev_best_sha = history[-2]["checkpoint_sha256"] if len(history) >= 2 else None
    for role, entry, prev_gen, prev_sha in (("LAST", commit["last"], None, None), ("BEST", commit["best"], prev_best_gen, prev_best_sha)):
        ptr = {
            "pointer_schema": POINTER_SCHEMA_VERSION,
            "role": role,
            "generation_id": entry["generation_id"],
            "generation_dir": entry["generation_dir"],
            "checkpoint_relpath": entry["checkpoint_relpath"],
            "manifest_relpath": entry["manifest_relpath"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "manifest_sha256": entry["manifest_sha256"],
            "completed": entry["completed"],
            "macro": entry["macro"],
            "previous_generation_id": prev_gen,
            "previous_checkpoint_sha256": prev_sha,
            "commit_seq": commit["commit_seq"],
            "commit_sha256": commit_sha256,
        }
        ptr_bytes = canonical_json_bytes(ptr, max_bytes=MAX_POINTER_BYTES)
        partial = root / f"CURRENT_{role}.partial"
        partial.write_bytes(ptr_bytes)
        _fsync_file(partial)
        os.replace(partial, root / f"CURRENT_{role}")
        _fsync_dir(root)


def commit_gpu_event(
    root: Path,
    generation_id: str,
    payload: bytes,
    *,
    is_new_best: bool,
    expected_provenance: dict,
    dev_report_bytes: bytes,
    release_audit: dict,
    expected_dev_events: list[int],
) -> dict:
    """Atomically commit a verified stca-v3-gpu generation + DEV report with historical BEST provenance binding."""
    gen_step = parse_generation_id(generation_id)
    decoded = unpack_gpu_checkpoint(payload)
    header = decoded["header"]
    if header.get("completed") != gen_step:
        raise ValueError("generation_id disagrees with header.completed")
    validate_event_and_best_history(header, expected_dev_events=expected_dev_events)
    prov = verify_gpu_provenance(header, expected_provenance, expected_generation_id=generation_id)
    macro_val = _req_finite_num(header.get("macro"), "header.macro")
    dev_doc = strict_json_loads(dev_report_bytes, max_bytes=MAX_DEV_REPORT_BYTES)
    verify_historical_dev_report_consistency(dev_doc, expected_macro=macro_val)
    dev_sha = hashlib.sha256(dev_report_bytes).hexdigest()
    if header.get("dev_report_sha256") != dev_sha:
        raise ValueError("header.dev_report_sha256 mismatch")
    if header.get("release_audit") != release_audit:
        raise ValueError("header.release_audit disagrees with active release audit")

    with _gpu_writer_lock(root):
        commit_path = root / "CURRENT_COMMIT"
        prev_commit = _parse_gpu_commit(root, _read_bounded(commit_path, max_bytes=MAX_COMMIT_BYTES, label="CURRENT_COMMIT")) if commit_path.exists() else None
        if prev_commit is not None:
            if prev_commit["effective_config_sha256"] != prov["effective_config_sha256"]:
                raise ValueError("CURRENT_COMMIT effective_config_sha256 mismatch")
            if gen_step <= prev_commit["last"]["completed"]:
                raise ValueError("GPU commit step must strictly advance")
            expected_best = macro_val < float(prev_commit["best"]["macro"])
            if is_new_best != expected_best:
                raise ValueError("is_new_best disagrees with earliest strict tie rule")
            if is_new_best:
                if header.get("best_step") != gen_step or float(header.get("best_macro")) != macro_val:
                    raise ValueError("header best disagrees with new BEST")
            else:
                if header.get("best_step") != prev_commit["best"]["completed"] or float(header.get("best_macro")) != float(prev_commit["best"]["macro"]):
                    raise ValueError("header best disagrees with preserved committed BEST")
        elif not is_new_best or header.get("best_step") != gen_step or float(header.get("best_macro")) != macro_val:
            raise ValueError("initial commit must set BEST to initial generation")

        last_rec = _materialize_gpu_gen(root, generation_id, "LAST", gen_step, payload, dev_report_bytes, header, prov, release_audit)
        if is_new_best:
            best_rec = _materialize_gpu_gen(root, generation_id, "BEST", gen_step, payload, dev_report_bytes, header, prov, release_audit)
            history = ([] if prev_commit is None else list(prev_commit["best_history"])) + [best_rec]
        else:
            assert prev_commit is not None
            best_rec = dict(prev_commit["best"])
            prev_best_prov = {**expected_provenance, "generation_id": best_rec["generation_id"]}
            best_ckpt = _read_bounded(_safe_rel_gen_path(root, best_rec["checkpoint_relpath"], "best.ckpt"), max_bytes=MAX_CHECKPOINT_BYTES, label="best.ckpt")
            if hashlib.sha256(best_ckpt).hexdigest() != best_rec["checkpoint_sha256"]:
                raise ValueError("preserved BEST checkpoint digest mismatch")
            verify_gpu_provenance(unpack_gpu_checkpoint(best_ckpt)["header"], prev_best_prov, expected_generation_id=best_rec["generation_id"])
            history = list(prev_commit["best_history"])

        expected_hdr_history = [{"step": rec["completed"], "macro": float(rec["macro"])} for rec in history]
        normalized_hdr_history = [{"step": int(it["step"]), "macro": float(it["macro"])} for it in header["best_history"]]
        if normalized_hdr_history != expected_hdr_history:
            raise ValueError("header.best_history disagrees with committed best_history")

        next_seq = 1 if prev_commit is None else int(prev_commit["commit_seq"]) + 1
        commit = {
            "commit_schema": GPU_COMMIT_SCHEMA_VERSION,
            "manifest_schema": MANIFEST_SCHEMA_VERSION,
            "pointer_schema": POINTER_SCHEMA_VERSION,
            "provenance_schema": GPU_PROVENANCE_SCHEMA_VERSION,
            "dev_report_schema": DEV_REPORT_SCHEMA_VERSION,
            "commit_seq": next_seq,
            "effective_config_sha256": prov["effective_config_sha256"],
            "last": last_rec,
            "best": best_rec,
            "best_history": history,
            "owner_trust_anchor_sha256": prov["owner_trust_anchor_sha256"],
            "owner_trust_root_status": prov["owner_trust_root_status"],
            "release_created": False,
        }
        commit_bytes = canonical_json_bytes(commit, max_bytes=MAX_COMMIT_BYTES)
        commit_sha = hashlib.sha256(commit_bytes).hexdigest()
        partial = root / "CURRENT_COMMIT.partial"
        partial.write_bytes(commit_bytes)
        _fsync_file(partial)
        os.replace(partial, commit_path)
        try:
            _fsync_dir(root)
        except Exception as exc:
            raise CommitDurabilityUnknownError(str(exc), generation_id=generation_id, commit_sha256=commit_sha) from exc
        try:
            _sync_gpu_mirrors(root, commit, commit_sha)
        except Exception as exc:
            raise PostCommitError(str(exc), generation_id=generation_id, commit_sha256=commit_sha, phase="mirror_sync") from exc
        return {
            "status": "committed",
            "commit_state": COMMIT_STATE_DURABLE,
            "commit_seq": next_seq,
            "commit_sha256": commit_sha,
            "generation_id": generation_id,
            "last": last_rec,
            "best": best_rec,
        }


def build_gpu_checkpoint_sections(model, opt, device, sampler_bytes: bytes, exposure_journal_bytes: bytes) -> dict[str, bytes]:
    import random

    import numpy
    import torch

    py_ver, py_tuple, py_gauss = random.getstate()
    py_list = [int(py_ver), [int(w) for w in py_tuple], None if py_gauss is None else float(py_gauss)]
    np_algo, np_keys, np_pos, np_hg, np_c = numpy.random.get_state()
    np_list = [str(np_algo), [int(w) for w in np_keys.tolist()], int(np_pos), int(np_hg), float(np_c)]

    return {
        "model": serialize_portable_model_bytes(model),
        "adamw": serialize_portable_adamw_bytes(opt),
        "python_rng": canonical_json_bytes(py_list, max_bytes=MAX_RNG_SECTION_BYTES),
        "numpy_rng": canonical_json_bytes(np_list, max_bytes=MAX_RNG_SECTION_BYTES),
        "torch_rng": bytes(torch.get_rng_state().tolist()),
        "cuda_rng": serialize_cuda_rng_bytes(device),
        "sampler": sampler_bytes,
        "exposure": exposure_journal_bytes,
    }


def model_params_digest_any_device(model) -> str:
    import torch

    hasher = hashlib.sha256(b"STCA-MODEL-PARAMS-V1\x00")
    sd = model.state_dict()
    for name in sorted(sd):
        t = sd[name].detach().to(device="cpu").contiguous()
        hasher.update(name.encode("utf-8") + b"\x00" + str(t.dtype).encode("utf-8") + str(tuple(t.shape)).encode("utf-8"))
        u8 = t.to(torch.uint8) if t.dtype == torch.bool else t.view(torch.uint8)
        hasher.update(u8.numpy().tobytes())
    return hasher.hexdigest()
