"""Authorized Stage A synthetic spine with external Owner trust, resolved config, and durable dev_report.json."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from . import CANDIDATE_SHA256, HORIZONS_SHA256, LOSS_SHA256, SAMPLER_SHA256
from .authorization import (
    ExecutionNotAuthorized,
    OWNER_TRUST_STATUS_PENDING,
    RESUME_POLICY_FRESH,
    RESUME_POLICY_LAST,
    assert_external_trust_boundary,
    canonical_resolved_path,
    finalize_release_nonce,
    load,
    observe_trusted_utc_now,
    reserve_release_nonce,
)
from .checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    COMMIT_SCHEMA_VERSION,
    COMMIT_STATE_DURABLE,
    COMMIT_STATE_PRE_COMMIT_FAILURE,
    COMMIT_STATE_VISIBLE_DURABILITY_UNKNOWN,
    DEV_METRIC_ID,
    DEV_METRIC_STATUS,
    DEV_METRIC_VERSION,
    DEV_REPORT_SCHEMA_VERSION,
    EXPECTED_ADAMW_AMSGRAD,
    EXPECTED_ADAMW_BETAS,
    EXPECTED_ADAMW_EPS,
    EXPECTED_ADAMW_LR,
    EXPECTED_ADAMW_WEIGHT_DECAY,
    MANIFEST_SCHEMA_VERSION,
    MAX_DEV_REPORT_BYTES,
    MAX_HORIZONS_FILE_BYTES,
    MAX_RNG_SECTION_BYTES,
    OWNER_TRUST_ROOT_STATUS,
    POINTER_SCHEMA_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    RNG_SCOPE,
    CommitDurabilityUnknownError,
    FatalRollbackError,
    PostCommitError,
    build_expected_provenance,
    capture_cpu_rng_snapshot,
    clone_optimizer_state_dict,
    commit_event,
    file_sha256,
    pack,
    recover_commit_state,
    resolve_commit,
    resolve_pointer,
    restore_cpu_rng_snapshot,
    stage_adamw_state,
    stage_model_state,
    stage_numpy_rng,
    stage_python_rng,
    stage_torch_rng,
    unpack,
    validate_event_and_best_history,
    validate_numpy_rng_bytes,
    validate_python_rng_bytes,
    validate_torch_rng_bytes,
    verify_checkpoint_provenance,
    write_failure_receipt,
)
from .dev import (
    ANCHOR_WEIGHT,
    BETA,
    DIAGNOSTICS_POLICY,
    H2_POLICY,
    LOCAL_WEIGHT,
    METRIC_AGGREGATION,
    METRIC_DIRECTION,
    METRIC_EPSILON,
    METRIC_NAME,
    METRIC_STATUS as DEV_METRIC_STATUS_VAL,
    METRIC_TIE_POLICY,
    METRIC_VERSION,
    EvaluatorMutationError,
    evaluate_with_diagnostics,
    expected_dev_digests,
    load_keys,
    load_rows,
)
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
from .forward_binding import REPRESENTATION_FIREWALL
from .model_adapter import load_class
from .plan import (
    DEV_TEMPORAL_AXIS_SEMANTICS,
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

CONFIG_SCHEMA_VERSION = "stage-a-config-v1"
TRAIN_TEMPORAL_AXIS_SEMANTICS = "SYNTHETIC_PINNED_LABELS_0_2_9_10_40"
FRESH_SEED_DETERMINISM_POLICY = "cpu_fresh_seed_python_numpy_torch_single_process_no_gpu_claim"


def _seed_fresh_cpu_rng(seed: int) -> None:
    """Bind Python, NumPy, and Torch CPU RNG initialization to resolved seed on fresh runs only."""
    import random

    import numpy
    import torch

    seed_val = require_exact_int(seed, "seed", minimum=0, maximum=MAX_SEED)
    random.seed(seed_val)
    numpy.random.seed(seed_val)
    torch.manual_seed(seed_val)


def _collect_runtime_environment() -> dict:
    import platform
    import sys

    import numpy
    import torch

    return {
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "numpy_version": str(numpy.__version__),
        "platform": platform.platform(),
        "device": "cpu",
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "gpu_determinism_claimed": False,
    }


def _model_params_sha256(model) -> str:
    import torch

    hasher = hashlib.sha256(b"STCA-MODEL-PARAMS-V1\x00")
    state = model.state_dict()
    for name in sorted(state.keys()):
        tensor = state[name].detach().contiguous().cpu()
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        raw_u8 = tensor.to(torch.uint8) if tensor.dtype == torch.bool else tensor.view(torch.uint8)
        hasher.update(raw_u8.numpy().tobytes())
    return hasher.hexdigest()

REQUIRED_CONFIG_KEYS = frozenset(
    {
        "schema",
        "package",
        "data",
        "scope",
        "device",
        "budget",
        "max_updates",
        "batch",
        "seed",
        "dev_events",
        "clip",
        "horizons_path",
        "output_root",
        "trust_anchor_path",
        "revocation_path",
        "nonce_ledger_path",
        "candidate_sha256",
        "loss_sha256",
        "horizons_sha256",
        "sampler_sha256",
        "exposure_schema",
        "sampler_schema",
        "checkpoint_schema",
        "provenance_schema",
        "manifest_schema",
        "pointer_schema",
        "commit_schema",
        "dev_report_schema",
        "rng_scope",
        "optimizer",
        "lr",
        "weight_decay",
        "betas",
        "eps",
        "amsgrad",
        "synthetic_source_id",
        "synthetic_train_size",
        "synthetic_train_ids_sha256",
        "synthetic_remainder_policy",
        "forward_binding",
        "representation_firewall",
        "train_temporal_axis_semantics",
        "dev_temporal_axis_semantics",
        "dev_metric_id",
        "dev_metric_version",
        "dev_metric_status",
        "dev_metric_beta",
        "dev_metric_local_weight",
        "dev_metric_anchor_weight",
        "dev_metric_aggregation",
        "dev_metric_direction",
        "dev_metric_tie_policy",
        "dev_metric_epsilon",
        "dev_diagnostics_policy",
        "dev_h2_policy",
        "release_created",
    }
)
OPTIONAL_RUNTIME_CONFIG_KEYS = frozenset({"status", "owner_trust_root_status", "resume_role", "resume_bytes"})


def _require_finite_config_number(value: object, name: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite numeric value")
    return float(value)


def _safe_write_failure_receipt(
    root: Path,
    stage: str,
    reason: str,
    *,
    generation_id: str | None = None,
    details: dict | None = None,
) -> dict | None:
    try:
        return write_failure_receipt(
            root,
            stage,
            reason,
            generation_id=generation_id,
            details=details,
        )
    except Exception:
        return None


def resolve_effective_config(config: dict) -> dict:
    """Strictly validate full `stage-a-config-v1` without fallback defaults or unprovisioned sentinels."""
    if type(config) is not dict:
        raise ValueError("config must be a dict")
    allowed = REQUIRED_CONFIG_KEYS | OPTIONAL_RUNTIME_CONFIG_KEYS
    unknown = sorted(set(config.keys()) - allowed)
    if unknown:
        raise ValueError(f"unexpected config keys rejected: {unknown}")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config.keys()))
    if missing:
        raise ValueError(f"config missing required stage-a-config-v1 keys: {missing}")

    if config["schema"] != CONFIG_SCHEMA_VERSION:
        raise ValueError("config.schema must be stage-a-config-v1")
    if config["package"] != "stage_a_integrated" or config["data"] != "synthetic":
        raise ValueError("config.package or config.data mismatch")
    if config["scope"] != "stage_a_synthetic_preflight" or config["device"] != "cpu":
        raise ValueError("config.scope or config.device mismatch")
    budget = config["budget"]
    if type(budget) is not str or not budget or budget != budget.strip():
        raise ValueError("config.budget must be a trimmed nonempty string")
    if config["release_created"] is not False:
        raise ValueError("config.release_created must be false")
    if "owner_trust_root_status" in config and config["owner_trust_root_status"] != OWNER_TRUST_STATUS_PENDING:
        raise ValueError("config.owner_trust_root_status must be PENDING_OWNER_C4")

    max_updates = require_exact_int(config["max_updates"], "config.max_updates", minimum=1)
    batch_size = require_exact_int(config["batch"], "config.batch", minimum=1, maximum=SYNTHETIC_TRAIN_SIZE)
    seed = require_exact_int(config["seed"], "config.seed", minimum=0, maximum=MAX_SEED)
    raw_events = config["dev_events"]
    if (
        type(raw_events) is not list
        or not raw_events
        or any(type(event) is not int or event < 1 for event in raw_events)
        or raw_events != sorted(set(raw_events))
    ):
        raise ValueError("config.dev_events must be a nonempty strictly increasing list of positive exact ints")
    dev_events = list(raw_events)
    clip = _require_finite_config_number(config["clip"], "config.clip")
    if clip <= 0.0:
        raise ValueError("config.clip must be strictly positive")

    horizons_path = canonical_resolved_path(config["horizons_path"], "config.horizons_path")
    output_root = canonical_resolved_path(config["output_root"], "config.output_root")
    trust_anchor_path = canonical_resolved_path(config["trust_anchor_path"], "config.trust_anchor_path")
    revocation_path = canonical_resolved_path(config["revocation_path"], "config.revocation_path")
    nonce_ledger_path = canonical_resolved_path(config["nonce_ledger_path"], "config.nonce_ledger_path")

    if config["amsgrad"] is not EXPECTED_ADAMW_AMSGRAD:
        raise ValueError("config.amsgrad does not match pinned contract")
    if _require_finite_config_number(config["lr"], "config.lr") != EXPECTED_ADAMW_LR:
        raise ValueError("config.lr does not match pinned contract")
    if _require_finite_config_number(config["weight_decay"], "config.weight_decay") != EXPECTED_ADAMW_WEIGHT_DECAY:
        raise ValueError("config.weight_decay does not match pinned contract")
    if _require_finite_config_number(config["eps"], "config.eps") != EXPECTED_ADAMW_EPS:
        raise ValueError("config.eps does not match pinned contract")
    raw_betas = config["betas"]
    if type(raw_betas) not in (list, tuple) or len(raw_betas) != 2:
        raise ValueError("config.betas must be a 2-element list/tuple")
    betas = (
        _require_finite_config_number(raw_betas[0], "config.betas[0]"),
        _require_finite_config_number(raw_betas[1], "config.betas[1]"),
    )
    if betas != EXPECTED_ADAMW_BETAS:
        raise ValueError("config.betas does not match pinned contract")

    if _require_finite_config_number(config["dev_metric_beta"], "config.dev_metric_beta") != BETA:
        raise ValueError("config.dev_metric_beta must equal 0.05")
    if _require_finite_config_number(config["dev_metric_local_weight"], "config.dev_metric_local_weight") != LOCAL_WEIGHT:
        raise ValueError("config.dev_metric_local_weight must equal 0.5")
    if _require_finite_config_number(config["dev_metric_anchor_weight"], "config.dev_metric_anchor_weight") != ANCHOR_WEIGHT:
        raise ValueError("config.dev_metric_anchor_weight must equal 0.5")
    if _require_finite_config_number(config["dev_metric_epsilon"], "config.dev_metric_epsilon") != METRIC_EPSILON:
        raise ValueError("config.dev_metric_epsilon must equal 0.0")

    pin_checks = {
        "candidate_sha256": CANDIDATE_SHA256,
        "loss_sha256": LOSS_SHA256,
        "horizons_sha256": HORIZONS_SHA256,
        "sampler_sha256": SAMPLER_SHA256,
        "exposure_schema": SCHEMA_VERSION,
        "sampler_schema": SAMPLER_SCHEMA_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "provenance_schema": PROVENANCE_SCHEMA_VERSION,
        "manifest_schema": MANIFEST_SCHEMA_VERSION,
        "pointer_schema": POINTER_SCHEMA_VERSION,
        "commit_schema": COMMIT_SCHEMA_VERSION,
        "dev_report_schema": DEV_REPORT_SCHEMA_VERSION,
        "rng_scope": RNG_SCOPE,
        "optimizer": "AdamW",
        "synthetic_source_id": SYNTHETIC_SOURCE_ID,
        "synthetic_train_size": SYNTHETIC_TRAIN_SIZE,
        "synthetic_train_ids_sha256": SYNTHETIC_TRAIN_IDS_SHA256,
        "synthetic_remainder_policy": REMAINDER_POLICY,
        "forward_binding": FORWARD_BINDING,
        "representation_firewall": REPRESENTATION_FIREWALL,
        "train_temporal_axis_semantics": TRAIN_TEMPORAL_AXIS_SEMANTICS,
        "dev_temporal_axis_semantics": DEV_TEMPORAL_AXIS_SEMANTICS,
        "dev_metric_id": METRIC_NAME,
        "dev_metric_version": METRIC_VERSION,
        "dev_metric_status": DEV_METRIC_STATUS_VAL,
        "dev_metric_aggregation": METRIC_AGGREGATION,
        "dev_metric_direction": METRIC_DIRECTION,
        "dev_metric_tie_policy": METRIC_TIE_POLICY,
        "dev_diagnostics_policy": DIAGNOSTICS_POLICY,
        "dev_h2_policy": H2_POLICY,
    }
    for key, expected_val in pin_checks.items():
        if config[key] != expected_val:
            raise ValueError(f"config.{key} does not match pinned Stage A C4 contract")

    horizons_file_sha256 = file_sha256(Path(horizons_path), max_bytes=MAX_HORIZONS_FILE_BYTES)
    if horizons_file_sha256 != HORIZONS_SHA256:
        raise ValueError("horizons file sha256 does not match pinned HORIZONS_SHA256")

    effective = {
        "schema": CONFIG_SCHEMA_VERSION,
        "package": "stage_a_integrated",
        "data": "synthetic",
        "scope": "stage_a_synthetic_preflight",
        "device": "cpu",
        "budget": budget,
        "max_updates": max_updates,
        "batch": batch_size,
        "seed": seed,
        "dev_events": list(dev_events),
        "clip": clip,
        "horizons_path": horizons_path,
        "output_root": output_root,
        "trust_anchor_path": trust_anchor_path,
        "revocation_path": revocation_path,
        "nonce_ledger_path": nonce_ledger_path,
        "candidate_sha256": CANDIDATE_SHA256,
        "loss_sha256": LOSS_SHA256,
        "horizons_sha256": HORIZONS_SHA256,
        "horizons_file_sha256": horizons_file_sha256,
        "sampler_sha256": SAMPLER_SHA256,
        "exposure_schema": SCHEMA_VERSION,
        "sampler_schema": SAMPLER_SCHEMA_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "provenance_schema": PROVENANCE_SCHEMA_VERSION,
        "manifest_schema": MANIFEST_SCHEMA_VERSION,
        "pointer_schema": POINTER_SCHEMA_VERSION,
        "commit_schema": COMMIT_SCHEMA_VERSION,
        "dev_report_schema": DEV_REPORT_SCHEMA_VERSION,
        "rng_scope": RNG_SCOPE,
        "optimizer": "AdamW",
        "lr": EXPECTED_ADAMW_LR,
        "weight_decay": EXPECTED_ADAMW_WEIGHT_DECAY,
        "betas": list(EXPECTED_ADAMW_BETAS),
        "eps": EXPECTED_ADAMW_EPS,
        "amsgrad": EXPECTED_ADAMW_AMSGRAD,
        "synthetic_source_id": SYNTHETIC_SOURCE_ID,
        "synthetic_train_size": SYNTHETIC_TRAIN_SIZE,
        "synthetic_train_ids_sha256": SYNTHETIC_TRAIN_IDS_SHA256,
        "synthetic_remainder_policy": REMAINDER_POLICY,
        "forward_binding": FORWARD_BINDING,
        "representation_firewall": REPRESENTATION_FIREWALL,
        "train_temporal_axis_semantics": TRAIN_TEMPORAL_AXIS_SEMANTICS,
        "dev_temporal_axis_semantics": DEV_TEMPORAL_AXIS_SEMANTICS,
        "dev_metric_id": DEV_METRIC_ID,
        "dev_metric_version": DEV_METRIC_VERSION,
        "dev_metric_status": DEV_METRIC_STATUS,
        "dev_metric_beta": BETA,
        "dev_metric_local_weight": LOCAL_WEIGHT,
        "dev_metric_anchor_weight": ANCHOR_WEIGHT,
        "dev_metric_aggregation": METRIC_AGGREGATION,
        "dev_metric_direction": METRIC_DIRECTION,
        "dev_metric_tie_policy": METRIC_TIE_POLICY,
        "dev_metric_epsilon": METRIC_EPSILON,
        "dev_diagnostics_policy": DIAGNOSTICS_POLICY,
        "dev_h2_policy": H2_POLICY,
        "release_created": False,
    }
    effective_config_bytes = canonical_json_bytes(effective, max_bytes=65536)
    effective["effective_config_bytes"] = effective_config_bytes
    effective["effective_config_sha256"] = hashlib.sha256(effective_config_bytes).hexdigest()
    return effective


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
    expected_prov = expected_config.get("expected_provenance")
    if type(expected_prov) is not dict:
        raise ValueError("expected_config must supply external expected_provenance")
    expected_dev_events = expected_config.get("dev_events")
    if expected_dev_events is None:
        raise ValueError("expected_config must supply external dev_events")

    decoded = unpack(payload)
    header = _admit(decoded, expected_dev_events)
    provenance = verify_checkpoint_provenance(header, expected_prov)
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


def _sync_expected_with_config(expected: dict, resolved: dict, *, resume_policy: str, resume_from_commit_sha256: str | None) -> dict:
    if type(expected) is not dict:
        raise ExecutionNotAuthorized("expected verification parameters must be a dict")
    verified = dict(expected)
    for key in ("scope", "device", "budget", "max_updates"):
        if verified.get(key) != resolved[key]:
            raise ExecutionNotAuthorized(f"expected.{key} disagrees with resolved effective config")
    if "output_root" in verified and canonical_resolved_path(verified["output_root"], "expected.output_root") != resolved["output_root"]:
        raise ExecutionNotAuthorized("expected.output_root disagrees with resolved effective config")
    verified["output_root"] = resolved["output_root"]
    if "revocation_path" in verified and canonical_resolved_path(verified["revocation_path"], "expected.revocation_path") != resolved["revocation_path"]:
        raise ExecutionNotAuthorized("expected.revocation_path disagrees with resolved effective config")
    verified["revocation_path"] = resolved["revocation_path"]
    if "nonce_ledger_path" in verified and canonical_resolved_path(verified["nonce_ledger_path"], "expected.nonce_ledger_path") != resolved["nonce_ledger_path"]:
        raise ExecutionNotAuthorized("expected.nonce_ledger_path disagrees with resolved effective config")
    verified["nonce_ledger_path"] = resolved["nonce_ledger_path"]
    if "effective_config_sha256" in verified and verified["effective_config_sha256"] != resolved["effective_config_sha256"]:
        raise ExecutionNotAuthorized("expected.effective_config_sha256 disagrees with resolved effective config")
    verified["effective_config_sha256"] = resolved["effective_config_sha256"]
    if "resume_policy" in verified and verified["resume_policy"] != resume_policy:
        raise ExecutionNotAuthorized("expected.resume_policy disagrees with run resume mode")
    verified["resume_policy"] = resume_policy
    if "resume_from_commit_sha256" in verified and verified["resume_from_commit_sha256"] != resume_from_commit_sha256:
        raise ExecutionNotAuthorized("expected.resume_from_commit_sha256 disagrees with committed LAST")
    verified["resume_from_commit_sha256"] = resume_from_commit_sha256
    return verified


def run_synthetic(
    release_path: str,
    trust_anchor_path: str,
    expected: dict,
    config: dict,
    *,
    allow_synthetic_test_trust: bool = False,
) -> dict:
    resolved = resolve_effective_config(config)
    if canonical_resolved_path(trust_anchor_path, "trust_anchor_path") != resolved["trust_anchor_path"]:
        raise ExecutionNotAuthorized("trust_anchor_path disagrees with resolved effective config")

    root = Path(resolved["output_root"])
    has_resume_bytes = config.get("resume_bytes") is not None
    has_resume_role = config.get("resume_role") is not None
    if has_resume_bytes and has_resume_role:
        raise ValueError("config cannot specify both resume_bytes and resume_role")

    commit_path = root / "CURRENT_COMMIT"
    if has_resume_role or has_resume_bytes:
        if has_resume_role:
            role = config["resume_role"]
            if role == "BEST":
                raise ValueError(
                    "BEST is a read-only selection/evaluation artifact; canonical continuation requires resume_role='LAST'"
                )
            if role != "LAST":
                raise ValueError("resume_role must be 'LAST' for canonical continuation")
        if not commit_path.is_file():
            raise ValueError("cannot resume LAST from an uncommitted root")
        resume_from_commit_sha256 = file_sha256(commit_path, max_bytes=131072)
        resume_policy = RESUME_POLICY_LAST
    else:
        if commit_path.exists():
            raise ValueError("cannot start unresumed training in an already-committed root; specify resume_role='LAST'")
        resume_from_commit_sha256 = None
        resume_policy = RESUME_POLICY_FRESH

    verified_expected = _sync_expected_with_config(
        expected,
        resolved,
        resume_policy=resume_policy,
        resume_from_commit_sha256=resume_from_commit_sha256,
    )
    release = load(
        release_path,
        trust_anchor_path,
        verified_expected,
        revocation_path=resolved["revocation_path"],
        allow_synthetic_test_trust=allow_synthetic_test_trust,
    )
    forbidden_boundary_roots = (
        (root, Path(release.checkout_root))
        if release.checkout_root is not None
        else (root,)
    )
    ledger_path = assert_external_trust_boundary(
        resolved["nonce_ledger_path"],
        label="nonce_ledger_path",
        forbidden_roots=forbidden_boundary_roots,
    )
    reserve_release_nonce(ledger_path, release, now=release.observed_now)

    run_succeeded = False
    final_commit_sha256: str | None = resume_from_commit_sha256
    try:
        max_updates = resolved["max_updates"]
        batch_size = resolved["batch"]
        seed = resolved["seed"]
        dev_events = resolved["dev_events"]
        clip = resolved["clip"]
        effective_config_bytes = resolved["effective_config_bytes"]
        horizons_path = Path(resolved["horizons_path"])
        horizons_file_sha256 = resolved["horizons_file_sha256"]
        sampler_digest = verify_sampler_pin()
        train_id_list = train_ids()
        source_digest = verify_train_ids_pin(train_id_list)
        train_id_set = set(train_id_list)

        base_provenance = build_expected_provenance(
            effective_config_bytes,
            horizons_path,
            owner_trust_anchor_sha256=release.trust_anchor_sha256,
            owner_trust_root_status=release.trust_status,
            generation_id=None,
            horizons_file_sha256=horizons_file_sha256,
        )

        resume_payload: bytes | None = None
        expected_resume_cfg = {
            "seed": seed,
            "batch": batch_size,
            "source_id": SYNTHETIC_SOURCE_ID,
            "source_sha256": source_digest,
            "sampler_sha256": sampler_digest,
            "expected_provenance": base_provenance,
            "dev_events": dev_events,
        }
        if has_resume_bytes or has_resume_role:
            try:
                recovery = recover_commit_state(
                    root,
                    expected_provenance=base_provenance,
                    expected_dev_events=dev_events,
                )
                if not recovery["committed"] or recovery["commit_sha256"] != resume_from_commit_sha256:
                    raise ValueError("recovered committed LAST disagrees with authorized resume_from_commit_sha256")
                resolved_ptr = resolve_pointer(
                    root,
                    "LAST",
                    expected_provenance=base_provenance,
                    expected_dev_events=dev_events,
                )
                if has_resume_role:
                    resume_payload = resolved_ptr["payload"]
                else:
                    raw_resume = config["resume_bytes"]
                    if type(raw_resume) is not bytes:
                        raise ValueError("config.resume_bytes must be bytes")
                    if (
                        hashlib.sha256(raw_resume).hexdigest() != resolved_ptr["pointer"]["checkpoint_sha256"]
                        or raw_resume != resolved_ptr["payload"]
                    ):
                        raise ValueError("resume_bytes does not match authoritative committed LAST")
                    resume_payload = raw_resume
                preflight = _preflight_and_reconcile(resume_payload, expected_resume_cfg)
                if int(preflight["reconciled"]["final_state"]["completed"]) > max_updates:
                    raise ValueError("checkpoint completed step exceeds config.max_updates")
            except CommitDurabilityUnknownError as exc:
                _safe_write_failure_receipt(
                    root,
                    "commit_durability_unknown",
                    str(exc),
                    generation_id=exc.generation_id,
                    details={
                        "phase": "resume_preflight",
                        "commit_state": COMMIT_STATE_VISIBLE_DURABILITY_UNKNOWN,
                        "commit_visible": True,
                        "commit_durable": False,
                        "commit_sha256": exc.commit_sha256,
                        "must_fail_stop": True,
                    },
                )
                raise
            except PostCommitError as exc:
                _safe_write_failure_receipt(
                    root,
                    "post_commit_failure",
                    str(exc),
                    generation_id=exc.generation_id,
                    details={
                        "phase": "resume_preflight",
                        "commit_state": COMMIT_STATE_DURABLE,
                        "commit_visible": True,
                        "commit_durable": True,
                        "post_commit_phase": exc.phase,
                        "commit_sha256": exc.commit_sha256,
                    },
                )
                raise
            except Exception as exc:
                raw_role = config.get("resume_role")
                _safe_write_failure_receipt(
                    root,
                    "resume_preflight",
                    str(exc),
                    details={"resume_role": raw_role if type(raw_role) in (str, type(None)) else repr(raw_role)},
                )
                raise

        import torch

        is_fresh_run = resume_payload is None
        runtime_environment = _collect_runtime_environment()
        caller_rng_snapshot: tuple | None = None
        if not is_fresh_run:
            try:
                caller_rng_snapshot = capture_cpu_rng_snapshot()
            except Exception as exc:
                _safe_write_failure_receipt(
                    root,
                    "resume_snapshot",
                    str(exc),
                    details={"phase": "caller_rng_snapshot", "rollback_attempted": False, "fail_stop": True},
                )
                raise
        try:
            if is_fresh_run:
                _seed_fresh_cpu_rng(seed)
            model_cls, config_cls = load_class()
            model = model_cls(config_cls())
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=EXPECTED_ADAMW_LR,
                weight_decay=EXPECTED_ADAMW_WEIGHT_DECAY,
                betas=EXPECTED_ADAMW_BETAS,
                eps=EXPECTED_ADAMW_EPS,
                amsgrad=EXPECTED_ADAMW_AMSGRAD,
            )
            rows = load_rows(horizons_path)
            keys = load_keys(horizons_path)
            dev_expected_digests = expected_dev_digests(rows)
        except Exception as exc:
            if caller_rng_snapshot is not None:
                rb_errors = restore_cpu_rng_snapshot(caller_rng_snapshot)
                if rb_errors:
                    _safe_write_failure_receipt(
                        root,
                        "fatal_rollback",
                        "RNG rollback failed after pre-resume setup failure",
                        details={
                            "original_error": str(exc),
                            "rollback_errors": rb_errors,
                            "fatal": True,
                            "must_terminate_process": True,
                        },
                    )
                    raise FatalRollbackError(
                        "fatal pre-resume RNG rollback failure",
                        original_error=str(exc),
                        rollback_errors=rb_errors,
                    ) from exc
                _safe_write_failure_receipt(
                    root,
                    "resume_stage",
                    str(exc),
                    details={"phase": "pre_resume_setup", "rollback_succeeded": True, "fail_stop": True},
                )
            raise

        completed = 0
        epoch = 0
        cursor = 0
        last_cursor_before: int | None = None
        last_receipt_sha256: str | None = None
        best = None
        best_history: list[dict] = []
        macros = {}
        dev_report_sha256s: dict[int, str] = {}
        signal_statuses: dict[int, str] = {}
        last_dev_report: dict | None = None
        events = []
        exposure_frames: list[bytes] = []
        if resume_payload is not None:
            restored = resume(
                resume_payload,
                model,
                opt,
                expected_resume_cfg,
                failure_root=root,
                caller_rng_snapshot=caller_rng_snapshot,
            )
            completed = int(restored["completed"])
            epoch = int(restored["epoch"])
            cursor = int(restored["cursor"])
            last_cursor_before = restored["last_cursor_before"]
            last_receipt_sha256 = restored["last_receipt_sha256"]
            best = restored["best"]
            best_history = [dict(item) for item in restored["best_history"]]
            events = [int(event) for event in restored["events"]]
            macros = {int(step): float(value) for step, value in restored["macros"].items()}
            exposure_frames = list(restored["exposure_frames"])
        initial_model_state_sha256 = _model_params_sha256(model)

        release_audit = {
            "ticket_id": release.get("ticket_id"),
            "release_sha256": release.release_sha256,
            "resume_policy": release.get("resume_policy"),
        }
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
            step_inc = optimizer_step(
                model,
                batch,
                opt,
                clip,
                expected_forward_digests=record["forward_input_sha256"],
            )
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
                    dev_report = evaluate_with_diagnostics(
                        model,
                        items,
                        keys,
                        expected_digests=dev_expected_digests,
                        forbidden_video_ids=train_id_set,
                        require_external_expectations=True,
                    )
                except EvaluatorMutationError as exc:
                    _safe_write_failure_receipt(
                        root,
                        "dev_evaluator_mutation",
                        str(exc),
                        generation_id=generation_id,
                        details={
                            "completed": completed,
                            "mutation_details": exc.mutation_details,
                            "original_error": exc.original_error,
                            "must_fail_stop": True,
                            "rollback_promised": False,
                        },
                    )
                    raise
                except Exception as exc:
                    _safe_write_failure_receipt(
                        root,
                        "dev_eval_failure",
                        str(exc),
                        generation_id=generation_id,
                        details={"completed": completed},
                    )
                    raise
                finally:
                    model.train()

                value = dev_report["macro"]
                if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)):
                    _safe_write_failure_receipt(
                        root,
                        "dev_nonfinite",
                        f"nonfinite DEV metric at {generation_id}",
                        generation_id=generation_id,
                        details={"completed": completed, "value": repr(value)},
                    )
                    raise ValueError(f"nonfinite DEV metric at step {completed}")
                value = float(value)
                dev_report_bytes = canonical_json_bytes(dev_report, max_bytes=MAX_DEV_REPORT_BYTES)
                dev_report_sha256 = hashlib.sha256(dev_report_bytes).hexdigest()

                next_macros = dict(macros)
                next_macros[completed] = value
                is_new_best = best is None or value < best[1]
                next_best = (completed, value) if is_new_best else best
                next_best_history = (
                    best_history + [{"step": completed, "macro": value}] if is_new_best else list(best_history)
                )
                next_events = list(events) + [completed]

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
                    owner_trust_anchor_sha256=release.trust_anchor_sha256,
                    owner_trust_root_status=release.trust_status,
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
                        "determinism_policy": FRESH_SEED_DETERMINISM_POLICY,
                        "runtime_environment": runtime_environment,
                        "provenance": step_provenance,
                        "macro": value,
                        "dev_report_sha256": dev_report_sha256,
                        "release_audit": release_audit,
                        "best_step": next_best[0],
                        "best_macro": next_best[1],
                        "best_history": next_best_history,
                        "events": next_events,
                        "macros": {str(step): metric for step, metric in next_macros.items()},
                    },
                    _sections(model, opt, sampler_bytes, exposure_journal_bytes),
                )
                try:
                    _admit(unpack(payload), dev_events)
                    committed = commit_event(
                        root,
                        generation_id,
                        payload,
                        is_new_best=is_new_best,
                        expected_provenance=step_provenance,
                        dev_report_bytes=dev_report_bytes,
                        release_audit=release_audit,
                        expected_dev_events=dev_events,
                    )
                    final_commit_sha256 = committed["commit_sha256"]
                except CommitDurabilityUnknownError as exc:
                    _safe_write_failure_receipt(
                        root,
                        "commit_durability_unknown",
                        str(exc),
                        generation_id=generation_id,
                        details={
                            "completed": completed,
                            "commit_state": COMMIT_STATE_VISIBLE_DURABILITY_UNKNOWN,
                            "commit_visible": True,
                            "commit_durable": False,
                            "commit_sha256": exc.commit_sha256,
                            "must_fail_stop": True,
                        },
                    )
                    raise
                except PostCommitError as exc:
                    final_commit_sha256 = exc.commit_sha256
                    _safe_write_failure_receipt(
                        root,
                        "post_commit_failure",
                        str(exc),
                        generation_id=generation_id,
                        details={
                            "completed": completed,
                            "commit_state": COMMIT_STATE_DURABLE,
                            "commit_visible": True,
                            "commit_durable": True,
                            "post_commit_phase": exc.phase,
                            "commit_sha256": exc.commit_sha256,
                        },
                    )
                    raise
                except Exception as exc:
                    _safe_write_failure_receipt(
                        root,
                        "checkpoint_publish",
                        str(exc),
                        generation_id=generation_id,
                        details={
                            "completed": completed,
                            "commit_state": COMMIT_STATE_PRE_COMMIT_FAILURE,
                            "commit_visible": False,
                            "commit_durable": False,
                        },
                    )
                    raise
                macros = next_macros
                dev_report_sha256s[completed] = dev_report_sha256
                signal_statuses[completed] = dev_report["signal_status"]
                last_dev_report = dev_report
                best = next_best
                best_history = next_best_history
                events = next_events

        run_succeeded = True
        return {
            "completed_updates": completed,
            "best_step": None if best is None else best[0],
            "last_step": completed,
            "macros": macros,
            "events": events,
            "dev_report_sha256s": dev_report_sha256s,
            "signal_statuses": signal_statuses,
            "last_dev_report": last_dev_report,
            "determinism_policy": FRESH_SEED_DETERMINISM_POLICY,
            "fresh_seed_applied": is_fresh_run,
            "runtime_environment": runtime_environment,
            "initial_model_state_sha256": initial_model_state_sha256,
            "admission_mode": release.admission_mode,
            "observed_code_commit_sha": release.observed_code_commit_sha,
            "observed_code_tree_sha": release.observed_code_tree_sha,
            "observed_admission_utc": release.observed_now,
            "git_clean": release.git_clean,
            "module_origin_verified": release.module_origin_verified,
            "release_ticket_id": release.get("ticket_id"),
            "release_sha256": release.release_sha256,
            "owner_trust_anchor_sha256": release.trust_anchor_sha256,
            "owner_trust_root_status": release.trust_status,
            "effective_config_sha256": resolved["effective_config_sha256"],
            "observed": False,
        }
    finally:
        outcome = "COMPLETED" if run_succeeded else "FAILED"
        finalize_now = (
            release.observed_now
            if (allow_synthetic_test_trust and expected.get("now") is not None)
            else observe_trusted_utc_now()
        )
        if run_succeeded:
            finalize_release_nonce(
                ledger_path,
                release,
                outcome=outcome,
                now=finalize_now,
                final_commit_sha256=final_commit_sha256,
            )
        else:
            try:
                finalize_release_nonce(
                    ledger_path,
                    release,
                    outcome=outcome,
                    now=finalize_now,
                    final_commit_sha256=final_commit_sha256,
                )
            except Exception:
                pass


def _admit(decoded: dict, expected_dev_events: list[int] | None = None) -> dict:
    if type(decoded) is not dict or type(decoded.get("header")) is not dict:
        raise ValueError("checkpoint payload missing header")
    if expected_dev_events is None:
        raise ValueError("_admit requires external expected_dev_events")
    return validate_event_and_best_history(
        decoded["header"],
        expected_dev_events=expected_dev_events,
    )


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
    rng_snapshot: tuple,
) -> list[str]:
    errors: list[str] = []
    for name, action in (
        ("model", lambda: model.load_state_dict(snap_model, strict=True)),
        ("adamw", lambda: opt.load_state_dict(snap_adamw)),
    ):
        try:
            action()
        except Exception as rb_exc:
            errors.append(f"{name}: {rb_exc}")
    errors.extend(restore_cpu_rng_snapshot(rng_snapshot))
    return errors


def resume(
    payload: bytes,
    model,
    opt,
    expected_config: dict,
    *,
    failure_root: Path | None = None,
    caller_rng_snapshot: tuple | None = None,
) -> dict:
    import random

    import numpy
    import torch

    if caller_rng_snapshot is not None:
        restore_rng = caller_rng_snapshot
    else:
        try:
            restore_rng = capture_cpu_rng_snapshot()
        except Exception as exc:
            if failure_root is not None:
                _safe_write_failure_receipt(
                    failure_root,
                    "resume_snapshot",
                    str(exc),
                    details={"phase": "entry_rng_snapshot", "rollback_attempted": False, "fail_stop": True},
                )
            raise

    generation_id: str | None = None
    stage_phase = "preflight"
    try:
        checked = _preflight_and_reconcile(payload, expected_config)
        decoded = checked["decoded"]
        header = checked["header"]
        reconciled = checked["reconciled"]
        final_state = reconciled["final_state"]
        generation_id = checked["provenance"]["generation_id"]

        stage_phase = "stage_state"
        staged_model = stage_model_state(decoded["model"], model)
        expected_updated_pids = {
            pid
            for grp_sd, grp in zip(opt.state_dict()["param_groups"], opt.param_groups)
            for pid, param in zip(grp_sd["params"], grp["params"])
            if getattr(param, "requires_grad", True)
        }
        staged_adamw = stage_adamw_state(
            decoded["adamw"],
            opt,
            completed=int(final_state["completed"]),
            expected_updated_param_ids=expected_updated_pids,
        )
        staged_py_rng = stage_python_rng(decoded["python_rng"])
        staged_np_rng = stage_numpy_rng(decoded["numpy_rng"])
        staged_torch_rng = stage_torch_rng(decoded["torch_rng"])

        admitted_result = {
            "completed": int(final_state["completed"]),
            "epoch": int(final_state["epoch"]),
            "cursor": int(final_state["cursor"]),
            "last_cursor_before": final_state["last_cursor_before"],
            "last_receipt_sha256": final_state["last_receipt_sha256"],
            "best": None if header["best_step"] is None else (int(header["best_step"]), float(header["best_macro"])),
            "best_history": [{"step": int(e["step"]), "macro": float(e["macro"])} for e in header["best_history"]],
            "events": [int(e) for e in header["events"]],
            "macros": {int(k): float(v) for k, v in header["macros"].items()},
            "exposure": reconciled["records"],
            "exposure_frames": list(reconciled["frames"]),
            "sampler": reconciled["sampler"],
            "provenance": checked["provenance"],
        }

        stage_phase = "live_snapshot"
        snap_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
        snap_adamw = clone_optimizer_state_dict(opt)
    except Exception as exc:
        rb_errors = restore_cpu_rng_snapshot(restore_rng)
        if rb_errors:
            if failure_root is not None:
                _safe_write_failure_receipt(
                    failure_root,
                    "fatal_rollback",
                    f"CPU RNG rollback failed during resume {stage_phase} failure",
                    generation_id=generation_id,
                    details={
                        "phase": stage_phase,
                        "original_error": str(exc),
                        "rollback_errors": rb_errors,
                        "fatal": True,
                        "must_terminate_process": True,
                    },
                )
            raise FatalRollbackError(
                "fatal resume RNG rollback failure; process state tainted",
                original_error=str(exc),
                rollback_errors=rb_errors,
            ) from exc
        if failure_root is not None:
            receipt_stage = "resume_snapshot" if stage_phase == "live_snapshot" else "resume_stage"
            _safe_write_failure_receipt(
                failure_root,
                receipt_stage,
                str(exc),
                generation_id=generation_id,
                details={"phase": stage_phase, "rollback_succeeded": True, "fail_stop": True},
            )
        raise

    try:
        model.load_state_dict(staged_model, strict=True)
        opt.load_state_dict(staged_adamw)
        random.setstate(staged_py_rng)
        numpy.random.set_state(staged_np_rng)
        torch.set_rng_state(staged_torch_rng)
    except Exception as exc:
        rb_errors = _rollback_live_state(model, opt, snap_model, snap_adamw, restore_rng)
        if rb_errors:
            if failure_root is not None:
                _safe_write_failure_receipt(
                    failure_root,
                    "fatal_rollback",
                    "live state/RNG rollback failed during resume apply failure",
                    generation_id=generation_id,
                    details={
                        "phase": "live_apply",
                        "original_error": str(exc),
                        "rollback_errors": rb_errors,
                        "fatal": True,
                        "must_terminate_process": True,
                    },
                )
            raise FatalRollbackError(
                "fatal resume apply rollback failure; process state tainted",
                original_error=str(exc),
                rollback_errors=rb_errors,
            ) from exc
        if failure_root is not None:
            _safe_write_failure_receipt(
                failure_root,
                "resume_apply",
                str(exc),
                generation_id=generation_id,
                details={"phase": "live_apply", "rollback_succeeded": True, "fail_stop": True},
            )
        raise

    return admitted_result


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
