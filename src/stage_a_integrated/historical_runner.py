"""Authorized CUDA Smoke (1-4 train updates) and Fresh Stage A 128-update Historical GPU Pilot (PREPARED_NOT_RUN)."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

from . import CANDIDATE_SHA256, HORIZONS_SHA256, LOSS_SHA256, SAMPLER_SHA256
from .authorization import (
    OWNER_TRUST_STATUS_PENDING,
    RESUME_POLICY_FRESH,
    ExecutionNotAuthorized,
    assert_external_trust_boundary,
    canonical_resolved_path,
    finalize_release_nonce,
    load,
    observe_trusted_utc_now,
    reserve_release_nonce,
)
from .checkpoints import (
    COMMIT_STATE_DURABLE,
    COMMIT_STATE_PRE_COMMIT_FAILURE,
    COMMIT_STATE_VISIBLE_DURABILITY_UNKNOWN,
    EXPECTED_ADAMW_AMSGRAD,
    EXPECTED_ADAMW_BETAS,
    EXPECTED_ADAMW_EPS,
    EXPECTED_ADAMW_LR,
    EXPECTED_ADAMW_WEIGHT_DECAY,
    MANIFEST_SCHEMA_VERSION,
    MAX_DEV_REPORT_BYTES,
    MAX_HORIZONS_FILE_BYTES,
    POINTER_SCHEMA_VERSION,
    CommitDurabilityUnknownError,
    PostCommitError,
    file_sha256,
    stage_numpy_rng,
    stage_python_rng,
    stage_torch_rng,
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
    METRIC_STATUS,
    METRIC_TIE_POLICY,
    METRIC_VERSION,
    EvaluatorMutationError,
)
from .exposure import (
    FORWARD_BINDING,
    HEX_CHARS,
    MAX_RECEIPT_BYTES,
    MAX_SAMPLER_BYTES,
    SAMPLER_SCHEMA_VERSION,
    SCHEMA_VERSION,
    canonical_json_bytes,
    strict_json_loads,
)
from .forward_binding import REPRESENTATION_FIREWALL
from .gpu_runtime import (
    ADAMW_STEP_DEVICE_POLICY,
    GPU_CHECKPOINT_SCHEMA_VERSION,
    GPU_COMMIT_SCHEMA_VERSION,
    GPU_DETERMINISM_POLICY,
    GPU_PROVENANCE_SCHEMA_VERSION,
    GPU_RNG_SCOPE,
    PRECISION_POLICY,
    adamw_step_expected_device,
    build_gpu_checkpoint_sections,
    build_gpu_provenance,
    capture_runtime_rng_snapshot,
    commit_gpu_event,
    configure_fp32_precision_and_determinism,
    create_pinned_adamw,
    evaluate_historical_dev_with_diagnostics,
    gpu_optimizer_step,
    model_params_digest_any_device,
    pack_gpu_checkpoint,
    resolve_runtime_device,
    rng_snapshots_equal,
    seed_fresh_runtime_rng,
    stage_cuda_rng,
    stage_portable_adamw_state,
    stage_portable_model_state,
    unpack_gpu_checkpoint,
    verify_model_device_and_dtype,
)
from .historical_corpus import (
    HISTORICAL_SOURCE_ID,
    HISTORICAL_TEMPORAL_AXIS_SEMANTICS,
    TEXT_MASK_POLICY,
    SelectiveSafetensorsReader,
    advance_historical_window,
    build_historical_dev_items,
    eligible_train_records,
    load_verified_portable_manifest,
    load_verified_split_ids,
    next_historical_train_batch,
)
from .model_adapter import load_class
from .plan import REMAINDER_POLICY, require_exact_int, verify_sampler_pin

SMOKE_CONFIG_SCHEMA = "stage-a-historical-gpu-smoke-config-v1"
SMOKE_SCOPE = "stage_a_historical_cuda_smoke"
PILOT_CONFIG_SCHEMA = "stage-a-historical-gpu-pilot-config-v1"
PILOT_SCOPE = "stage_a_historical_gpu_pilot_128"

MAX_CONFIG_BYTES = 65_536
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024

COMMON_GPU_KEYS = frozenset(
    {
        "schema",
        "package",
        "data",
        "scope",
        "device",
        "precision",
        "precision_policy",
        "rng_scope",
        "budget",
        "max_updates",
        "batch",
        "seed",
        "dev_events",
        "clip",
        "mode",
        "strength_mode",
        "heldout_sealed",
        "corpus_manifest_path",
        "corpus_shard_path",
        "split_ids_dir",
        "horizons_path",
        "preflight_receipt_path",
        "preflight_receipt_sha256",
        "expected_eligible_train_count",
        "expected_eligible_train_ids_sha256",
        "output_root",
        "trust_anchor_path",
        "revocation_path",
        "nonce_ledger_path",
        "candidate_sha256",
        "loss_sha256",
        "horizons_sha256",
        "sampler_sha256",
        "optimizer",
        "lr",
        "weight_decay",
        "betas",
        "eps",
        "amsgrad",
        "adamw_capturable",
        "adamw_fused",
        "forward_binding",
        "representation_firewall",
        "temporal_axis_semantics",
        "text_mask_policy",
        "release_created",
    }
)
SMOKE_REQUIRED_KEYS = COMMON_GPU_KEYS | frozenset({"dev_sealed", "smoke_checkpoint_roundtrip"})
PILOT_REQUIRED_KEYS = COMMON_GPU_KEYS | frozenset(
    {
        "single_seed_exploratory",
        "checkpoint_schema",
        "provenance_schema",
        "manifest_schema",
        "pointer_schema",
        "commit_schema",
        "dev_report_schema",
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
    }
)
OPTIONAL_STATUS_KEYS = frozenset({"status", "owner_trust_root_status"})


def _req_finite(v: object, name: str) -> float:
    if type(v) not in (int, float) or isinstance(v, bool) or not math.isfinite(float(v)):
        raise ValueError(f"{name} must be finite numeric")
    return float(v)


def _req_sha(v: object, name: str) -> str:
    if type(v) is not str or len(v) != 64 or any(c not in HEX_CHARS for c in v):
        raise ValueError(f"{name} must be 64-char hex sha256")
    return v


def _atomic_write_json(path: Path, obj: dict) -> str:
    raw = canonical_json_bytes(obj, max_bytes=MAX_ARTIFACT_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.partial")
    part.write_bytes(raw)
    fd = os.open(part, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(part, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return digest


def _validate_common_gpu_config(config: dict, *, required_keys: frozenset[str], schema: str, scope: str, data: str) -> dict:
    if type(config) is not dict:
        raise ValueError("GPU config must be a dict")
    unknown = sorted(set(config.keys()) - (required_keys | OPTIONAL_STATUS_KEYS))
    if unknown:
        raise ValueError(f"unexpected GPU config keys rejected: {unknown}")
    missing = sorted(required_keys - set(config.keys()))
    if missing:
        raise ValueError(f"missing required GPU config keys: {missing}")

    if config["schema"] != schema or config["package"] != "stage_a_integrated" or config["scope"] != scope or config["data"] != data:
        raise ValueError("GPU config identity mismatch")
    budget = config["budget"]
    if type(budget) is not str or not budget or budget != budget.strip():
        raise ValueError("GPU config.budget must be a trimmed nonempty str")
    dev = config["device"]
    if type(dev) is not str or not (dev == "cuda" or dev.startswith("cuda:")):
        raise ValueError("GPU config requires CUDA device ('cuda' or 'cuda:<idx>')")
    if config["precision"] != "fp32" or config["precision_policy"] != PRECISION_POLICY or config["rng_scope"] != GPU_RNG_SCOPE:
        raise ValueError("GPU config precision/rng_scope mismatch")
    if config["heldout_sealed"] is not True or config["release_created"] is not False:
        raise ValueError("GPU config heldout_sealed must be True and release_created must be False")
    if "owner_trust_root_status" in config and config["owner_trust_root_status"] != OWNER_TRUST_STATUS_PENDING:
        raise ValueError("owner_trust_root_status must remain PENDING_OWNER_C4 in static config")
    if config["mode"] != "hybrid" or config["strength_mode"] != "schedule":
        raise ValueError("GPU config must bind mode='hybrid' and strength_mode='schedule'")
    if config["optimizer"] != "AdamW" or config["amsgrad"] is not EXPECTED_ADAMW_AMSGRAD:
        raise ValueError("GPU config AdamW mismatch")
    if _req_finite(config["lr"], "lr") != EXPECTED_ADAMW_LR or _req_finite(config["weight_decay"], "wd") != EXPECTED_ADAMW_WEIGHT_DECAY:
        raise ValueError("GPU config LR/WD mismatch")
    raw_betas = config["betas"]
    if type(raw_betas) not in (list, tuple) or len(raw_betas) != 2:
        raise ValueError("GPU config.betas must be a 2-element list/tuple")
    betas = (_req_finite(raw_betas[0], "betas[0]"), _req_finite(raw_betas[1], "betas[1]"))
    if _req_finite(config["eps"], "eps") != EXPECTED_ADAMW_EPS or betas != EXPECTED_ADAMW_BETAS:
        raise ValueError("GPU config betas/eps mismatch")
    if config["adamw_capturable"] is not False or config["adamw_fused"] is not False:
        raise ValueError("GPU config adamw_capturable and adamw_fused must be False")
    if _req_finite(config["clip"], "clip") != 1.0:
        raise ValueError("GPU config clip must equal 1.0")
    require_exact_int(config["batch"], "batch", minimum=4, maximum=4)
    require_exact_int(config["seed"], "seed", minimum=17, maximum=17)

    for k, exp_v in (
        ("candidate_sha256", CANDIDATE_SHA256),
        ("loss_sha256", LOSS_SHA256),
        ("horizons_sha256", HORIZONS_SHA256),
        ("sampler_sha256", SAMPLER_SHA256),
        ("forward_binding", FORWARD_BINDING),
        ("representation_firewall", REPRESENTATION_FIREWALL),
        ("temporal_axis_semantics", HISTORICAL_TEMPORAL_AXIS_SEMANTICS),
        ("text_mask_policy", TEXT_MASK_POLICY),
    ):
        if config[k] != exp_v:
            raise ValueError(f"GPU config.{k} pin mismatch")

    paths = {
        k: canonical_resolved_path(config[k], f"config.{k}")
        for k in (
            "corpus_manifest_path",
            "corpus_shard_path",
            "split_ids_dir",
            "horizons_path",
            "preflight_receipt_path",
            "output_root",
            "trust_anchor_path",
            "revocation_path",
            "nonce_ledger_path",
        )
    }
    preflight_sha = _req_sha(config["preflight_receipt_sha256"], "config.preflight_receipt_sha256")
    elig_cnt = require_exact_int(config["expected_eligible_train_count"], "config.expected_eligible_train_count", minimum=4, maximum=400)
    elig_sha = _req_sha(config["expected_eligible_train_ids_sha256"], "config.expected_eligible_train_ids_sha256")

    if file_sha256(Path(paths["horizons_path"]), max_bytes=MAX_HORIZONS_FILE_BYTES) != HORIZONS_SHA256:
        raise ValueError("horizons_path SHA mismatch")
    verify_sampler_pin()

    return {
        **paths,
        "device": dev,
        "budget": str(config["budget"]),
        "clip": 1.0,
        "batch": 4,
        "seed": 17,
        "preflight_receipt_sha256": preflight_sha,
        "expected_eligible_train_count": elig_cnt,
        "expected_eligible_train_ids_sha256": elig_sha,
    }


def resolve_historical_cuda_smoke_config(config: dict) -> dict:
    base = _validate_common_gpu_config(
        config,
        required_keys=SMOKE_REQUIRED_KEYS,
        schema=SMOKE_CONFIG_SCHEMA,
        scope=SMOKE_SCOPE,
        data="historical_500_1_train_only",
    )
    max_updates = require_exact_int(config["max_updates"], "config.max_updates", minimum=1, maximum=4)
    if config["dev_events"] != [] or config["dev_sealed"] is not True:
        raise ValueError("CUDA smoke must seal DEV (dev_sealed=True, dev_events=[])")
    if type(config["smoke_checkpoint_roundtrip"]) is not bool:
        raise ValueError("smoke_checkpoint_roundtrip must be exact bool")

    effective = {
        **{k: config[k] for k in SMOKE_REQUIRED_KEYS},
        "corpus_manifest_path": base["corpus_manifest_path"],
        "corpus_shard_path": base["corpus_shard_path"],
        "split_ids_dir": base["split_ids_dir"],
        "horizons_path": base["horizons_path"],
        "preflight_receipt_path": base["preflight_receipt_path"],
        "output_root": base["output_root"],
        "trust_anchor_path": base["trust_anchor_path"],
        "revocation_path": base["revocation_path"],
        "nonce_ledger_path": base["nonce_ledger_path"],
        "max_updates": max_updates,
        "clip": 1.0,
        "lr": EXPECTED_ADAMW_LR,
        "weight_decay": EXPECTED_ADAMW_WEIGHT_DECAY,
        "eps": EXPECTED_ADAMW_EPS,
        "betas": list(EXPECTED_ADAMW_BETAS),
    }
    eff_bytes = canonical_json_bytes(effective, max_bytes=MAX_CONFIG_BYTES)
    return {
        **effective,
        "effective_config_bytes": eff_bytes,
        "effective_config_sha256": hashlib.sha256(eff_bytes).hexdigest(),
    }


def resolve_historical_gpu_pilot_config(config: dict) -> dict:
    base = _validate_common_gpu_config(
        config,
        required_keys=PILOT_REQUIRED_KEYS,
        schema=PILOT_CONFIG_SCHEMA,
        scope=PILOT_SCOPE,
        data="historical_500_1_train_dev",
    )
    require_exact_int(config["max_updates"], "config.max_updates", minimum=128, maximum=128)
    if config["dev_events"] != [32, 64, 96, 128] or config["single_seed_exploratory"] is not True:
        raise ValueError("GPU pilot must specify dev_events=[32, 64, 96, 128] and single_seed_exploratory=True")
    for k, exp_v in (
        ("checkpoint_schema", GPU_CHECKPOINT_SCHEMA_VERSION),
        ("provenance_schema", GPU_PROVENANCE_SCHEMA_VERSION),
        ("manifest_schema", MANIFEST_SCHEMA_VERSION),
        ("pointer_schema", POINTER_SCHEMA_VERSION),
        ("commit_schema", GPU_COMMIT_SCHEMA_VERSION),
        ("dev_report_schema", "dev-report-v1"),
        ("dev_metric_id", METRIC_NAME),
        ("dev_metric_version", METRIC_VERSION),
        ("dev_metric_status", METRIC_STATUS),
        ("dev_metric_aggregation", METRIC_AGGREGATION),
        ("dev_metric_direction", METRIC_DIRECTION),
        ("dev_metric_tie_policy", METRIC_TIE_POLICY),
        ("dev_diagnostics_policy", DIAGNOSTICS_POLICY),
        ("dev_h2_policy", H2_POLICY),
    ):
        if config[k] != exp_v:
            raise ValueError(f"GPU pilot config.{k} mismatch")
    if _req_finite(config["dev_metric_beta"], "beta") != BETA:
        raise ValueError("GPU pilot beta != 0.05")
    if _req_finite(config["dev_metric_local_weight"], "lw") != LOCAL_WEIGHT or _req_finite(config["dev_metric_anchor_weight"], "aw") != ANCHOR_WEIGHT:
        raise ValueError("GPU pilot head weights != 0.5")
    if _req_finite(config["dev_metric_epsilon"], "eps") != METRIC_EPSILON:
        raise ValueError("GPU pilot epsilon != 0.0")

    effective = {
        **{k: config[k] for k in PILOT_REQUIRED_KEYS},
        "corpus_manifest_path": base["corpus_manifest_path"],
        "corpus_shard_path": base["corpus_shard_path"],
        "split_ids_dir": base["split_ids_dir"],
        "horizons_path": base["horizons_path"],
        "preflight_receipt_path": base["preflight_receipt_path"],
        "output_root": base["output_root"],
        "trust_anchor_path": base["trust_anchor_path"],
        "revocation_path": base["revocation_path"],
        "nonce_ledger_path": base["nonce_ledger_path"],
        "clip": 1.0,
        "lr": EXPECTED_ADAMW_LR,
        "weight_decay": EXPECTED_ADAMW_WEIGHT_DECAY,
        "eps": EXPECTED_ADAMW_EPS,
        "betas": list(EXPECTED_ADAMW_BETAS),
        "dev_metric_beta": BETA,
        "dev_metric_local_weight": LOCAL_WEIGHT,
        "dev_metric_anchor_weight": ANCHOR_WEIGHT,
        "dev_metric_epsilon": METRIC_EPSILON,
    }
    eff_bytes = canonical_json_bytes(effective, max_bytes=MAX_CONFIG_BYTES)
    return {
        **effective,
        "effective_config_bytes": eff_bytes,
        "effective_config_sha256": hashlib.sha256(eff_bytes).hexdigest(),
    }


def _verify_preflight_prerequisite(resolved: dict) -> dict:
    p_path = Path(resolved["preflight_receipt_path"])
    if not p_path.is_file():
        raise ValueError("preflight receipt file missing")
    raw = p_path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != resolved["preflight_receipt_sha256"]:
        raise ValueError("preflight_receipt_sha256 disagrees with resolved config pin")
    doc = strict_json_loads(raw, max_bytes=MAX_ARTIFACT_BYTES)
    if doc.get("overall_verdict") != "PREFLIGHT_PASS" or doc.get("heldout_sealed") is not True:
        raise ValueError("historical ABI preflight did not PASS cleanly")
    held_audit = doc.get("heldout_audit", {})
    if held_audit.get("heldout_keys_opened") != 0 or held_audit.get("heldout_payload_bytes_read") != 0:
        raise ValueError("preflight receipt indicates heldout access")
    elig = doc.get("train_eligibility_and_schedule", {})
    if elig.get("eligible_train_count") != resolved["expected_eligible_train_count"] or elig.get("eligible_train_ids_sha256") != resolved["expected_eligible_train_ids_sha256"]:
        raise ValueError("config eligible train count/hash disagrees with verified preflight receipt")
    return doc


def _sync_owner_expected(expected: dict, resolved: dict) -> dict:
    if type(expected) is not dict:
        raise ExecutionNotAuthorized("expected verification parameters must be a dict")
    verified = dict(expected)
    for k in ("scope", "device", "budget", "max_updates"):
        if verified.get(k) != resolved[k]:
            raise ExecutionNotAuthorized(f"expected.{k} disagrees with resolved GPU config")
    for pk in ("output_root", "revocation_path", "nonce_ledger_path"):
        if pk in verified and canonical_resolved_path(verified[pk], f"expected.{pk}") != resolved[pk]:
            raise ExecutionNotAuthorized(f"expected.{pk} disagrees with resolved GPU config")
        verified[pk] = resolved[pk]
    if "effective_config_sha256" in verified and verified["effective_config_sha256"] != resolved["effective_config_sha256"]:
        raise ExecutionNotAuthorized("expected.effective_config_sha256 disagrees with resolved GPU config")
    verified["effective_config_sha256"] = resolved["effective_config_sha256"]
    if "resume_policy" in verified and verified["resume_policy"] != RESUME_POLICY_FRESH:
        raise ExecutionNotAuthorized("expected.resume_policy disagrees with FRESH_ONLY")
    verified["resume_policy"] = RESUME_POLICY_FRESH
    if "resume_from_commit_sha256" in verified and verified["resume_from_commit_sha256"] is not None:
        raise ExecutionNotAuthorized("expected.resume_from_commit_sha256 must be None under FRESH_ONLY")
    verified["resume_from_commit_sha256"] = None
    return verified


def _build_historical_exposure_receipt_bytes(batch: list[dict], completed: int, epoch: int, before: int, after: int, seed: int, batch_size: int) -> tuple[bytes, dict]:
    body = {
        "schema": SCHEMA_VERSION,
        "completed": completed,
        "provider": "HISTORICAL_500_1_TRAIN",
        "source_id": batch[0]["source_id"],
        "source_sha256": batch[0]["source_sha256"],
        "sampler_sha256": batch[0]["sampler_sha256"],
        "remainder_policy": REMAINDER_POLICY,
        "forward_binding": FORWARD_BINDING,
        "seed": seed,
        "epoch": epoch,
        "batch_size": batch_size,
        "cursor_before": before,
        "cursor_after": after,
        "videos": [it["id_movi"] for it in batch],
        "anchor_ordinal": [it["anchor_ordinal"] for it in batch],
        "anchor": [it["anchor"] for it in batch],
        "selected_ordinals": [it["selected_ordinals"] for it in batch],
        "selected_labels": [it["selected_labels"] for it in batch],
        "current_ordinal": [it["current_ordinal"] for it in batch],
        "target_ordinal": [it["target_ordinal"] for it in batch],
        "current": [it["current"] for it in batch],
        "target": [it["target"] for it in batch],
        "ids_frames": [it["ids_frames"] for it in batch],
        "horizon": [it["horizon"] for it in batch],
        "forward_input_sha256": [it["forward_input_sha256"] for it in batch],
    }
    body_bytes = canonical_json_bytes(body, max_bytes=MAX_RECEIPT_BYTES)
    rec = dict(body)
    rec["receipt_sha256"] = hashlib.sha256(body_bytes).hexdigest()
    return canonical_json_bytes(rec, max_bytes=MAX_RECEIPT_BYTES), rec


def _build_historical_sampler_bytes(*, completed: int, epoch: int, cursor: int, last_before: int | None, seed: int, batch_size: int, eligible_ids_sha256: str, journal_sha256: str, last_receipt_sha256: str | None) -> bytes:
    payload = {
        "schema": SAMPLER_SCHEMA_VERSION,
        "exposure_schema": SCHEMA_VERSION,
        "completed": completed,
        "epoch": epoch,
        "cursor": cursor,
        "last_cursor_before": last_before,
        "seed": seed,
        "batch_size": batch_size,
        "source_id": HISTORICAL_SOURCE_ID,
        "source_sha256": eligible_ids_sha256,
        "sampler_sha256": SAMPLER_SHA256,
        "remainder_policy": REMAINDER_POLICY,
        "exposure_journal_sha256": journal_sha256,
        "last_receipt_sha256": last_receipt_sha256,
    }
    return canonical_json_bytes(payload, max_bytes=MAX_SAMPLER_BYTES)


def run_historical_cuda_smoke(
    release_path: str,
    trust_anchor_path: str,
    expected: dict,
    config: dict,
    *,
    allow_synthetic_test_trust: bool = False,
) -> dict:
    """Execute bounded 1-4 update CUDA smoke on actual TRAIN records only (DEV + held-out sealed)."""
    resolved = resolve_historical_cuda_smoke_config(config)
    if canonical_resolved_path(trust_anchor_path, "trust_anchor_path") != resolved["trust_anchor_path"]:
        raise ExecutionNotAuthorized("trust_anchor_path mismatch")

    root = Path(resolved["output_root"])
    release = load(
        release_path,
        trust_anchor_path,
        _sync_owner_expected(expected, resolved),
        revocation_path=resolved["revocation_path"],
        allow_synthetic_test_trust=allow_synthetic_test_trust,
    )
    forbidden_roots = (root, Path(release.checkout_root)) if release.checkout_root is not None else (root,)
    ledger_path = assert_external_trust_boundary(resolved["nonce_ledger_path"], label="nonce_ledger_path", forbidden_roots=forbidden_roots)
    reserve_release_nonce(ledger_path, release, now=release.observed_now)

    succeeded = False
    receipt_sha: str | None = None
    try:
        import random

        import numpy
        import torch

        _verify_preflight_prerequisite(resolved)
        splits = load_verified_split_ids(Path(resolved["split_ids_dir"]))
        manifest = load_verified_portable_manifest(Path(resolved["corpus_manifest_path"]), splits)
        elig_recs, _, elig_ids_sha = eligible_train_records(manifest)
        if len(elig_recs) != resolved["expected_eligible_train_count"] or elig_ids_sha != resolved["expected_eligible_train_ids_sha256"]:
            raise ValueError("eligible train pool disagrees with preflight config pin")

        device = resolve_runtime_device(resolved["device"])
        runtime_env = configure_fp32_precision_and_determinism(device)
        seed_fresh_runtime_rng(resolved["seed"], device)

        model_cls, cfg_cls = load_class()
        model = model_cls(cfg_cls()).to(device=device, dtype=torch.float32)
        verify_model_device_and_dtype(model, device)
        opt = create_pinned_adamw(model)
        before_params = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        step_records = []
        completed = 0
        epoch = 0
        cursor = 0
        last_cursor_before = None
        last_receipt_sha = None
        exposure_frames: list[bytes] = []
        tensor_cache: dict[str, tuple] = {}

        with SelectiveSafetensorsReader(Path(resolved["corpus_shard_path"]), manifest, allowed_splits=frozenset({"train"})) as reader:
            while completed < resolved["max_updates"]:
                step_epoch, cursor_before, expected_after = advance_historical_window(epoch, cursor, resolved["batch"], len(elig_recs))
                batch, cursor_after, chosen = next_historical_train_batch(
                    reader, elig_recs, eligible_ids_sha256=elig_ids_sha, seed=resolved["seed"], epoch=step_epoch, cursor_before=cursor_before, batch_size=resolved["batch"], tensor_cache=tensor_cache
                )
                if cursor_after != expected_after:
                    raise ValueError("cursor_after mismatch")
                completed += 1
                epoch, cursor, last_cursor_before = step_epoch, cursor_after, cursor_before
                exp_bytes, exp_rec = _build_historical_exposure_receipt_bytes(batch, completed, step_epoch, cursor_before, cursor_after, resolved["seed"], resolved["batch"])
                exposure_frames.append(exp_bytes)
                last_receipt_sha = exp_rec["receipt_sha256"]

                tel = gpu_optimizer_step(
                    model,
                    batch,
                    opt,
                    resolved["clip"],
                    device=device,
                    expected_forward_digests=[it["forward_input_sha256"] for it in batch],
                )
                step_records.append(
                    {
                        "step": completed,
                        "epoch": step_epoch,
                        "videos": chosen,
                        "pre_clip_grad_norm": tel["pre_clip_grad_norm"],
                        "post_clip_grad_norm": tel["post_clip_grad_norm"],
                        "loss_means": tel["loss_means"],
                    }
                )
            counters = reader.audit_counters()

        if counters["heldout_keys_opened"] != 0 or counters["heldout_payload_bytes_read"] != 0 or counters["opened_key_counts_by_split"]["dev"] != 0:
            raise PermissionError("CUDA smoke opened sealed DEV or held-out keys")

        changed_tensors = sum(1 for k, v in model.state_dict().items() if not bool(torch.equal(v.detach().cpu(), before_params[k])))
        if changed_tensors == 0:
            raise RuntimeError("CUDA smoke updates produced zero parameter tensor changes")

        sample_param = next(p for p in model.parameters() if p.requires_grad)
        sample_slot = opt.state_dict()["state"][0] if 0 in opt.state_dict()["state"] else next(iter(opt.state_dict()["state"].values()))
        actual_step_device = str(sample_slot["step"].device) if isinstance(sample_slot["step"], torch.Tensor) else type(sample_slot["step"]).__name__
        expected_step_device = adamw_step_expected_device(opt.param_groups[0], sample_param)
        if actual_step_device != expected_step_device:
            raise ValueError(f"AdamW step device {actual_step_device} != expected {expected_step_device}")

        roundtrip_report = {"executed": False}
        if resolved["smoke_checkpoint_roundtrip"]:
            journal_bytes = b"[" + b",".join(exposure_frames) + b"]"
            journal_sha = hashlib.sha256(journal_bytes).hexdigest()
            sampler_bytes = _build_historical_sampler_bytes(
                completed=completed,
                epoch=epoch,
                cursor=cursor,
                last_before=last_cursor_before,
                seed=resolved["seed"],
                batch_size=resolved["batch"],
                eligible_ids_sha256=elig_ids_sha,
                journal_sha256=journal_sha,
                last_receipt_sha256=last_receipt_sha,
            )
            gen_id = f"u{completed}"
            preflight_sha = resolved["preflight_receipt_sha256"]
            prov = build_gpu_provenance(
                resolved["effective_config_sha256"],
                eligible_train_ids_sha256=elig_ids_sha,
                eligible_train_count=len(elig_recs),
                preflight_receipt_sha256=preflight_sha,
                owner_trust_anchor_sha256=release.trust_anchor_sha256,
                owner_trust_root_status=release.trust_status,
                generation_id=gen_id,
            )
            orig_rng = capture_runtime_rng_snapshot(device)
            payload = pack_gpu_checkpoint(
                {"completed": completed, "provenance": prov, "events": [], "macros": {}, "best_step": None, "best_macro": None, "best_history": []},
                build_gpu_checkpoint_sections(model, opt, device, sampler_bytes, journal_bytes),
            )
            rt_path = root / "smoke_roundtrip" / "checkpoint.bin"
            rt_path.parent.mkdir(parents=True, exist_ok=True)
            rt_path.write_bytes(payload)
            decoded = unpack_gpu_checkpoint(rt_path.read_bytes())

            fresh_model = model_cls(cfg_cls()).to(device=device, dtype=torch.float32)
            fresh_opt = create_pinned_adamw(fresh_model)
            staged_m = stage_portable_model_state(decoded["model"], fresh_model, device)
            staged_o = stage_portable_adamw_state(decoded["adamw"], fresh_opt, device, completed=completed)
            staged_py = stage_python_rng(decoded["python_rng"])
            staged_np = stage_numpy_rng(decoded["numpy_rng"])
            staged_tc = stage_torch_rng(decoded["torch_rng"])
            staged_cu = stage_cuda_rng(decoded["cuda_rng"], device)
            fresh_model.load_state_dict(staged_m, strict=True)
            fresh_opt.load_state_dict(staged_o)
            random.setstate(staged_py)
            numpy.random.set_state(staged_np)
            torch.set_rng_state(staged_tc)
            torch.cuda.set_rng_state(staged_cu, device=int(device.index or 0))
            max_param_diff = max(float((fresh_model.state_dict()[k] - model.state_dict()[k]).abs().max().item()) for k in model.state_dict())
            orig_opt_state = opt.state_dict()["state"]
            rest_opt_state = fresh_opt.state_dict()["state"]
            if set(orig_opt_state.keys()) != set(rest_opt_state.keys()):
                raise RuntimeError("smoke checkpoint roundtrip AdamW param key mismatch")
            for pid in orig_opt_state:
                for slot_key in ("step", "exp_avg", "exp_avg_sq"):
                    ov = orig_opt_state[pid][slot_key]
                    rv = rest_opt_state[pid][slot_key]
                    if isinstance(ov, torch.Tensor):
                        if not isinstance(rv, torch.Tensor) or ov.device != rv.device or not bool(torch.equal(ov, rv)):
                            raise RuntimeError(f"smoke checkpoint roundtrip AdamW slot mismatch at pid={pid}, key={slot_key}")
                    elif ov != rv:
                        raise RuntimeError(f"smoke checkpoint roundtrip AdamW int step mismatch at pid={pid}")
            rng_restored_eq = rng_snapshots_equal(orig_rng, capture_runtime_rng_snapshot(device))
            if max_param_diff != 0.0 or not rng_restored_eq:
                raise RuntimeError(f"smoke checkpoint roundtrip mismatch: max_diff={max_param_diff}, rng={rng_restored_eq}")
            roundtrip_report = {"executed": True, "checkpoint_sha256": hashlib.sha256(payload).hexdigest(), "max_param_diff": max_param_diff, "adamw_slots_exact": True, "cpu_cuda_rng_restored": rng_restored_eq}

        receipt = {
            "schema": "historical-cuda-smoke-v1",
            "verdict": "CUDA_SMOKE_PASS",
            "completed_updates": completed,
            "changed_param_tensors": changed_tensors,
            "runtime_environment": runtime_env,
            "adamw_step_device_policy": ADAMW_STEP_DEVICE_POLICY,
            "observed_adamw_step_device": actual_step_device,
            "steps": step_records,
            "checkpoint_roundtrip": roundtrip_report,
            "dev_scored": False,
            "heldout_sealed": True,
            "heldout_audit": counters,
            "provenance": {
                "code_commit_sha": release.observed_code_commit_sha,
                "code_tree_sha": release.observed_code_tree_sha,
                "effective_config_sha256": resolved["effective_config_sha256"],
                "release_ticket_id": release.get("ticket_id"),
                "release_sha256": release.release_sha256,
            },
        }
        receipt_sha = _atomic_write_json(root / "cuda_smoke_receipt.json", receipt)
        succeeded = True
        receipt["receipt_sha256"] = receipt_sha
        return receipt
    finally:
        fin_now = release.observed_now if (allow_synthetic_test_trust and expected.get("now") is not None) else observe_trusted_utc_now()
        try:
            finalize_release_nonce(ledger_path, release, outcome="COMPLETED" if succeeded else "FAILED", now=fin_now, final_commit_sha256=receipt_sha)
        except Exception:
            if succeeded:
                raise


def run_historical_gpu_pilot(
    release_path: str,
    trust_anchor_path: str,
    expected: dict,
    config: dict,
    *,
    allow_synthetic_test_trust: bool = False,
) -> dict:
    """Execute fresh Stage A 128-update historical GPU pilot under frozen protocol and stage-a-dev-v1."""
    resolved = resolve_historical_gpu_pilot_config(config)
    if canonical_resolved_path(trust_anchor_path, "trust_anchor_path") != resolved["trust_anchor_path"]:
        raise ExecutionNotAuthorized("trust_anchor_path mismatch")

    root = Path(resolved["output_root"])
    if (root / "CURRENT_COMMIT").exists():
        raise ValueError("fresh Stage A 128-update GPU pilot requires an uncommitted fresh output_root")

    release = load(
        release_path,
        trust_anchor_path,
        _sync_owner_expected(expected, resolved),
        revocation_path=resolved["revocation_path"],
        allow_synthetic_test_trust=allow_synthetic_test_trust,
    )
    forbidden_roots = (root, Path(release.checkout_root)) if release.checkout_root is not None else (root,)
    ledger_path = assert_external_trust_boundary(resolved["nonce_ledger_path"], label="nonce_ledger_path", forbidden_roots=forbidden_roots)
    reserve_release_nonce(ledger_path, release, now=release.observed_now)

    succeeded = False
    final_commit_sha: str | None = None
    try:
        import torch

        _verify_preflight_prerequisite(resolved)
        splits = load_verified_split_ids(Path(resolved["split_ids_dir"]))
        manifest = load_verified_portable_manifest(Path(resolved["corpus_manifest_path"]), splits)
        elig_recs, inelig_recs, elig_ids_sha = eligible_train_records(manifest)
        if len(elig_recs) != resolved["expected_eligible_train_count"] or elig_ids_sha != resolved["expected_eligible_train_ids_sha256"]:
            raise ValueError("eligible train pool disagrees with preflight config pin")

        device = resolve_runtime_device(resolved["device"])
        runtime_env = configure_fp32_precision_and_determinism(device)
        seed_fresh_runtime_rng(resolved["seed"], device)

        model_cls, cfg_cls = load_class()
        model = model_cls(cfg_cls()).to(device=device, dtype=torch.float32)
        verify_model_device_and_dtype(model, device)
        opt = create_pinned_adamw(model)
        initial_params_sha256 = model_params_digest_any_device(model)

        dev_events = list(resolved["dev_events"])
        max_updates = int(resolved["max_updates"])
        completed = 0
        epoch = 0
        cursor = 0
        last_cursor_before: int | None = None
        last_receipt_sha256: str | None = None
        best = None
        best_history: list[dict] = []
        events: list[int] = []
        macros: dict[int, float] = {}
        dev_report_shas: dict[int, str] = {}
        signal_statuses: dict[int, str] = {}
        last_dev_report: dict | None = None
        exposure_frames: list[bytes] = []
        train_telemetry: list[dict] = []
        tensor_cache: dict[str, tuple] = {}
        release_audit = {"ticket_id": release.get("ticket_id"), "release_sha256": release.release_sha256, "resume_policy": release.get("resume_policy")}
        forbidden_dev_ids = set(splits["train_set"] | splits["heldout_set"])

        with SelectiveSafetensorsReader(Path(resolved["corpus_shard_path"]), manifest, allowed_splits=frozenset({"train", "dev"})) as reader:
            dev_bundle = build_historical_dev_items(reader, manifest, Path(resolved["horizons_path"]), verify_target_leak_invariance=True)

            while completed < max_updates:
                step_epoch, cursor_before, expected_after = advance_historical_window(epoch, cursor, resolved["batch"], len(elig_recs))
                batch, cursor_after, chosen = next_historical_train_batch(
                    reader, elig_recs, eligible_ids_sha256=elig_ids_sha, seed=resolved["seed"], epoch=step_epoch, cursor_before=cursor_before, batch_size=resolved["batch"], tensor_cache=tensor_cache
                )
                if cursor_after != expected_after:
                    raise ValueError("cursor_after mismatch")
                completed_next = completed + 1
                exp_bytes, exp_rec = _build_historical_exposure_receipt_bytes(batch, completed_next, step_epoch, cursor_before, cursor_after, resolved["seed"], resolved["batch"])

                tel = gpu_optimizer_step(
                    model,
                    batch,
                    opt,
                    resolved["clip"],
                    device=device,
                    expected_forward_digests=[it["forward_input_sha256"] for it in batch],
                )
                completed = completed_next
                epoch = step_epoch
                cursor = cursor_after
                last_cursor_before = cursor_before
                last_receipt_sha256 = exp_rec["receipt_sha256"]
                exposure_frames.append(exp_bytes)
                train_telemetry.append({"step": completed, "epoch": step_epoch, "videos": chosen, **tel})

                if completed in dev_events:
                    gen_id = f"u{completed}"
                    try:
                        dev_report = evaluate_historical_dev_with_diagnostics(
                            model,
                            dev_bundle["items"],
                            dev_bundle["expected_keys"],
                            device=device,
                            expected_digests=dev_bundle["expected_digests"],
                            forbidden_video_ids=forbidden_dev_ids,
                        )
                    except EvaluatorMutationError as exc:
                        write_failure_receipt(root, "dev_evaluator_mutation", str(exc), generation_id=gen_id, details={"mutation_details": exc.mutation_details, "must_fail_stop": True})
                        raise
                    except Exception as exc:
                        write_failure_receipt(root, "dev_eval_failure", str(exc), generation_id=gen_id)
                        raise
                    finally:
                        model.train()

                    val = float(dev_report["macro"])
                    dev_bytes = canonical_json_bytes(dev_report, max_bytes=MAX_DEV_REPORT_BYTES)
                    dev_sha = hashlib.sha256(dev_bytes).hexdigest()

                    next_macros = {**macros, completed: val}
                    is_new_best = best is None or val < best[1]
                    next_best = (completed, val) if is_new_best else best
                    next_history = (best_history + [{"step": completed, "macro": val}]) if is_new_best else list(best_history)
                    next_events = list(events) + [completed]

                    journal_bytes = b"[" + b",".join(exposure_frames) + b"]"
                    journal_sha = hashlib.sha256(journal_bytes).hexdigest()
                    sampler_bytes = _build_historical_sampler_bytes(
                        completed=completed,
                        epoch=epoch,
                        cursor=cursor,
                        last_before=last_cursor_before,
                        seed=resolved["seed"],
                        batch_size=resolved["batch"],
                        eligible_ids_sha256=elig_ids_sha,
                        journal_sha256=journal_sha,
                        last_receipt_sha256=last_receipt_sha256,
                    )
                    step_prov = build_gpu_provenance(
                        resolved["effective_config_sha256"],
                        eligible_train_ids_sha256=elig_ids_sha,
                        eligible_train_count=len(elig_recs),
                        preflight_receipt_sha256=resolved["preflight_receipt_sha256"],
                        owner_trust_anchor_sha256=release.trust_anchor_sha256,
                        owner_trust_root_status=release.trust_status,
                        generation_id=gen_id,
                    )
                    payload = pack_gpu_checkpoint(
                        {
                            "completed": completed,
                            "epoch": epoch,
                            "cursor": cursor,
                            "seed": resolved["seed"],
                            "batch_size": resolved["batch"],
                            "source_id": HISTORICAL_SOURCE_ID,
                            "source_sha256": elig_ids_sha,
                            "sampler_sha256": SAMPLER_SHA256,
                            "exposure_schema": SCHEMA_VERSION,
                            "exposure_journal_sha256": journal_sha,
                            "last_receipt_sha256": last_receipt_sha256,
                            "determinism_policy": GPU_DETERMINISM_POLICY,
                            "runtime_environment": runtime_env,
                            "provenance": step_prov,
                            "macro": val,
                            "dev_report_sha256": dev_sha,
                            "release_audit": release_audit,
                            "best_step": next_best[0],
                            "best_macro": next_best[1],
                            "best_history": next_history,
                            "events": next_events,
                            "macros": {str(s): m for s, m in next_macros.items()},
                        },
                        build_gpu_checkpoint_sections(model, opt, device, sampler_bytes, journal_bytes),
                    )
                    try:
                        committed = commit_gpu_event(
                            root,
                            gen_id,
                            payload,
                            is_new_best=is_new_best,
                            expected_provenance=step_prov,
                            dev_report_bytes=dev_bytes,
                            release_audit=release_audit,
                            expected_dev_events=dev_events,
                        )
                        final_commit_sha = committed["commit_sha256"]
                    except CommitDurabilityUnknownError as exc:
                        write_failure_receipt(root, "commit_durability_unknown", str(exc), generation_id=gen_id, details={"commit_state": COMMIT_STATE_VISIBLE_DURABILITY_UNKNOWN, "commit_sha256": exc.commit_sha256})
                        raise
                    except PostCommitError as exc:
                        final_commit_sha = exc.commit_sha256
                        write_failure_receipt(root, "post_commit_failure", str(exc), generation_id=gen_id, details={"commit_state": COMMIT_STATE_DURABLE, "commit_sha256": exc.commit_sha256})
                        raise
                    except Exception as exc:
                        write_failure_receipt(root, "checkpoint_publish", str(exc), generation_id=gen_id, details={"commit_state": COMMIT_STATE_PRE_COMMIT_FAILURE})
                        raise

                    macros = next_macros
                    dev_report_shas[completed] = dev_sha
                    signal_statuses[completed] = dev_report["signal_status"]
                    last_dev_report = dev_report
                    best = next_best
                    best_history = next_history
                    events = next_events

            counters = reader.audit_counters()

        if counters["heldout_keys_opened"] != 0 or counters["heldout_payload_bytes_read"] != 0:
            raise PermissionError("GPU pilot opened held-out keys")

        best_step = None if best is None else best[0]
        best_macro = None if best is None else best[1]
        assert last_dev_report is not None and best_macro is not None
        nm_macro = float(last_dev_report["no_motion_macro"])
        pilot_verdict = (
            "BEST_BELOW_NO_MOTION_SINGLE_SEED_EXPLORATORY"
            if best_macro < nm_macro
            else "NO_POSITIVE_DELTA_SIGNAL"
        )
        telemetry_sha = _atomic_write_json(root / "train_telemetry.json", {"steps": train_telemetry})
        summary = {
            "schema": "historical-gpu-pilot-summary-v1",
            "completed_updates": completed,
            "best_step": best_step,
            "best_macro": best_macro,
            "no_motion_macro": nm_macro,
            "pilot_signal_verdict": pilot_verdict,
            "no_alternate_best_or_retry": True,
            "single_seed_exploratory": True,
            "last_step": completed,
            "macros": {str(s): m for s, m in macros.items()},
            "dev_report_sha256s": {str(s): d for s, d in dev_report_shas.items()},
            "signal_statuses": {str(s): st for s, st in signal_statuses.items()},
            "final_commit_sha256": final_commit_sha,
            "initial_model_state_sha256": initial_params_sha256,
            "eligible_train_count": len(elig_recs),
            "ineligible_train_count": len(inelig_recs),
            "eligible_train_ids_sha256": elig_ids_sha,
            "train_telemetry_sha256": telemetry_sha,
            "heldout_sealed": True,
            "heldout_audit": counters,
            "runtime_environment": runtime_env,
            "provenance": {
                "code_commit_sha": release.observed_code_commit_sha,
                "code_tree_sha": release.observed_code_tree_sha,
                "effective_config_sha256": resolved["effective_config_sha256"],
                "release_ticket_id": release.get("ticket_id"),
                "release_sha256": release.release_sha256,
                "owner_trust_anchor_sha256": release.trust_anchor_sha256,
                "owner_trust_root_status": release.trust_status,
            },
        }
        summary["summary_sha256"] = _atomic_write_json(root / "pilot_summary.json", summary)
        succeeded = True
        return summary
    finally:
        fin_now = release.observed_now if (allow_synthetic_test_trust and expected.get("now") is not None) else observe_trusted_utc_now()
        try:
            finalize_release_nonce(ledger_path, release, outcome="COMPLETED" if succeeded else "FAILED", now=fin_now, final_commit_sha256=final_commit_sha)
        except Exception:
            if succeeded:
                raise
