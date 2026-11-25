"""Read-only Historical 500_1 ABI Preflight (`owner-preflight-release-v1`, TRAIN/DEV only, ZERO held-out read)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from . import CANDIDATE_SHA256, HORIZONS_SHA256, LOSS_SHA256, SAMPLER_SHA256
from .authorization import (
    ADMISSION_MODE_PRODUCTION,
    ADMISSION_MODE_SYNTHETIC_FIXTURE,
    MAX_LEDGER_BYTES,
    MAX_LEDGER_ENTRIES,
    OWNER_TRUST_STATUS_PENDING,
    ExecutionNotAuthorized,
    LedgerDurabilityUnknownError,
    assert_external_trust_boundary,
    canonical_resolved_path,
    load_revocations,
    load_trust_anchor,
    observe_clean_git_checkout,
    observe_trusted_utc_now,
)
from .checkpoints import DEV_EXPECTED_HORIZON_COUNTS, file_sha256
from .dev import EXPECTED_DEV_PAIRS, EXPECTED_DEV_VIDEOS, PREDECLARED_ABSENT_MID_VIDEOS
from .exposure import HEX_CHARS, canonical_json_bytes, strict_json_loads
from .historical_corpus import (
    CONVERSION_READBACK_LOG_SHA256,
    CONVERSION_STAGE1_SHA256,
    CONVERSION_STAGE2_SHA256,
    EXPECTED_EFFECTIVE_TRAIN_COUNT,
    EXPECTED_EFFECTIVE_TRAIN_IDS_SHA256,
    EXPECTED_EXCLUDED_TRAIN_COUNT,
    EXPECTED_RAW_TRAIN_IDS_SHA256,
    HISTORICAL_SOURCE_ID,
    HISTORICAL_TEMPORAL_AXIS_SEMANTICS,
    MIN_ELIGIBLE_TRAIN_FRAMES,
    ORIGINAL_DEV_FORWARD_DIGESTS_SHA256,
    ORIGINAL_DEV_PAYLOAD_SHA256,
    ORIGINAL_HEADER_SHA256,
    ORIGINAL_RAW_TRAIN_PAYLOAD_SHA256,
    PORTABLE_MANIFEST_SHA256,
    PORTABLE_SHARD_ATTESTED_SHA256,
    SOURCE_PTH_BYTES,
    SOURCE_PTH_SHA256,
    TEXT_MASK_POLICY,
    TRAIN_EXCLUSION_MANIFEST_SHA256,
    TRAIN_IDS_SHA256,
    DEV_IDS_SHA256,
    HELDOUT_IDS_SHA256,
    EXPECTED_SPLIT_COUNTS,
    SelectiveSafetensorsReader,
    assert_historical_no_target_leak,
    build_historical_dev_items,
    build_historical_item,
    eligible_train_records,
    epoch_remainder_video_ids,
    load_verified_portable_manifest,
    load_verified_split_ids,
    load_verified_train_exclusion_manifest,
    summarize_tensor_mask_and_digests,
)
from .plan import require_exact_int, verify_sampler_pin

PREFLIGHT_CONFIG_SCHEMA = "stage-a-historical-preflight-config-v1"
PREFLIGHT_RELEASE_SCHEMA = "owner-preflight-release-v1"
PREFLIGHT_RELEASE_DOMAIN_PREFIX = b"STCA-OWNER-PREFLIGHT-RELEASE-V1\x00"
PREFLIGHT_RECEIPT_SCHEMA = "historical-abi-preflight-v1"
PREFLIGHT_SCOPE = "stage_a_historical_abi_preflight"

MAX_AUTH_BYTES = 65_536
MAX_CONFIG_BYTES = 65_536
MAX_RECEIPT_WRITE_BYTES = 4 * 1024 * 1024
TRAIN_SAMPLE_DIGEST_LIMIT = 8
PILOT_MAX_UPDATES = 128
PILOT_BATCH_SIZE = 4
PILOT_SEED = 17

REQUIRED_PREFLIGHT_CONFIG_KEYS = frozenset(
    {
        "schema",
        "package",
        "scope",
        "device",
        "budget",
        "data",
        "max_updates",
        "heldout_sealed",
        "model_execution_allowed",
        "scoring_allowed",
        "optimizer_allowed",
        "corpus_manifest_path",
        "corpus_shard_path",
        "split_ids_dir",
        "horizons_path",
        "train_exclusion_manifest_path",
        "output_root",
        "trust_anchor_path",
        "revocation_path",
        "nonce_ledger_path",
        "candidate_sha256",
        "loss_sha256",
        "horizons_sha256",
        "sampler_sha256",
        "source_id",
        "source_pth_sha256",
        "portable_manifest_sha256",
        "portable_shard_attested_sha256",
        "train_exclusion_manifest_sha256",
        "train_ids_sha256",
        "dev_ids_sha256",
        "heldout_ids_sha256",
        "min_eligible_train_frames",
        "temporal_axis_semantics",
        "text_mask_policy",
        "release_created",
    }
)
OPTIONAL_PREFLIGHT_CONFIG_KEYS = frozenset({"status", "owner_trust_root_status"})

PREFLIGHT_RELEASE_BODY_KEYS = frozenset(
    {
        "schema",
        "allows_readonly_train_dev_preflight",
        "allows_heldout_payload_read",
        "allows_optimizer",
        "package",
        "scope",
        "device",
        "budget",
        "key_id",
        "ticket_id",
        "nonce",
        "issued_at",
        "not_before",
        "expires",
        "max_updates",
        "output_root",
        "code_commit_sha",
        "code_tree_sha",
        "effective_config_sha256",
        "corpus_manifest_sha256",
        "corpus_shard_attested_sha256",
        "horizons_sha256",
        "resume_policy",
        "resume_from_commit_sha256",
    }
)
PREFLIGHT_RELEASE_KEYS = frozenset((*PREFLIGHT_RELEASE_BODY_KEYS, "signature"))
FORBIDDEN_EMBEDDED_KEYS = frozenset({"public_key", "public_key_hex", "private_key", "signing_key", "trust_anchor", "verify_key"})


def _require_str(value: object, name: str, *, max_len: int = 256) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ExecutionNotAuthorized(f"{name} must be a trimmed nonempty str")
    if len(value) > max_len:
        raise ExecutionNotAuthorized(f"{name} exceeds max length {max_len}")
    return value


def _require_hex(value: object, name: str, *, length: int) -> str:
    txt = _require_str(value, name, max_len=length)
    if len(txt) != length or any(ch not in HEX_CHARS for ch in txt):
        raise ExecutionNotAuthorized(f"{name} must be {length}-char lowercase hex")
    return txt


def _verify_ed25519(public_bytes: bytes, message: bytes, signature: bytes) -> None:
    if type(public_bytes) is not bytes or len(public_bytes) != 32 or type(signature) is not bytes or len(signature) != 64:
        raise ExecutionNotAuthorized("malformed Ed25519 key or signature length")
    try:
        from nacl.exceptions import BadSignatureError
        from nacl.signing import VerifyKey
    except ImportError:
        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError as exc:
            raise ExecutionNotAuthorized("Ed25519 verification requires PyNaCl or cryptography") from exc
        try:
            Ed25519PublicKey.from_public_bytes(public_bytes).verify(signature, message)
            return
        except (InvalidSignature, ValueError, TypeError) as exc:
            raise ExecutionNotAuthorized("preflight release Ed25519 signature verification failed") from exc
    try:
        VerifyKey(public_bytes).verify(message, signature)
    except (BadSignatureError, ValueError, TypeError) as exc:
        raise ExecutionNotAuthorized("preflight release Ed25519 signature verification failed") from exc


def resolve_historical_preflight_config(config: dict) -> dict:
    """Strictly validate read-only historical ABI preflight configuration without defaults or unprovisioned sentinels."""
    if type(config) is not dict:
        raise ValueError("preflight config must be a dict")
    allowed = REQUIRED_PREFLIGHT_CONFIG_KEYS | OPTIONAL_PREFLIGHT_CONFIG_KEYS
    unknown = sorted(set(config.keys()) - allowed)
    if unknown:
        raise ValueError(f"unexpected preflight config keys rejected: {unknown}")
    missing = sorted(REQUIRED_PREFLIGHT_CONFIG_KEYS - set(config.keys()))
    if missing:
        raise ValueError(f"preflight config missing required keys: {missing}")

    if config["schema"] != PREFLIGHT_CONFIG_SCHEMA:
        raise ValueError("preflight config.schema mismatch")
    if config["scope"] != PREFLIGHT_SCOPE or config["device"] != "cpu" or config["data"] != "historical_500_1_train_dev_only":
        raise ValueError("preflight config scope/device/data mismatch")
    budget = config["budget"]
    if type(budget) is not str or not budget or budget != budget.strip():
        raise ValueError("preflight config.budget must be a trimmed nonempty str")
    require_exact_int(config["max_updates"], "config.max_updates", minimum=0, maximum=0)
    if config["heldout_sealed"] is not True:
        raise ValueError("heldout_sealed must be exact True")
    if config["model_execution_allowed"] is not False or config["scoring_allowed"] is not False or config["optimizer_allowed"] is not False:
        raise ValueError("model_execution_allowed, scoring_allowed, and optimizer_allowed must all be False in preflight")
    if config["release_created"] is not False:
        raise ValueError("config.release_created must be False")
    if "owner_trust_root_status" in config and config["owner_trust_root_status"] != OWNER_TRUST_STATUS_PENDING:
        raise ValueError("config.owner_trust_root_status must remain PENDING_OWNER_C4")

    corpus_manifest_path = canonical_resolved_path(config["corpus_manifest_path"], "config.corpus_manifest_path")
    corpus_shard_path = canonical_resolved_path(config["corpus_shard_path"], "config.corpus_shard_path")
    split_ids_dir = canonical_resolved_path(config["split_ids_dir"], "config.split_ids_dir")
    horizons_path = canonical_resolved_path(config["horizons_path"], "config.horizons_path")
    train_exclusion_manifest_path = canonical_resolved_path(
        config["train_exclusion_manifest_path"], "config.train_exclusion_manifest_path"
    )
    output_root = canonical_resolved_path(config["output_root"], "config.output_root")
    trust_anchor_path = canonical_resolved_path(config["trust_anchor_path"], "config.trust_anchor_path")
    revocation_path = canonical_resolved_path(config["revocation_path"], "config.revocation_path")
    nonce_ledger_path = canonical_resolved_path(config["nonce_ledger_path"], "config.nonce_ledger_path")

    pin_checks = {
        "candidate_sha256": CANDIDATE_SHA256,
        "loss_sha256": LOSS_SHA256,
        "horizons_sha256": HORIZONS_SHA256,
        "sampler_sha256": SAMPLER_SHA256,
        "source_id": HISTORICAL_SOURCE_ID,
        "source_pth_sha256": SOURCE_PTH_SHA256,
        "portable_manifest_sha256": PORTABLE_MANIFEST_SHA256,
        "portable_shard_attested_sha256": PORTABLE_SHARD_ATTESTED_SHA256,
        "train_exclusion_manifest_sha256": TRAIN_EXCLUSION_MANIFEST_SHA256,
        "train_ids_sha256": TRAIN_IDS_SHA256,
        "dev_ids_sha256": DEV_IDS_SHA256,
        "heldout_ids_sha256": HELDOUT_IDS_SHA256,
        "min_eligible_train_frames": MIN_ELIGIBLE_TRAIN_FRAMES,
        "temporal_axis_semantics": HISTORICAL_TEMPORAL_AXIS_SEMANTICS,
        "text_mask_policy": TEXT_MASK_POLICY,
    }
    for k, expected_val in pin_checks.items():
        if config[k] != expected_val:
            raise ValueError(f"preflight config.{k} disagrees with pinned Stage A historical contract")

    horizons_file_sha256 = file_sha256(Path(horizons_path), max_bytes=16 * 1024 * 1024)
    if horizons_file_sha256 != HORIZONS_SHA256:
        raise ValueError("horizons file sha256 disagrees with pinned HORIZONS_SHA256")
    manifest_file_sha256 = file_sha256(Path(corpus_manifest_path), max_bytes=8 * 1024 * 1024)
    if manifest_file_sha256 != PORTABLE_MANIFEST_SHA256:
        raise ValueError("portable manifest file sha256 disagrees with pinned PORTABLE_MANIFEST_SHA256")
    exclusion_manifest_file_sha256 = file_sha256(Path(train_exclusion_manifest_path), max_bytes=1 * 1024 * 1024)
    if exclusion_manifest_file_sha256 != TRAIN_EXCLUSION_MANIFEST_SHA256:
        raise ValueError("train exclusion manifest file sha256 disagrees with pinned TRAIN_EXCLUSION_MANIFEST_SHA256")
    verify_sampler_pin()

    if config["package"] != "stage_a_integrated":
        raise ValueError("preflight config.package mismatch")
    effective = {
        "schema": PREFLIGHT_CONFIG_SCHEMA,
        "package": "stage_a_integrated",
        "scope": PREFLIGHT_SCOPE,
        "device": "cpu",
        "budget": budget,
        "data": "historical_500_1_train_dev_only",
        "max_updates": 0,
        "heldout_sealed": True,
        "model_execution_allowed": False,
        "scoring_allowed": False,
        "optimizer_allowed": False,
        "corpus_manifest_path": corpus_manifest_path,
        "corpus_shard_path": corpus_shard_path,
        "split_ids_dir": split_ids_dir,
        "horizons_path": horizons_path,
        "train_exclusion_manifest_path": train_exclusion_manifest_path,
        "output_root": output_root,
        "trust_anchor_path": trust_anchor_path,
        "revocation_path": revocation_path,
        "nonce_ledger_path": nonce_ledger_path,
        **pin_checks,
        "horizons_file_sha256": horizons_file_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "train_exclusion_manifest_file_sha256": exclusion_manifest_file_sha256,
        "release_created": False,
    }
    eff_bytes = canonical_json_bytes(effective, max_bytes=MAX_CONFIG_BYTES)
    effective["effective_config_bytes"] = eff_bytes
    effective["effective_config_sha256"] = hashlib.sha256(eff_bytes).hexdigest()
    return effective


def _load_preflight_release(
    release_path: str | Path,
    trust_anchor_path: str | Path,
    expected: dict,
    *,
    revocation_path: str | Path,
    resolved: dict,
    allow_synthetic_test_trust: bool = False,
) -> dict:
    exp_root = canonical_resolved_path(resolved["output_root"], "output_root")
    exp_root_path = Path(exp_root)
    if type(expected) is not dict:
        raise ExecutionNotAuthorized("expected verification parameters must be a dict")
    if not allow_synthetic_test_trust:
        for ov in ("now", "code_commit_sha", "code_tree_sha"):
            if expected.get(ov) is not None:
                raise ExecutionNotAuthorized(f"override expected.{ov} forbidden in production preflight admission")
        obs = observe_clean_git_checkout(Path(__file__).resolve().parent, verify_loaded_modules=True)
        now = observe_trusted_utc_now()
        exp_commit = obs["code_commit_sha"]
        exp_tree = obs["code_tree_sha"]
        checkout_root: str | None = obs["checkout_root"]
        git_clean: bool | None = True
        mod_verified = True
        adm_mode = ADMISSION_MODE_PRODUCTION
        forbidden_roots: tuple[Path, ...] = (exp_root_path, Path(checkout_root))
    else:
        now = require_exact_int(expected["now"], "expected.now", minimum=1) if expected.get("now") is not None else observe_trusted_utc_now()
        exp_commit = _require_hex(expected.get("code_commit_sha"), "expected.code_commit_sha", length=40)
        exp_tree = _require_hex(expected.get("code_tree_sha"), "expected.code_tree_sha", length=40)
        checkout_root = None
        git_clean = None
        mod_verified = False
        adm_mode = ADMISSION_MODE_SYNTHETIC_FIXTURE
        forbidden_roots = (exp_root_path,)

    assert_external_trust_boundary(trust_anchor_path, label="trust_anchor_path", forbidden_roots=forbidden_roots)
    assert_external_trust_boundary(revocation_path, label="revocation_path", forbidden_roots=forbidden_roots)
    assert_external_trust_boundary(resolved["nonce_ledger_path"], label="nonce_ledger_path", forbidden_roots=forbidden_roots)

    ta = load_trust_anchor(trust_anchor_path, expected.get("trust_anchor_sha256"), allow_synthetic_test_trust=allow_synthetic_test_trust)
    revs = load_revocations(revocation_path, ta, expected_revocation_sha256=expected.get("revocation_sha256"))

    rel_p = Path(canonical_resolved_path(str(release_path), "release_path"))
    if not rel_p.is_file() or rel_p.stat().st_size > MAX_AUTH_BYTES:
        raise ExecutionNotAuthorized("preflight release file missing or oversized")
    rel_raw = rel_p.read_bytes()
    rel_sha = hashlib.sha256(rel_raw).hexdigest()
    try:
        release = strict_json_loads(rel_raw, max_bytes=MAX_AUTH_BYTES)
    except ValueError as exc:
        raise ExecutionNotAuthorized(f"preflight release JSON invalid: {exc}") from exc
    if type(release) is not dict or any(k in release for k in FORBIDDEN_EMBEDDED_KEYS):
        raise ExecutionNotAuthorized("malformed preflight release or forbidden embedded key")
    if set(release.keys()) != PREFLIGHT_RELEASE_KEYS:
        raise ExecutionNotAuthorized("release does not match owner-preflight-release-v1")
    if release["schema"] != PREFLIGHT_RELEASE_SCHEMA or release["package"] != "stage_a_integrated":
        raise ExecutionNotAuthorized("preflight release schema/package mismatch")
    if release["allows_readonly_train_dev_preflight"] is not True:
        raise ExecutionNotAuthorized("preflight release must set allows_readonly_train_dev_preflight=True")
    if release["allows_heldout_payload_read"] is not False or release["allows_optimizer"] is not False:
        raise ExecutionNotAuthorized("preflight release must forbid held-out reads and optimizer execution")
    if release["key_id"] != ta["key_id"]:
        raise ExecutionNotAuthorized("preflight release key_id disagrees with trust anchor")

    ticket_id = _require_str(release["ticket_id"], "release.ticket_id", max_len=128)
    nonce = _require_str(release["nonce"], "release.nonce", max_len=128)
    if ticket_id in revs["revoked_ticket_ids"] or nonce in revs["revoked_nonces"]:
        raise ExecutionNotAuthorized("preflight release ticket or nonce revoked")

    issued_at = require_exact_int(release["issued_at"], "release.issued_at", minimum=1)
    not_before = require_exact_int(release["not_before"], "release.not_before", minimum=1)
    expires = require_exact_int(release["expires"], "release.expires", minimum=1)
    if not (issued_at <= not_before < expires) or now < not_before or now > expires:
        raise ExecutionNotAuthorized("preflight release outside validity window")

    if release["scope"] != PREFLIGHT_SCOPE or release["device"] != "cpu":
        raise ExecutionNotAuthorized("preflight release scope/device mismatch")
    if release["budget"] != resolved["budget"]:
        raise ExecutionNotAuthorized("preflight release budget mismatch")
    for ek in ("scope", "device", "budget", "max_updates"):
        if ek in expected and expected.get(ek) != resolved[ek]:
            raise ExecutionNotAuthorized(f"expected.{ek} disagrees with resolved preflight config")
    for ep in ("output_root", "revocation_path", "nonce_ledger_path"):
        if ep in expected and canonical_resolved_path(expected[ep], f"expected.{ep}") != resolved[ep]:
            raise ExecutionNotAuthorized(f"expected.{ep} disagrees with resolved preflight config")
    if "effective_config_sha256" in expected and expected.get("effective_config_sha256") != resolved["effective_config_sha256"]:
        raise ExecutionNotAuthorized("expected.effective_config_sha256 disagrees with resolved preflight config")
    if require_exact_int(release["max_updates"], "release.max_updates", minimum=0, maximum=0) != 0:
        raise ExecutionNotAuthorized("preflight release max_updates must be 0")
    if canonical_resolved_path(release["output_root"], "release.output_root") != exp_root:
        raise ExecutionNotAuthorized("preflight release output_root mismatch")
    if _require_hex(release["code_commit_sha"], "release.code_commit_sha", length=40) != exp_commit:
        raise ExecutionNotAuthorized("preflight release code_commit_sha mismatch")
    if _require_hex(release["code_tree_sha"], "release.code_tree_sha", length=40) != exp_tree:
        raise ExecutionNotAuthorized("preflight release code_tree_sha mismatch")
    if _require_hex(release["effective_config_sha256"], "release.effective_config_sha256", length=64) != resolved["effective_config_sha256"]:
        raise ExecutionNotAuthorized("preflight release effective_config_sha256 mismatch")
    if release["corpus_manifest_sha256"] != PORTABLE_MANIFEST_SHA256:
        raise ExecutionNotAuthorized("preflight release corpus_manifest_sha256 mismatch")
    if release["corpus_shard_attested_sha256"] != PORTABLE_SHARD_ATTESTED_SHA256:
        raise ExecutionNotAuthorized("preflight release corpus_shard_attested_sha256 mismatch")
    if release["horizons_sha256"] != HORIZONS_SHA256:
        raise ExecutionNotAuthorized("preflight release horizons_sha256 mismatch")
    if release["resume_policy"] != "FRESH_ONLY" or release["resume_from_commit_sha256"] is not None:
        raise ExecutionNotAuthorized("preflight release must specify FRESH_ONLY with null resume_from_commit_sha256")

    sig_hex = _require_hex(release["signature"], "release.signature", length=128)
    body = {k: release[k] for k in PREFLIGHT_RELEASE_BODY_KEYS}
    _verify_ed25519(ta["public_bytes"], PREFLIGHT_RELEASE_DOMAIN_PREFIX + canonical_json_bytes(body, max_bytes=MAX_AUTH_BYTES), bytes.fromhex(sig_hex))

    normalized = dict(release)
    normalized["output_root"] = exp_root
    return {
        "receipt": normalized,
        "release_sha256": rel_sha,
        "trust_anchor_sha256": ta["public_key_sha256"],
        "trust_status": ta["trust_status"],
        "revocation_sha256": revs["revocation_sha256"],
        "observed_now": now,
        "observed_code_commit_sha": exp_commit,
        "observed_code_tree_sha": exp_tree,
        "admission_mode": adm_mode,
        "checkout_root": checkout_root,
        "git_clean": git_clean,
        "module_origin_verified": mod_verified,
    }


def _atomic_write_json(path: Path, doc: dict) -> tuple[bytes, str]:
    payload = canonical_json_bytes(doc, max_bytes=MAX_RECEIPT_WRITE_BYTES)
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_bytes(payload)
    fd = os.open(partial, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    if partial.read_bytes() != payload:
        raise ValueError("staged preflight receipt readback mismatch")
    os.replace(partial, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    if path.read_bytes() != payload:
        raise ValueError("durable preflight receipt readback mismatch")
    return payload, digest


def _scan_reverse_and_symmetry(
    fwd_index: dict[str, list[dict]],
    endpoint_index: dict[tuple[str, str], list[dict]],
    authorized_digests: list[dict],
) -> tuple[list[dict], list[str]]:
    reverse_collisions: list[dict] = []
    self_symmetric_ids: list[str] = []
    for entry in authorized_digests:
        vid = entry["id_movi"]
        fwd_matches = [m for m in (fwd_index.get(entry["rev_img_f32_sha256"]) or []) if m["id_movi"] != vid]
        if fwd_matches:
            reverse_collisions.append({"source": vid, "source_split": entry["split"], "matches": fwd_matches})
        elif fwd_index.get(entry["rev_img_f32_sha256"]):
            self_symmetric_ids.append(vid)
        rev_pair = (entry["last_frame_sha256"], entry["first_frame_sha256"])
        ep_matches = [m for m in (endpoint_index.get(rev_pair) or []) if m["id_movi"] != vid]
        if ep_matches:
            reverse_collisions.append(
                {
                    "source": vid,
                    "source_split": entry["split"],
                    "endpoint_reverse_matches": ep_matches,
                }
            )
    return reverse_collisions, sorted(set(self_symmetric_ids))


def run_historical_abi_preflight(
    release_path: str,
    trust_anchor_path: str,
    expected: dict,
    config: dict,
    *,
    allow_synthetic_test_trust: bool = False,
) -> dict:
    """Execute read-only 500_1 TRAIN/DEV ABI preflight after explicit Owner release verification."""
    resolved = resolve_historical_preflight_config(config)
    if canonical_resolved_path(trust_anchor_path, "trust_anchor_path") != resolved["trust_anchor_path"]:
        raise ExecutionNotAuthorized("trust_anchor_path disagrees with resolved preflight config")

    rel_info = _load_preflight_release(
        release_path,
        trust_anchor_path,
        expected,
        revocation_path=resolved["revocation_path"],
        resolved=resolved,
        allow_synthetic_test_trust=allow_synthetic_test_trust,
    )
    # Build an authorized release-nonce gate on external nonce_ledger_path using strict JSON ledger logic via authorization
    # Note: reserve_release_nonce requires VerifiedRelease; preflight uses a separate ledger entry namespace or dedicated check.
    ledger_path = assert_external_trust_boundary(
        resolved["nonce_ledger_path"],
        label="nonce_ledger_path",
        forbidden_roots=(Path(resolved["output_root"]),) if rel_info["checkout_root"] is None else (Path(resolved["output_root"]), Path(rel_info["checkout_root"])),
    )

    from .stage_a_sampler_v2 import sample

    # Enforce atomic single-use nonce reservation on the external preflight nonce ledger
    _preflight_nonce_reserve(ledger_path, rel_info)

    succeeded = False
    receipt_sha256: str | None = None
    try:
        splits = load_verified_split_ids(Path(resolved["split_ids_dir"]))
        manifest = load_verified_portable_manifest(Path(resolved["corpus_manifest_path"]), splits)
        excl_manifest = load_verified_train_exclusion_manifest(
            Path(resolved["train_exclusion_manifest_path"]), splits
        )
        excluded_set = excl_manifest["excluded_train_set"]

        # Metadata-only duplicate/reverse ID check across all 500 IDs
        norm_groups: dict[str, list[dict]] = {}
        for rec in manifest["records"]:
            raw_id = rec["id_movi"]
            norm = raw_id.strip().lstrip("0") or "0"
            norm_groups.setdefault(norm, []).append({"id_movi": raw_id, "split": rec["split"]})
        metadata_id_collisions = {k: v for k, v in norm_groups.items() if len(v) > 1}

        raw_elig, raw_inelig, raw_train_ids_sha256 = eligible_train_records(
            manifest, min_frames=MIN_ELIGIBLE_TRAIN_FRAMES
        )
        elig_recs, inelig_recs, eligible_ids_sha256 = eligible_train_records(
            manifest,
            min_frames=MIN_ELIGIBLE_TRAIN_FRAMES,
            excluded_train_ids=excluded_set,
        )
        eff_train_lengths = [
            len(r["ids_frames"])
            for r in manifest["records"]
            if r["split"] == "train" and r["id_movi"] not in excluded_set
        ]
        train_lengths_sorted = sorted(eff_train_lengths)
        median_len = train_lengths_sorted[len(train_lengths_sorted) // 2]

        n_elig = len(elig_recs)
        steps_per_epoch = n_elig // PILOT_BATCH_SIZE if n_elig >= PILOT_BATCH_SIZE else 0
        remainder_per_epoch = n_elig % PILOT_BATCH_SIZE
        full_epochs = PILOT_MAX_UPDATES // steps_per_epoch if steps_per_epoch > 0 else 0
        tail_steps = PILOT_MAX_UPDATES % steps_per_epoch if steps_per_epoch > 0 else 0
        ep0_remainder_ids = epoch_remainder_video_ids(
            elig_recs, seed=PILOT_SEED, epoch=0, batch_size=PILOT_BATCH_SIZE
        )
        ep1_remainder_ids = epoch_remainder_video_ids(
            elig_recs, seed=PILOT_SEED, epoch=1, batch_size=PILOT_BATCH_SIZE
        )

        failures: list[str] = []
        if len(excluded_set) != EXPECTED_EXCLUDED_TRAIN_COUNT:
            failures.append(f"excluded_train_count_mismatch: {len(excluded_set)} != {EXPECTED_EXCLUDED_TRAIN_COUNT}")
        if len(raw_elig) != EXPECTED_SPLIT_COUNTS["train"] or raw_inelig or raw_train_ids_sha256 != EXPECTED_RAW_TRAIN_IDS_SHA256:
            failures.append("raw_train_pool_or_digest_mismatch")
        if n_elig != EXPECTED_EFFECTIVE_TRAIN_COUNT:
            failures.append(f"effective_train_count_mismatch: {n_elig} != {EXPECTED_EFFECTIVE_TRAIN_COUNT}")
        if inelig_recs:
            failures.append(f"effective_ineligible_train_records: {len(inelig_recs)}")
        if eligible_ids_sha256 != EXPECTED_EFFECTIVE_TRAIN_IDS_SHA256:
            failures.append("effective_train_ids_sha256_mismatch")
        if not excluded_set.isdisjoint({r["id_movi"] for r in elig_recs}):
            failures.append("excluded_train_id_present_in_effective_pool")
        if n_elig < PILOT_BATCH_SIZE:
            failures.append(f"eligible_train_count_below_batch: {n_elig} < {PILOT_BATCH_SIZE}")

        train_sample_checks: list[dict] = []
        frame_label_audit = {
            "train": {"identity_0_to_t_minus_1": 0, "non_identity": 0},
            "dev": {"identity_0_to_t_minus_1": 0, "non_identity": 0},
            "heldout": {"identity_0_to_t_minus_1": 0, "non_identity": 0},
        }
        for rec in manifest["records"]:
            is_id = rec["ids_frames"] == list(range(len(rec["ids_frames"])))
            bucket = "identity_0_to_t_minus_1" if is_id else "non_identity"
            frame_label_audit[rec["split"]][bucket] += 1

        with SelectiveSafetensorsReader(
            Path(resolved["corpus_shard_path"]),
            manifest,
            allowed_splits=frozenset({"train", "dev"}),
            unbuffered=True,
        ) as reader:
            hdr_summary = reader.header_summary()

            authorized_recs = [r for r in manifest["records"] if r["split"] in ("train", "dev")]
            raw_fwd_index: dict[str, list[dict]] = {}
            raw_endpoint_index: dict[tuple[str, str], list[dict]] = {}
            raw_authorized_digests: list[dict] = []
            eff_fwd_index: dict[str, list[dict]] = {}
            eff_endpoint_index: dict[tuple[str, str], list[dict]] = {}
            eff_authorized_digests: list[dict] = []

            payload_digests_by_split: dict[str, list[str]] = {"train": [], "dev": []}
            eff_train_payload_digests: list[str] = []
            zero_mask_summary = {
                "stored_text_mask_present": False,
                "text_mask_policy": TEXT_MASK_POLICY,
                "train_records_with_zero_rows": 0,
                "dev_records_with_zero_rows": 0,
                "max_zero_rows_observed": 0,
                "records_with_trailing_repeated_rows": 0,
                "min_trailing_repeated_rows": 77,
                "max_trailing_repeated_rows": 0,
            }

            for rec in sorted(authorized_recs, key=lambda r: (r["split"], r["key"])):
                info = summarize_tensor_mask_and_digests(rec, reader)
                payload_digests_by_split[rec["split"]].append(info["record_payload_sha256"])
                is_excluded = rec["split"] == "train" and rec["id_movi"] in excluded_set
                is_effective = rec["split"] == "dev" or (rec["split"] == "train" and not is_excluded)

                if rec["split"] == "train" and not is_excluded:
                    eff_train_payload_digests.append(info["record_payload_sha256"])

                member_entry = {"id_movi": rec["id_movi"], "split": rec["split"]}
                digest_entry = {
                    "id_movi": rec["id_movi"],
                    "split": rec["split"],
                    "rev_img_f32_sha256": info["rev_img_f32_sha256"],
                    "first_frame_sha256": info["first_frame_sha256"],
                    "last_frame_sha256": info["last_frame_sha256"],
                }
                raw_fwd_index.setdefault(info["fwd_img_f32_sha256"], []).append(member_entry)
                raw_endpoint_index.setdefault((info["first_frame_sha256"], info["last_frame_sha256"]), []).append(member_entry)
                raw_authorized_digests.append(digest_entry)

                if is_effective:
                    eff_fwd_index.setdefault(info["fwd_img_f32_sha256"], []).append(member_entry)
                    eff_endpoint_index.setdefault((info["first_frame_sha256"], info["last_frame_sha256"]), []).append(member_entry)
                    eff_authorized_digests.append(digest_entry)

                if info["text_zero_token_rows"] > 0:
                    zero_mask_summary[f"{rec['split']}_records_with_zero_rows"] += 1
                zero_mask_summary["max_zero_rows_observed"] = max(
                    zero_mask_summary["max_zero_rows_observed"],
                    info["text_zero_token_rows"],
                )
                if info["text_trailing_repeated_rows"] > 0:
                    zero_mask_summary["records_with_trailing_repeated_rows"] += 1
                zero_mask_summary["min_trailing_repeated_rows"] = min(
                    zero_mask_summary["min_trailing_repeated_rows"],
                    info["text_trailing_repeated_rows"],
                )
                zero_mask_summary["max_trailing_repeated_rows"] = max(
                    zero_mask_summary["max_trailing_repeated_rows"],
                    info["text_trailing_repeated_rows"],
                )

                # Build sample forward digests + counterfactual target-leak invariance on retained effective train items ONLY
                if (
                    rec["split"] == "train"
                    and not is_excluded
                    and len(rec["ids_frames"]) >= MIN_ELIGIBLE_TRAIN_FRAMES
                    and len(train_sample_checks) < TRAIN_SAMPLE_DIGEST_LIMIT
                ):
                    ordinals, history = sample(rec["ids_frames"], rec["id_movi"], 0, PILOT_SEED)
                    cur_ord, tgt_ord = int(ordinals[-2]), int(ordinals[-1])
                    hist_ords = [int(x) for x in history[cur_ord]]
                    sample_item = build_historical_item(
                        rec["id_movi"],
                        rec["ids_frames"],
                        hist_ords,
                        cur_ord,
                        tgt_ord,
                        "near",
                        image_f32=info["image_f32"],
                        unclip_f32=info["unclip_f32"],
                        text_hidden_f32=info["text_hidden_f32"],
                    )
                    leak_res = assert_historical_no_target_leak(
                        sample_item,
                        image_f32=info["image_f32"],
                        unclip_f32=info["unclip_f32"],
                        text_hidden_f32=info["text_hidden_f32"],
                    )
                    train_sample_checks.append(
                        {
                            "id_movi": rec["id_movi"],
                            "history_len": len(hist_ords),
                            "current_ordinal": cur_ord,
                            "target_ordinal": tgt_ord,
                            "forward_input_sha256": leak_res["forward_input_sha256"],
                            "supervision_sha256": leak_res["clean_supervision_sha256"],
                            "target_mutation_preserved_digest": leak_res["target_mutation_preserved_digest"],
                        }
                    )

            raw_exact_img_collisions = {k: v for k, v in raw_fwd_index.items() if len(v) > 1}
            exp_by_digest = excl_manifest["expected_collision_group_by_digest"]
            raw_groups_exact_match = set(raw_exact_img_collisions.keys()) == set(exp_by_digest.keys())
            if raw_groups_exact_match:
                for digest, exp_group in exp_by_digest.items():
                    obs_members = {(m["id_movi"], m["split"]) for m in raw_exact_img_collisions[digest]}
                    exp_members = {(m["id_movi"], m["split"]) for m in exp_group["members"]}
                    if obs_members != exp_members:
                        raw_groups_exact_match = False
                        break
            if not raw_groups_exact_match:
                failures.append("raw_exact_collisions_deviate_from_exclusion_manifest")

            raw_reverse_img_collisions, raw_self_symmetric_ids = _scan_reverse_and_symmetry(
                raw_fwd_index, raw_endpoint_index, raw_authorized_digests
            )
            allowed_raw_reverse_pair = {"16379", "16380"}
            for col in raw_reverse_img_collisions:
                peers = col.get("matches") or col.get("endpoint_reverse_matches") or []
                involved = {col["source"], *(m["id_movi"] for m in peers)}
                if not involved.issubset(allowed_raw_reverse_pair):
                    failures.append(f"unexpected_raw_reverse_collision: {sorted(involved)}")

            eff_exact_img_collisions = {k: v for k, v in eff_fwd_index.items() if len(v) > 1}
            eff_reverse_img_collisions, eff_self_symmetric_ids = _scan_reverse_and_symmetry(
                eff_fwd_index, eff_endpoint_index, eff_authorized_digests
            )
            if eff_exact_img_collisions:
                failures.append(f"effective_exact_image_trajectory_collisions: {len(eff_exact_img_collisions)}")
            if eff_reverse_img_collisions:
                failures.append(f"effective_reverse_or_bothway_collisions: {len(eff_reverse_img_collisions)}")
            if metadata_id_collisions:
                failures.append(f"normalized_id_collisions: {len(metadata_id_collisions)}")
            if not train_sample_checks:
                failures.append("no_train_forward_samples_verified")

            try:
                dev_bundle = build_historical_dev_items(
                    reader,
                    manifest,
                    Path(resolved["horizons_path"]),
                    verify_target_leak_invariance=True,
                )
            except Exception as exc:
                failures.append(f"dev_manifest_verification_failed: {exc}")
                dev_bundle = None

            raw_train_payload_sha256 = hashlib.sha256("|".join(payload_digests_by_split["train"]).encode("ascii")).hexdigest()
            effective_train_payload_sha256 = hashlib.sha256("|".join(eff_train_payload_digests).encode("ascii")).hexdigest()
            dev_payload_sha256 = hashlib.sha256("|".join(payload_digests_by_split["dev"]).encode("ascii")).hexdigest()
            counters = reader.audit_counters()

        header_preserved = hdr_summary["header_sha256"] == ORIGINAL_HEADER_SHA256
        raw_train_preserved = raw_train_payload_sha256 == ORIGINAL_RAW_TRAIN_PAYLOAD_SHA256
        dev_payload_preserved = dev_payload_sha256 == ORIGINAL_DEV_PAYLOAD_SHA256
        dev_forward_preserved = dev_bundle is not None and dev_bundle["dev_forward_digests_sha256"] == ORIGINAL_DEV_FORWARD_DIGESTS_SHA256
        if not header_preserved:
            failures.append("original_header_sha256_mismatch")
        if not raw_train_preserved:
            failures.append("original_raw_train_payload_sha256_mismatch")
        if not dev_payload_preserved:
            failures.append("original_dev_payload_sha256_mismatch")
        if not dev_forward_preserved:
            failures.append("original_dev_forward_digests_sha256_mismatch")
        if len(eff_train_payload_digests) != EXPECTED_EFFECTIVE_TRAIN_COUNT:
            failures.append(f"effective_train_payload_count_mismatch: {len(eff_train_payload_digests)} != {EXPECTED_EFFECTIVE_TRAIN_COUNT}")

        if counters["heldout_keys_opened"] != 0 or counters["heldout_payload_bytes_read"] != 0:
            failures.append("heldout_seal_counter_nonzero")

        dev_audit = {
            "verified": dev_bundle is not None and dev_forward_preserved,
            "horizons_sha256": HORIZONS_SHA256,
            "expected_videos": EXPECTED_DEV_VIDEOS,
            "expected_pairs": EXPECTED_DEV_PAIRS,
            "expected_horizon_counts": dict(DEV_EXPECTED_HORIZON_COUNTS),
            "predeclared_absent_mid_videos": sorted(PREDECLARED_ABSENT_MID_VIDEOS),
            "dev_forward_digests_sha256": None if dev_bundle is None else dev_bundle["dev_forward_digests_sha256"],
            "invariance_checked_pairs": 0 if dev_bundle is None else dev_bundle["invariance_checked_pairs"],
            "dev_scored": False,
        }

        verdict = "PREFLIGHT_PASS" if not failures else "PREFLIGHT_FAIL_STOP"
        receipt = {
            "receipt_schema": PREFLIGHT_RECEIPT_SCHEMA,
            "overall_verdict": verdict,
            "failures": failures,
            "execution_mode": "READONLY_TRAIN_DEV_PREFLIGHT_NO_MODEL_NO_OPTIMIZER_NO_SCORING",
            "model_executed": False,
            "dev_scored": False,
            "optimizer_steps": 0,
            "heldout_sealed": True,
            "heldout_audit": counters,
            "reused_phase1a_evidence": {
                "source_pth_sha256": SOURCE_PTH_SHA256,
                "source_pth_bytes": SOURCE_PTH_BYTES,
                "conversion_stage1_sha256": CONVERSION_STAGE1_SHA256,
                "conversion_stage2_sha256": CONVERSION_STAGE2_SHA256,
                "conversion_readback_log_sha256": CONVERSION_READBACK_LOG_SHA256,
                "portable_manifest_sha256": manifest["manifest_sha256"],
                "portable_shard_attested_sha256": manifest["attested_shard_sha256"],
                "full_shard_rehash_skipped_reason": "avoids_reading_50_heldout_records_preserving_heldout_seal",
            },
            "split_audit": {
                "counts": manifest["split_counts"],
                "digests": splits["digests"],
                "train_dev_disjoint": True,
                "train_heldout_disjoint": True,
                "dev_heldout_disjoint": True,
                "metadata_id_collisions": metadata_id_collisions,
            },
            "train_exclusion_audit": {
                "manifest_sha256": excl_manifest["manifest_sha256"],
                "decision_id": excl_manifest["decision_id"],
                "option": excl_manifest["option"],
                "tie_policy": excl_manifest["tie_policy"],
                "source_fail_receipt_sha256": excl_manifest["source_fail_receipt_sha256"],
                "excluded_train_count": len(excluded_set),
                "excluded_train_ids": excl_manifest["excluded_train_ids"],
                "raw_collision_groups_exact_match": raw_groups_exact_match,
                "excluded_ids_in_effective_pool": 0,
            },
            "safetensors_header_audit": hdr_summary,
            "leakage_audit_raw_train_dev": {
                "authorized_records_checked": len(raw_authorized_digests),
                "heldout_records_payload_unchecked_by_policy": manifest["split_counts"]["heldout"],
                "expected_collision_groups_exact_match": raw_groups_exact_match,
                "exact_image_collisions": raw_exact_img_collisions,
                "reverse_or_bothway_collisions": raw_reverse_img_collisions,
                "self_symmetric_trajectory_ids": raw_self_symmetric_ids,
            },
            "leakage_audit_effective_train_dev": {
                "authorized_records_checked": len(eff_authorized_digests),
                "effective_train_records": len(elig_recs),
                "frozen_dev_records": manifest["split_counts"]["dev"],
                "heldout_records_payload_unchecked_by_policy": manifest["split_counts"]["heldout"],
                "exact_image_collisions": eff_exact_img_collisions,
                "reverse_or_bothway_collisions": eff_reverse_img_collisions,
                "self_symmetric_trajectory_ids": eff_self_symmetric_ids,
                "claim_scope_limitation": "BOUNDED_TO_EXACT_FORWARD_REVERSE_AND_ENDPOINT_IMAGE_EQUALITY_ON_EFFECTIVE_TRAIN387_PLUS_DEV50_HELDOUT_AND_NEAR_DUP_UNTESTED",
            },
            "train_eligibility_and_schedule": {
                "min_eligible_train_frames": MIN_ELIGIBLE_TRAIN_FRAMES,
                "raw_total_train_videos": len(raw_elig) + len(raw_inelig),
                "raw_train_ids_sha256": raw_train_ids_sha256,
                "excluded_train_count": len(excluded_set),
                "excluded_train_ids": excl_manifest["excluded_train_ids"],
                "eligible_train_count": n_elig,
                "ineligible_train_count": len(inelig_recs),
                "ineligible_train_ids": [r["id_movi"] for r in inelig_recs],
                "eligible_train_ids_sha256": eligible_ids_sha256,
                "length_min": train_lengths_sorted[0],
                "length_median": median_len,
                "length_max": train_lengths_sorted[-1],
                "population_basis": "OPTION_A_TRAIN_387_AFTER_PINNED_EXCLUSION_MANIFEST",
                "population_changed_from_387": n_elig != EXPECTED_EFFECTIVE_TRAIN_COUNT,
                "requires_ds_review_before_pilot_if_changed": True,
                "epoch_0_remainder_video_ids": ep0_remainder_ids,
                "epoch_1_remainder_video_ids": ep1_remainder_ids,
                "schedule_implications_at_batch4_128updates": {
                    "batch_size": PILOT_BATCH_SIZE,
                    "max_updates": PILOT_MAX_UPDATES,
                    "steps_per_epoch": steps_per_epoch,
                    "remainder_per_epoch": remainder_per_epoch,
                    "full_epochs_in_128_updates": full_epochs,
                    "tail_steps_in_final_epoch": tail_steps,
                },
            },
            "temporal_axis_audit": {
                "semantics": HISTORICAL_TEMPORAL_AXIS_SEMANTICS,
                "no_fps_no_physical_time_claim": True,
                "per_split_ids_frames_kind": frame_label_audit,
            },
            "tensor_and_mask_audit": {
                "raw_train_payload_sha256": raw_train_payload_sha256,
                "effective_train_payload_sha256": effective_train_payload_sha256,
                "effective_train_count_hashed": len(eff_train_payload_digests),
                "dev_payload_sha256": dev_payload_sha256,
                "original_identity_checks": {
                    "header_sha256_preserved": header_preserved,
                    "raw_train_payload_preserved": raw_train_preserved,
                    "dev_payload_preserved": dev_payload_preserved,
                    "dev_forward_digests_preserved": dev_forward_preserved,
                },
                "expected_dtypes_converted": "stored_float16_to_cpu_float32_all_finite",
                "text_mask_audit": zero_mask_summary,
            },
            "train_forward_binding_samples": train_sample_checks,
            "dev_frozen_audit": dev_audit,
            "provenance": {
                "code_commit_sha": rel_info["observed_code_commit_sha"],
                "code_tree_sha": rel_info["observed_code_tree_sha"],
                "effective_config_sha256": resolved["effective_config_sha256"],
                "train_exclusion_manifest_sha256": excl_manifest["manifest_sha256"],
                "release_ticket_id": rel_info["receipt"]["ticket_id"],
                "release_sha256": rel_info["release_sha256"],
                "owner_trust_anchor_sha256": rel_info["trust_anchor_sha256"],
                "owner_trust_root_status": rel_info["trust_status"],
                "admission_mode": rel_info["admission_mode"],
            },
        }
        out_root = Path(resolved["output_root"])
        receipt_path = out_root / "historical_abi_preflight_receipt.json"
        _, receipt_sha256 = _atomic_write_json(receipt_path, receipt)
        sha_path = out_root / "historical_abi_preflight_receipt.sha256"
        sha_path.write_text(f"{receipt_sha256}  {receipt_path.name}\n", encoding="utf-8")

        succeeded = verdict == "PREFLIGHT_PASS"
        receipt["receipt_path"] = str(receipt_path)
        receipt["receipt_sha256"] = receipt_sha256
        return receipt
    finally:
        fin_now = rel_info["observed_now"] if (allow_synthetic_test_trust and expected.get("now") is not None) else observe_trusted_utc_now()
        try:
            _preflight_nonce_finalize(
                ledger_path,
                rel_info,
                outcome="COMPLETED" if succeeded else "FAILED",
                now=fin_now,
                result_sha256=receipt_sha256,
            )
        except Exception:
            if succeeded:
                raise


def _write_preflight_ledger_atomic(ledger_path: Path, updated: dict, *, nonce: str, ticket_id: str) -> None:
    payload = canonical_json_bytes(updated, max_bytes=MAX_LEDGER_BYTES)
    partial = ledger_path.with_name(f".{ledger_path.name}.partial")
    partial.write_bytes(payload)
    pfd = os.open(partial, os.O_RDONLY)
    try:
        os.fsync(pfd)
    finally:
        os.close(pfd)
    if partial.read_bytes() != payload:
        raise ExecutionNotAuthorized("preflight nonce ledger staged readback mismatch")
    os.replace(partial, ledger_path)
    try:
        dfd = os.open(ledger_path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception as exc:
        raise LedgerDurabilityUnknownError(
            f"preflight nonce ledger {nonce} replaced but parent directory fsync failed: {exc}",
            nonce=nonce,
            ticket_id=ticket_id,
        ) from exc
    if ledger_path.read_bytes() != payload:
        raise ExecutionNotAuthorized("preflight nonce ledger durable readback mismatch")


def _preflight_nonce_reserve(ledger_path: Path, rel_info: dict) -> None:
    import fcntl

    lock_path = ledger_path.with_name(ledger_path.name + ".lock")
    if not ledger_path.is_file():
        raise ExecutionNotAuthorized("nonce ledger file missing; refusing preflight without durable ledger")
    if ledger_path.stat().st_size > MAX_LEDGER_BYTES:
        raise ExecutionNotAuthorized("nonce ledger exceeds MAX_LEDGER_BYTES")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        raise ExecutionNotAuthorized("nonce ledger is locked") from exc
    try:
        raw = ledger_path.read_bytes()
        try:
            doc = strict_json_loads(raw, max_bytes=MAX_LEDGER_BYTES)
        except ValueError as exc:
            raise ExecutionNotAuthorized(f"nonce ledger JSON invalid: {exc}") from exc
        if type(doc) is not dict or set(doc.keys()) != {"ledger_schema", "ledger_seq", "entries"} or doc.get("ledger_schema") != "nonce-ledger-v1":
            raise ExecutionNotAuthorized("nonce ledger format invalid")
        if type(doc.get("ledger_seq")) is not int or doc["ledger_seq"] < 0 or type(doc.get("entries")) is not dict:
            raise ExecutionNotAuthorized("nonce ledger header or entries invalid")
        entries = dict(doc["entries"])
        if len(entries) >= MAX_LEDGER_ENTRIES:
            raise ExecutionNotAuthorized("nonce ledger capacity exceeded")
        nonce = rel_info["receipt"]["nonce"]
        ticket_id = rel_info["receipt"]["ticket_id"]
        if nonce in entries or any(isinstance(e, dict) and e.get("ticket_id") == ticket_id for e in entries.values()):
            raise ExecutionNotAuthorized("preflight release nonce or ticket_id already recorded")
        entries[nonce] = {
            "ticket_id": ticket_id,
            "nonce": nonce,
            "release_sha256": rel_info["release_sha256"],
            "effective_config_sha256": rel_info["receipt"]["effective_config_sha256"],
            "output_root": rel_info["receipt"]["output_root"],
            "resume_policy": "FRESH_ONLY",
            "state": "RESERVED",
            "reserved_at": rel_info["observed_now"],
            "finalized_at": None,
            "outcome": None,
            "final_commit_sha256": None,
        }
        updated = {
            "ledger_schema": doc["ledger_schema"],
            "ledger_seq": int(doc["ledger_seq"]) + 1,
            "entries": entries,
        }
        _write_preflight_ledger_atomic(ledger_path, updated, nonce=nonce, ticket_id=ticket_id)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _preflight_nonce_finalize(
    ledger_path: Path,
    rel_info: dict,
    *,
    outcome: str,
    now: int,
    result_sha256: str | None,
) -> None:
    import fcntl

    lock_path = ledger_path.with_name(ledger_path.name + ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        raise ExecutionNotAuthorized("nonce ledger is locked") from exc
    try:
        raw = ledger_path.read_bytes()
        try:
            doc = strict_json_loads(raw, max_bytes=MAX_LEDGER_BYTES)
        except ValueError as exc:
            raise ExecutionNotAuthorized(f"nonce ledger JSON invalid: {exc}") from exc
        if type(doc) is not dict or set(doc.keys()) != {"ledger_schema", "ledger_seq", "entries"} or doc.get("ledger_schema") != "nonce-ledger-v1":
            raise ExecutionNotAuthorized("nonce ledger format invalid")
        if type(doc.get("ledger_seq")) is not int or doc["ledger_seq"] < 0 or type(doc.get("entries")) is not dict:
            raise ExecutionNotAuthorized("nonce ledger header or entries invalid")
        entries = dict(doc["entries"])
        nonce = rel_info["receipt"]["nonce"]
        ticket_id = rel_info["receipt"]["ticket_id"]
        existing = entries.get(nonce)
        if existing is None or not isinstance(existing, dict) or existing.get("state") != "RESERVED":
            raise ExecutionNotAuthorized("preflight nonce reservation missing or already finalized")
        entries[nonce] = {
            **existing,
            "state": "CONSUMED" if outcome == "COMPLETED" else "ABORTED",
            "finalized_at": int(now),
            "outcome": outcome,
            "final_commit_sha256": result_sha256,
        }
        updated = {
            "ledger_schema": doc["ledger_schema"],
            "ledger_seq": int(doc["ledger_seq"]) + 1,
            "entries": entries,
        }
        _write_preflight_ledger_atomic(ledger_path, updated, nonce=nonce, ticket_id=ticket_id)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
