"""Integrated C1-C4 & A-F synthetic fixture suite (PREPARED_NOT_RUN; default-deny)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import tempfile
from unittest.mock import patch

from stage_a_integrated import CANDIDATE_SHA256, HORIZONS_SHA256, LOSS_SHA256, SAMPLER_SHA256
from stage_a_integrated.authorization import (
    RELEASE_DOMAIN_PREFIX,
    RELEASE_SCHEMA,
    RESUME_POLICY_FRESH,
    RESUME_POLICY_LAST,
    REVOCATION_DOMAIN_PREFIX,
    REVOCATION_SCHEMA,
    TRUST_ANCHOR_SCHEMA,
    ExecutionNotAuthorized,
    LedgerDurabilityUnknownError,
    init_empty_nonce_ledger,
    load as load_release,
    read_nonce_ledger,
)
from stage_a_integrated.checkpoints import (
    CommitDurabilityUnknownError,
    FatalRollbackError,
    PostCommitError,
    capture_cpu_rng_snapshot,
    recover_commit_state,
    resolve_commit,
    resolve_pointer,
    restore_cpu_rng_snapshot,
    unpack,
)
import stage_a_integrated.authorization as auth_mod
from stage_a_integrated.cli import main as cli_main
from stage_a_integrated.dev import (
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
    evaluate,
    evaluate_with_diagnostics,
    expected_dev_digests,
    load_keys,
    load_rows,
)
from stage_a_integrated.exposure import (
    FORWARD_BINDING,
    canonical_json_bytes,
    reconcile_checkpoint_state,
    require_record,
)
from stage_a_integrated.forward_binding import (
    REPRESENTATION_FIREWALL,
    expected_synthetic_supervision_digest,
    verify_forward_target_invariance,
)
from stage_a_integrated.model_adapter import load_class
from stage_a_integrated.plan import dev_items, train_ids
from stage_a_integrated.samples import (
    assert_causal_input_sensitivity,
    assert_future_unselected_invariance,
    assert_no_target_leak,
    inject_target_into_forward_kwargs,
    synthetic_item,
)
from stage_a_integrated.spine import resolve_effective_config, resume, run_synthetic
from stage_a_integrated.trainer import optimizer_step

A_MACROS = {32: 0.50, 64: 0.40, 96: 0.45, 128: 0.47}
DEV_ORDER_TOL = 1e-6
RESUME_MACRO_TOL = 1e-6
FIXTURE_SEED = 17
FIXTURE_BATCH = 4
FIXTURE_MAX_UPDATES = 128
FIXTURE_DEV_EVENTS = [32, 64, 96, 128]
FIXTURE_CODE_COMMIT = "0" * 40
FIXTURE_CODE_TREE = "1" * 40
FIXTURE_NOW = 1_795_520_000
SUITE_SCHEMA_VERSION = "stage-a-af-receipt-v1"


def case_a(result: dict, best: bytes, last: bytes) -> None:
    if result["macros"] != A_MACROS or unpack(best)["header"]["completed"] != 64 or unpack(last)["header"]["completed"] != 128:
        raise ValueError("A mismatch")


def case_b(previous: bytes, after: bytes) -> None:
    if unpack(after)["header"]["completed"] != unpack(previous)["header"]["completed"]:
        raise ValueError("B replaced previous BEST")


def case_c(payload: bytes) -> None:
    unpack(payload)


def case_d(left: float, right: float) -> None:
    if abs(left - right) > DEV_ORDER_TOL:
        raise ValueError("D macro changed")


def case_e(left: dict, right: dict) -> None:
    for key in ("model", "adamw", "python_rng", "numpy_rng", "torch_rng", "sampler", "exposure"):
        if left[key] != right[key]:
            raise ValueError(f"E {key} mismatch")


def case_f(events: list[int]) -> None:
    if events != FIXTURE_DEV_EVENTS:
        raise ValueError("F events are not completed-update boundaries")


def dispatch(release_path: str, trust_anchor_path: str, expected: dict, config: dict) -> dict:
    result = run_synthetic(release_path, trust_anchor_path, expected, config)
    case_f(result["events"])
    return result


def run_all(*_args) -> None:
    raise ExecutionNotAuthorized("A-F NOT_RUN")


def _ed25519_ephemeral_signer():
    """Ephemeral in-memory Ed25519 test signer for non-authoritative synthetic fixtures ONLY."""
    try:
        from nacl.signing import SigningKey
    except ImportError:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        sk = Ed25519PrivateKey.generate()
        pub_bytes = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return pub_bytes, lambda msg: sk.sign(msg)

    sk = SigningKey.generate()
    return bytes(sk.verify_key), lambda msg: sk.sign(msg).signature


def _write_json(path: Path, payload: dict) -> Path:
    path.write_bytes(canonical_json_bytes(payload, max_bytes=65536))
    return path


def _synthetic_test_trust_bundle(root: Path, horizons_path: Path) -> dict:
    """Create ephemeral non-authoritative fixture trust root, revocations, ledger, and signer."""
    auth_dir = root / "fixture_auth"
    auth_dir.mkdir(parents=True, exist_ok=True)
    pub_bytes, sign_fn = _ed25519_ephemeral_signer()
    pub_sha = hashlib.sha256(pub_bytes).hexdigest()
    key_id = "synthetic-fixture-ed25519-k1"

    trust_path = auth_dir / "trust_anchor.json"
    _write_json(
        trust_path,
        {
            "schema": TRUST_ANCHOR_SCHEMA,
            "key_id": key_id,
            "algorithm": "ed25519",
            "public_key_hex": pub_bytes.hex(),
            "authority": "SYNTHETIC_FIXTURE_ONLY",
            "non_authoritative_test_key": True,
        },
    )

    rev_path = auth_dir / "revocations.json"
    def write_revocations(*, revoked_keys=(), revoked_tickets=(), revoked_nonces=()) -> str:
        rev_body = {
            "schema": REVOCATION_SCHEMA,
            "key_id": key_id,
            "issued_at": FIXTURE_NOW - 60,
            "revoked_key_ids": list(revoked_keys),
            "revoked_ticket_ids": list(revoked_tickets),
            "revoked_nonces": list(revoked_nonces),
        }
        rev_sig = sign_fn(REVOCATION_DOMAIN_PREFIX + canonical_json_bytes(rev_body, max_bytes=65536))
        _write_json(rev_path, {**rev_body, "signature": rev_sig.hex()})
        return hashlib.sha256(rev_path.read_bytes()).hexdigest()

    rev_sha = write_revocations()
    ledger_path = init_empty_nonce_ledger(auth_dir / "nonce_ledger.json")
    output_root = root / "ckpt_out"
    output_root.mkdir(parents=True, exist_ok=True)

    repo_cfg = json.loads((Path(__file__).resolve().parents[2] / "configs" / "stage_a_synthetic.json").read_text("utf-8"))

    def build_config(
        *,
        max_updates: int = FIXTURE_MAX_UPDATES,
        dev_events: list[int] | None = None,
        resume_role: str | None = None,
        resume_bytes: bytes | None = None,
    ) -> dict:
        cfg = dict(repo_cfg)
        cfg["max_updates"] = max_updates
        cfg["budget"] = f"{max_updates}_updates_synthetic_cpu"
        cfg["dev_events"] = list(FIXTURE_DEV_EVENTS if dev_events is None else dev_events)
        cfg["horizons_path"] = str(horizons_path.resolve())
        cfg["output_root"] = str(output_root.resolve())
        cfg["trust_anchor_path"] = str(trust_path.resolve())
        cfg["revocation_path"] = str(rev_path.resolve())
        cfg["nonce_ledger_path"] = str(ledger_path.resolve())
        if resume_role is not None:
            cfg["resume_role"] = resume_role
        if resume_bytes is not None:
            cfg["resume_bytes"] = resume_bytes
        return cfg

    ticket_counter = {"n": 0}

    def mint_release(
        cfg: dict,
        *,
        ticket_id: str | None = None,
        nonce: str | None = None,
        max_updates: int | None = None,
        resume_policy: str = RESUME_POLICY_FRESH,
        resume_from_commit_sha256: str | None = None,
        not_before: int = FIXTURE_NOW - 10,
        expires: int = FIXTURE_NOW + 3600,
        tamper_sig: bool = False,
        embed_public_key: bool = False,
        filename: str | None = None,
    ) -> tuple[Path, dict]:
        ticket_counter["n"] += 1
        idx = ticket_counter["n"]
        resolved = resolve_effective_config(cfg)
        tid = ticket_id or f"syn-ticket-{idx:04d}"
        nc = nonce or f"syn-nonce-{idx:04d}"
        body = {
            "schema": RELEASE_SCHEMA,
            "allows_execution": True,
            "package": "stage_a_integrated",
            "scope": resolved["scope"],
            "key_id": key_id,
            "ticket_id": tid,
            "nonce": nc,
            "issued_at": FIXTURE_NOW - 60,
            "not_before": not_before,
            "expires": expires,
            "device": resolved["device"],
            "budget": resolved["budget"],
            "max_updates": resolved["max_updates"] if max_updates is None else max_updates,
            "output_root": resolved["output_root"],
            "code_commit_sha": FIXTURE_CODE_COMMIT,
            "code_tree_sha": FIXTURE_CODE_TREE,
            "effective_config_sha256": resolved["effective_config_sha256"],
            "candidate_sha256": CANDIDATE_SHA256,
            "loss_sha256": LOSS_SHA256,
            "sampler_sha256": SAMPLER_SHA256,
            "data_manifest_sha256": HORIZONS_SHA256,
            "resume_policy": resume_policy,
            "resume_from_commit_sha256": resume_from_commit_sha256,
        }
        sig = bytearray(sign_fn(RELEASE_DOMAIN_PREFIX + canonical_json_bytes(body, max_bytes=65536)))
        if tamper_sig:
            sig[0] ^= 0xFF
        payload = {**body, "signature": bytes(sig).hex()}
        if embed_public_key:
            payload["public_key_hex"] = pub_bytes.hex()
        rel_path = auth_dir / (filename or f"release_{idx:04d}.json")
        rel_path.write_bytes(canonical_json_bytes(payload, max_bytes=65536))
        expected = {
            "trust_anchor_sha256": pub_sha,
            "revocation_sha256": hashlib.sha256(rev_path.read_bytes()).hexdigest(),
            "scope": resolved["scope"],
            "device": resolved["device"],
            "budget": resolved["budget"],
            "max_updates": resolved["max_updates"],
            "output_root": resolved["output_root"],
            "revocation_path": resolved["revocation_path"],
            "nonce_ledger_path": resolved["nonce_ledger_path"],
            "effective_config_sha256": resolved["effective_config_sha256"],
            "code_commit_sha": FIXTURE_CODE_COMMIT,
            "code_tree_sha": FIXTURE_CODE_TREE,
            "resume_policy": resume_policy,
            "resume_from_commit_sha256": resume_from_commit_sha256,
            "now": FIXTURE_NOW,
        }
        return rel_path, expected

    return {
        "trust_path": trust_path,
        "rev_path": rev_path,
        "ledger_path": ledger_path,
        "output_root": output_root,
        "trust_sha": pub_sha,
        "rev_sha": rev_sha,
        "key_id": key_id,
        "write_revocations": write_revocations,
        "build_config": build_config,
        "mint_release": mint_release,
        "repo_cfg": repo_cfg,
    }


def _expect_raises(exc_type, fn, label: str) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"{label} did not raise {exc_type.__name__}")


def _fake_report_with_macro(base_report: dict, target_macro: float) -> dict:
    updated = dict(base_report)
    nm = float(updated["no_motion_macro"])
    updated["macro"] = float(target_macro)
    updated["delta_vs_no_motion"] = float(target_macro) - nm
    updated["beats_no_motion"] = bool(float(target_macro) < nm)
    updated["signal_status"] = "STRICTLY_BELOW_NO_MOTION" if float(target_macro) < nm else "NO_POSITIVE_DELTA_SIGNAL"
    return updated


def fixture_authorization_positive_and_adversarial(horizons_path: Path) -> dict:
    _expect_raises(ExecutionNotAuthorized, lambda: cli_main([]), "cli default deny")
    _expect_raises(ExecutionNotAuthorized, lambda: run_all(), "run_all default deny")
    with tempfile.TemporaryDirectory() as tmp:
        bundle = _synthetic_test_trust_bundle(Path(tmp), horizons_path)
        _expect_raises(ValueError, lambda: resolve_effective_config(bundle["repo_cfg"]), "unprovisioned repo cfg")
        cfg = bundle["build_config"]()
        rel_ok, exp_ok = bundle["mint_release"](cfg)

        # Production default must reject non_authoritative_test_key=True.
        _expect_raises(
            ExecutionNotAuthorized,
            lambda: load_release(rel_ok, bundle["trust_path"], exp_ok, allow_synthetic_test_trust=False),
            "reject synthetic trust in default production path",
        )
        # Valid synthetic trust acceptance in explicit fixture path.
        loaded = load_release(rel_ok, bundle["trust_path"], exp_ok, allow_synthetic_test_trust=True)
        if loaded.get("max_updates") != FIXTURE_MAX_UPDATES:
            raise AssertionError("verified release max_updates mismatch")

        # Adversarial release checks.
        rel_emb, exp_emb = bundle["mint_release"](cfg, embed_public_key=True)
        _expect_raises(
            ExecutionNotAuthorized,
            lambda: load_release(rel_emb, bundle["trust_path"], exp_emb, allow_synthetic_test_trust=True),
            "reject embedded key in release",
        )
        rel_sig, exp_sig = bundle["mint_release"](cfg, tamper_sig=True)
        _expect_raises(
            ExecutionNotAuthorized,
            lambda: load_release(rel_sig, bundle["trust_path"], exp_sig, allow_synthetic_test_trust=True),
            "reject tampered Ed25519 signature",
        )
        rel_budget, exp_budget = bundle["mint_release"](cfg, max_updates=64)
        _expect_raises(
            ExecutionNotAuthorized,
            lambda: load_release(rel_budget, bundle["trust_path"], exp_budget, allow_synthetic_test_trust=True),
            "reject budget slack (64 != 128)",
        )
        exp_bad_pin = {**exp_ok, "trust_anchor_sha256": "0" * 64}
        _expect_raises(
            ExecutionNotAuthorized,
            lambda: load_release(rel_ok, bundle["trust_path"], exp_bad_pin, allow_synthetic_test_trust=True),
            "reject wrong trust anchor pin",
        )
        rel_expired, exp_expired = bundle["mint_release"](cfg, not_before=FIXTURE_NOW - 100, expires=FIXTURE_NOW - 1)
        _expect_raises(
            ExecutionNotAuthorized,
            lambda: load_release(rel_expired, bundle["trust_path"], exp_expired, allow_synthetic_test_trust=True),
            "reject expired ticket",
        )
        bundle["write_revocations"](revoked_tickets=[loaded.get("ticket_id")])
        _expect_raises(
            ExecutionNotAuthorized,
            lambda: load_release(rel_ok, bundle["trust_path"], {**exp_ok, "revocation_sha256": None}, allow_synthetic_test_trust=True),
            "reject revoked ticket",
        )
        bundle["write_revocations"]()

        # Ledger dir-fsync failure must raise LedgerDurabilityUnknownError fail-stop.
        rel_ldg, exp_ldg = bundle["mint_release"](cfg)
        real_auth_fsync_dir = auth_mod._fsync_dir

        def fail_ledger_dir_fsync(path: Path) -> None:
            if path == bundle["ledger_path"].parent:
                raise OSError("simulated ledger dir fsync failure")
            real_auth_fsync_dir(path)

        with patch.object(auth_mod, "_fsync_dir", side_effect=fail_ledger_dir_fsync):
            _expect_raises(
                LedgerDurabilityUnknownError,
                lambda: run_synthetic(str(rel_ldg), str(bundle["trust_path"]), exp_ldg, cfg, allow_synthetic_test_trust=True),
                "ledger dir fsync failure must fail closed with LedgerDurabilityUnknownError",
            )
    return {"status": "FIXTURE_PASS"}


def fixture_case_a_best_last_and_tie(horizons_path: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bundle = _synthetic_test_trust_bundle(root / "primary", horizons_path)
        cfg = bundle["build_config"]()
        rel, exp = bundle["mint_release"](cfg)
        seq = iter([0.50, 0.40, 0.45, 0.47])
        real_eval = evaluate_with_diagnostics

        def fake_eval(*args, **kwargs):
            report = real_eval(*args, **kwargs)
            return _fake_report_with_macro(report, next(seq))

        with patch("stage_a_integrated.spine.evaluate_with_diagnostics", side_effect=fake_eval):
            result = run_synthetic(str(rel), str(bundle["trust_path"]), exp, cfg, allow_synthetic_test_trust=True)

        # Replay of same ticket nonce must fail closed.
        _expect_raises(
            ExecutionNotAuthorized,
            lambda: run_synthetic(str(rel), str(bundle["trust_path"]), exp, cfg, allow_synthetic_test_trust=True),
            "reject single-use nonce replay",
        )
        ledger = read_nonce_ledger(bundle["ledger_path"])
        entry = ledger["entries"][load_release(rel, bundle["trust_path"], exp, allow_synthetic_test_trust=True).get("nonce")]
        if entry["state"] != "CONSUMED":
            raise AssertionError("nonce state did not finalize to CONSUMED")

        best_ptr = resolve_pointer(bundle["output_root"], "BEST")
        last_ptr = resolve_pointer(bundle["output_root"], "LAST")
        case_a(result, best_ptr["payload"], last_ptr["payload"])
        case_f(result["events"])
        if best_ptr["dev_report"]["macro"] != 0.40 or last_ptr["dev_report"]["macro"] != 0.47:
            raise AssertionError("persisted dev_report.json macro mismatch")

        # Earliest-tie sub-run: [0.50, 0.40, 0.40, 0.45] must keep BEST at step 64.
        tie_bundle = _synthetic_test_trust_bundle(root / "earliest_tie", horizons_path)
        tie_cfg = tie_bundle["build_config"]()
        tie_rel, tie_exp = tie_bundle["mint_release"](tie_cfg)
        tie_seq = iter([0.50, 0.40, 0.40, 0.45])

        def fake_tie_eval(*args, **kwargs):
            report = real_eval(*args, **kwargs)
            return _fake_report_with_macro(report, next(tie_seq))

        with patch("stage_a_integrated.spine.evaluate_with_diagnostics", side_effect=fake_tie_eval):
            tie_res = run_synthetic(str(tie_rel), str(tie_bundle["trust_path"]), tie_exp, tie_cfg, allow_synthetic_test_trust=True)
        tie_best = resolve_pointer(tie_bundle["output_root"], "BEST")
        tie_last = resolve_pointer(tie_bundle["output_root"], "LAST")
        if tie_res["best_step"] != 64 or tie_best["manifest"]["completed"] != 64 or tie_last["manifest"]["completed"] != 128:
            raise AssertionError("earliest-tie BEST preservation failed")
        if [rec["completed"] for rec in tie_best["commit"]["best_history"]] != [32, 64]:
            raise AssertionError("earliest-tie best_history recorded non-strict improvement")
    return {"status": "FIXTURE_PASS"}


def fixture_case_b_nonfinite_dev(horizons_path: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        bundle = _synthetic_test_trust_bundle(Path(tmp), horizons_path)
        cfg = bundle["build_config"]()
        rel, exp = bundle["mint_release"](cfg)
        macros = iter([0.50, 0.40, float("nan"), 0.30])
        real_eval = evaluate_with_diagnostics

        def fake_eval(*args, **kwargs):
            report = real_eval(*args, **kwargs)
            return _fake_report_with_macro(report, next(macros))

        with patch("stage_a_integrated.spine.evaluate_with_diagnostics", side_effect=fake_eval):
            _expect_raises(
                ValueError,
                lambda: run_synthetic(str(rel), str(bundle["trust_path"]), exp, cfg, allow_synthetic_test_trust=True),
                "nonfinite dev at step 96 must fail closed",
            )
        best_ptr = resolve_pointer(bundle["output_root"], "BEST")
        last_ptr = resolve_pointer(bundle["output_root"], "LAST")
        if best_ptr["manifest"]["completed"] != 64 or last_ptr["manifest"]["completed"] != 64:
            raise AssertionError("nonfinite dev replaced prior committed BEST/LAST")
        ledger = read_nonce_ledger(bundle["ledger_path"])
        only_entry = next(iter(ledger["entries"].values()))
        if only_entry["state"] != "ABORTED":
            raise AssertionError("failed run nonce must remain permanently ABORTED")
    return {"status": "FIXTURE_PASS"}


def fixture_case_c_corrupted_artifacts(horizons_path: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        bundle = _synthetic_test_trust_bundle(Path(tmp), horizons_path)
        cfg = bundle["build_config"](max_updates=32, dev_events=[32])
        rel, exp = bundle["mint_release"](cfg)
        run_synthetic(str(rel), str(bundle["trust_path"]), exp, cfg, allow_synthetic_test_trust=True)
        resolved = resolve_commit(bundle["output_root"])
        dev_rel = resolved["last"]["record"]["dev_report_relpath"]
        dev_file = bundle["output_root"] / dev_rel
        orig_dev = dev_file.read_bytes()
        dev_file.write_bytes(b"{}")
        _expect_raises(ValueError, lambda: resolve_commit(bundle["output_root"]), "corrupted dev_report.json must fail")
        dev_file.write_bytes(orig_dev)
        ckpt_file = bundle["output_root"] / resolved["last"]["record"]["checkpoint_relpath"]
        raw_ckpt = ckpt_file.read_bytes()
        ckpt_file.write_bytes(raw_ckpt[:-4] + b"XXXX")
        _expect_raises(ValueError, lambda: case_c(ckpt_file.read_bytes()), "corrupted checkpoint.bin must fail")
        _expect_raises(ValueError, lambda: resolve_commit(bundle["output_root"]), "corrupted checkpoint must fail resolve")
    return {"status": "FIXTURE_PASS"}


def fixture_case_d_dev_order_invariance(horizons_path: Path) -> dict:
    rows = load_rows(horizons_path)
    keys = load_keys(horizons_path)
    digests = expected_dev_digests(rows)
    items_canon = dev_items(rows)
    items_rev = list(reversed(items_canon))
    model_cls, config_cls = load_class()
    model = model_cls(config_cls())
    train_set = set(train_ids())
    rep_canon = evaluate_with_diagnostics(
        model,
        items_canon,
        keys,
        expected_digests=digests,
        forbidden_video_ids=train_set,
        require_external_expectations=True,
    )
    rep_rev = evaluate_with_diagnostics(
        model,
        items_rev,
        keys,
        expected_digests=digests,
        forbidden_video_ids=train_set,
        require_external_expectations=True,
    )
    case_d(rep_canon["macro"], rep_rev["macro"])
    if rep_canon["dev_forward_digests_sha256"] != rep_rev["dev_forward_digests_sha256"]:
        raise AssertionError("DEV digest chain changed under item permutation")
    return {"status": "FIXTURE_PASS", "macro": rep_canon["macro"]}


def fixture_case_e_last_resume_parity(horizons_path: Path) -> dict:
    import numpy
    import torch

    def _reset_fixture_rng() -> None:
        random.seed(FIXTURE_SEED)
        numpy.random.seed(FIXTURE_SEED)
        torch.manual_seed(FIXTURE_SEED)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bundle_full = _synthetic_test_trust_bundle(root / "full", horizons_path)
        cfg_full = bundle_full["build_config"](max_updates=64, dev_events=[32, 64])
        rel_full, exp_full = bundle_full["mint_release"](cfg_full)
        _reset_fixture_rng()
        res_full = run_synthetic(str(rel_full), str(bundle_full["trust_path"]), exp_full, cfg_full, allow_synthetic_test_trust=True)
        last_full = resolve_pointer(bundle_full["output_root"], "LAST")

        # Resumed run uses the exact same max_updates=64 config, interrupted on step 33 after u32 commit.
        bundle_res = _synthetic_test_trust_bundle(root / "resumed", horizons_path)
        cfg_p1 = bundle_res["build_config"](max_updates=64, dev_events=[32, 64])
        rel_p1, exp_p1 = bundle_res["mint_release"](cfg_p1)
        real_opt_step = optimizer_step
        call_count = {"n": 0}

        def interrupt_at_step_33(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 33:
                raise RuntimeError("simulated interruption after u32 commit")
            return real_opt_step(*args, **kwargs)

        _reset_fixture_rng()
        with patch("stage_a_integrated.spine.optimizer_step", side_effect=interrupt_at_step_33):
            _expect_raises(
                RuntimeError,
                lambda: run_synthetic(str(rel_p1), str(bundle_res["trust_path"]), exp_p1, cfg_p1, allow_synthetic_test_trust=True),
                "simulated interruption at step 33",
            )

        commit_p1 = resolve_commit(bundle_res["output_root"])
        cfg_p2 = bundle_res["build_config"](max_updates=64, dev_events=[32, 64], resume_role="LAST")
        rel_p2, exp_p2 = bundle_res["mint_release"](
            cfg_p2,
            resume_policy=RESUME_POLICY_LAST,
            resume_from_commit_sha256=commit_p1["commit_sha256"],
        )
        res_p2 = run_synthetic(str(rel_p2), str(bundle_res["trust_path"]), exp_p2, cfg_p2, allow_synthetic_test_trust=True)
        last_p2 = resolve_pointer(bundle_res["output_root"], "LAST")

        case_e(last_full["decoded"], last_p2["decoded"])
        if abs(res_full["macros"][64] - res_p2["macros"][64]) > RESUME_MACRO_TOL:
            raise AssertionError("Case E resumed macro drifted beyond tolerance")

        cfg_best = bundle_res["build_config"](max_updates=64, dev_events=[32, 64], resume_role="BEST")
        _expect_raises(
            ValueError,
            lambda: run_synthetic(str(rel_p2), str(bundle_res["trust_path"]), exp_p2, cfg_best, allow_synthetic_test_trust=True),
            "BEST continuation must fail",
        )
    return {"status": "FIXTURE_PASS"}


def fixture_case_f_exact_milestones(horizons_path: Path) -> dict:
    # Covered cleanly by Case A (128 updates, [32, 64, 96, 128]); explicit milestone contract helper check.
    case_f([32, 64, 96, 128])
    _expect_raises(ValueError, lambda: case_f([32, 64, 95, 128]), "misaligned milestone rejected")
    return {"status": "FIXTURE_PASS"}


def fixture_c2_durability_and_rollback_faults(horizons_path: Path) -> dict:
    import stage_a_integrated.checkpoints as ckpt_mod
    import stage_a_integrated.spine as spine_mod
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bundle = _synthetic_test_trust_bundle(root / "dur", horizons_path)
        cfg = bundle["build_config"](max_updates=32, dev_events=[32])
        rel, exp = bundle["mint_release"](cfg)
        real_fsync_dir = ckpt_mod._fsync_dir

        def fail_on_root_fsync(path: Path) -> None:
            if path == bundle["output_root"] and (path / "CURRENT_COMMIT").exists():
                raise OSError("simulated dir fsync failure")
            real_fsync_dir(path)

        with patch.object(ckpt_mod, "_fsync_dir", side_effect=fail_on_root_fsync):
            _expect_raises(
                CommitDurabilityUnknownError,
                lambda: run_synthetic(str(rel), str(bundle["trust_path"]), exp, cfg, allow_synthetic_test_trust=True),
                "post-replace dir fsync failure must raise CommitDurabilityUnknownError",
            )
        recovered = recover_commit_state(bundle["output_root"])
        if not recovered["commit_durable"]:
            raise AssertionError("recover_commit_state failed to establish durability")

        # Post-commit mirror sync fault & FatalRollbackError on resume failure.
        bundle_rb = _synthetic_test_trust_bundle(root / "rb", horizons_path)
        cfg_rb = bundle_rb["build_config"](max_updates=32, dev_events=[32])
        rel_rb, exp_rb = bundle_rb["mint_release"](cfg_rb)
        with patch.object(ckpt_mod, "_sync_derived_mirrors", side_effect=OSError("mirror write fault")):
            _expect_raises(
                PostCommitError,
                lambda: run_synthetic(str(rel_rb), str(bundle_rb["trust_path"]), exp_rb, cfg_rb, allow_synthetic_test_trust=True),
                "mirror fault post-fsync must raise PostCommitError",
            )
        recovered_rb = recover_commit_state(bundle_rb["output_root"])
        if not recovered_rb["commit_durable"]:
            raise AssertionError("recover_commit_state after PostCommitError failed")
        last_ptr = resolve_pointer(bundle_rb["output_root"], "LAST")
        resolved_rb = resolve_effective_config(cfg_rb)
        model_cls, config_cls = load_class()
        model = model_cls(config_cls())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
        bad_resume_cfg = {
            "seed": FIXTURE_SEED,
            "batch": FIXTURE_BATCH,
            "expected_provenance": last_ptr["provenance"],
            "dev_events": [32, 64],
        }
        with patch.object(spine_mod, "restore_cpu_rng_snapshot", return_value=["torch_rng: simulated rollback fail"]):
            _expect_raises(
                FatalRollbackError,
                lambda: resume(b"corrupt_payload", model, opt, bad_resume_cfg, failure_root=bundle_rb["output_root"]),
                "failed RNG rollback during resume fault must raise FatalRollbackError",
            )
    return {"status": "FIXTURE_PASS"}


def fixture_c3_forward_leakage_and_dev_mutation(horizons_path: Path) -> dict:
    import torch

    rows = load_rows(horizons_path)
    keys = load_keys(horizons_path)
    digests = expected_dev_digests(rows)
    items = dev_items(rows)
    train_set = set(train_ids())
    model_cls, config_cls = load_class()
    model = model_cls(config_cls())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)

    # 1. Paired RNG/state invariance & unselected future invariance across origin (t=0) & non-origin (t>0).
    origin_item = next(it for it in items if it["current"] == it["ids_frames"][0] and it["ids_frames"][-1] > it["target"])
    non_origin_idx, non_origin_item = next(
        (idx, it) for idx, it in enumerate(items) if it["current"] > it["ids_frames"][0] and it["ids_frames"][-1] > it["target"]
    )
    for it in (origin_item, non_origin_item):
        assert_no_target_leak(it, verify_canonical_oracle=True)
        assert_causal_input_sensitivity(it)
        assert_future_unselected_invariance(it)
        verify_forward_target_invariance(model, it)

    # 2. Negative target injection must fail BEFORE any model.forward or opt.zero_grad on both train & DEV.
    for field in ("current_latent", "history_latents", "transition", "__extra_target_key__"):
        leaked = inject_target_into_forward_kwargs(non_origin_item, field=field)
        bad_items = list(items)
        bad_items[non_origin_idx] = leaked
        with patch.object(model, "forward", wraps=model.forward) as fwd_spy, patch.object(opt, "zero_grad", wraps=opt.zero_grad) as zg_spy:
            _expect_raises(
                ValueError,
                lambda: optimizer_step(model, [leaked], opt, 1.0, expected_forward_digests=[non_origin_item["forward_input_sha256"]]),
                f"train injection {field} failed to reject pre-forward",
            )
            _expect_raises(
                ValueError,
                lambda: evaluate_with_diagnostics(model, bad_items, keys, expected_digests=digests, forbidden_video_ids=train_set, require_external_expectations=True),
                f"dev injection {field} failed to reject pre-forward",
            )
            if fwd_spy.call_count != 0 or zg_spy.call_count != 0:
                raise AssertionError("target leak invoked forward or zero_grad")

    # 3. Missing external expectations, mixed mode, EvaluatorMutationError on success & exception, zero pred raw_cosine=None.
    _expect_raises(
        ValueError,
        lambda: evaluate_with_diagnostics(model, items, keys, require_external_expectations=True),
        "missing external expectations must reject",
    )
    model.train()
    model.local_direction_head.eval()
    _expect_raises(
        ValueError,
        lambda: evaluate_with_diagnostics(model, items, keys, expected_digests=digests, forbidden_video_ids=train_set, require_external_expectations=True),
        "mixed module mode must reject",
    )
    model.train()

    orig_fwd = model.forward

    def mutating_success(*args, **kwargs):
        torch.rand(1)
        return orig_fwd(*args, **kwargs)

    with patch.object(model, "forward", side_effect=mutating_success):
        _expect_raises(
            EvaluatorMutationError,
            lambda: evaluate_with_diagnostics(model, items, keys, expected_digests=digests, forbidden_video_ids=train_set, require_external_expectations=True),
            "RNG mutation on success must raise EvaluatorMutationError",
        )

    def mutating_exception(*args, **kwargs):
        torch.rand(1)
        raise RuntimeError("injected forward crash")

    with patch.object(model, "forward", side_effect=mutating_exception):
        _expect_raises(
            EvaluatorMutationError,
            lambda: evaluate_with_diagnostics(model, items, keys, expected_digests=digests, forbidden_video_ids=train_set, require_external_expectations=True),
            "RNG mutation on exception must raise EvaluatorMutationError",
        )

    def zero_delta_forward(*args, **kwargs):
        return {
            "delta_local": torch.zeros(1, 1280, dtype=torch.float32),
            "delta_anchor": torch.zeros(1, 1280, dtype=torch.float32),
        }

    with patch.object(model, "forward", side_effect=zero_delta_forward):
        rep_zero = evaluate_with_diagnostics(
            model,
            items,
            keys,
            expected_digests=digests,
            forbidden_video_ids=train_set,
            require_external_expectations=True,
        )
        if rep_zero["raw_cosine"]["mean"] is not None or rep_zero["signal_status"] != "NO_POSITIVE_DELTA_SIGNAL":
            raise AssertionError("zero-delta output must report raw_cosine=None and NO_POSITIVE_DELTA_SIGNAL")
    return {"status": "FIXTURE_PASS"}


def run_authorized_synthetic_suite(horizons_path: str | Path, *, allow_synthetic_fixture_execution: bool = False) -> dict:
    """Consolidated C4 synthetic A-F & C1-C3 test entrypoint (strictly default-deny & corpus-isolated)."""
    if allow_synthetic_fixture_execution is not True:
        raise ExecutionNotAuthorized("A-F synthetic suite requires explicit allow_synthetic_fixture_execution=True")
    hpath = Path(horizons_path).resolve()
    if "500_1_safetensors" in str(hpath) or "bothway_embed_data_min_500_1" in str(hpath):
        raise ExecutionNotAuthorized("historical 500_1 corpus path forbidden in synthetic A-F suite")

    results = {
        "auth": fixture_authorization_positive_and_adversarial(hpath),
        "case_a": fixture_case_a_best_last_and_tie(hpath),
        "case_b": fixture_case_b_nonfinite_dev(hpath),
        "case_c": fixture_case_c_corrupted_artifacts(hpath),
        "case_d": fixture_case_d_dev_order_invariance(hpath),
        "case_e": fixture_case_e_last_resume_parity(hpath),
        "case_f": fixture_case_f_exact_milestones(hpath),
        "c2_durability_rollback": fixture_c2_durability_and_rollback_faults(hpath),
        "c3_c4_leakage_dev_mutation": fixture_c3_forward_leakage_and_dev_mutation(hpath),
    }
    return {
        "suite_schema": SUITE_SCHEMA_VERSION,
        "status": "ALL_FIXTURES_PASSED",
        "historical_corpus_read": False,
        "results": results,
    }
