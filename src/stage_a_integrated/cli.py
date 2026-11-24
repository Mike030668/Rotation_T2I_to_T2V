"""Fail-closed CLI for authorized Stage A synthetic execution. Default invocation raises."""

from __future__ import annotations

import argparse
from pathlib import Path

from .authorization import ExecutionNotAuthorized, canonical_resolved_path
from .exposure import strict_json_loads
from .spine import run_synthetic

MAX_CONFIG_BYTES = 65_536


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stage_a_integrated",
        description="Fail-closed Stage A synthetic runner (requires external Owner Ed25519 trust + signed release).",
        add_help=True,
    )
    parser.add_argument("--config", required=False, help="Absolute path to provisioned stage-a-config-v1 JSON")
    parser.add_argument("--release", required=False, help="Absolute path to signed owner-release-v1 JSON")
    parser.add_argument("--trust-anchor-sha256", required=False, help="Expected external Owner Ed25519 public key SHA-256")
    parser.add_argument("--code-commit", required=False, help="Expected 40-char git commit SHA bound by Owner release")
    parser.add_argument("--code-tree", required=False, help="Expected 40-char git tree SHA bound by Owner release")
    parser.add_argument("--now", required=False, type=int, help="Exact unix timestamp for release validity check")
    parser.add_argument("--resume-last", action="store_true", help="Resume canonically from committed LAST pointer")
    parser.add_argument(
        "--execute-authorized-release",
        action="store_true",
        help="Explicit acknowledgement flag; still requires valid external Owner trust, revocations, release, and ledger",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.execute_authorized_release:
        raise ExecutionNotAuthorized("NOT_RUN")
    missing = [
        flag
        for flag, val in (
            ("--config", args.config),
            ("--release", args.release),
            ("--trust-anchor-sha256", args.trust_anchor_sha256),
            ("--code-commit", args.code_commit),
            ("--code-tree", args.code_tree),
            ("--now", args.now),
        )
        if val is None
    ]
    if missing:
        raise ExecutionNotAuthorized(f"missing required authorization arguments: {', '.join(missing)}")

    config_path = Path(canonical_resolved_path(args.config, "--config"))
    if not config_path.is_file():
        raise ExecutionNotAuthorized("config file does not exist")
    raw = config_path.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise ExecutionNotAuthorized("config file exceeds bounded limit")
    config = strict_json_loads(raw, max_bytes=MAX_CONFIG_BYTES)
    if type(config) is not dict:
        raise ExecutionNotAuthorized("config must decode to a JSON object")
    if args.resume_last:
        config = dict(config)
        config["resume_role"] = "LAST"

    expected = {
        "trust_anchor_sha256": args.trust_anchor_sha256,
        "scope": config.get("scope"),
        "device": config.get("device"),
        "budget": config.get("budget"),
        "max_updates": config.get("max_updates"),
        "output_root": config.get("output_root"),
        "revocation_path": config.get("revocation_path"),
        "nonce_ledger_path": config.get("nonce_ledger_path"),
        "code_commit_sha": args.code_commit,
        "code_tree_sha": args.code_tree,
        "now": args.now,
    }
    return run_synthetic(
        args.release,
        str(config.get("trust_anchor_path")),
        expected,
        config,
        allow_synthetic_test_trust=False,
    )
