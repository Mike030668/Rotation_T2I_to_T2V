"""Fail-closed CLI for authorized Historical 500_1 ABI preflight, CUDA smoke, and 128-update GPU pilot. Default raises NOT_RUN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .authorization import ExecutionNotAuthorized, canonical_resolved_path
from .exposure import strict_json_loads
from .historical_abi import run_historical_abi_preflight
from .historical_runner import run_historical_cuda_smoke, run_historical_gpu_pilot

MAX_CONFIG_BYTES = 65_536


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stage_a_integrated.historical_cli",
        description=(
            "Fail-closed Stage A Historical 500_1 ABI preflight & GPU runner (requires external Owner Ed25519 "
            "trust anchor, signed release, clean Git checkout, and durable nonce ledger)."
        ),
        add_help=True,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("preflight", "cuda-smoke", "gpu-pilot"),
        help="Execution mode: preflight (read-only TRAIN/DEV), cuda-smoke (1-4 train updates), gpu-pilot (128 updates)",
    )
    parser.add_argument("--config", required=False, help="Absolute path to provisioned config JSON")
    parser.add_argument("--release", required=False, help="Absolute path to signed Owner release JSON")
    parser.add_argument("--trust-anchor-sha256", required=False, help="Expected external Owner Ed25519 public key SHA-256")
    parser.add_argument(
        "--execute-authorized-release",
        action="store_true",
        help="Explicit execution acknowledgement; still requires valid external Owner trust, revocations, release, and ledger",
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
            ("mode", args.mode),
            ("--config", args.config),
            ("--release", args.release),
            ("--trust-anchor-sha256", args.trust_anchor_sha256),
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

    expected = {
        "trust_anchor_sha256": args.trust_anchor_sha256,
        "scope": config.get("scope"),
        "device": config.get("device"),
        "budget": config.get("budget"),
        "max_updates": config.get("max_updates"),
        "output_root": config.get("output_root"),
        "revocation_path": config.get("revocation_path"),
        "nonce_ledger_path": config.get("nonce_ledger_path"),
    }
    trust_anchor_path = str(config.get("trust_anchor_path"))

    if args.mode == "preflight":
        result = run_historical_abi_preflight(
            args.release,
            trust_anchor_path,
            expected,
            config,
            allow_synthetic_test_trust=False,
        )
    elif args.mode == "cuda-smoke":
        result = run_historical_cuda_smoke(
            args.release,
            trust_anchor_path,
            expected,
            config,
            allow_synthetic_test_trust=False,
        )
    else:
        result = run_historical_gpu_pilot(
            args.release,
            trust_anchor_path,
            expected,
            config,
            allow_synthetic_test_trust=False,
        )
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
