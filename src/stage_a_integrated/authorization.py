"""Fail-closed external release. This package contains no signing key."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_TOKEN = object()


class ExecutionNotAuthorized(RuntimeError):
    pass


class VerifiedRelease:
    def __init__(self, receipt: dict, token: object):
        if token is not _TOKEN:
            raise ExecutionNotAuthorized("caller cannot mint a release")
        object.__setattr__(self, "_receipt", dict(receipt))

    def get(self, name: str):
        return self._receipt[name]


def load(release_path: str, trust_anchor_path: str, expected: dict) -> VerifiedRelease:
    if not Path(trust_anchor_path).is_file():
        raise ExecutionNotAuthorized("Owner trust anchor is absent")
    header = Path(trust_anchor_path).read_text(encoding="utf-8").splitlines()[0]
    if not header.startswith("ed25519 "):
        raise ExecutionNotAuthorized("trust anchor is not ed25519")
    public = bytes.fromhex(header.split(" ", 1)[1])
    if hashlib.sha256(public).hexdigest() != expected["trust_anchor_sha256"]:
        raise ExecutionNotAuthorized("trust anchor is not the Owner pin")
    release = json.loads(Path(release_path).read_text(encoding="utf-8"))
    required = ("allows_execution", "package", "scope", "device", "budget", "max_updates", "output_root", "nonce", "expires", "signature")
    if any(key not in release for key in required):
        raise ExecutionNotAuthorized("release receipt is incomplete")
    if release["allows_execution"] is not True or release["scope"] != expected["scope"] or release["device"] != expected["device"]:
        raise ExecutionNotAuthorized("release scope mismatch")
    if release["expires"] < expected["now"] or release["nonce"] in expected.get("used_nonces", []):
        raise ExecutionNotAuthorized("release expired or replayed")
    body = {key: release[key] for key in sorted(release) if key != "signature"}
    from nacl.signing import VerifyKey

    VerifyKey(public).verify(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8"), bytes.fromhex(release["signature"]))
    return VerifiedRelease(release, _TOKEN)
