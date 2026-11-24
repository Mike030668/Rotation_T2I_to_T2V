"""Fail-closed external Owner Ed25519 trust root, revocations, signed releases, and durable nonce ledger."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path

from . import CANDIDATE_SHA256, HORIZONS_SHA256, LOSS_SHA256, SAMPLER_SHA256
from .exposure import HEX_CHARS, canonical_json_bytes, strict_json_loads
from .plan import require_exact_int

_TOKEN = object()

TRUST_ANCHOR_SCHEMA = "owner-trust-v1"
REVOCATION_SCHEMA = "owner-revocations-v1"
RELEASE_SCHEMA = "owner-release-v1"
NONCE_LEDGER_SCHEMA = "nonce-ledger-v1"

RELEASE_DOMAIN_PREFIX = b"STCA-OWNER-RELEASE-V1\x00"
REVOCATION_DOMAIN_PREFIX = b"STCA-OWNER-REVOCATIONS-V1\x00"

OWNER_TRUST_STATUS_PENDING = "PENDING_OWNER_C4"
OWNER_TRUST_STATUS_VERIFIED = "OWNER_ED25519_VERIFIED"
OWNER_TRUST_STATUS_SYNTHETIC_TEST = "SYNTHETIC_NON_AUTHORITATIVE_TEST_TRUST"

RESUME_POLICY_FRESH = "FRESH_ONLY"
RESUME_POLICY_LAST = "RESUME_LAST_ONLY"
ALLOWED_RESUME_POLICIES = frozenset({RESUME_POLICY_FRESH, RESUME_POLICY_LAST})

NONCE_STATE_RESERVED = "RESERVED"
NONCE_STATE_CONSUMED = "CONSUMED"
NONCE_STATE_ABORTED = "ABORTED"
ALLOWED_NONCE_STATES = frozenset({NONCE_STATE_RESERVED, NONCE_STATE_CONSUMED, NONCE_STATE_ABORTED})

MAX_AUTH_FILE_BYTES = 65_536
MAX_LEDGER_BYTES = 4 * 1024 * 1024
MAX_LEDGER_ENTRIES = 10_000
MAX_REVOCATION_ITEMS = 4_096

FORBIDDEN_RELEASE_EMBEDDED_KEY_FIELDS = frozenset(
    {
        "public_key",
        "public_key_hex",
        "private_key",
        "signing_key",
        "trust_anchor",
        "verify_key",
    }
)
TRUST_ANCHOR_KEYS = frozenset(
    {"schema", "key_id", "algorithm", "public_key_hex", "authority", "non_authoritative_test_key"}
)
REVOCATION_BODY_KEYS = frozenset({"schema", "key_id", "issued_at", "revoked_key_ids", "revoked_ticket_ids", "revoked_nonces"})
REVOCATION_KEYS = frozenset((*REVOCATION_BODY_KEYS, "signature"))
RELEASE_BODY_KEYS = frozenset(
    {
        "schema",
        "allows_execution",
        "package",
        "scope",
        "key_id",
        "ticket_id",
        "nonce",
        "issued_at",
        "not_before",
        "expires",
        "device",
        "budget",
        "max_updates",
        "output_root",
        "code_commit_sha",
        "code_tree_sha",
        "effective_config_sha256",
        "candidate_sha256",
        "loss_sha256",
        "sampler_sha256",
        "data_manifest_sha256",
        "resume_policy",
        "resume_from_commit_sha256",
    }
)
RELEASE_KEYS = frozenset((*RELEASE_BODY_KEYS, "signature"))
LEDGER_KEYS = frozenset({"ledger_schema", "ledger_seq", "entries"})
LEDGER_ENTRY_KEYS = frozenset(
    {
        "ticket_id",
        "nonce",
        "release_sha256",
        "effective_config_sha256",
        "output_root",
        "resume_policy",
        "state",
        "reserved_at",
        "finalized_at",
        "outcome",
        "final_commit_sha256",
    }
)


class ExecutionNotAuthorized(RuntimeError):
    pass


class LedgerDurabilityUnknownError(ExecutionNotAuthorized):
    """Raised when nonce ledger atomic replace succeeded but directory fsync failed."""

    def __init__(self, message: str, *, nonce: str, ticket_id: str) -> None:
        super().__init__(message)
        self.nonce = nonce
        self.ticket_id = ticket_id
        self.durability_confirmed = False
        self.must_fail_stop = True


class VerifiedRelease:
    def __init__(
        self,
        receipt: dict,
        token: object,
        *,
        release_sha256: str,
        trust_anchor_sha256: str,
        trust_status: str,
        revocation_sha256: str,
    ) -> None:
        if token is not _TOKEN:
            raise ExecutionNotAuthorized("caller cannot mint a release")
        object.__setattr__(self, "_receipt", dict(receipt))
        object.__setattr__(self, "release_sha256", release_sha256)
        object.__setattr__(self, "trust_anchor_sha256", trust_anchor_sha256)
        object.__setattr__(self, "trust_status", trust_status)
        object.__setattr__(self, "revocation_sha256", revocation_sha256)

    def __setattr__(self, name: str, value: object) -> None:
        raise ExecutionNotAuthorized("VerifiedRelease is immutable")

    def get(self, name: str):
        return self._receipt[name]

    def to_dict(self) -> dict:
        return dict(self._receipt)


def _require_nonempty_str(value: object, name: str, *, max_len: int = 256) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ExecutionNotAuthorized(f"{name} must be a trimmed nonempty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ExecutionNotAuthorized(f"{name} is not valid UTF-8") from exc
    if len(value) > max_len:
        raise ExecutionNotAuthorized(f"{name} exceeds max length {max_len}")
    return value


def _require_exact_hex(value: object, name: str, *, length: int) -> str:
    text = _require_nonempty_str(value, name, max_len=length)
    if len(text) != length or any(ch not in HEX_CHARS for ch in text):
        raise ExecutionNotAuthorized(f"{name} must be {length}-char lowercase hex")
    return text


def _require_auth_int(value: object, name: str, *, minimum: int = 0) -> int:
    try:
        return require_exact_int(value, name, minimum=minimum)
    except ValueError as exc:
        raise ExecutionNotAuthorized(str(exc)) from exc


def canonical_resolved_path(raw: object, name: str) -> str:
    path_str = _require_nonempty_str(raw, name, max_len=4096)
    if "UNPROVISIONED_" in path_str:
        raise ExecutionNotAuthorized(f"{name} is unprovisioned")
    path = Path(path_str)
    if not path.is_absolute():
        raise ExecutionNotAuthorized(f"{name} must be an absolute path")
    return str(path.resolve())


def _read_bounded_auth_bytes(path: Path, *, max_bytes: int, label: str) -> bytes:
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_file():
        raise ExecutionNotAuthorized(f"{label} file is missing")
    size = path.stat().st_size
    if size < 2 or size > max_bytes:
        raise ExecutionNotAuthorized(f"{label} size out of bounds")
    raw = path.read_bytes()
    if len(raw) != size or len(raw) > max_bytes:
        raise ExecutionNotAuthorized(f"{label} read size mismatch")
    return raw


def _verify_ed25519_signature(public_bytes: bytes, message: bytes, signature: bytes, *, context: str) -> None:
    if type(public_bytes) is not bytes or len(public_bytes) != 32:
        raise ExecutionNotAuthorized(f"{context} public key must be 32 bytes")
    if type(signature) is not bytes or len(signature) != 64:
        raise ExecutionNotAuthorized(f"{context} signature must be 64 bytes")
    try:
        from nacl.exceptions import BadSignatureError
        from nacl.signing import VerifyKey
    except ImportError:
        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError as exc:
            raise ExecutionNotAuthorized(
                "Ed25519 verification requires PyNaCl or cryptography; fail closed"
            ) from exc
        try:
            Ed25519PublicKey.from_public_bytes(public_bytes).verify(signature, message)
            return
        except (InvalidSignature, ValueError, TypeError) as exc:
            raise ExecutionNotAuthorized(f"{context} Ed25519 signature verification failed") from exc
    try:
        VerifyKey(public_bytes).verify(message, signature)
    except (BadSignatureError, ValueError, TypeError) as exc:
        raise ExecutionNotAuthorized(f"{context} Ed25519 signature verification failed") from exc


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
def _ledger_lock(ledger_path: Path):
    lock_path = ledger_path.with_name(ledger_path.name + ".lock")
    if not ledger_path.parent.is_dir():
        raise ExecutionNotAuthorized("nonce ledger parent directory does not exist")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        raise ExecutionNotAuthorized("nonce ledger is locked by another process") from exc
    try:
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def load_trust_anchor(
    trust_anchor_path: str | Path,
    expected_trust_anchor_sha256: str,
    *,
    allow_synthetic_test_trust: bool = False,
) -> dict:
    """Load external Owner Ed25519 trust anchor independently of release payload."""
    expected_sha = _require_exact_hex(expected_trust_anchor_sha256, "expected.trust_anchor_sha256", length=64)
    path = Path(canonical_resolved_path(str(trust_anchor_path), "trust_anchor_path"))
    raw = _read_bounded_auth_bytes(path, max_bytes=MAX_AUTH_FILE_BYTES, label="Owner trust anchor")
    file_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        doc = strict_json_loads(raw, max_bytes=MAX_AUTH_FILE_BYTES)
    except ValueError as exc:
        raise ExecutionNotAuthorized(f"trust anchor JSON invalid: {exc}") from exc

    if type(doc) is not dict or set(doc.keys()) != TRUST_ANCHOR_KEYS:
        raise ExecutionNotAuthorized("trust anchor does not match owner-trust-v1 field set")
    if doc["schema"] != TRUST_ANCHOR_SCHEMA:
        raise ExecutionNotAuthorized("trust anchor schema mismatch")
    if doc["algorithm"] != "ed25519":
        raise ExecutionNotAuthorized("trust anchor algorithm must be ed25519")
    key_id = _require_nonempty_str(doc["key_id"], "trust_anchor.key_id", max_len=128)
    pub_hex = _require_exact_hex(doc["public_key_hex"], "trust_anchor.public_key_hex", length=64)
    public_bytes = bytes.fromhex(pub_hex)
    if hashlib.sha256(public_bytes).hexdigest() != expected_sha:
        raise ExecutionNotAuthorized("trust anchor public key digest does not match Owner pin")

    is_synthetic = doc["non_authoritative_test_key"]
    if type(is_synthetic) is not bool:
        raise ExecutionNotAuthorized("trust_anchor.non_authoritative_test_key must be exact bool")
    authority = _require_nonempty_str(doc["authority"], "trust_anchor.authority", max_len=64)
    if is_synthetic:
        if not allow_synthetic_test_trust or authority != "SYNTHETIC_FIXTURE_ONLY":
            raise ExecutionNotAuthorized("non-authoritative fixture trust anchor rejected in execution path")
        trust_status = OWNER_TRUST_STATUS_SYNTHETIC_TEST
    else:
        if authority != "OWNER_EXTERNAL":
            raise ExecutionNotAuthorized("authoritative trust anchor must declare authority OWNER_EXTERNAL")
        trust_status = OWNER_TRUST_STATUS_VERIFIED

    return {
        "schema": TRUST_ANCHOR_SCHEMA,
        "key_id": key_id,
        "public_bytes": public_bytes,
        "public_key_sha256": expected_sha,
        "file_sha256": file_sha256,
        "trust_status": trust_status,
        "non_authoritative_test_key": is_synthetic,
    }


def _validate_revocation_list(values: object, name: str) -> list[str]:
    if type(values) is not list or len(values) > MAX_REVOCATION_ITEMS:
        raise ExecutionNotAuthorized(f"revocations.{name} must be a bounded list")
    cleaned = []
    for entry in values:
        cleaned.append(_require_nonempty_str(entry, f"revocations.{name}[]", max_len=128))
    return cleaned


def load_revocations(
    revocation_path: str | Path,
    trust_anchor: dict,
    *,
    expected_revocation_sha256: str | None = None,
) -> dict:
    """Load and verify signed Owner revocation list under external trust anchor."""
    path = Path(canonical_resolved_path(str(revocation_path), "revocation_path"))
    raw = _read_bounded_auth_bytes(path, max_bytes=MAX_AUTH_FILE_BYTES, label="Owner revocations")
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_revocation_sha256 is not None:
        exp_sha = _require_exact_hex(expected_revocation_sha256, "expected.revocation_sha256", length=64)
        if file_sha256 != exp_sha:
            raise ExecutionNotAuthorized("revocations file sha256 mismatch")
    try:
        doc = strict_json_loads(raw, max_bytes=MAX_AUTH_FILE_BYTES)
    except ValueError as exc:
        raise ExecutionNotAuthorized(f"revocations JSON invalid: {exc}") from exc

    if type(doc) is not dict or set(doc.keys()) != REVOCATION_KEYS:
        raise ExecutionNotAuthorized("revocations does not match owner-revocations-v1 field set")
    if doc["schema"] != REVOCATION_SCHEMA:
        raise ExecutionNotAuthorized("revocations schema mismatch")
    if doc["key_id"] != trust_anchor["key_id"]:
        raise ExecutionNotAuthorized("revocations key_id disagrees with trust anchor")
    _require_auth_int(doc["issued_at"], "revocations.issued_at", minimum=1)
    revoked_keys = set(_validate_revocation_list(doc["revoked_key_ids"], "revoked_key_ids"))
    revoked_tickets = set(_validate_revocation_list(doc["revoked_ticket_ids"], "revoked_ticket_ids"))
    revoked_nonces = set(_validate_revocation_list(doc["revoked_nonces"], "revoked_nonces"))
    sig_hex = _require_exact_hex(doc["signature"], "revocations.signature", length=128)
    body = {key: doc[key] for key in REVOCATION_BODY_KEYS}
    _verify_ed25519_signature(
        trust_anchor["public_bytes"],
        REVOCATION_DOMAIN_PREFIX + canonical_json_bytes(body, max_bytes=MAX_AUTH_FILE_BYTES),
        bytes.fromhex(sig_hex),
        context="revocations",
    )
    if trust_anchor["key_id"] in revoked_keys:
        raise ExecutionNotAuthorized("trust anchor key_id is revoked")
    return {
        "revocation_sha256": file_sha256,
        "issued_at": doc["issued_at"],
        "revoked_key_ids": revoked_keys,
        "revoked_ticket_ids": revoked_tickets,
        "revoked_nonces": revoked_nonces,
    }


def load(
    release_path: str | Path,
    trust_anchor_path: str | Path,
    expected: dict,
    *,
    revocation_path: str | Path | None = None,
    allow_synthetic_test_trust: bool = False,
) -> VerifiedRelease:
    """Verify external Owner trust root, signed revocations, and signed single-use release."""
    if type(expected) is not dict:
        raise ExecutionNotAuthorized("expected verification parameters must be a dict")
    rev_path = revocation_path if revocation_path is not None else expected.get("revocation_path")
    if rev_path is None:
        raise ExecutionNotAuthorized("revocation_path is required for Owner release verification")

    trust_anchor = load_trust_anchor(
        trust_anchor_path,
        expected.get("trust_anchor_sha256"),
        allow_synthetic_test_trust=allow_synthetic_test_trust,
    )
    revocations = load_revocations(
        rev_path,
        trust_anchor,
        expected_revocation_sha256=expected.get("revocation_sha256"),
    )

    rel_path = Path(canonical_resolved_path(str(release_path), "release_path"))
    rel_raw = _read_bounded_auth_bytes(rel_path, max_bytes=MAX_AUTH_FILE_BYTES, label="Owner release")
    release_sha256 = hashlib.sha256(rel_raw).hexdigest()
    try:
        release = strict_json_loads(rel_raw, max_bytes=MAX_AUTH_FILE_BYTES)
    except ValueError as exc:
        raise ExecutionNotAuthorized(f"release JSON invalid: {exc}") from exc

    if type(release) is not dict:
        raise ExecutionNotAuthorized("release payload must be an object")
    if any(forbidden in release for forbidden in FORBIDDEN_RELEASE_EMBEDDED_KEY_FIELDS):
        raise ExecutionNotAuthorized("release contains forbidden embedded trust/key field")
    if set(release.keys()) != RELEASE_KEYS:
        raise ExecutionNotAuthorized("release fields do not match owner-release-v1")

    if release["schema"] != RELEASE_SCHEMA:
        raise ExecutionNotAuthorized("release schema mismatch")
    if release["allows_execution"] is not True or release["package"] != "stage_a_integrated":
        raise ExecutionNotAuthorized("release package or execution flag mismatch")
    if release["key_id"] != trust_anchor["key_id"]:
        raise ExecutionNotAuthorized("release key_id disagrees with external trust anchor")

    ticket_id = _require_nonempty_str(release["ticket_id"], "release.ticket_id", max_len=128)
    nonce = _require_nonempty_str(release["nonce"], "release.nonce", max_len=128)
    if ticket_id in revocations["revoked_ticket_ids"] or nonce in revocations["revoked_nonces"]:
        raise ExecutionNotAuthorized("release ticket or nonce has been revoked")
    used_nonces = expected.get("used_nonces")
    if used_nonces is not None:
        if type(used_nonces) not in (list, tuple, set, frozenset):
            raise ExecutionNotAuthorized("expected.used_nonces must be a sequence or set when provided")
        if nonce in used_nonces:
            raise ExecutionNotAuthorized("release nonce replayed in used_nonces")

    issued_at = _require_auth_int(release["issued_at"], "release.issued_at", minimum=1)
    not_before = _require_auth_int(release["not_before"], "release.not_before", minimum=1)
    expires = _require_auth_int(release["expires"], "release.expires", minimum=1)
    if not (issued_at <= not_before < expires):
        raise ExecutionNotAuthorized("release validity window is malformed")
    now = _require_auth_int(expected.get("now"), "expected.now", minimum=1)
    if now < not_before or now > expires:
        raise ExecutionNotAuthorized("release is outside its authorized validity window")

    exp_scope = _require_nonempty_str(expected.get("scope"), "expected.scope", max_len=64)
    exp_device = _require_nonempty_str(expected.get("device"), "expected.device", max_len=32)
    exp_budget = _require_nonempty_str(expected.get("budget"), "expected.budget", max_len=64)
    exp_max_updates = _require_auth_int(expected.get("max_updates"), "expected.max_updates", minimum=1)
    if release["scope"] != exp_scope or release["device"] != exp_device or release["budget"] != exp_budget:
        raise ExecutionNotAuthorized("release scope/device/budget mismatch")
    rel_max_updates = _require_auth_int(release["max_updates"], "release.max_updates", minimum=1)
    if rel_max_updates != exp_max_updates:
        raise ExecutionNotAuthorized("release max_updates must equal config max_updates exactly")

    rel_root = canonical_resolved_path(release["output_root"], "release.output_root")
    exp_root = canonical_resolved_path(expected.get("output_root"), "expected.output_root")
    if rel_root != exp_root:
        raise ExecutionNotAuthorized("release output_root disagrees with expected resolved output_root")

    if _require_exact_hex(release["code_commit_sha"], "release.code_commit_sha", length=40) != _require_exact_hex(
        expected.get("code_commit_sha"), "expected.code_commit_sha", length=40
    ):
        raise ExecutionNotAuthorized("release code_commit_sha mismatch")
    if _require_exact_hex(release["code_tree_sha"], "release.code_tree_sha", length=40) != _require_exact_hex(
        expected.get("code_tree_sha"), "expected.code_tree_sha", length=40
    ):
        raise ExecutionNotAuthorized("release code_tree_sha mismatch")
    if _require_exact_hex(
        release["effective_config_sha256"], "release.effective_config_sha256", length=64
    ) != _require_exact_hex(expected.get("effective_config_sha256"), "expected.effective_config_sha256", length=64):
        raise ExecutionNotAuthorized("release effective_config_sha256 mismatch")

    if release["candidate_sha256"] != CANDIDATE_SHA256:
        raise ExecutionNotAuthorized("release candidate_sha256 mismatch")
    if release["loss_sha256"] != LOSS_SHA256:
        raise ExecutionNotAuthorized("release loss_sha256 mismatch")
    if release["sampler_sha256"] != SAMPLER_SHA256:
        raise ExecutionNotAuthorized("release sampler_sha256 mismatch")
    if release["data_manifest_sha256"] != HORIZONS_SHA256:
        raise ExecutionNotAuthorized("release data_manifest_sha256 mismatch")

    resume_policy = release["resume_policy"]
    if resume_policy not in ALLOWED_RESUME_POLICIES:
        raise ExecutionNotAuthorized("release resume_policy unsupported")
    exp_resume_policy = expected.get("resume_policy", RESUME_POLICY_FRESH)
    if resume_policy != exp_resume_policy:
        raise ExecutionNotAuthorized("release resume_policy disagrees with requested run mode")
    if resume_policy == RESUME_POLICY_FRESH:
        if release["resume_from_commit_sha256"] is not None:
            raise ExecutionNotAuthorized("FRESH_ONLY release cannot bind resume_from_commit_sha256")
        if expected.get("resume_from_commit_sha256") is not None:
            raise ExecutionNotAuthorized("expected resume_from_commit_sha256 present under FRESH_ONLY")
    else:
        rel_commit = _require_exact_hex(release["resume_from_commit_sha256"], "release.resume_from_commit_sha256", length=64)
        exp_commit = _require_exact_hex(expected.get("resume_from_commit_sha256"), "expected.resume_from_commit_sha256", length=64)
        if rel_commit != exp_commit:
            raise ExecutionNotAuthorized("release resume_from_commit_sha256 disagrees with authoritative LAST commit")

    sig_hex = _require_exact_hex(release["signature"], "release.signature", length=128)
    body = {key: release[key] for key in RELEASE_BODY_KEYS}
    _verify_ed25519_signature(
        trust_anchor["public_bytes"],
        RELEASE_DOMAIN_PREFIX + canonical_json_bytes(body, max_bytes=MAX_AUTH_FILE_BYTES),
        bytes.fromhex(sig_hex),
        context="release",
    )

    normalized = dict(release)
    normalized["output_root"] = rel_root
    return VerifiedRelease(
        normalized,
        _TOKEN,
        release_sha256=release_sha256,
        trust_anchor_sha256=trust_anchor["public_key_sha256"],
        trust_status=trust_anchor["trust_status"],
        revocation_sha256=revocations["revocation_sha256"],
    )


def _parse_nonce_ledger(raw: bytes) -> dict:
    try:
        doc = strict_json_loads(raw, max_bytes=MAX_LEDGER_BYTES)
    except ValueError as exc:
        raise ExecutionNotAuthorized(f"nonce ledger JSON invalid: {exc}") from exc
    if type(doc) is not dict or set(doc.keys()) != LEDGER_KEYS:
        raise ExecutionNotAuthorized("nonce ledger does not match nonce-ledger-v1 keys")
    if doc["ledger_schema"] != NONCE_LEDGER_SCHEMA:
        raise ExecutionNotAuthorized("nonce ledger schema mismatch")
    _require_auth_int(doc["ledger_seq"], "nonce_ledger.ledger_seq", minimum=0)
    entries = doc["entries"]
    if type(entries) is not dict or len(entries) > MAX_LEDGER_ENTRIES:
        raise ExecutionNotAuthorized("nonce ledger entries invalid or oversized")
    for nonce, entry in entries.items():
        _require_nonempty_str(nonce, "nonce_ledger.key", max_len=128)
        if type(entry) is not dict or set(entry.keys()) != LEDGER_ENTRY_KEYS:
            raise ExecutionNotAuthorized("nonce ledger entry keys invalid")
        if entry["nonce"] != nonce or entry["state"] not in ALLOWED_NONCE_STATES:
            raise ExecutionNotAuthorized("nonce ledger entry state invalid")
        _require_nonempty_str(entry["ticket_id"], "nonce_ledger.ticket_id", max_len=128)
        _require_exact_hex(entry["release_sha256"], "nonce_ledger.release_sha256", length=64)
        _require_exact_hex(entry["effective_config_sha256"], "nonce_ledger.effective_config_sha256", length=64)
        canonical_resolved_path(entry["output_root"], "nonce_ledger.output_root")
        if entry["resume_policy"] not in ALLOWED_RESUME_POLICIES:
            raise ExecutionNotAuthorized("nonce ledger entry resume_policy invalid")
        _require_auth_int(entry["reserved_at"], "nonce_ledger.reserved_at", minimum=1)
        if entry["finalized_at"] is not None:
            _require_auth_int(entry["finalized_at"], "nonce_ledger.finalized_at", minimum=1)
        if entry["final_commit_sha256"] is not None:
            _require_exact_hex(entry["final_commit_sha256"], "nonce_ledger.final_commit_sha256", length=64)
    return doc


def _write_ledger_atomic(ledger_path: Path, ledger: dict, *, nonce: str, ticket_id: str) -> bytes:
    payload = canonical_json_bytes(ledger, max_bytes=MAX_LEDGER_BYTES)
    partial = ledger_path.with_name(f".{ledger_path.name}.partial")
    partial.write_bytes(payload)
    _fsync_file(partial)
    if partial.read_bytes() != payload:
        raise ExecutionNotAuthorized("nonce ledger staged readback mismatch")
    os.replace(partial, ledger_path)
    try:
        _fsync_dir(ledger_path.parent)
    except Exception as exc:
        raise LedgerDurabilityUnknownError(
            f"nonce ledger {nonce} replaced but parent directory fsync failed: {exc}",
            nonce=nonce,
            ticket_id=ticket_id,
        ) from exc
    if ledger_path.read_bytes() != payload:
        raise ExecutionNotAuthorized("nonce ledger durable readback mismatch")
    return payload


def init_empty_nonce_ledger(ledger_path: str | Path) -> Path:
    """Initialize an empty durable nonce ledger without ever overwriting an existing ledger."""
    path = Path(canonical_resolved_path(str(ledger_path), "nonce_ledger_path"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ledger_lock(path):
        if path.exists():
            raise ExecutionNotAuthorized("refusing to overwrite existing nonce ledger")
        initial = {"ledger_schema": NONCE_LEDGER_SCHEMA, "ledger_seq": 0, "entries": {}}
        _write_ledger_atomic(path, initial, nonce="__init__", ticket_id="__init__")
    return path


def read_nonce_ledger(ledger_path: str | Path) -> dict:
    path = Path(canonical_resolved_path(str(ledger_path), "nonce_ledger_path"))
    raw = _read_bounded_auth_bytes(path, max_bytes=MAX_LEDGER_BYTES, label="Nonce ledger")
    return _parse_nonce_ledger(raw)


def reserve_release_nonce(ledger_path: str | Path, release: VerifiedRelease, *, now: int) -> dict:
    """Atomically reserve single-use release nonce & ticket_id before any state mutation or step."""
    if not isinstance(release, VerifiedRelease):
        raise ExecutionNotAuthorized("reserve_release_nonce requires VerifiedRelease")
    now_val = _require_auth_int(now, "now", minimum=1)
    path = Path(canonical_resolved_path(str(ledger_path), "nonce_ledger_path"))
    if not path.is_file():
        raise ExecutionNotAuthorized("nonce ledger file is absent; refusing replay-unsafe execution")

    nonce = release.get("nonce")
    ticket_id = release.get("ticket_id")
    with _ledger_lock(path):
        raw = _read_bounded_auth_bytes(path, max_bytes=MAX_LEDGER_BYTES, label="Nonce ledger")
        ledger = _parse_nonce_ledger(raw)
        entries = ledger["entries"]
        if nonce in entries:
            raise ExecutionNotAuthorized(f"release nonce {nonce} already recorded in durable ledger")
        if any(entry["ticket_id"] == ticket_id for entry in entries.values()):
            raise ExecutionNotAuthorized(f"release ticket_id {ticket_id} already recorded in durable ledger")
        if len(entries) >= MAX_LEDGER_ENTRIES:
            raise ExecutionNotAuthorized("nonce ledger capacity exceeded")

        entry = {
            "ticket_id": ticket_id,
            "nonce": nonce,
            "release_sha256": release.release_sha256,
            "effective_config_sha256": release.get("effective_config_sha256"),
            "output_root": release.get("output_root"),
            "resume_policy": release.get("resume_policy"),
            "state": NONCE_STATE_RESERVED,
            "reserved_at": now_val,
            "finalized_at": None,
            "outcome": None,
            "final_commit_sha256": None,
        }
        updated = {
            "ledger_schema": NONCE_LEDGER_SCHEMA,
            "ledger_seq": int(ledger["ledger_seq"]) + 1,
            "entries": {**entries, nonce: entry},
        }
        _write_ledger_atomic(path, updated, nonce=nonce, ticket_id=ticket_id)
        return dict(entry)


def finalize_release_nonce(
    ledger_path: str | Path,
    release: VerifiedRelease,
    *,
    outcome: str,
    now: int,
    final_commit_sha256: str | None = None,
) -> dict:
    """Atomically finalize a reserved single-use nonce to CONSUMED or ABORTED."""
    if not isinstance(release, VerifiedRelease):
        raise ExecutionNotAuthorized("finalize_release_nonce requires VerifiedRelease")
    if outcome not in {"COMPLETED", "FAILED"}:
        raise ExecutionNotAuthorized("finalize_release_nonce outcome must be COMPLETED or FAILED")
    now_val = _require_auth_int(now, "now", minimum=1)
    commit_hex = (
        None
        if final_commit_sha256 is None
        else _require_exact_hex(final_commit_sha256, "final_commit_sha256", length=64)
    )
    path = Path(canonical_resolved_path(str(ledger_path), "nonce_ledger_path"))
    nonce = release.get("nonce")
    ticket_id = release.get("ticket_id")

    with _ledger_lock(path):
        raw = _read_bounded_auth_bytes(path, max_bytes=MAX_LEDGER_BYTES, label="Nonce ledger")
        ledger = _parse_nonce_ledger(raw)
        entries = dict(ledger["entries"])
        existing = entries.get(nonce)
        if existing is None:
            raise ExecutionNotAuthorized("cannot finalize unreserved nonce")
        if existing["ticket_id"] != ticket_id or existing["release_sha256"] != release.release_sha256:
            raise ExecutionNotAuthorized("nonce ledger reservation mismatch")
        if existing["state"] != NONCE_STATE_RESERVED:
            raise ExecutionNotAuthorized("nonce ledger entry is already finalized")

        final_state = NONCE_STATE_CONSUMED if outcome == "COMPLETED" else NONCE_STATE_ABORTED
        updated_entry = {
            **existing,
            "state": final_state,
            "finalized_at": now_val,
            "outcome": outcome,
            "final_commit_sha256": commit_hex,
        }
        entries[nonce] = updated_entry
        updated = {
            "ledger_schema": NONCE_LEDGER_SCHEMA,
            "ledger_seq": int(ledger["ledger_seq"]) + 1,
            "entries": entries,
        }
        _write_ledger_atomic(path, updated, nonce=nonce, ticket_id=ticket_id)
        return dict(updated_entry)
