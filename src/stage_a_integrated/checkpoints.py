"""Bounded stca-v2 checkpoints, semantic staging, external prov-v1, and atomic BEST/LAST commits."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import io
import math
import os
from pathlib import Path

from . import CANDIDATE_SHA256, HORIZONS_SHA256, LOSS_SHA256, SAMPLER_SHA256
from .authorization import (
    ALLOWED_RESUME_POLICIES,
    OWNER_TRUST_STATUS_SYNTHETIC_TEST,
    OWNER_TRUST_STATUS_VERIFIED,
)
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
from .plan import (
    DEV_TEMPORAL_AXIS_SEMANTICS,
    REMAINDER_POLICY,
    SYNTHETIC_SOURCE_ID,
    SYNTHETIC_TRAIN_IDS_SHA256,
    require_exact_int,
)

MAGIC = b"STCA"
CHECKPOINT_SCHEMA_VERSION = "stca-v2"
MANIFEST_SCHEMA_VERSION = "ckpt-manifest-v2"
PROVENANCE_SCHEMA_VERSION = "prov-v2"
POINTER_SCHEMA_VERSION = "ptr-v1"
COMMIT_SCHEMA_VERSION = "commit-v2"
DEV_REPORT_SCHEMA_VERSION = "dev-report-v1"
DEV_METRIC_ID = "dev_delta_huber_macro"
DEV_METRIC_VERSION = "stage-a-dev-v1"
DEV_METRIC_STATUS = "DS_FROZEN_STAGE_A_V1"
DEV_METRIC_BETA = 0.05
DEV_METRIC_LOCAL_WEIGHT = 0.5
DEV_METRIC_ANCHOR_WEIGHT = 0.5
DEV_METRIC_AGGREGATION = "equal_video_macro_50_over_147_pairs"
DEV_METRIC_DIRECTION = "minimize"
DEV_METRIC_TIE_POLICY = "earliest_strict_improvement_eps_0"
DEV_METRIC_EPSILON = 0.0
DEV_DIAGNOSTICS_POLICY = "mandatory_no_motion_norms_cosine_magnitude_no_aux_selection"
DEV_H2_POLICY = "secondary_contrast_deferred_not_run"
DEV_EXPECTED_VIDEOS = 50
DEV_EXPECTED_PAIRS = 147
DEV_HORIZON_NAMES = ("near", "mid", "far")
DEV_EXPECTED_HORIZON_COUNTS = {"near": 50, "mid": 47, "far": 50}
DEV_PREDECLARED_ABSENT_MID_VIDEOS = frozenset({"28060", "37577", "17858"})
CONTROL_PLANE_FIXTURE_LABEL = "CONTROL_PLANE_INJECTED_MACRO_FIXTURE"
DEV_REPORT_RECOMPUTE_TOL = 1e-12
FAILURE_RECEIPT_SCHEMA_VERSION = "fail-receipt-v1"
OWNER_TRUST_ROOT_STATUS = "PENDING_OWNER_C4"
ALLOWED_ACTIVE_TRUST_STATUSES = frozenset({OWNER_TRUST_STATUS_VERIFIED, OWNER_TRUST_STATUS_SYNTHETIC_TEST})
RNG_SCOPE = "cpu_only"

COMMIT_STATE_PRE_COMMIT_FAILURE = "PRE_COMMIT_FAILURE"
COMMIT_STATE_VISIBLE_DURABILITY_UNKNOWN = "COMMIT_VISIBLE_DURABILITY_UNKNOWN"
COMMIT_STATE_DURABLE = "COMMIT_DURABLE"

EXPECTED_ADAMW_LR = 1e-4
EXPECTED_ADAMW_WEIGHT_DECAY = 1e-4
EXPECTED_ADAMW_BETAS = (0.9, 0.999)
EXPECTED_ADAMW_EPS = 1e-8
EXPECTED_ADAMW_AMSGRAD = False

SECTIONS = ("model", "adamw", "python_rng", "numpy_rng", "torch_rng", "sampler", "exposure")

MAX_HEADER_BYTES = 1 * 1024 * 1024
MAX_MODEL_SECTION_BYTES = 128 * 1024 * 1024
MAX_ADAMW_SECTION_BYTES = 128 * 1024 * 1024
MAX_RNG_SECTION_BYTES = 65536
MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024
MAX_POINTER_BYTES = 16384
MAX_MANIFEST_BYTES = 65536
MAX_COMMIT_BYTES = 131072
MAX_DEV_REPORT_BYTES = 1 * 1024 * 1024
MAX_BEST_HISTORY = 1024
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

PROVENANCE_KEYS = frozenset(
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
        "rng_scope",
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
        "owner_trust_anchor_sha256",
        "owner_trust_root_status",
        "release_created",
    }
)


class CommitDurabilityUnknownError(RuntimeError):
    """Raised when CURRENT_COMMIT is visible via os.replace but directory/commit fsync failed."""

    def __init__(self, message: str, *, generation_id: str, commit_sha256: str) -> None:
        super().__init__(message)
        self.generation_id = generation_id
        self.commit_sha256 = commit_sha256
        self.commit_state = COMMIT_STATE_VISIBLE_DURABILITY_UNKNOWN
        self.commit_visible = True
        self.commit_durable = False
        self.durability_confirmed = False


class PostCommitError(RuntimeError):
    """Raised after CURRENT_COMMIT durability is confirmed via fsync, during readback or mirror sync."""

    def __init__(
        self,
        message: str,
        *,
        generation_id: str,
        commit_sha256: str,
        phase: str = "mirror_sync",
    ) -> None:
        super().__init__(message)
        self.generation_id = generation_id
        self.commit_sha256 = commit_sha256
        self.phase = phase
        self.commit_state = COMMIT_STATE_DURABLE
        self.commit_visible = True
        self.commit_durable = True
        self.durability_confirmed = True


class FatalRollbackError(RuntimeError):
    """Raised when state/RNG rollback itself fails; process state is tainted and must terminate."""

    def __init__(self, message: str, *, original_error: str, rollback_errors: list[str]) -> None:
        super().__init__(message)
        self.original_error = original_error
        self.rollback_errors = list(rollback_errors)
        self.fatal = True
        self.must_terminate_process = True


def _is_hex64(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(ch in HEX_CHARS for ch in value)


def _require_hex64(value: object, name: str) -> str:
    if not _is_hex64(value):
        raise ValueError(f"{name} must be a 64-char lowercase hex sha256")
    return str(value)


def _require_finite_number(value: object, name: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite int or float")
    return float(value)


def _reject_root_if_scored(root: Path) -> Path:
    if not isinstance(root, Path):
        root = Path(root)
    if "_scored_event_pkg" in str(root):
        raise ValueError("checkpoint root rejected")
    return root


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


def validate_event_and_best_history(
    header: dict,
    *,
    expected_dev_events: list[int] | None = None,
) -> dict:
    """Validate event schedule prefix, finite macros, and strict-improvement best_history."""
    if type(header) is not dict:
        raise ValueError("checkpoint header must be a dict")
    completed = require_exact_int(header.get("completed"), "header.completed", minimum=0)
    events = header.get("events")
    macros = header.get("macros")
    if type(events) is not list or type(macros) is not dict:
        raise ValueError("checkpoint header missing events or macros")
    if events != sorted(set(events)) or any(type(event) is not int or event < 1 for event in events):
        raise ValueError("event journal is not a strictly increasing positive int sequence")
    if any(event > completed for event in events):
        raise ValueError("event is after completed step")
    if any(type(step) is not str or not step.isdigit() or str(int(step)) != step for step in macros):
        raise ValueError("macro keys are not canonical decimal strings")
    if set(events) != {int(step) for step in macros}:
        raise ValueError("macro keys do not match event ids")
    for value in macros.values():
        if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("nonfinite restored macro")
    numeric = {int(step): float(value) for step, value in macros.items()}

    if expected_dev_events is not None:
        if (
            type(expected_dev_events) is not list
            or any(type(event) is not int or event < 1 for event in expected_dev_events)
            or expected_dev_events != sorted(set(expected_dev_events))
        ):
            raise ValueError("expected_dev_events must be a strictly increasing list of positive ints")
        expected_prefix = [event for event in expected_dev_events if event <= completed]
        if events != expected_prefix:
            raise ValueError("checkpoint events do not match external dev_events prefix through completed")

    best_step = header.get("best_step")
    best_macro = header.get("best_macro")
    macro = header.get("macro")
    best_history = header.get("best_history")
    if events and best_step is None:
        raise ValueError("best is missing while events exist")
    if not events:
        if best_step is not None or best_macro is not None or macro is not None:
            raise ValueError("best/macro exists while events are empty")
        if type(best_history) is not list or best_history:
            raise ValueError("best_history must be empty list while events are empty")
        return header

    if type(best_step) is not int or isinstance(best_step, bool):
        raise ValueError("best step or macro has invalid type")
    try:
        best_macro_val = _require_finite_number(best_macro, "header.best_macro")
    except ValueError as exc:
        raise ValueError("best step or macro has invalid type") from exc
    macro_val = _require_finite_number(macro, "header.macro")
    if completed == events[-1] and macro_val != numeric[events[-1]]:
        raise ValueError("header.macro disagrees with latest event in macros")

    expected_best_history: list[dict] = []
    running_best_macro: float | None = None
    for step in events:
        step_macro = numeric[step]
        if running_best_macro is None or step_macro < running_best_macro:
            expected_best_history.append({"step": step, "macro": step_macro})
            running_best_macro = step_macro

    if best_step != expected_best_history[-1]["step"] or best_macro_val != expected_best_history[-1]["macro"]:
        raise ValueError("best is not the earliest finite minimum")
    if type(best_history) is not list or not best_history or len(best_history) > len(events):
        raise ValueError("best_history must be a non-empty bounded list when events exist")

    normalized_history: list[dict] = []
    for entry in best_history:
        if type(entry) is not dict or set(entry.keys()) != {"step", "macro"}:
            raise ValueError("best_history entry malformed")
        h_step = entry["step"]
        if type(h_step) is not int or isinstance(h_step, bool) or h_step not in numeric:
            raise ValueError("best_history step invalid or not in events")
        h_val = _require_finite_number(entry["macro"], "best_history.macro")
        normalized_history.append({"step": h_step, "macro": h_val})
    if normalized_history != expected_best_history:
        raise ValueError("best_history does not match strict-improvement reconstruction from macros")
    return header


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


@contextmanager
def _single_writer_lock(root: Path):
    root = _reject_root_if_scored(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".writer.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        raise RuntimeError("checkpoint root is locked by another writer") from exc
    try:
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def verify_dev_report_consistency(
    report: object,
    *,
    expected_macro: float | None = None,
    require_scientific_metric: bool = False,
) -> dict:
    """Verify frozen `stage-a-dev-v1` schema, recompute aggregates/diagnostics, and check fixture tags."""
    if type(report) is not dict:
        raise ValueError("dev_report must be a dict")
    required = {
        "dev_report_schema",
        "metric_name",
        "metric_version",
        "metric_status",
        "beta",
        "local_weight",
        "anchor_weight",
        "aggregation",
        "direction",
        "tie_policy",
        "epsilon",
        "diagnostics_policy",
        "h2_policy",
        "temporal_axis_semantics",
        "horizons_sha256",
        "macro",
        "no_motion_macro",
        "delta_vs_no_motion",
        "beats_no_motion",
        "signal_status",
        "video_count",
        "pair_count",
        "predeclared_absent_mid_videos",
        "per_video_macro",
        "per_video_no_motion_macro",
        "per_video_delta_vs_no_motion",
        "per_video_counts",
        "per_horizon_macro",
        "per_horizon_no_motion_macro",
        "per_horizon_delta_vs_no_motion",
        "per_horizon_counts",
        "directional_cosine_loss",
        "raw_cosine",
        "magnitude_log_ratio_loss",
        "norms_and_ratios",
        "dev_forward_digests_sha256",
    }
    if not required.issubset(report.keys()):
        raise ValueError("dev_report missing required frozen metric fields")

    if "control_plane_fixture" in report:
        if type(report["control_plane_fixture"]) is not bool:
            raise ValueError("dev_report.control_plane_fixture must be exact bool")
        is_cp_fixture = report["control_plane_fixture"]
    else:
        is_cp_fixture = False
    if "scientific_metric_valid" in report:
        if type(report["scientific_metric_valid"]) is not bool:
            raise ValueError("dev_report.scientific_metric_valid must be exact bool")
        sci_valid = report["scientific_metric_valid"]
    else:
        sci_valid = not is_cp_fixture
    if is_cp_fixture:
        if sci_valid is not False:
            raise ValueError("control-plane injected macro fixture must set scientific_metric_valid=False")
        if report.get("control_plane_fixture_label") != CONTROL_PLANE_FIXTURE_LABEL:
            raise ValueError("control-plane injected macro fixture missing required label")
    else:
        if sci_valid is not True:
            raise ValueError("non-fixture dev_report must set scientific_metric_valid=True")
    if require_scientific_metric and (is_cp_fixture or not sci_valid):
        raise ValueError("control-plane injected macro fixture cannot satisfy scientific metric gate")

    if report["dev_report_schema"] != DEV_REPORT_SCHEMA_VERSION:
        raise ValueError("dev_report schema mismatch")
    if (
        report["metric_name"] != DEV_METRIC_ID
        or report["metric_version"] != DEV_METRIC_VERSION
        or report["metric_status"] != DEV_METRIC_STATUS
    ):
        raise ValueError("dev_report frozen metric identity mismatch")
    if _require_finite_number(report["beta"], "dev_report.beta") != DEV_METRIC_BETA:
        raise ValueError("dev_report.beta disagrees with frozen 0.05")
    if _require_finite_number(report["local_weight"], "dev_report.local_weight") != DEV_METRIC_LOCAL_WEIGHT:
        raise ValueError("dev_report.local_weight disagrees with frozen 0.5")
    if _require_finite_number(report["anchor_weight"], "dev_report.anchor_weight") != DEV_METRIC_ANCHOR_WEIGHT:
        raise ValueError("dev_report.anchor_weight disagrees with frozen 0.5")
    if report["aggregation"] != DEV_METRIC_AGGREGATION:
        raise ValueError("dev_report.aggregation mismatch")
    if report["direction"] != DEV_METRIC_DIRECTION:
        raise ValueError("dev_report.direction mismatch")
    if report["tie_policy"] != DEV_METRIC_TIE_POLICY:
        raise ValueError("dev_report.tie_policy mismatch")
    if _require_finite_number(report["epsilon"], "dev_report.epsilon") != DEV_METRIC_EPSILON:
        raise ValueError("dev_report.epsilon mismatch")
    if report["diagnostics_policy"] != DEV_DIAGNOSTICS_POLICY:
        raise ValueError("dev_report.diagnostics_policy mismatch")
    if report["h2_policy"] != DEV_H2_POLICY:
        raise ValueError("dev_report.h2_policy mismatch")
    if report["temporal_axis_semantics"] != DEV_TEMPORAL_AXIS_SEMANTICS:
        raise ValueError("dev_report.temporal_axis_semantics mismatch")
    if report["horizons_sha256"] != HORIZONS_SHA256:
        raise ValueError("dev_report.horizons_sha256 mismatch")
    _require_hex64(report["dev_forward_digests_sha256"], "dev_report.dev_forward_digests_sha256")

    video_count = require_exact_int(report["video_count"], "dev_report.video_count", minimum=DEV_EXPECTED_VIDEOS, maximum=DEV_EXPECTED_VIDEOS)
    pair_count = require_exact_int(report["pair_count"], "dev_report.pair_count", minimum=DEV_EXPECTED_PAIRS, maximum=DEV_EXPECTED_PAIRS)
    if report["predeclared_absent_mid_videos"] != sorted(DEV_PREDECLARED_ABSENT_MID_VIDEOS):
        raise ValueError("dev_report.predeclared_absent_mid_videos mismatch")

    pvm = report["per_video_macro"]
    pvnm = report["per_video_no_motion_macro"]
    pvd = report["per_video_delta_vs_no_motion"]
    pvc = report["per_video_counts"]
    for name, container in (
        ("per_video_macro", pvm),
        ("per_video_no_motion_macro", pvnm),
        ("per_video_delta_vs_no_motion", pvd),
        ("per_video_counts", pvc),
    ):
        if type(container) is not dict or len(container) != video_count:
            raise ValueError(f"dev_report.{name} must cover exact {video_count} videos")
    if not (set(pvm.keys()) == set(pvnm.keys()) == set(pvd.keys()) == set(pvc.keys())):
        raise ValueError("dev_report per-video maps have mismatched video keys")
    if not DEV_PREDECLARED_ABSENT_MID_VIDEOS.issubset(pvm.keys()):
        raise ValueError("dev_report missing predeclared absent-mid videos")

    total_pairs_from_videos = 0
    per_video_sum = 0.0
    per_video_nm_sum = 0.0
    for vid in sorted(pvm.keys()):
        if type(vid) is not str or not vid:
            raise ValueError("dev_report per_video key must be nonempty str")
        expected_vcount = 2 if vid in DEV_PREDECLARED_ABSENT_MID_VIDEOS else 3
        vcount = require_exact_int(pvc[vid], f"per_video_counts.{vid}", minimum=expected_vcount, maximum=expected_vcount)
        total_pairs_from_videos += vcount
        vm = _require_finite_number(pvm[vid], f"per_video_macro.{vid}")
        vnm = _require_finite_number(pvnm[vid], f"per_video_no_motion_macro.{vid}")
        vd = _require_finite_number(pvd[vid], f"per_video_delta_vs_no_motion.{vid}")
        if vm < 0.0 or vnm < 0.0:
            raise ValueError(f"negative Smooth L1 loss for video {vid}")
        if abs((vm - vnm) - vd) > DEV_REPORT_RECOMPUTE_TOL:
            raise ValueError(f"per_video_delta_vs_no_motion inconsistency for {vid}")
        per_video_sum += vm
        per_video_nm_sum += vnm
    if total_pairs_from_videos != pair_count:
        raise ValueError("sum(per_video_counts) does not equal 147")

    recomputed_macro = per_video_sum / float(video_count)
    recomputed_nm = per_video_nm_sum / float(video_count)
    recomputed_delta = recomputed_macro - recomputed_nm
    macro_val = _require_finite_number(report["macro"], "dev_report.macro")
    nm_val = _require_finite_number(report["no_motion_macro"], "dev_report.no_motion_macro")
    delta_val = _require_finite_number(report["delta_vs_no_motion"], "dev_report.delta_vs_no_motion")
    if abs(macro_val - recomputed_macro) > DEV_REPORT_RECOMPUTE_TOL:
        raise ValueError("dev_report.macro disagrees with recomputed mean(per_video_macro)")
    if abs(nm_val - recomputed_nm) > DEV_REPORT_RECOMPUTE_TOL:
        raise ValueError("dev_report.no_motion_macro disagrees with recomputed mean(per_video_no_motion_macro)")
    if abs(delta_val - (macro_val - nm_val)) > DEV_REPORT_RECOMPUTE_TOL:
        raise ValueError("dev_report.delta_vs_no_motion disagrees with macro - no_motion_macro")
    if expected_macro is not None and macro_val != float(expected_macro):
        raise ValueError("dev_report.macro disagrees with checkpoint header/commit macro")
    if report["beats_no_motion"] is not (macro_val < nm_val):
        raise ValueError("dev_report.beats_no_motion disagrees with macro vs no_motion_macro")
    expected_signal = "STRICTLY_BELOW_NO_MOTION" if (macro_val < nm_val) else "NO_POSITIVE_DELTA_SIGNAL"
    if report["signal_status"] != expected_signal:
        raise ValueError("dev_report.signal_status disagrees with comparator")

    phm = report["per_horizon_macro"]
    phnm = report["per_horizon_no_motion_macro"]
    phd = report["per_horizon_delta_vs_no_motion"]
    phc = report["per_horizon_counts"]
    horizon_keys = set(DEV_HORIZON_NAMES)
    for name, container in (
        ("per_horizon_macro", phm),
        ("per_horizon_no_motion_macro", phnm),
        ("per_horizon_delta_vs_no_motion", phd),
        ("per_horizon_counts", phc),
    ):
        if type(container) is not dict or set(container.keys()) != horizon_keys:
            raise ValueError(f"dev_report.{name} keys must match near/mid/far")
    for horizon in DEV_HORIZON_NAMES:
        expected_hc = DEV_EXPECTED_HORIZON_COUNTS[horizon]
        require_exact_int(phc[horizon], f"per_horizon_counts.{horizon}", minimum=expected_hc, maximum=expected_hc)
        hm = _require_finite_number(phm[horizon], f"per_horizon_macro.{horizon}")
        hnm = _require_finite_number(phnm[horizon], f"per_horizon_no_motion_macro.{horizon}")
        hd = _require_finite_number(phd[horizon], f"per_horizon_delta_vs_no_motion.{horizon}")
        if hm < 0.0 or hnm < 0.0:
            raise ValueError(f"negative horizon Smooth L1 for {horizon}")
        if abs((hm - hnm) - hd) > DEV_REPORT_RECOMPUTE_TOL:
            raise ValueError(f"per_horizon_delta_vs_no_motion inconsistency for {horizon}")

    dir_block = report["directional_cosine_loss"]
    dir_keys = {
        "definition",
        "local_mean",
        "anchor_mean",
        "mean",
        "valid_target_count_local",
        "valid_target_count_anchor",
    }
    if type(dir_block) is not dict or set(dir_block.keys()) != dir_keys:
        raise ValueError("dev_report.directional_cosine_loss keys invalid")
    if type(dir_block["definition"]) is not str or not dir_block["definition"]:
        raise ValueError("dev_report.directional_cosine_loss.definition invalid")
    d_loc = _require_finite_number(dir_block["local_mean"], "directional_cosine_loss.local_mean")
    d_anc = _require_finite_number(dir_block["anchor_mean"], "directional_cosine_loss.anchor_mean")
    d_mean = _require_finite_number(dir_block["mean"], "directional_cosine_loss.mean")
    if d_loc < 0.0 or d_anc < 0.0 or abs(0.5 * (d_loc + d_anc) - d_mean) > DEV_REPORT_RECOMPUTE_TOL:
        raise ValueError("dev_report.directional_cosine_loss mean inconsistent")
    d_vt_loc = require_exact_int(dir_block["valid_target_count_local"], "directional.valid_target_count_local", minimum=0, maximum=pair_count)
    d_vt_anc = require_exact_int(dir_block["valid_target_count_anchor"], "directional.valid_target_count_anchor", minimum=0, maximum=pair_count)

    rc_block = report["raw_cosine"]
    rc_keys = {
        "definition",
        "local_mean",
        "anchor_mean",
        "mean",
        "valid_both_count_local",
        "valid_both_count_anchor",
        "undefined_count_local",
        "undefined_count_anchor",
    }
    if type(rc_block) is not dict or set(rc_block.keys()) != rc_keys:
        raise ValueError("dev_report.raw_cosine keys invalid")
    if type(rc_block["definition"]) is not str or not rc_block["definition"]:
        raise ValueError("dev_report.raw_cosine.definition invalid")
    vb_loc = require_exact_int(rc_block["valid_both_count_local"], "raw_cosine.valid_both_count_local", minimum=0, maximum=pair_count)
    vb_anc = require_exact_int(rc_block["valid_both_count_anchor"], "raw_cosine.valid_both_count_anchor", minimum=0, maximum=pair_count)
    un_loc = require_exact_int(rc_block["undefined_count_local"], "raw_cosine.undefined_count_local", minimum=0, maximum=pair_count)
    un_anc = require_exact_int(rc_block["undefined_count_anchor"], "raw_cosine.undefined_count_anchor", minimum=0, maximum=pair_count)
    if vb_loc + un_loc != pair_count or vb_anc + un_anc != pair_count:
        raise ValueError("dev_report.raw_cosine valid + undefined counts do not sum to 147")
    rc_loc = rc_block["local_mean"]
    rc_anc = rc_block["anchor_mean"]
    rc_mean = rc_block["mean"]
    for head_name, count_val, raw_val in (("local", vb_loc, rc_loc), ("anchor", vb_anc, rc_anc)):
        if count_val == 0:
            if raw_val is not None:
                raise ValueError(f"raw_cosine.{head_name}_mean must be None when valid_both_count == 0")
        else:
            hv = _require_finite_number(raw_val, f"raw_cosine.{head_name}_mean")
            if hv < -1.0 - 1e-9 or hv > 1.0 + 1e-9:
                raise ValueError(f"raw_cosine.{head_name}_mean out of [-1, 1]")
    if rc_loc is not None and rc_anc is not None:
        rm = _require_finite_number(rc_mean, "raw_cosine.mean")
        if abs(0.5 * (float(rc_loc) + float(rc_anc)) - rm) > DEV_REPORT_RECOMPUTE_TOL:
            raise ValueError("raw_cosine.mean inconsistent with head means")
    elif rc_mean is not None:
        raise ValueError("raw_cosine.mean must be None when either head is undefined")

    mag_block = report["magnitude_log_ratio_loss"]
    mag_keys = {"definition", "local_mean", "anchor_mean", "mean"}
    if type(mag_block) is not dict or set(mag_block.keys()) != mag_keys:
        raise ValueError("dev_report.magnitude_log_ratio_loss keys invalid")
    if type(mag_block["definition"]) is not str or not mag_block["definition"]:
        raise ValueError("dev_report.magnitude_log_ratio_loss.definition invalid")
    m_loc = _require_finite_number(mag_block["local_mean"], "magnitude.local_mean")
    m_anc = _require_finite_number(mag_block["anchor_mean"], "magnitude.anchor_mean")
    m_mean = _require_finite_number(mag_block["mean"], "magnitude.mean")
    if m_loc < 0.0 or m_anc < 0.0 or abs(0.5 * (m_loc + m_anc) - m_mean) > DEV_REPORT_RECOMPUTE_TOL:
        raise ValueError("magnitude_log_ratio_loss mean inconsistent")

    nr_block = report["norms_and_ratios"]
    nr_keys = {
        "pred_norm_mean_local",
        "pred_norm_mean_anchor",
        "target_norm_mean_local",
        "target_norm_mean_anchor",
        "raw_norm_ratio_mean_local",
        "raw_norm_ratio_mean_anchor",
        "zero_pred_count_local",
        "zero_pred_count_anchor",
        "zero_target_count_local",
        "zero_target_count_anchor",
        "undefined_ratio_count_local",
        "undefined_ratio_count_anchor",
    }
    if type(nr_block) is not dict or set(nr_block.keys()) != nr_keys:
        raise ValueError("dev_report.norms_and_ratios keys invalid")
    for k in ("pred_norm_mean_local", "pred_norm_mean_anchor", "target_norm_mean_local", "target_norm_mean_anchor"):
        if _require_finite_number(nr_block[k], f"norms_and_ratios.{k}") < 0.0:
            raise ValueError(f"norms_and_ratios.{k} must be non-negative")
    require_exact_int(nr_block["zero_pred_count_local"], "norms_and_ratios.zero_pred_count_local", minimum=0, maximum=pair_count)
    require_exact_int(nr_block["zero_pred_count_anchor"], "norms_and_ratios.zero_pred_count_anchor", minimum=0, maximum=pair_count)
    zt_loc = require_exact_int(nr_block["zero_target_count_local"], "norms_and_ratios.zero_target_count_local", minimum=0, maximum=pair_count)
    zt_anc = require_exact_int(nr_block["zero_target_count_anchor"], "norms_and_ratios.zero_target_count_anchor", minimum=0, maximum=pair_count)
    ur_loc = require_exact_int(nr_block["undefined_ratio_count_local"], "norms_and_ratios.undefined_ratio_count_local", minimum=0, maximum=pair_count)
    ur_anc = require_exact_int(nr_block["undefined_ratio_count_anchor"], "norms_and_ratios.undefined_ratio_count_anchor", minimum=0, maximum=pair_count)
    if zt_loc != ur_loc or zt_anc != ur_anc:
        raise ValueError("zero_target_count must equal undefined_ratio_count")
    if d_vt_loc + zt_loc != pair_count or d_vt_anc + zt_anc != pair_count:
        raise ValueError("valid_target_count + zero_target_count must equal 147")
    for head_name, zero_t in (("local", zt_loc), ("anchor", zt_anc)):
        ratio_val = nr_block[f"raw_norm_ratio_mean_{head_name}"]
        if zero_t == pair_count:
            if ratio_val is not None:
                raise ValueError(f"raw_norm_ratio_mean_{head_name} must be None when all targets are zero")
        elif _require_finite_number(ratio_val, f"norms_and_ratios.raw_norm_ratio_mean_{head_name}") < 0.0:
            raise ValueError(f"norms_and_ratios.raw_norm_ratio_mean_{head_name} must be non-negative")

    return {
        "recomputed_macro": float(recomputed_macro),
        "recomputed_no_motion_macro": float(recomputed_nm),
        "recomputed_delta_vs_no_motion": float(recomputed_delta),
        "video_count": video_count,
        "pair_count": pair_count,
        "control_plane_fixture": is_cp_fixture,
        "scientific_metric_valid": sci_valid,
    }


def _validate_dev_report_bytes(raw: bytes, *, expected_macro: float) -> tuple[dict, str]:
    if type(raw) is not bytes or len(raw) < 2 or len(raw) > MAX_DEV_REPORT_BYTES:
        raise ValueError("dev_report.json must be bounded canonical bytes")
    report = strict_json_loads(raw, max_bytes=MAX_DEV_REPORT_BYTES)
    if type(report) is not dict:
        raise ValueError("dev_report.json must decode to an object")
    if canonical_json_bytes(report, max_bytes=MAX_DEV_REPORT_BYTES) != raw:
        raise ValueError("dev_report.json bytes are not canonical")
    verify_dev_report_consistency(
        report,
        expected_macro=float(expected_macro),
        require_scientific_metric=False,
    )
    return report, hashlib.sha256(raw).hexdigest()


def _validate_release_audit(audit: object) -> dict:
    if type(audit) is not dict or set(audit.keys()) != {"ticket_id", "release_sha256", "resume_policy"}:
        raise ValueError("release_audit must contain exact keys {ticket_id, release_sha256, resume_policy}")
    ticket_id = audit.get("ticket_id")
    if type(ticket_id) is not str or not ticket_id or ticket_id != ticket_id.strip():
        raise ValueError("release_audit.ticket_id must be a trimmed nonempty string")
    _require_hex64(audit.get("release_sha256"), "release_audit.release_sha256")
    if audit.get("resume_policy") not in ALLOWED_RESUME_POLICIES:
        raise ValueError("release_audit.resume_policy invalid")
    return {"ticket_id": ticket_id, "release_sha256": audit["release_sha256"], "resume_policy": audit["resume_policy"]}


def build_expected_provenance(
    effective_config_bytes: bytes,
    horizons_path: Path,
    *,
    owner_trust_anchor_sha256: str,
    owner_trust_root_status: str,
    generation_id: str | None = None,
    horizons_file_sha256: str | None = None,
) -> dict:
    if type(effective_config_bytes) is not bytes or not effective_config_bytes:
        raise ValueError("effective_config_bytes must be non-empty bytes")
    if generation_id is not None:
        parse_generation_id(generation_id)
    trust_sha = _require_hex64(owner_trust_anchor_sha256, "owner_trust_anchor_sha256")
    if owner_trust_root_status not in ALLOWED_ACTIVE_TRUST_STATUSES:
        raise ValueError("owner_trust_root_status must reflect verified external or explicit test trust")
    horizons_digest = (
        _require_hex64(horizons_file_sha256, "horizons_file_sha256")
        if horizons_file_sha256 is not None
        else file_sha256(horizons_path, max_bytes=MAX_HORIZONS_FILE_BYTES)
    )
    if horizons_digest != HORIZONS_SHA256:
        raise ValueError("horizons file sha256 does not match pinned HORIZONS_SHA256")
    return {
        "provenance_schema": PROVENANCE_SCHEMA_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "manifest_schema": MANIFEST_SCHEMA_VERSION,
        "pointer_schema": POINTER_SCHEMA_VERSION,
        "commit_schema": COMMIT_SCHEMA_VERSION,
        "dev_report_schema": DEV_REPORT_SCHEMA_VERSION,
        "dev_metric_id": DEV_METRIC_ID,
        "dev_metric_version": DEV_METRIC_VERSION,
        "dev_metric_status": DEV_METRIC_STATUS,
        "failure_receipt_schema": FAILURE_RECEIPT_SCHEMA_VERSION,
        "rng_scope": RNG_SCOPE,
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
        "owner_trust_anchor_sha256": trust_sha,
        "owner_trust_root_status": owner_trust_root_status,
        "release_created": False,
    }


def verify_baseline_provenance(header: dict) -> dict:
    if type(header) is not dict:
        raise ValueError("header must be a dict")
    completed = require_exact_int(header.get("completed"), "header.completed", minimum=0)
    prov = header.get("provenance")
    if type(prov) is not dict:
        raise ValueError("checkpoint header missing prov-v2 provenance")
    if set(prov) != PROVENANCE_KEYS:
        raise ValueError("checkpoint provenance field set invalid")
    expected_pins = {
        "provenance_schema": PROVENANCE_SCHEMA_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "manifest_schema": MANIFEST_SCHEMA_VERSION,
        "pointer_schema": POINTER_SCHEMA_VERSION,
        "commit_schema": COMMIT_SCHEMA_VERSION,
        "dev_report_schema": DEV_REPORT_SCHEMA_VERSION,
        "dev_metric_id": DEV_METRIC_ID,
        "dev_metric_version": DEV_METRIC_VERSION,
        "dev_metric_status": DEV_METRIC_STATUS,
        "failure_receipt_schema": FAILURE_RECEIPT_SCHEMA_VERSION,
        "rng_scope": RNG_SCOPE,
        "source_id": SYNTHETIC_SOURCE_ID,
        "source_sha256": SYNTHETIC_TRAIN_IDS_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "loss_sha256": LOSS_SHA256,
        "horizons_sha256": HORIZONS_SHA256,
        "horizons_file_sha256": HORIZONS_SHA256,
        "sampler_sha256": SAMPLER_SHA256,
        "exposure_schema": SCHEMA_VERSION,
        "sampler_schema": SAMPLER_SCHEMA_VERSION,
        "remainder_policy": REMAINDER_POLICY,
        "forward_binding": FORWARD_BINDING,
    }
    for key, expected_val in expected_pins.items():
        if prov.get(key) != expected_val or type(prov.get(key)) is not str:
            raise ValueError(f"checkpoint provenance.{key} mismatch")
    if prov.get("owner_trust_root_status") not in ALLOWED_ACTIVE_TRUST_STATUSES:
        raise ValueError("checkpoint provenance missing verified owner_trust_root_status")
    _require_hex64(prov.get("owner_trust_anchor_sha256"), "provenance.owner_trust_anchor_sha256")
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


def verify_checkpoint_provenance(
    header: dict,
    expected_provenance: dict,
    *,
    expected_generation_id: str | None = None,
) -> dict:
    if type(expected_provenance) is not dict:
        raise ValueError("expected_provenance must be a dict")
    if set(expected_provenance) != PROVENANCE_KEYS:
        raise ValueError("expected_provenance field set invalid")
    prov = verify_baseline_provenance(header)
    for key, expected_val in expected_provenance.items():
        if key == "generation_id":
            if expected_val is not None:
                parse_generation_id(expected_val)
                if prov[key] != expected_val:
                    raise ValueError("checkpoint provenance.generation_id mismatch")
        elif prov[key] != expected_val or type(prov[key]) is not type(expected_val):
            raise ValueError(f"checkpoint provenance.{key} mismatch")
    if expected_generation_id is not None:
        parse_generation_id(expected_generation_id)
        if prov["generation_id"] != expected_generation_id:
            raise ValueError("checkpoint provenance.generation_id disagrees with expected_generation_id")
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
        gauss = _require_finite_number(gauss, "python_rng.gauss")
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
    validated_cached = _require_finite_number(cached, "numpy_rng.cached")
    return (algo, validated_keys, validated_pos, validated_has_gauss, validated_cached)


def validate_torch_rng_bytes(raw: bytes, *, expected_len: int | None = None) -> bytes:
    if type(raw) is not bytes or len(raw) < 1 or len(raw) > MAX_RNG_SECTION_BYTES:
        raise ValueError("torch_rng section must be bounded non-empty bytes")
    if expected_len is not None and len(raw) != expected_len:
        raise ValueError(f"torch_rng byte length mismatch ({len(raw)} != {expected_len})")
    return raw


def capture_cpu_rng_snapshot() -> tuple:
    import random

    import numpy
    import torch

    py_state = random.getstate()
    np_algo, np_keys, np_pos, np_has_gauss, np_cached = numpy.random.get_state()
    np_state = (np_algo, np_keys.copy(), np_pos, np_has_gauss, np_cached)
    torch_state = torch.get_rng_state().clone()
    return (py_state, np_state, torch_state)


def restore_cpu_rng_snapshot(snapshot: tuple) -> list[str]:
    import random

    import numpy
    import torch

    py_state, np_state, torch_state = snapshot
    errors: list[str] = []
    for name, action in (
        ("python_rng", lambda: random.setstate(py_state)),
        ("numpy_rng", lambda: numpy.random.set_state(np_state)),
        ("torch_rng", lambda: torch.set_rng_state(torch_state)),
    ):
        try:
            action()
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    return errors


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
        if ref_tensor.device.type != "cpu":
            raise ValueError(f"live model state {key} must be on cpu under C2 contract")
        tensor = loaded[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"model state {key} is not a Tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"model state {key} must be on cpu")
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


def _validate_adamw_slot(slot: object, live_param, *, completed: int) -> dict:
    import torch

    if type(slot) is not dict:
        raise ValueError("adamw state entry must be a dict")
    if set(slot.keys()) != {"step", "exp_avg", "exp_avg_sq"}:
        raise ValueError("adamw state entry must contain exact keys {step, exp_avg, exp_avg_sq}")
    if live_param.device.type != "cpu" or not live_param.is_floating_point():
        raise ValueError("live parameter must be a floating-point CPU tensor")

    validated = {}
    for field in ("exp_avg", "exp_avg_sq"):
        tensor = slot[field]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"adamw state {field} must be a Tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"adamw state {field} must be on cpu")
        if tensor.shape != live_param.shape or tensor.dtype != live_param.dtype or not tensor.is_floating_point():
            raise ValueError(f"adamw state {field} shape/dtype mismatch")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"adamw state {field} contains nonfinite values")
        if field == "exp_avg_sq" and not bool((tensor >= 0).all().item()):
            raise ValueError("adamw state exp_avg_sq contains negative entries")
        validated[field] = tensor.detach().clone()

    step_val = slot["step"]
    if isinstance(step_val, torch.Tensor):
        if step_val.device.type != "cpu" or step_val.ndim != 0 or step_val.dtype == torch.bool or step_val.is_complex():
            raise ValueError("adamw state step tensor must be non-bool real CPU scalar")
        if not bool(torch.isfinite(step_val).item()):
            raise ValueError("adamw state step tensor is nonfinite")
        step_num = float(step_val.item())
        if not step_num.is_integer() or int(step_num) != completed:
            raise ValueError("adamw state step tensor must be integral and equal completed")
        validated["step"] = step_val.detach().clone()
    elif type(step_val) is int:
        if step_val != completed:
            raise ValueError("adamw state step int must equal completed")
        validated["step"] = step_val
    else:
        raise ValueError("adamw state step must be an integral scalar Tensor or exact int")

    return validated


def stage_adamw_state(
    raw: bytes,
    opt,
    *,
    completed: int,
    expected_updated_param_ids: set[int] | None = None,
) -> dict:
    import torch

    completed = require_exact_int(completed, "completed", minimum=0)
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
    for grp, live_grp, live_sd_grp in zip(raw_groups, opt.param_groups, live_sd_groups):
        if type(grp) is not dict:
            raise ValueError("adamw param_group must be a dict")
        if set(grp.keys()) != set(live_sd_grp.keys()):
            raise ValueError("adamw param_group option key set mismatch")

        lr_val = _require_finite_number(grp["lr"], "adamw.lr")
        wd_val = _require_finite_number(grp["weight_decay"], "adamw.weight_decay")
        eps_val = _require_finite_number(grp["eps"], "adamw.eps")
        if lr_val != EXPECTED_ADAMW_LR or lr_val != float(live_sd_grp["lr"]):
            raise ValueError("adamw param_group lr disagrees with fixed contract")
        if wd_val != EXPECTED_ADAMW_WEIGHT_DECAY or wd_val != float(live_sd_grp["weight_decay"]):
            raise ValueError("adamw param_group weight_decay disagrees with fixed contract")
        if eps_val != EXPECTED_ADAMW_EPS or eps_val != float(live_sd_grp["eps"]):
            raise ValueError("adamw param_group eps disagrees with fixed contract")

        betas_raw = grp["betas"]
        if type(betas_raw) not in (tuple, list) or len(betas_raw) != 2:
            raise ValueError("adamw param_group betas must be a 2-element tuple/list")
        betas_val = (
            _require_finite_number(betas_raw[0], "adamw.betas[0]"),
            _require_finite_number(betas_raw[1], "adamw.betas[1]"),
        )
        live_betas = tuple(float(x) for x in live_sd_grp["betas"])
        if betas_val != EXPECTED_ADAMW_BETAS or betas_val != live_betas:
            raise ValueError("adamw param_group betas disagree with fixed contract")

        if grp["amsgrad"] is not EXPECTED_ADAMW_AMSGRAD or live_sd_grp.get("amsgrad") is not EXPECTED_ADAMW_AMSGRAD:
            raise ValueError("adamw param_group amsgrad disagrees with fixed contract")

        for opt_key, live_opt_val in live_sd_grp.items():
            if opt_key in ("params", "lr", "weight_decay", "betas", "eps", "amsgrad"):
                continue
            ckpt_opt_val = grp[opt_key]
            if type(ckpt_opt_val) is not type(live_opt_val) or ckpt_opt_val != live_opt_val:
                raise ValueError(f"adamw param_group option {opt_key} disagrees with live optimizer")

        raw_pids = grp.get("params")
        live_pids = live_sd_grp["params"]
        if type(raw_pids) is not list or len(raw_pids) != len(live_grp["params"]) or len(raw_pids) != len(live_pids):
            raise ValueError("adamw param_group parameter count mismatch")
        pids = [require_exact_int(pid, "adamw.param_id", minimum=0) for pid in raw_pids]
        if pids != [int(pid) for pid in live_pids]:
            raise ValueError("adamw param_group parameter mapping/order mismatch")
        saved_param_ids.extend(pids)
        live_params.extend(live_grp["params"])

    if len(set(saved_param_ids)) != len(saved_param_ids):
        raise ValueError("adamw duplicate parameter id in param_groups")

    # Preserve live fixed optimizer settings rather than adopting untrusted group dicts.
    staged_groups = _clone_state_value(live_sd_groups)

    if completed == 0:
        if raw_state:
            raise ValueError("adamw state must be empty when completed == 0")
        return {"state": {}, "param_groups": staged_groups}

    id_to_param = dict(zip(saved_param_ids, live_params))
    # Every parameter with requires_grad=True in SplitterV3 participates unconditionally in
    # SplitterV3.forward() -> SplitterV3Loss.forward()["total"], unless an explicit subset is given.
    expected_pids = (
        set(expected_updated_param_ids)
        if expected_updated_param_ids is not None
        else {pid for pid, param in id_to_param.items() if getattr(param, "requires_grad", True)}
    )
    if any(type(pid) is not int for pid in raw_state.keys()):
        raise ValueError("adamw state keys must be exact ints")
    if set(raw_state.keys()) != expected_pids:
        raise ValueError("adamw state entry set mismatch after completed updates")

    staged_state = {}
    for pid in sorted(expected_pids):
        staged_state[pid] = _validate_adamw_slot(raw_state[pid], id_to_param[pid], completed=completed)
    return {"state": staged_state, "param_groups": staged_groups}


def clone_optimizer_state_dict(opt) -> dict:
    return _clone_state_value(opt.state_dict())


def _safe_generation_relpath(root: Path, relpath: object, *, label: str) -> Path:
    if type(relpath) is not str or not relpath or relpath.startswith("/") or "\\" in relpath:
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    parts = relpath.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{label} contains invalid path segment")
    generations_root = (root / "generations").resolve()
    resolved = (root / relpath).resolve()
    try:
        resolved.relative_to(generations_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes generations root") from exc
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
        "commit_seq",
        "commit_sha256",
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
    _require_hex64(ptr.get("commit_sha256"), "pointer.commit_sha256")
    require_exact_int(ptr.get("commit_seq"), "pointer.commit_seq", minimum=1)
    if ptr.get("generation_dir") != f"generations/{gen_id}-{role}":
        raise ValueError("pointer.generation_dir mismatch")
    if ptr.get("checkpoint_relpath") != f"generations/{gen_id}-{role}/checkpoint.bin":
        raise ValueError("pointer.checkpoint_relpath mismatch")
    if ptr.get("manifest_relpath") != f"generations/{gen_id}-{role}/manifest.json":
        raise ValueError("pointer.manifest_relpath mismatch")
    _safe_generation_relpath(root, ptr["checkpoint_relpath"], label="pointer.checkpoint_relpath")
    _safe_generation_relpath(root, ptr["manifest_relpath"], label="pointer.manifest_relpath")
    prev_gen = ptr.get("previous_generation_id")
    prev_sha = ptr.get("previous_checkpoint_sha256")
    if (prev_gen is None) != (prev_sha is None):
        raise ValueError("pointer previous generation pairing incomplete")
    if prev_gen is not None:
        parse_generation_id(prev_gen)
        _require_hex64(prev_sha, "pointer.previous_checkpoint_sha256")
    macro = ptr.get("macro")
    if macro is not None:
        _require_finite_number(macro, "pointer.macro")
    return ptr


def _validate_generation_record(root: Path, entry: object, expected_role: str) -> dict:
    required = {
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
    if type(entry) is not dict or set(entry) != required:
        raise ValueError("commit generation record field set invalid")
    if entry.get("role") != expected_role:
        raise ValueError(f"commit record role must be {expected_role}")
    gen_id = entry.get("generation_id")
    gen_step = parse_generation_id(gen_id)
    completed = require_exact_int(entry.get("completed"), f"commit.{expected_role}.completed", minimum=0)
    if completed != gen_step:
        raise ValueError("commit generation_id/completed mismatch")
    _require_hex64(entry.get("checkpoint_sha256"), f"commit.{expected_role}.checkpoint_sha256")
    _require_hex64(entry.get("dev_report_sha256"), f"commit.{expected_role}.dev_report_sha256")
    _require_hex64(entry.get("manifest_sha256"), f"commit.{expected_role}.manifest_sha256")
    _require_hex64(entry.get("provenance_sha256"), f"commit.{expected_role}.provenance_sha256")
    _require_finite_number(entry.get("macro"), f"commit.{expected_role}.macro")
    if entry.get("generation_dir") != f"generations/{gen_id}-{expected_role}":
        raise ValueError("commit record generation_dir mismatch")
    if entry.get("checkpoint_relpath") != f"generations/{gen_id}-{expected_role}/checkpoint.bin":
        raise ValueError("commit record checkpoint_relpath mismatch")
    if entry.get("dev_report_relpath") != f"generations/{gen_id}-{expected_role}/dev_report.json":
        raise ValueError("commit record dev_report_relpath mismatch")
    if entry.get("manifest_relpath") != f"generations/{gen_id}-{expected_role}/manifest.json":
        raise ValueError("commit record manifest_relpath mismatch")
    _safe_generation_relpath(root, entry["checkpoint_relpath"], label="commit.checkpoint_relpath")
    _safe_generation_relpath(root, entry["dev_report_relpath"], label="commit.dev_report_relpath")
    _safe_generation_relpath(root, entry["manifest_relpath"], label="commit.manifest_relpath")
    return entry


def _parse_commit_bytes(root: Path, raw: bytes) -> dict:
    commit = strict_json_loads(raw, max_bytes=MAX_COMMIT_BYTES)
    required = {
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
    if type(commit) is not dict or set(commit) != required:
        raise ValueError("CURRENT_COMMIT field set invalid")
    if commit.get("commit_schema") != COMMIT_SCHEMA_VERSION:
        raise ValueError("CURRENT_COMMIT commit_schema mismatch")
    if commit.get("manifest_schema") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("CURRENT_COMMIT manifest_schema mismatch")
    if commit.get("pointer_schema") != POINTER_SCHEMA_VERSION:
        raise ValueError("CURRENT_COMMIT pointer_schema mismatch")
    if commit.get("provenance_schema") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("CURRENT_COMMIT provenance_schema mismatch")
    if commit.get("dev_report_schema") != DEV_REPORT_SCHEMA_VERSION:
        raise ValueError("CURRENT_COMMIT dev_report_schema mismatch")
    require_exact_int(commit.get("commit_seq"), "commit.commit_seq", minimum=1)
    _require_hex64(commit.get("effective_config_sha256"), "commit.effective_config_sha256")
    _require_hex64(commit.get("owner_trust_anchor_sha256"), "commit.owner_trust_anchor_sha256")
    if commit.get("owner_trust_root_status") not in ALLOWED_ACTIVE_TRUST_STATUSES:
        raise ValueError("CURRENT_COMMIT owner_trust_root_status mismatch")
    if commit.get("release_created") is not False:
        raise ValueError("CURRENT_COMMIT release_created must be false")

    last_entry = _validate_generation_record(root, commit.get("last"), "LAST")
    best_entry = _validate_generation_record(root, commit.get("best"), "BEST")
    if best_entry["completed"] > last_entry["completed"]:
        raise ValueError("CURRENT_COMMIT best step cannot exceed last step")

    raw_history = commit.get("best_history")
    if type(raw_history) is not list or not raw_history or len(raw_history) > MAX_BEST_HISTORY:
        raise ValueError("CURRENT_COMMIT best_history invalid")
    prev_step = -1
    prev_macro: float | None = None
    for item in raw_history:
        rec = _validate_generation_record(root, item, "BEST")
        step = rec["completed"]
        macro = float(rec["macro"])
        if step <= prev_step:
            raise ValueError("CURRENT_COMMIT best_history steps must strictly increase")
        if prev_macro is not None and not (macro < prev_macro):
            raise ValueError("CURRENT_COMMIT best_history must strictly improve per earliest-tie rule")
        prev_step = step
        prev_macro = macro
    if raw_history[-1] != best_entry:
        raise ValueError("CURRENT_COMMIT best_history[-1] must match committed best")
    return commit


def _verify_materialized_record(
    root: Path,
    entry: dict,
    role: str,
    *,
    expected_provenance: dict | None = None,
) -> dict:
    gen_dir = _safe_generation_relpath(root, entry["generation_dir"], label=f"{role}.generation_dir")
    if not gen_dir.is_dir():
        raise ValueError(f"{role} generation directory missing")
    man_path = _safe_generation_relpath(root, entry["manifest_relpath"], label=f"{role}.manifest_relpath")
    man_bytes = _read_bounded_bytes(man_path, max_bytes=MAX_MANIFEST_BYTES, label=f"{role}.manifest.json")
    if hashlib.sha256(man_bytes).hexdigest() != entry["manifest_sha256"]:
        raise ValueError(f"{role} manifest sha256 mismatch")

    manifest = strict_json_loads(man_bytes, max_bytes=MAX_MANIFEST_BYTES)
    required_man = {
        "manifest_schema",
        "checkpoint_schema",
        "provenance_schema",
        "dev_report_schema",
        "generation_id",
        "role",
        "completed",
        "checkpoint_bytes",
        "checkpoint_sha256",
        "dev_report_bytes",
        "dev_report_sha256",
        "header_sha256",
        "provenance_sha256",
        "provenance",
        "release_audit",
    }
    if type(manifest) is not dict or set(manifest) != required_man:
        raise ValueError(f"{role} manifest field set invalid")
    if manifest.get("manifest_schema") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"{role} manifest schema mismatch")
    if manifest.get("checkpoint_schema") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"{role} manifest checkpoint_schema mismatch")
    if manifest.get("provenance_schema") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(f"{role} manifest provenance_schema mismatch")
    if manifest.get("dev_report_schema") != DEV_REPORT_SCHEMA_VERSION:
        raise ValueError(f"{role} manifest dev_report_schema mismatch")
    if manifest.get("generation_id") != entry["generation_id"] or manifest.get("role") != role:
        raise ValueError(f"{role} manifest identity mismatch")
    if manifest.get("completed") != entry["completed"] or manifest.get("checkpoint_sha256") != entry["checkpoint_sha256"]:
        raise ValueError(f"{role} manifest digest/step mismatch")
    if manifest.get("dev_report_sha256") != entry["dev_report_sha256"]:
        raise ValueError(f"{role} manifest dev_report_sha256 mismatch")
    if manifest.get("provenance_sha256") != entry["provenance_sha256"]:
        raise ValueError(f"{role} manifest provenance_sha256 mismatch")
    _validate_release_audit(manifest.get("release_audit"))

    ckpt_path = _safe_generation_relpath(root, entry["checkpoint_relpath"], label=f"{role}.checkpoint_relpath")
    payload = _read_bounded_bytes(ckpt_path, max_bytes=MAX_CHECKPOINT_BYTES, label=f"{role}.checkpoint.bin")
    if len(payload) != manifest.get("checkpoint_bytes"):
        raise ValueError(f"{role} checkpoint byte count mismatch")
    if hashlib.sha256(payload).hexdigest() != entry["checkpoint_sha256"]:
        raise ValueError(f"{role} checkpoint sha256 mismatch")

    dev_path = _safe_generation_relpath(root, entry["dev_report_relpath"], label=f"{role}.dev_report_relpath")
    dev_raw = _read_bounded_bytes(dev_path, max_bytes=MAX_DEV_REPORT_BYTES, label=f"{role}.dev_report.json")
    if len(dev_raw) != manifest.get("dev_report_bytes"):
        raise ValueError(f"{role} dev_report byte count mismatch")
    dev_report, dev_sha = _validate_dev_report_bytes(dev_raw, expected_macro=float(entry["macro"]))
    if dev_sha != entry["dev_report_sha256"] or dev_sha != manifest.get("dev_report_sha256"):
        raise ValueError(f"{role} dev_report sha256 mismatch")

    decoded = unpack(payload)
    header = decoded["header"]
    if header.get("completed") != entry["completed"]:
        raise ValueError(f"{role} header.completed disagrees with commit record")
    if header.get("dev_report_sha256") != dev_sha:
        raise ValueError(f"{role} header.dev_report_sha256 disagrees with dev_report.json")
    if _validate_release_audit(header.get("release_audit")) != manifest.get("release_audit"):
        raise ValueError(f"{role} header.release_audit disagrees with manifest")
    header_sha = hashlib.sha256(canonical_json_bytes(header, max_bytes=MAX_HEADER_BYTES)).hexdigest()
    if header_sha != manifest.get("header_sha256"):
        raise ValueError(f"{role} header sha256 disagrees with manifest")

    prov = (
        verify_checkpoint_provenance(
            header,
            expected_provenance,
            expected_generation_id=entry["generation_id"],
        )
        if expected_provenance is not None
        else verify_baseline_provenance(header)
    )
    if prov.get("generation_id") != entry["generation_id"]:
        raise ValueError(f"{role} provenance generation_id mismatch")
    if manifest.get("provenance") != prov:
        raise ValueError(f"{role} manifest provenance disagrees with header provenance")
    prov_sha = hashlib.sha256(canonical_json_bytes(prov, max_bytes=MAX_HEADER_BYTES)).hexdigest()
    if prov_sha != entry["provenance_sha256"]:
        raise ValueError(f"{role} provenance sha256 mismatch")
    macro_val = _require_finite_number(header.get("macro"), f"{role}.header.macro")
    if macro_val != float(entry["macro"]):
        raise ValueError(f"{role} header.macro disagrees with commit record")

    return {
        "record": entry,
        "manifest": manifest,
        "payload": payload,
        "dev_report_bytes": dev_raw,
        "dev_report": dev_report,
        "decoded": decoded,
        "provenance": prov,
    }


def _sync_derived_mirrors(root: Path, commit: dict, commit_sha256: str) -> None:
    history = commit["best_history"]
    prev_best_gen: str | None = history[-2]["generation_id"] if len(history) >= 2 else None
    prev_best_sha: str | None = history[-2]["checkpoint_sha256"] if len(history) >= 2 else None
    last_entry = commit["last"]
    best_entry = commit["best"]

    mirrors = (
        ("LAST", last_entry, None, None),
        ("BEST", best_entry, prev_best_gen, prev_best_sha),
    )
    for role, entry, prev_gen, prev_sha in mirrors:
        pointer_name = f"CURRENT_{role}"
        pointer_path = root / pointer_name
        if role == "LAST" and pointer_path.exists():
            try:
                old_raw = _read_bounded_bytes(pointer_path, max_bytes=MAX_POINTER_BYTES, label=pointer_name)
                old_ptr = _parse_pointer_bytes(root, old_raw, expected_role="LAST")
                if old_ptr["generation_id"] != entry["generation_id"]:
                    prev_gen = old_ptr["generation_id"]
                    prev_sha = old_ptr["checkpoint_sha256"]
                else:
                    prev_gen = old_ptr["previous_generation_id"]
                    prev_sha = old_ptr["previous_checkpoint_sha256"]
            except ValueError:
                prev_gen = None
                prev_sha = None
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
        partial = root / f"{pointer_name}.partial"
        partial.write_bytes(ptr_bytes)
        _fsync_file(partial)
        if partial.read_bytes() != ptr_bytes:
            raise ValueError(f"{pointer_name} staged readback mismatch")
        os.replace(partial, pointer_path)
        _fsync_dir(root)
        if pointer_path.read_bytes() != ptr_bytes:
            raise ValueError(f"{pointer_name} readback mismatch")


def _fsync_committed_state(root: Path, commit: dict) -> None:
    """Establish on-disk durability for CURRENT_COMMIT and all referenced generations."""
    unique_records: dict[str, dict] = {}
    for entry in (commit["last"], commit["best"], *commit["best_history"]):
        unique_records[entry["generation_dir"]] = entry
    for entry in unique_records.values():
        gen_dir = _safe_generation_relpath(root, entry["generation_dir"], label="recovery.generation_dir")
        ckpt_path = _safe_generation_relpath(root, entry["checkpoint_relpath"], label="recovery.checkpoint_relpath")
        dev_path = _safe_generation_relpath(root, entry["dev_report_relpath"], label="recovery.dev_report_relpath")
        man_path = _safe_generation_relpath(root, entry["manifest_relpath"], label="recovery.manifest_relpath")
        _fsync_file(ckpt_path)
        _fsync_file(dev_path)
        _fsync_file(man_path)
        _fsync_dir(gen_dir)
    _fsync_dir(root / "generations")
    _fsync_file(root / "CURRENT_COMMIT")
    _fsync_dir(root)


def resolve_commit(
    root: Path,
    *,
    expected_provenance: dict | None = None,
    expected_dev_events: list[int] | None = None,
) -> dict:
    root = _reject_root_if_scored(root)
    commit_path = root / "CURRENT_COMMIT"
    commit_bytes = _read_bounded_bytes(commit_path, max_bytes=MAX_COMMIT_BYTES, label="CURRENT_COMMIT")
    commit_sha256 = hashlib.sha256(commit_bytes).hexdigest()
    commit = _parse_commit_bytes(root, commit_bytes)

    last_verified = _verify_materialized_record(
        root,
        commit["last"],
        "LAST",
        expected_provenance=expected_provenance,
    )
    best_verified = _verify_materialized_record(
        root,
        commit["best"],
        "BEST",
        expected_provenance=expected_provenance,
    )
    if last_verified["provenance"]["effective_config_sha256"] != commit["effective_config_sha256"]:
        raise ValueError("LAST effective_config_sha256 disagrees with CURRENT_COMMIT")
    if best_verified["provenance"]["effective_config_sha256"] != commit["effective_config_sha256"]:
        raise ValueError("BEST effective_config_sha256 disagrees with CURRENT_COMMIT")
    if last_verified["provenance"]["owner_trust_anchor_sha256"] != commit["owner_trust_anchor_sha256"]:
        raise ValueError("LAST owner_trust_anchor_sha256 disagrees with CURRENT_COMMIT")
    if best_verified["provenance"]["owner_trust_anchor_sha256"] != commit["owner_trust_anchor_sha256"]:
        raise ValueError("BEST owner_trust_anchor_sha256 disagrees with CURRENT_COMMIT")

    last_header = last_verified["decoded"]["header"]
    best_header = best_verified["decoded"]["header"]
    validate_event_and_best_history(last_header, expected_dev_events=expected_dev_events)
    validate_event_and_best_history(best_header, expected_dev_events=expected_dev_events)
    if last_header.get("best_step") != commit["best"]["completed"]:
        raise ValueError("LAST header.best_step disagrees with committed BEST")
    if _require_finite_number(last_header.get("best_macro"), "last.best_macro") != float(commit["best"]["macro"]):
        raise ValueError("LAST header.best_macro disagrees with committed BEST")
    if best_header.get("best_step") != commit["best"]["completed"]:
        raise ValueError("BEST header.best_step disagrees with committed BEST")
    expected_hdr_history = [{"step": rec["completed"], "macro": float(rec["macro"])} for rec in commit["best_history"]]
    normalized_last_history = [
        {"step": int(item["step"]), "macro": float(item["macro"])} for item in last_header["best_history"]
    ]
    if normalized_last_history != expected_hdr_history:
        raise ValueError("LAST header.best_history disagrees with committed best_history")

    # Verify every historical BEST generation remains preserved on disk with matching bytes.
    for hist_entry in commit["best_history"]:
        hist_verified = _verify_materialized_record(
            root,
            hist_entry,
            "BEST",
            expected_provenance=expected_provenance,
        )
        validate_event_and_best_history(
            hist_verified["decoded"]["header"],
            expected_dev_events=expected_dev_events,
        )

    return {
        "commit": commit,
        "commit_sha256": commit_sha256,
        "last": last_verified,
        "best": best_verified,
    }


def resolve_pointer(
    root: Path,
    role: str,
    *,
    expected_provenance: dict | None = None,
    expected_dev_events: list[int] | None = None,
) -> dict:
    if role not in {"BEST", "LAST"}:
        raise ValueError("pointer role rejected")
    resolved = resolve_commit(
        root,
        expected_provenance=expected_provenance,
        expected_dev_events=expected_dev_events,
    )
    chosen = resolved["best"] if role == "BEST" else resolved["last"]
    return {
        "commit": resolved["commit"],
        "commit_sha256": resolved["commit_sha256"],
        "pointer": chosen["record"],
        "manifest": chosen["manifest"],
        "payload": chosen["payload"],
        "dev_report": chosen["dev_report"],
        "dev_report_bytes": chosen["dev_report_bytes"],
        "decoded": chosen["decoded"],
        "provenance": chosen["provenance"],
    }


def _inventory_unreferenced_paths(root: Path, commit: dict | None) -> tuple[list[str], list[str]]:
    referenced: set[str] = set()
    if commit is not None:
        referenced.add(Path(commit["last"]["generation_dir"]).name)
        referenced.add(Path(commit["best"]["generation_dir"]).name)
        for item in commit["best_history"]:
            referenced.add(Path(item["generation_dir"]).name)
    generations = root / "generations"
    stage_root = root / "stage"
    unreferenced = (
        sorted(p.name for p in generations.iterdir() if p.name not in referenced)
        if generations.is_dir()
        else []
    )
    orphaned = sorted(p.name for p in stage_root.iterdir()) if stage_root.is_dir() else []
    return unreferenced, orphaned


def recover_commit_state(
    root: Path,
    *,
    expected_provenance: dict | None = None,
    expected_dev_events: list[int] | None = None,
) -> dict:
    root = _reject_root_if_scored(root)
    with _single_writer_lock(root):
        commit_path = root / "CURRENT_COMMIT"
        if not commit_path.exists():
            unreferenced, orphaned = _inventory_unreferenced_paths(root, None)
            # Never guess or promote uncommitted generations when CURRENT_COMMIT does not exist.
            return {
                "committed": False,
                "commit_state": COMMIT_STATE_PRE_COMMIT_FAILURE,
                "commit_durable": False,
                "commit": None,
                "commit_sha256": None,
                "unreferenced_generations": unreferenced,
                "orphaned_stages": orphaned,
            }
        resolved = resolve_commit(
            root,
            expected_provenance=expected_provenance,
            expected_dev_events=expected_dev_events,
        )
        commit_record = resolved["commit"]
        commit_sha256 = resolved["commit_sha256"]
        generation_id = commit_record["last"]["generation_id"]
        try:
            _fsync_committed_state(root, commit_record)
        except Exception as exc:
            raise CommitDurabilityUnknownError(
                f"recovered CURRENT_COMMIT {generation_id} durability fsync failed: {exc}",
                generation_id=generation_id,
                commit_sha256=commit_sha256,
            ) from exc
        try:
            _sync_derived_mirrors(root, commit_record, commit_sha256)
        except Exception as exc:
            raise PostCommitError(
                f"post-commit mirror_sync failure after durable CURRENT_COMMIT {generation_id}: {exc}",
                generation_id=generation_id,
                commit_sha256=commit_sha256,
                phase="mirror_sync",
            ) from exc
        unreferenced, orphaned = _inventory_unreferenced_paths(root, commit_record)
        return {
            "committed": True,
            "commit_state": COMMIT_STATE_DURABLE,
            "commit_durable": True,
            "commit": commit_record,
            "commit_sha256": commit_sha256,
            "unreferenced_generations": unreferenced,
            "orphaned_stages": orphaned,
        }


def _build_generation_manifest(
    generation_id: str,
    role: str,
    completed: int,
    payload: bytes,
    dev_report_bytes: bytes,
    header: dict,
    prov: dict,
    release_audit: dict,
) -> tuple[dict, bytes, str, str, str]:
    checkpoint_sha256 = hashlib.sha256(payload).hexdigest()
    dev_report_sha256 = hashlib.sha256(dev_report_bytes).hexdigest()
    header_sha256 = hashlib.sha256(canonical_json_bytes(header, max_bytes=MAX_HEADER_BYTES)).hexdigest()
    provenance_sha256 = hashlib.sha256(canonical_json_bytes(prov, max_bytes=MAX_HEADER_BYTES)).hexdigest()
    manifest = {
        "manifest_schema": MANIFEST_SCHEMA_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "provenance_schema": PROVENANCE_SCHEMA_VERSION,
        "dev_report_schema": DEV_REPORT_SCHEMA_VERSION,
        "generation_id": generation_id,
        "role": role,
        "completed": completed,
        "checkpoint_bytes": len(payload),
        "checkpoint_sha256": checkpoint_sha256,
        "dev_report_bytes": len(dev_report_bytes),
        "dev_report_sha256": dev_report_sha256,
        "header_sha256": header_sha256,
        "provenance_sha256": provenance_sha256,
        "provenance": prov,
        "release_audit": release_audit,
    }
    manifest_bytes = canonical_json_bytes(manifest, max_bytes=MAX_MANIFEST_BYTES)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    return manifest, manifest_bytes, checkpoint_sha256, dev_report_sha256, manifest_sha256


def _materialize_generation(
    root: Path,
    generation_id: str,
    role: str,
    completed: int,
    payload: bytes,
    dev_report_bytes: bytes,
    header: dict,
    prov: dict,
    *,
    expected_provenance: dict,
    release_audit: dict,
) -> dict:
    generations = root / "generations"
    stage_root = root / "stage"
    generations.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)
    _fsync_dir(root)
    if generations.stat().st_dev != stage_root.stat().st_dev:
        raise ValueError("stage and generations must be on the same filesystem")

    manifest, manifest_bytes, checkpoint_sha256, dev_report_sha256, manifest_sha256 = _build_generation_manifest(
        generation_id, role, completed, payload, dev_report_bytes, header, prov, release_audit
    )
    final = generations / f"{generation_id}-{role}"
    if final.exists():
        # Idempotent recovery of post-rename unreferenced or retried generation: verify exact match; refuse conflicts.
        if not final.is_dir():
            raise ValueError("existing generation path is not a directory")
        entries = sorted(p.name for p in final.iterdir())
        if entries != ["checkpoint.bin", "dev_report.json", "manifest.json"]:
            raise ValueError("conflicting files in existing generation directory")
        existing_ckpt = _read_bounded_bytes(final / "checkpoint.bin", max_bytes=MAX_CHECKPOINT_BYTES, label="existing checkpoint")
        existing_dev = _read_bounded_bytes(final / "dev_report.json", max_bytes=MAX_DEV_REPORT_BYTES, label="existing dev_report")
        existing_man = _read_bounded_bytes(final / "manifest.json", max_bytes=MAX_MANIFEST_BYTES, label="existing manifest")
        if existing_ckpt != payload or existing_dev != dev_report_bytes or existing_man != manifest_bytes:
            raise ValueError("conflicting existing generation; immutable artifact refuses overwrite")
        decoded_existing = unpack(existing_ckpt)
        verify_checkpoint_provenance(
            decoded_existing["header"],
            expected_provenance,
            expected_generation_id=generation_id,
        )
        _fsync_file(final / "checkpoint.bin")
        _fsync_file(final / "dev_report.json")
        _fsync_file(final / "manifest.json")
        _fsync_dir(final)
        _fsync_dir(generations)
        return {
            "status": "reused_verified",
            "record": {
                "generation_id": generation_id,
                "role": role,
                "generation_dir": f"generations/{generation_id}-{role}",
                "checkpoint_relpath": f"generations/{generation_id}-{role}/checkpoint.bin",
                "dev_report_relpath": f"generations/{generation_id}-{role}/dev_report.json",
                "manifest_relpath": f"generations/{generation_id}-{role}/manifest.json",
                "checkpoint_sha256": checkpoint_sha256,
                "dev_report_sha256": dev_report_sha256,
                "manifest_sha256": manifest_sha256,
                "provenance_sha256": manifest["provenance_sha256"],
                "completed": completed,
                "macro": _require_finite_number(header.get("macro"), "header.macro"),
            },
        }

    stage: Path | None = None
    for seq in range(1, MAX_FAILURE_RECEIPTS + 1):
        candidate = stage_root / f"{generation_id}-{role}.tmp-{seq:04d}"
        if not candidate.exists():
            stage = candidate
            break
    if stage is None:
        raise ValueError("stage directory reached MAX_FAILURE_RECEIPTS")
    stage.mkdir(parents=False, exist_ok=False)
    _fsync_dir(stage_root)

    target = stage / "checkpoint.bin"
    target.write_bytes(payload)
    _fsync_file(target)
    dev_stage = stage / "dev_report.json"
    dev_stage.write_bytes(dev_report_bytes)
    _fsync_file(dev_stage)
    man_stage = stage / "manifest.json"
    man_stage.write_bytes(manifest_bytes)
    _fsync_file(man_stage)
    _fsync_dir(stage)

    staged_readback = target.read_bytes()
    if staged_readback != payload or hashlib.sha256(staged_readback).hexdigest() != checkpoint_sha256:
        raise ValueError("staged readback mismatch")
    if dev_stage.read_bytes() != dev_report_bytes:
        raise ValueError("staged dev_report readback mismatch")
    if man_stage.read_bytes() != manifest_bytes:
        raise ValueError("staged manifest readback mismatch")
    unpack(staged_readback)

    os.rename(stage, final)
    _fsync_dir(generations)
    _fsync_dir(stage_root)

    if (final / "checkpoint.bin").read_bytes() != payload:
        raise ValueError("checkpoint readback mismatch")
    if (final / "dev_report.json").read_bytes() != dev_report_bytes:
        raise ValueError("dev_report readback mismatch")
    if (final / "manifest.json").read_bytes() != manifest_bytes:
        raise ValueError("manifest readback mismatch")

    return {
        "status": "created",
        "record": {
            "generation_id": generation_id,
            "role": role,
            "generation_dir": f"generations/{generation_id}-{role}",
            "checkpoint_relpath": f"generations/{generation_id}-{role}/checkpoint.bin",
            "dev_report_relpath": f"generations/{generation_id}-{role}/dev_report.json",
            "manifest_relpath": f"generations/{generation_id}-{role}/manifest.json",
            "checkpoint_sha256": checkpoint_sha256,
            "dev_report_sha256": dev_report_sha256,
            "manifest_sha256": manifest_sha256,
            "provenance_sha256": manifest["provenance_sha256"],
            "completed": completed,
            "macro": _require_finite_number(header.get("macro"), "header.macro"),
        },
    }


def write_failure_receipt(
    root: Path,
    stage: str,
    reason: str,
    *,
    generation_id: str | None = None,
    details: dict | None = None,
) -> dict:
    root = _reject_root_if_scored(root)
    if type(stage) is not str or not stage:
        raise ValueError("failure stage must be a non-empty string")
    if type(reason) is not str or not reason:
        reason = "unspecified failure"
    if generation_id is not None:
        parse_generation_id(generation_id)
    if details is not None and type(details) is not dict:
        raise ValueError("failure details must be a dict or None")

    # Never acquire .writer.lock here so failure paths under lock cannot deadlock.
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


def commit_event(
    root: Path,
    generation_id: str,
    payload: bytes,
    *,
    is_new_best: bool,
    expected_provenance: dict,
    dev_report_bytes: bytes,
    release_audit: dict,
    expected_dev_events: list[int] | None = None,
) -> dict:
    root = _reject_root_if_scored(root)
    if type(is_new_best) is not bool:
        raise ValueError("is_new_best must be an exact bool")
    if type(expected_provenance) is not dict:
        raise ValueError("expected_provenance must be a dict")
    validated_audit = _validate_release_audit(release_audit)
    gen_step = parse_generation_id(generation_id)

    decoded = unpack(payload)
    header = decoded["header"]
    if header.get("completed") != gen_step:
        raise ValueError("commit generation_id does not match header.completed")
    validate_event_and_best_history(header, expected_dev_events=expected_dev_events)
    if not header["events"] or header["events"][-1] != gen_step:
        raise ValueError("commit payload events must end at committed generation step")
    prov = verify_checkpoint_provenance(
        header,
        expected_provenance,
        expected_generation_id=generation_id,
    )
    macro_val = _require_finite_number(header.get("macro"), "header.macro")
    _, dev_report_sha256 = _validate_dev_report_bytes(dev_report_bytes, expected_macro=macro_val)
    if header.get("dev_report_sha256") != dev_report_sha256:
        raise ValueError("header.dev_report_sha256 disagrees with committed dev_report.json")
    if _validate_release_audit(header.get("release_audit")) != validated_audit:
        raise ValueError("header.release_audit disagrees with active release audit")
    best_step = require_exact_int(header.get("best_step"), "header.best_step", minimum=0)
    best_macro = _require_finite_number(header.get("best_macro"), "header.best_macro")
    payload_sha256 = hashlib.sha256(payload).hexdigest()

    with _single_writer_lock(root):
        commit_path = root / "CURRENT_COMMIT"
        prev_commit: dict | None = None
        if commit_path.exists():
            prev_bytes = _read_bounded_bytes(commit_path, max_bytes=MAX_COMMIT_BYTES, label="CURRENT_COMMIT")
            prev_commit = _parse_commit_bytes(root, prev_bytes)
            if prev_commit["effective_config_sha256"] != prov["effective_config_sha256"]:
                raise ValueError("CURRENT_COMMIT effective_config_sha256 mismatch")

            if prev_commit["last"]["generation_id"] == generation_id:
                prev_best_is_this = prev_commit["best"]["generation_id"] == generation_id
                if (
                    prev_commit["last"]["checkpoint_sha256"] != payload_sha256
                    or prev_commit["last"]["dev_report_sha256"] != dev_report_sha256
                    or prev_best_is_this != is_new_best
                    or (is_new_best and prev_commit["best"]["checkpoint_sha256"] != payload_sha256)
                ):
                    raise ValueError("conflicting commit for already-committed generation")
                last_mat = _materialize_generation(
                    root, generation_id, "LAST", gen_step, payload, dev_report_bytes, header, prov,
                    expected_provenance=expected_provenance,
                    release_audit=validated_audit,
                )
                if is_new_best:
                    _materialize_generation(
                        root, generation_id, "BEST", gen_step, payload, dev_report_bytes, header, prov,
                        expected_provenance=expected_provenance,
                        release_audit=validated_audit,
                    )
                else:
                    _verify_materialized_record(
                        root, prev_commit["best"], "BEST", expected_provenance=expected_provenance
                    )
                for hist_entry in prev_commit["best_history"]:
                    hist_verified = _verify_materialized_record(
                        root, hist_entry, "BEST", expected_provenance=expected_provenance
                    )
                    validate_event_and_best_history(
                        hist_verified["decoded"]["header"],
                        expected_dev_events=expected_dev_events,
                    )
                prev_commit_sha = hashlib.sha256(prev_bytes).hexdigest()
                try:
                    _fsync_committed_state(root, prev_commit)
                except Exception as exc:
                    raise CommitDurabilityUnknownError(
                        f"commit durability fsync failed on idempotent retry {generation_id}: {exc}",
                        generation_id=generation_id,
                        commit_sha256=prev_commit_sha,
                    ) from exc
                try:
                    _sync_derived_mirrors(root, prev_commit, prev_commit_sha)
                except Exception as exc:
                    raise PostCommitError(
                        f"post-commit mirror_sync failure after durable commit {generation_id}: {exc}",
                        generation_id=generation_id,
                        commit_sha256=prev_commit_sha,
                        phase="mirror_sync",
                    ) from exc
                return {
                    "status": "idempotent_reused",
                    "commit_state": COMMIT_STATE_DURABLE,
                    "commit_durable": True,
                    "commit_seq": prev_commit["commit_seq"],
                    "commit_sha256": prev_commit_sha,
                    "generation_id": generation_id,
                    "last": last_mat["record"],
                    "best": prev_commit["best"],
                }

            prev_completed = prev_commit["last"]["completed"]
            if gen_step <= prev_completed:
                raise ValueError("commit step must strictly advance past previous committed step")
            expected_is_new_best = macro_val < float(prev_commit["best"]["macro"])
            if is_new_best != expected_is_new_best:
                raise ValueError("is_new_best disagrees with earliest-tie BEST comparator")
            if is_new_best:
                if best_step != gen_step or best_macro != macro_val:
                    raise ValueError("header best disagrees with new BEST")
            else:
                if best_step != prev_commit["best"]["completed"] or best_macro != float(prev_commit["best"]["macro"]):
                    raise ValueError("header best disagrees with preserved committed BEST")
        else:
            if not is_new_best or best_step != gen_step or best_macro != macro_val:
                raise ValueError("initial commit must set header best to initial generation")

        last_mat = _materialize_generation(
            root, generation_id, "LAST", gen_step, payload, dev_report_bytes, header, prov,
            expected_provenance=expected_provenance,
            release_audit=validated_audit,
        )
        if is_new_best:
            best_mat = _materialize_generation(
                root, generation_id, "BEST", gen_step, payload, dev_report_bytes, header, prov,
                expected_provenance=expected_provenance,
                release_audit=validated_audit,
            )
            best_record = best_mat["record"]
            history = ([] if prev_commit is None else list(prev_commit["best_history"])) + [best_record]
        else:
            assert prev_commit is not None
            best_verified = _verify_materialized_record(
                root, prev_commit["best"], "BEST", expected_provenance=expected_provenance
            )
            best_record = best_verified["record"]
            history = list(prev_commit["best_history"])

        if len(history) > MAX_BEST_HISTORY:
            raise ValueError("best_history exceeds MAX_BEST_HISTORY")
        expected_hdr_history = [{"step": rec["completed"], "macro": float(rec["macro"])} for rec in history]
        normalized_hdr_history = [
            {"step": int(item["step"]), "macro": float(item["macro"])} for item in header["best_history"]
        ]
        if normalized_hdr_history != expected_hdr_history:
            raise ValueError("header.best_history disagrees with committed best_history")

        next_seq = 1 if prev_commit is None else int(prev_commit["commit_seq"]) + 1
        commit = {
            "commit_schema": COMMIT_SCHEMA_VERSION,
            "manifest_schema": MANIFEST_SCHEMA_VERSION,
            "pointer_schema": POINTER_SCHEMA_VERSION,
            "provenance_schema": PROVENANCE_SCHEMA_VERSION,
            "dev_report_schema": DEV_REPORT_SCHEMA_VERSION,
            "commit_seq": next_seq,
            "effective_config_sha256": prov["effective_config_sha256"],
            "last": last_mat["record"],
            "best": best_record,
            "best_history": history,
            "owner_trust_anchor_sha256": prov["owner_trust_anchor_sha256"],
            "owner_trust_root_status": prov["owner_trust_root_status"],
            "release_created": False,
        }
        commit_bytes = canonical_json_bytes(commit, max_bytes=MAX_COMMIT_BYTES)
        commit_sha256 = hashlib.sha256(commit_bytes).hexdigest()

        partial = root / "CURRENT_COMMIT.partial"
        partial.write_bytes(commit_bytes)
        _fsync_file(partial)
        if partial.read_bytes() != commit_bytes:
            raise ValueError("CURRENT_COMMIT staged readback mismatch")

        # Single atomic visibility point across LAST + BEST; directory fsync establishes durability.
        os.replace(partial, commit_path)
        try:
            _fsync_dir(root)
        except Exception as exc:
            raise CommitDurabilityUnknownError(
                f"CURRENT_COMMIT visible at {generation_id} but root directory fsync failed (durability unknown): {exc}",
                generation_id=generation_id,
                commit_sha256=commit_sha256,
            ) from exc

        post_phase = "commit_readback"
        try:
            if commit_path.read_bytes() != commit_bytes:
                raise ValueError("CURRENT_COMMIT readback mismatch")
            post_phase = "mirror_sync"
            _sync_derived_mirrors(root, commit, commit_sha256)
        except Exception as exc:
            raise PostCommitError(
                f"post-commit {post_phase} failure after durable CURRENT_COMMIT {generation_id}: {exc}",
                generation_id=generation_id,
                commit_sha256=commit_sha256,
                phase=post_phase,
            ) from exc

        return {
            "status": "committed",
            "commit_state": COMMIT_STATE_DURABLE,
            "commit_durable": True,
            "commit_seq": next_seq,
            "commit_sha256": commit_sha256,
            "generation_id": generation_id,
            "last": last_mat["record"],
            "best": best_record,
        }
