"""Frozen Stage A DEV evaluator (`stage-a-dev-v1`) with mandatory diagnostics and mutation guards."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from . import HORIZONS_SHA256, LOSS_SHA256
from .checkpoints import (
    MAX_HORIZONS_FILE_BYTES,
    _read_bounded_bytes,
    capture_cpu_rng_snapshot,
    restore_cpu_rng_snapshot,
    verify_dev_report_consistency,
)
from .exposure import strict_json_loads
from .forward_binding import (
    WIDTH,
    expected_synthetic_forward_digest,
    prepare_verified_forward_call,
)
from .plan import DEV_TEMPORAL_AXIS_SEMANTICS, require_exact_int, train_ids
from .samples import assert_no_target_leak, synthetic_item

BETA = 0.05
LOCAL_WEIGHT = 0.5
ANCHOR_WEIGHT = 0.5
METRIC_NAME = "dev_delta_huber_macro"
METRIC_VERSION = "stage-a-dev-v1"
METRIC_STATUS = "DS_FROZEN_STAGE_A_V1"
METRIC_AGGREGATION = "equal_video_macro_50_over_147_pairs"
METRIC_DIRECTION = "minimize"
METRIC_TIE_POLICY = "earliest_strict_improvement_eps_0"
METRIC_EPSILON = 0.0
DIAGNOSTICS_POLICY = "mandatory_no_motion_norms_cosine_magnitude_no_aux_selection"
H2_POLICY = "secondary_contrast_deferred_not_run"
DEV_REPORT_SCHEMA_VERSION = "dev-report-v1"
DIAG_EPS = 1e-8

EXPECTED_DEV_VIDEOS = 50
EXPECTED_DEV_PAIRS = 147
HORIZON_NAMES = ("near", "mid", "far")
PREDECLARED_ABSENT_MID_VIDEOS = frozenset({"28060", "37577", "17858"})


class EvaluatorMutationError(RuntimeError):
    """Raised fail-stop if DEV evaluation mutates model mode, state, grads, or CPU RNG."""

    def __init__(
        self,
        message: str,
        *,
        mutation_details: dict,
        original_error: str | None = None,
    ) -> None:
        super().__init__(message)
        self.mutation_details = dict(mutation_details)
        self.original_error = original_error
        self.must_fail_stop = True
        self.rollback_promised = False


def load_rows(path: Path) -> list[dict]:
    if not isinstance(path, Path):
        path = Path(path)
    raw = _read_bounded_bytes(path, max_bytes=MAX_HORIZONS_FILE_BYTES, label="horizons")
    if hashlib.sha256(raw).hexdigest() != HORIZONS_SHA256:
        raise ValueError("horizons SHA mismatch")
    doc = strict_json_loads(raw, max_bytes=MAX_HORIZONS_FILE_BYTES)
    if type(doc) is not dict:
        raise ValueError("horizons manifest must decode to an object")
    if doc.get("n") != EXPECTED_DEV_VIDEOS or doc.get("pairs") != EXPECTED_DEV_PAIRS:
        raise ValueError("dev manifest header counts do not match 50 videos / 147 pairs")
    if doc.get("scored") is not False:
        raise ValueError("dev manifest scored flag must be false")
    raw_mid_missing = doc.get("mid_missing")
    if type(raw_mid_missing) is not list or set(raw_mid_missing) != PREDECLARED_ABSENT_MID_VIDEOS:
        raise ValueError("dev manifest mid_missing does not match predeclared absent set")
    rows = doc.get("rows")
    if type(rows) is not list or len(rows) != EXPECTED_DEV_VIDEOS:
        raise ValueError("dev manifest is not 50 videos")

    train_set = set(train_ids())
    seen_videos: set[str] = set()
    observed_mid_missing: set[str] = set()
    pair_count = 0
    for row in rows:
        if type(row) is not dict:
            raise ValueError("dev manifest row must be an object")
        id_movi = row.get("id_movi")
        if type(id_movi) is not str or not id_movi:
            raise ValueError("dev row id_movi must be a nonempty str")
        if id_movi in seen_videos:
            raise ValueError("duplicate id_movi in dev manifest")
        if id_movi in train_set:
            raise ValueError("train/dev video ID overlap detected")
        seen_videos.add(id_movi)
        if row.get("scored") is not False:
            raise ValueError("dev row scored flag must be false")

        n_frames = require_exact_int(row.get("n"), "dev.row.n", minimum=2)
        current = require_exact_int(row.get("current"), "dev.row.current", minimum=0, maximum=n_frames - 2)
        history = row.get("history")
        if (
            type(history) is not list
            or not history
            or any(type(frame) is not int or frame < 0 or frame >= n_frames for frame in history)
            or history != sorted(set(history))
            or history[0] != 0
            or history[-1] != current
        ):
            raise ValueError("dev row causal history is malformed")

        row_pairs = 0
        for horizon in HORIZON_NAMES:
            target = row.get(horizon)
            if target is None:
                if horizon == "mid":
                    observed_mid_missing.add(id_movi)
                else:
                    raise ValueError(f"dev row {id_movi} silently omitted required horizon {horizon}")
                continue
            target_val = require_exact_int(target, f"dev.row.{horizon}", minimum=current + 1, maximum=n_frames - 1)
            if target_val in history:
                raise ValueError("dev row future target present in causal history")
            row_pairs += 1
            pair_count += 1
        if row_pairs == 0:
            raise ValueError("dev row has zero target horizons")

    if observed_mid_missing != PREDECLARED_ABSENT_MID_VIDEOS:
        raise ValueError("dev rows with missing mid do not match predeclared absent set")
    if pair_count != EXPECTED_DEV_PAIRS:
        raise ValueError("dev manifest total pairs is not 147")
    return rows


def load_keys(path: Path) -> set[tuple]:
    rows = load_rows(path)
    keys = []
    for row in rows:
        for horizon in HORIZON_NAMES:
            if row.get(horizon) is None:
                continue
            keys.append((str(row["id_movi"]), int(row["current"]), int(row[horizon]), horizon))
    if len(keys) != EXPECTED_DEV_PAIRS or len(set(keys)) != EXPECTED_DEV_PAIRS or len({key[0] for key in keys}) != EXPECTED_DEV_VIDEOS:
        raise ValueError("dev support is not 147 pairs over 50 videos")
    return set(keys)


def expected_dev_digests(rows: list[dict]) -> dict[tuple, str]:
    """Independently compute expected canonical forward digests from verified DEV manifest rows."""
    if type(rows) is not list or len(rows) != EXPECTED_DEV_VIDEOS:
        raise ValueError("expected_dev_digests requires exact 50 manifest rows")
    digests: dict[tuple, str] = {}
    for row in rows:
        if type(row) is not dict:
            raise ValueError("dev manifest row must be a dict")
        id_movi = row.get("id_movi")
        if type(id_movi) is not str or not id_movi:
            raise ValueError("dev row id_movi must be a nonempty str")
        n_frames = require_exact_int(row.get("n"), "dev.row.n", minimum=2)
        labels = list(range(n_frames))
        raw_history = row.get("history")
        if type(raw_history) is not list or not raw_history:
            raise ValueError("dev row history must be a nonempty list")
        history = [require_exact_int(frame, "dev.row.history", minimum=0, maximum=n_frames - 1) for frame in raw_history]
        current = require_exact_int(row.get("current"), "dev.row.current", minimum=0, maximum=n_frames - 2)
        for horizon in HORIZON_NAMES:
            if row.get(horizon) is None:
                continue
            target = require_exact_int(row.get(horizon), f"dev.row.{horizon}", minimum=current + 1, maximum=n_frames - 1)
            key = (id_movi, current, target, horizon)
            if key in digests:
                raise ValueError("duplicate key while building expected DEV digests")
            digests[key] = expected_synthetic_forward_digest(id_movi, labels, history, current, target, horizon)
    if len(digests) != EXPECTED_DEV_PAIRS:
        raise ValueError("expected DEV digests do not cover 147 pairs")
    return digests


def _load_loss_diagnostics():
    loss_path = Path(__file__).with_name("pinned_losses_v3.py")
    if hashlib.sha256(loss_path.read_bytes()).hexdigest() != LOSS_SHA256:
        raise ValueError("prepared loss SHA mismatch")
    from . import pinned_losses_v3 as module

    return module.LossConfig().eps, module.directional_cosine_per_sample, module.magnitude_log_ratio_per_sample


def _model_state_and_grad_digest(model) -> str:
    import torch

    hasher = hashlib.sha256(b"STCA-DEV-STATE-V1\x00")
    for name, param in model.named_parameters():
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\x00P")
        hasher.update(b"1" if param.requires_grad else b"0")
        p_cpu = param.detach().contiguous().cpu()
        hasher.update(str(p_cpu.dtype).encode("utf-8"))
        hasher.update(str(tuple(p_cpu.shape)).encode("utf-8"))
        p_u8 = p_cpu.to(torch.uint8) if p_cpu.dtype == torch.bool else p_cpu.view(torch.uint8)
        hasher.update(p_u8.numpy().tobytes())
        if param.grad is None:
            hasher.update(b"\x00G0")
        else:
            hasher.update(b"\x00G1")
            g_cpu = param.grad.detach().contiguous().cpu()
            g_u8 = g_cpu.to(torch.uint8) if g_cpu.dtype == torch.bool else g_cpu.view(torch.uint8)
            hasher.update(g_u8.numpy().tobytes())
    for name, buf in model.named_buffers():
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\x00B")
        b_cpu = buf.detach().contiguous().cpu()
        hasher.update(str(b_cpu.dtype).encode("utf-8"))
        hasher.update(str(tuple(b_cpu.shape)).encode("utf-8"))
        b_u8 = b_cpu.to(torch.uint8) if b_cpu.dtype == torch.bool else b_cpu.view(torch.uint8)
        hasher.update(b_u8.numpy().tobytes())
    return hasher.hexdigest()


def _rng_snapshots_equal(left: tuple, right: tuple) -> bool:
    import numpy
    import torch

    py_l, np_l, th_l = left
    py_r, np_r, th_r = right
    if py_l != py_r:
        return False
    if (
        np_l[0] != np_r[0]
        or not bool(numpy.array_equal(np_l[1], np_r[1]))
        or np_l[2:] != np_r[2:]
    ):
        return False
    return bool(torch.equal(th_l, th_r))


def _assert_uniform_module_mode(model) -> bool:
    modes = {bool(mod.training) for _, mod in model.named_modules()}
    if not modes:
        return bool(model.training)
    if len(modes) != 1:
        raise ValueError("mixed module training/eval modes are not supported by DEV evaluator")
    return next(iter(modes))


def evaluate_with_diagnostics(
    model,
    items: list[dict],
    expected: set[tuple],
    *,
    expected_digests: dict[tuple, str] | None = None,
    forbidden_video_ids: set[str] | None = None,
    require_external_expectations: bool = False,
) -> dict:
    """Evaluate exact 147 DEV pairs under frozen `stage-a-dev-v1` with mutation checks on success/exception."""
    import torch

    if type(expected) is not set or len(expected) != EXPECTED_DEV_PAIRS:
        raise ValueError("expected DEV key set must contain exact 147 pairs")
    expected_videos = {key[0] for key in expected}
    if len(expected_videos) != EXPECTED_DEV_VIDEOS:
        raise ValueError("expected DEV key set must span exact 50 videos")

    pinned_train_ids = set(train_ids())
    if require_external_expectations:
        if expected_digests is None or forbidden_video_ids is None:
            raise ValueError("canonical DEV evaluation requires external expected_digests and forbidden_video_ids")
        if set(forbidden_video_ids) != pinned_train_ids:
            raise ValueError("forbidden_video_ids disagrees with pinned 400-video train split")

    forbidden = pinned_train_ids if forbidden_video_ids is None else set(forbidden_video_ids)
    if not expected_videos.isdisjoint(forbidden) or not expected_videos.isdisjoint(pinned_train_ids):
        raise ValueError("DEV evaluation contains train/forbidden video IDs")

    if expected_digests is not None:
        if type(expected_digests) is not dict or set(expected_digests.keys()) != expected:
            raise ValueError("expected_digests keys do not match frozen 147 DEV pairs")

    if type(items) is not list or len(items) != EXPECTED_DEV_PAIRS:
        raise ValueError("dev items length must equal 147 with zero silent drops or duplicates")

    # Pre-verify and freeze ALL 147 items before ANY model.forward invocation.
    seen: set[tuple] = set()
    verified_entries: list[tuple[tuple, dict, dict, str]] = []
    for item in items:
        if type(item) is not dict:
            raise ValueError("dev item must be a dict")
        key = (item.get("id_movi"), item.get("current"), item.get("target"), item.get("horizon"))
        if key not in expected or key in seen:
            raise ValueError("dev pair is missing from or duplicated relative to frozen 147")
        if key[0] in forbidden:
            raise ValueError("dev item belongs to forbidden train split")
        seen.add(key)
        exp_digest = (
            expected_digests[key]
            if expected_digests is not None
            else expected_synthetic_forward_digest(
                item.get("id_movi"),
                item.get("ids_frames"),
                item.get("history"),
                item.get("current"),
                item.get("target"),
                item.get("horizon"),
            )
        )
        frozen_kwargs = prepare_verified_forward_call(
            item,
            expected_digest=exp_digest,
            context="dev",
            verify_canonical_supervision=True,
        )
        assert_no_target_leak(item, verify_canonical_oracle=True)
        verified_entries.append((key, frozen_kwargs, item["supervision"], exp_digest))

    if seen != expected:
        raise ValueError("dev coverage mismatch prior to evaluation")

    eps, dir_cosine_fn, mag_ratio_fn = _load_loss_diagnostics()
    was_training = _assert_uniform_module_mode(model)
    had_last_sem = hasattr(model, "_last_semantic_context")
    prev_last_sem = getattr(model, "_last_semantic_context", None)
    if isinstance(prev_last_sem, torch.Tensor):
        prev_last_sem = prev_last_sem.detach().clone()
    pre_state_digest = _model_state_and_grad_digest(model)
    pre_rng = capture_cpu_rng_snapshot()

    grouped: dict[str, list[float]] = {}
    grouped_no_motion: dict[str, list[float]] = {}
    horizon_values: dict[str, list[float]] = {h: [] for h in HORIZON_NAMES}
    horizon_no_motion: dict[str, list[float]] = {h: [] for h in HORIZON_NAMES}

    dir_loss_local: list[float] = []
    dir_loss_anchor: list[float] = []
    dir_valid_tgt_local = 0
    dir_valid_tgt_anchor = 0

    raw_cos_local: list[float] = []
    raw_cos_anchor: list[float] = []
    raw_cos_undef_local = 0
    raw_cos_undef_anchor = 0

    mag_loss_local: list[float] = []
    mag_loss_anchor: list[float] = []

    pred_norms_local: list[float] = []
    pred_norms_anchor: list[float] = []
    tgt_norms_local: list[float] = []
    tgt_norms_anchor: list[float] = []
    norm_ratios_local: list[float] = []
    norm_ratios_anchor: list[float] = []
    zero_pred_local = 0
    zero_pred_anchor = 0
    zero_tgt_local = 0
    zero_tgt_anchor = 0

    sorted_entries = sorted(verified_entries, key=lambda entry: entry[0])
    digest_chain = hashlib.sha256()
    for _, _, _, exp_digest in sorted_entries:
        digest_chain.update(exp_digest.encode("ascii"))

    eval_exc: Exception | None = None
    try:
        model.eval()
        with torch.no_grad():
            for key, frozen_kwargs, supervision, _ in verified_entries:
                output = model.forward(**frozen_kwargs)
                if type(output) is not dict or "delta_local" not in output or "delta_anchor" not in output:
                    raise ValueError("model.forward output missing delta_local/delta_anchor")
                pred_local = output["delta_local"]
                pred_anchor = output["delta_anchor"]
                for pred_name, pred_tensor in (("delta_local", pred_local), ("delta_anchor", pred_anchor)):
                    if (
                        not isinstance(pred_tensor, torch.Tensor)
                        or pred_tensor.device.type != "cpu"
                        or pred_tensor.dtype != torch.float32
                        or tuple(pred_tensor.shape) != (1, WIDTH)
                        or not bool(torch.isfinite(pred_tensor).all().item())
                    ):
                        raise ValueError(f"nonfinite or malformed DEV output {pred_name}")

                tgt_local = supervision["target_delta_local"]
                tgt_anchor = supervision["target_delta_anchor"]
                local = torch.nn.functional.smooth_l1_loss(pred_local, tgt_local, beta=BETA)
                anchor = torch.nn.functional.smooth_l1_loss(pred_anchor, tgt_anchor, beta=BETA)
                if not bool(torch.isfinite(local).all().item()) or not bool(torch.isfinite(anchor).all().item()):
                    raise ValueError("nonfinite dev loss")
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

                scalars = (pair_loss, pair_nm, d_loc, d_anc, m_loc, m_anc, p_norm_loc, p_norm_anc, t_norm_loc, t_norm_anc)
                if not all(math.isfinite(x) for x in scalars):
                    raise ValueError("nonfinite DEV diagnostic scalar")

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
                        if not math.isfinite(cos_loc):
                            raise ValueError("nonfinite raw cosine local")
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
                        if not math.isfinite(cos_anc):
                            raise ValueError("nonfinite raw cosine anchor")
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
        post_modes = {bool(mod.training) for _, mod in model.named_modules()}
        mode_ok = (post_modes == {was_training}) if post_modes else (bool(model.training) == was_training)
        state_ok = _model_state_and_grad_digest(model) == pre_state_digest
        rng_ok = _rng_snapshots_equal(pre_rng, capture_cpu_rng_snapshot())
        if not (mode_ok and state_ok and rng_ok):
            raise EvaluatorMutationError(
                "DEV evaluation mutated model mode, parameters, buffers, gradients, or CPU RNG state",
                mutation_details={
                    "mode_restored": mode_ok,
                    "state_unchanged": state_ok,
                    "rng_unchanged": rng_ok,
                },
                original_error=None if eval_exc is None else str(eval_exc),
            ) from eval_exc

    if eval_exc is not None:
        raise eval_exc

    if set(grouped.keys()) != expected_videos or len(grouped) != EXPECTED_DEV_VIDEOS:
        raise ValueError("dev coverage mismatch")

    ordered_vids = sorted(grouped.keys())
    per_video_macro = {vid: sum(grouped[vid]) / len(grouped[vid]) for vid in ordered_vids}
    per_video_no_motion = {vid: sum(grouped_no_motion[vid]) / len(grouped_no_motion[vid]) for vid in ordered_vids}
    per_video_delta = {vid: per_video_macro[vid] - per_video_no_motion[vid] for vid in ordered_vids}
    per_video_counts = {vid: len(grouped[vid]) for vid in ordered_vids}

    macro = sum(per_video_macro.values()) / float(EXPECTED_DEV_VIDEOS)
    no_motion_macro = sum(per_video_no_motion.values()) / float(EXPECTED_DEV_VIDEOS)
    if not math.isfinite(macro) or not math.isfinite(no_motion_macro):
        raise ValueError("nonfinite dev aggregate scalar")

    raw_cos_loc_mean = (sum(raw_cos_local) / len(raw_cos_local)) if raw_cos_local else None
    raw_cos_anc_mean = (sum(raw_cos_anchor) / len(raw_cos_anchor)) if raw_cos_anchor else None
    raw_cos_mean = (
        0.5 * (raw_cos_loc_mean + raw_cos_anc_mean)
        if raw_cos_loc_mean is not None and raw_cos_anc_mean is not None
        else None
    )
    ratio_loc_mean = (sum(norm_ratios_local) / len(norm_ratios_local)) if norm_ratios_local else None
    ratio_anc_mean = (sum(norm_ratios_anchor) / len(norm_ratios_anchor)) if norm_ratios_anchor else None

    per_horizon_macro = {h: (sum(horizon_values[h]) / len(horizon_values[h])) for h in HORIZON_NAMES}
    per_horizon_no_motion = {h: (sum(horizon_no_motion[h]) / len(horizon_no_motion[h])) for h in HORIZON_NAMES}
    per_horizon_delta = {h: per_horizon_macro[h] - per_horizon_no_motion[h] for h in HORIZON_NAMES}

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
        "temporal_axis_semantics": DEV_TEMPORAL_AXIS_SEMANTICS,
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
        "per_video_no_motion_macro": per_video_no_motion,
        "per_video_delta_vs_no_motion": per_video_delta,
        "per_video_counts": per_video_counts,
        "per_horizon_macro": per_horizon_macro,
        "per_horizon_no_motion_macro": per_horizon_no_motion,
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
            "raw_norm_ratio_mean_local": ratio_loc_mean,
            "raw_norm_ratio_mean_anchor": ratio_anc_mean,
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


def collect_singleton_dev_pair_records(
    model,
    items: list[dict],
    expected: set[tuple],
    *,
    expected_digests: dict[tuple, str] | None = None,
    forbidden_video_ids: set[str] | None = None,
    require_external_expectations: bool = False,
) -> list[dict]:
    """Evaluate exact 147 DEV pairs via verified singleton forward calls and return per-pair records."""
    import torch

    if type(expected) is not set or len(expected) != EXPECTED_DEV_PAIRS:
        raise ValueError("expected DEV key set must contain exact 147 pairs")
    expected_videos = {key[0] for key in expected}
    if len(expected_videos) != EXPECTED_DEV_VIDEOS:
        raise ValueError("expected DEV key set must span exact 50 videos")

    pinned_train_ids = set(train_ids())
    if require_external_expectations:
        if expected_digests is None or forbidden_video_ids is None:
            raise ValueError("canonical DEV evaluation requires external expected_digests and forbidden_video_ids")
        if set(forbidden_video_ids) != pinned_train_ids:
            raise ValueError("forbidden_video_ids disagrees with pinned 400-video train split")

    forbidden = pinned_train_ids if forbidden_video_ids is None else set(forbidden_video_ids)
    if not expected_videos.isdisjoint(forbidden) or not expected_videos.isdisjoint(pinned_train_ids):
        raise ValueError("DEV evaluation contains train/forbidden video IDs")

    if expected_digests is not None:
        if type(expected_digests) is not dict or set(expected_digests.keys()) != expected:
            raise ValueError("expected_digests keys do not match frozen 147 DEV pairs")

    if type(items) is not list or len(items) != EXPECTED_DEV_PAIRS:
        raise ValueError("dev items length must equal 147 with zero silent drops or duplicates")

    seen: set[tuple] = set()
    verified_entries: list[tuple[tuple, dict, dict, str]] = []
    for item in items:
        if type(item) is not dict:
            raise ValueError("dev item must be a dict")
        key = (item.get("id_movi"), item.get("current"), item.get("target"), item.get("horizon"))
        if key not in expected or key in seen:
            raise ValueError("dev pair is missing from or duplicated relative to frozen 147")
        if key[0] in forbidden:
            raise ValueError("dev item belongs to forbidden train split")
        seen.add(key)
        exp_digest = (
            expected_digests[key]
            if expected_digests is not None
            else expected_synthetic_forward_digest(
                item.get("id_movi"),
                item.get("ids_frames"),
                item.get("history"),
                item.get("current"),
                item.get("target"),
                item.get("horizon"),
            )
        )
        frozen_kwargs = prepare_verified_forward_call(
            item,
            expected_digest=exp_digest,
            context="dev",
            verify_canonical_supervision=True,
        )
        assert_no_target_leak(item, verify_canonical_oracle=True)
        verified_entries.append((key, frozen_kwargs, item["supervision"], exp_digest))

    was_training = _assert_uniform_module_mode(model)
    had_last_sem = hasattr(model, "_last_semantic_context")
    prev_last_sem = getattr(model, "_last_semantic_context", None)
    if isinstance(prev_last_sem, torch.Tensor):
        prev_last_sem = prev_last_sem.detach().clone()
    pre_state_digest = _model_state_and_grad_digest(model)
    pre_rng = capture_cpu_rng_snapshot()

    pair_records: list[dict] = []
    eval_exc: Exception | None = None
    try:
        model.eval()
        with torch.no_grad():
            for key, frozen_kwargs, supervision, exp_digest in verified_entries:
                output = model.forward(**frozen_kwargs)
                if type(output) is not dict or "delta_local" not in output or "delta_anchor" not in output:
                    raise ValueError("model.forward output missing delta_local/delta_anchor")
                pred_local = output["delta_local"]
                pred_anchor = output["delta_anchor"]
                for pred_name, pred_tensor in (("delta_local", pred_local), ("delta_anchor", pred_anchor)):
                    if (
                        not isinstance(pred_tensor, torch.Tensor)
                        or pred_tensor.device.type != "cpu"
                        or pred_tensor.dtype != torch.float32
                        or tuple(pred_tensor.shape) != (1, WIDTH)
                        or not bool(torch.isfinite(pred_tensor).all().item())
                    ):
                        raise ValueError(f"nonfinite or malformed DEV output {pred_name}")
                tgt_local = supervision["target_delta_local"]
                tgt_anchor = supervision["target_delta_anchor"]
                local = torch.nn.functional.smooth_l1_loss(pred_local, tgt_local, beta=BETA)
                anchor = torch.nn.functional.smooth_l1_loss(pred_anchor, tgt_anchor, beta=BETA)
                pair_loss = float(LOCAL_WEIGHT * float(local.item()) + ANCHOR_WEIGHT * float(anchor.item()))
                zeros = torch.zeros_like(tgt_local)
                nm_local = float(torch.nn.functional.smooth_l1_loss(zeros, tgt_local, beta=BETA).item())
                nm_anchor = float(torch.nn.functional.smooth_l1_loss(zeros, tgt_anchor, beta=BETA).item())
                pair_nm = LOCAL_WEIGHT * nm_local + ANCHOR_WEIGHT * nm_anchor
                if not math.isfinite(pair_loss) or not math.isfinite(pair_nm):
                    raise ValueError("nonfinite DEV pair scalar")
                id_movi, current, target, horizon = key
                pair_records.append(
                    {
                        "id_movi": id_movi,
                        "current": current,
                        "target": target,
                        "horizon": horizon,
                        "pair_loss": pair_loss,
                        "pair_no_motion": pair_nm,
                        "forward_digest": exp_digest,
                    }
                )
    except Exception as exc:
        eval_exc = exc
    finally:
        model.train(was_training)
        if had_last_sem:
            model._last_semantic_context = prev_last_sem
        elif hasattr(model, "_last_semantic_context"):
            delattr(model, "_last_semantic_context")
        post_modes = {bool(mod.training) for _, mod in model.named_modules()}
        mode_ok = (post_modes == {was_training}) if post_modes else (bool(model.training) == was_training)
        state_ok = _model_state_and_grad_digest(model) == pre_state_digest
        rng_ok = _rng_snapshots_equal(pre_rng, capture_cpu_rng_snapshot())
        if not (mode_ok and state_ok and rng_ok):
            raise EvaluatorMutationError(
                "DEV singleton pair collection mutated model mode, state, grads, or CPU RNG state",
                mutation_details={"mode_restored": mode_ok, "state_unchanged": state_ok, "rng_unchanged": rng_ok},
                original_error=None if eval_exc is None else str(eval_exc),
            ) from eval_exc
    if eval_exc is not None:
        raise eval_exc
    return pair_records


def aggregate_partitioned_dev_losses(pair_records: list[dict], partitions: list[list[int]]) -> dict:
    """Aggregate fixed singleton DEV pair losses over an arbitrary partition/ordering of the 147 predictions."""
    if type(pair_records) is not list or len(pair_records) != EXPECTED_DEV_PAIRS:
        raise ValueError("pair_records must contain exact 147 pairs")
    if type(partitions) is not list or not partitions:
        raise ValueError("partitions must be a nonempty list of index chunks")

    seen_keys: set[tuple] = set()
    for rec in pair_records:
        key = (rec["id_movi"], rec["current"], rec["target"], rec["horizon"])
        if key in seen_keys:
            raise ValueError("duplicate pair in pair_records")
        seen_keys.add(key)

    seen_indices: list[int] = []
    video_loss_sum: dict[str, float] = {}
    video_nm_sum: dict[str, float] = {}
    video_counts: dict[str, int] = {}
    for chunk in partitions:
        if type(chunk) not in (list, tuple) or not chunk:
            raise ValueError("each partition chunk must be nonempty")
        chunk_loss_sum: dict[str, float] = {}
        chunk_nm_sum: dict[str, float] = {}
        chunk_counts: dict[str, int] = {}
        for raw_idx in chunk:
            idx = require_exact_int(raw_idx, "partition.index", minimum=0, maximum=EXPECTED_DEV_PAIRS - 1)
            seen_indices.append(idx)
            rec = pair_records[idx]
            vid = rec["id_movi"]
            chunk_loss_sum[vid] = chunk_loss_sum.get(vid, 0.0) + float(rec["pair_loss"])
            chunk_nm_sum[vid] = chunk_nm_sum.get(vid, 0.0) + float(rec["pair_no_motion"])
            chunk_counts[vid] = chunk_counts.get(vid, 0) + 1
        for vid in sorted(chunk_counts):
            video_loss_sum[vid] = video_loss_sum.get(vid, 0.0) + chunk_loss_sum[vid]
            video_nm_sum[vid] = video_nm_sum.get(vid, 0.0) + chunk_nm_sum[vid]
            video_counts[vid] = video_counts.get(vid, 0) + chunk_counts[vid]

    if len(seen_indices) != EXPECTED_DEV_PAIRS or set(seen_indices) != set(range(EXPECTED_DEV_PAIRS)):
        raise ValueError("partitions must cover 0..146 exactly once")
    if len(video_counts) != EXPECTED_DEV_VIDEOS or sum(video_counts.values()) != EXPECTED_DEV_PAIRS:
        raise ValueError("aggregated predictions do not span 50 videos / 147 pairs")

    ordered_vids = sorted(video_counts.keys())
    per_video_macro = {vid: video_loss_sum[vid] / video_counts[vid] for vid in ordered_vids}
    per_video_nm = {vid: video_nm_sum[vid] / video_counts[vid] for vid in ordered_vids}
    macro = sum(per_video_macro.values()) / float(EXPECTED_DEV_VIDEOS)
    nm_macro = sum(per_video_nm.values()) / float(EXPECTED_DEV_VIDEOS)
    return {
        "macro": float(macro),
        "no_motion_macro": float(nm_macro),
        "video_count": EXPECTED_DEV_VIDEOS,
        "pair_count": EXPECTED_DEV_PAIRS,
        "per_video_macro": per_video_macro,
        "per_video_counts": {vid: video_counts[vid] for vid in ordered_vids},
        "partition_sizes": [len(chunk) for chunk in partitions],
        "singleton_forward_only": True,
        "batched_forward_claimed": False,
    }


def evaluate(
    model,
    items: list[dict],
    expected: set[tuple],
    *,
    expected_digests: dict[tuple, str] | None = None,
    forbidden_video_ids: set[str] | None = None,
    require_external_expectations: bool = False,
) -> float:
    return float(
        evaluate_with_diagnostics(
            model,
            items,
            expected,
            expected_digests=expected_digests,
            forbidden_video_ids=forbidden_video_ids,
            require_external_expectations=require_external_expectations,
        )["macro"]
    )


def build_h2_contrast_pairs(rows: list[dict]) -> list[dict]:
    """Build secondary deferred H2 matched-history contrast pairs (source/design only; NOT run in Stage A)."""
    if type(rows) is not list:
        raise ValueError("rows must be a list")
    pairs: list[dict] = []
    for row in rows:
        history = list(row["history"])
        if len(history) <= 2:
            continue
        shortened_history = [history[0], history[-1]]
        if shortened_history == history:
            continue
        id_movi = str(row["id_movi"])
        labels = list(range(int(row["n"])))
        current = int(row["current"])
        for horizon in HORIZON_NAMES:
            if row.get(horizon) is None:
                continue
            target = int(row[horizon])
            full_item = synthetic_item(id_movi, labels, history, current, target, horizon)
            short_item = synthetic_item(id_movi, labels, shortened_history, current, target, horizon)
            pairs.append(
                {
                    "id_movi": id_movi,
                    "horizon": horizon,
                    "full_item": full_item,
                    "shortened_item": short_item,
                }
            )
    return pairs


def evaluate_h2_contrast(model, contrast_pairs: list[dict]) -> dict:
    """Prepared deferred secondary H2 contrast (`shortened - full`); NOT invoked on primary checkpoint selection."""
    import torch

    if not contrast_pairs:
        raise ValueError("contrast_pairs must contain at least one eligible distinct-history pair")
    was_training = _assert_uniform_module_mode(model)
    had_last_sem = hasattr(model, "_last_semantic_context")
    prev_last_sem = getattr(model, "_last_semantic_context", None)
    if isinstance(prev_last_sem, torch.Tensor):
        prev_last_sem = prev_last_sem.detach().clone()
    pre_state = _model_state_and_grad_digest(model)
    pre_rng = capture_cpu_rng_snapshot()

    full_by_video: dict[str, list[float]] = {}
    short_by_video: dict[str, list[float]] = {}
    eval_exc: Exception | None = None
    try:
        model.eval()
        with torch.no_grad():
            for entry in contrast_pairs:
                vid = entry["id_movi"]
                f_item = entry["full_item"]
                s_item = entry["shortened_item"]
                f_kwargs = prepare_verified_forward_call(
                    f_item,
                    expected_digest=f_item["forward_input_sha256"],
                    context="h2_full",
                    verify_canonical_supervision=True,
                )
                s_kwargs = prepare_verified_forward_call(
                    s_item,
                    expected_digest=s_item["forward_input_sha256"],
                    context="h2_shortened",
                    verify_canonical_supervision=True,
                )
                f_out = model.forward(**f_kwargs)
                s_out = model.forward(**s_kwargs)
                f_val = float(
                    LOCAL_WEIGHT
                    * float(torch.nn.functional.smooth_l1_loss(f_out["delta_local"], f_item["supervision"]["target_delta_local"], beta=BETA).item())
                    + ANCHOR_WEIGHT
                    * float(torch.nn.functional.smooth_l1_loss(f_out["delta_anchor"], f_item["supervision"]["target_delta_anchor"], beta=BETA).item())
                )
                s_val = float(
                    LOCAL_WEIGHT
                    * float(torch.nn.functional.smooth_l1_loss(s_out["delta_local"], s_item["supervision"]["target_delta_local"], beta=BETA).item())
                    + ANCHOR_WEIGHT
                    * float(torch.nn.functional.smooth_l1_loss(s_out["delta_anchor"], s_item["supervision"]["target_delta_anchor"], beta=BETA).item())
                )
                full_by_video.setdefault(vid, []).append(f_val)
                short_by_video.setdefault(vid, []).append(s_val)
    except Exception as exc:
        eval_exc = exc
    finally:
        model.train(was_training)
        if had_last_sem:
            model._last_semantic_context = prev_last_sem
        elif hasattr(model, "_last_semantic_context"):
            delattr(model, "_last_semantic_context")
        post_modes = {bool(mod.training) for _, mod in model.named_modules()}
        mode_ok = (post_modes == {was_training}) if post_modes else (bool(model.training) == was_training)
        state_ok = _model_state_and_grad_digest(model) == pre_state
        rng_ok = _rng_snapshots_equal(pre_rng, capture_cpu_rng_snapshot())
        if not (mode_ok and state_ok and rng_ok):
            raise EvaluatorMutationError(
                "H2 contrast evaluation mutated model mode, state, grads, or CPU RNG",
                mutation_details={"mode_restored": mode_ok, "state_unchanged": state_ok, "rng_unchanged": rng_ok},
                original_error=None if eval_exc is None else str(eval_exc),
            ) from eval_exc
    if eval_exc is not None:
        raise eval_exc

    vids = sorted(full_by_video.keys())
    full_macro = sum(sum(full_by_video[v]) / len(full_by_video[v]) for v in vids) / len(vids)
    short_macro = sum(sum(short_by_video[v]) / len(short_by_video[v]) for v in vids) / len(vids)
    return {
        "h2_policy": H2_POLICY,
        "eligible_video_count": len(vids),
        "eligible_pair_count": len(contrast_pairs),
        "full_macro": float(full_macro),
        "shortened_macro": float(short_macro),
        "shortened_minus_full": float(short_macro - full_macro),
    }
