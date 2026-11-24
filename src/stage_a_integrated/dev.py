"""Hardened 147-key DEV evaluator proposal with C3 forward-input binding. Not DS-frozen."""

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
)
from .exposure import strict_json_loads
from .forward_binding import (
    WIDTH,
    expected_synthetic_forward_digest,
    prepare_verified_forward_call,
)
from .plan import require_exact_int, train_ids
from .samples import assert_no_target_leak

BETA = 0.05
METRIC_NAME = "dev_delta_huber_macro"
METRIC_STATUS = "PROPOSED_NOT_DS_FROZEN"
EXPECTED_DEV_VIDEOS = 50
EXPECTED_DEV_PAIRS = 147
HORIZON_NAMES = ("near", "mid", "far")


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
    rows = doc.get("rows")
    if type(rows) is not list or len(rows) != EXPECTED_DEV_VIDEOS:
        raise ValueError("dev manifest is not 50 videos")

    train_set = set(train_ids())
    seen_videos: set[str] = set()
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
                continue
            target_val = require_exact_int(target, f"dev.row.{horizon}", minimum=current + 1, maximum=n_frames - 1)
            if target_val in history:
                raise ValueError("dev row future target present in causal history")
            row_pairs += 1
            pair_count += 1
        if row_pairs == 0:
            raise ValueError("dev row has zero target horizons")

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


def evaluate_with_diagnostics(
    model,
    items: list[dict],
    expected: set[tuple],
    *,
    expected_digests: dict[tuple, str] | None = None,
    forbidden_video_ids: set[str] | None = None,
) -> dict:
    """Evaluate exact 147 DEV pairs with pre-forward binding, mode/state/RNG guards, and diagnostics."""
    import torch

    if type(expected) is not set or len(expected) != EXPECTED_DEV_PAIRS:
        raise ValueError("expected DEV key set must contain exact 147 pairs")
    expected_videos = {key[0] for key in expected}
    if len(expected_videos) != EXPECTED_DEV_VIDEOS:
        raise ValueError("expected DEV key set must span exact 50 videos")
    forbidden = set(train_ids()) if forbidden_video_ids is None else set(forbidden_video_ids)
    if not expected_videos.isdisjoint(forbidden):
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
        frozen_kwargs = prepare_verified_forward_call(item, expected_digest=exp_digest, context="dev")
        assert_no_target_leak(item, verify_canonical_oracle=True)
        verified_entries.append((key, frozen_kwargs, item["supervision"], exp_digest))

    if seen != expected:
        raise ValueError("dev coverage mismatch prior to evaluation")

    eps, dir_cosine_fn, mag_ratio_fn = _load_loss_diagnostics()
    was_training = bool(model.training)
    had_last_sem = hasattr(model, "_last_semantic_context")
    prev_last_sem = getattr(model, "_last_semantic_context", None)
    if isinstance(prev_last_sem, torch.Tensor):
        prev_last_sem = prev_last_sem.detach().clone()
    pre_state_digest = _model_state_and_grad_digest(model)
    pre_rng = capture_cpu_rng_snapshot()

    grouped: dict[str, list[float]] = {}
    grouped_no_motion: dict[str, list[float]] = {}
    horizon_values: dict[str, list[float]] = {h: [] for h in HORIZON_NAMES}
    direction_values: list[float] = []
    magnitude_values: list[float] = []
    digest_chain = hashlib.sha256()

    try:
        model.eval()
        with torch.no_grad():
            for key, frozen_kwargs, supervision, exp_digest in verified_entries:
                digest_chain.update(exp_digest.encode("ascii"))
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
                pair_loss = float(0.5 * (float(local.item()) + float(anchor.item())))

                zeros = torch.zeros_like(tgt_local)
                nm_local = float(torch.nn.functional.smooth_l1_loss(zeros, tgt_local, beta=BETA).item())
                nm_anchor = float(torch.nn.functional.smooth_l1_loss(zeros, tgt_anchor, beta=BETA).item())
                pair_nm = 0.5 * (nm_local + nm_anchor)

                d_loc = float(dir_cosine_fn(pred_local, tgt_local, eps=eps).item())
                d_anc = float(dir_cosine_fn(pred_anchor, tgt_anchor, eps=eps).item())
                m_loc = float(mag_ratio_fn(pred_local, tgt_local, eps=eps).item())
                m_anc = float(mag_ratio_fn(pred_anchor, tgt_anchor, eps=eps).item())
                if not all(math.isfinite(x) for x in (pair_loss, pair_nm, d_loc, d_anc, m_loc, m_anc)):
                    raise ValueError("nonfinite DEV diagnostic scalar")

                id_movi, _, _, horizon = key
                grouped.setdefault(id_movi, []).append(pair_loss)
                grouped_no_motion.setdefault(id_movi, []).append(pair_nm)
                horizon_values[horizon].append(pair_loss)
                direction_values.append(0.5 * (d_loc + d_anc))
                magnitude_values.append(0.5 * (m_loc + m_anc))
    finally:
        model.train(was_training)
        if had_last_sem:
            model._last_semantic_context = prev_last_sem
        elif hasattr(model, "_last_semantic_context"):
            delattr(model, "_last_semantic_context")

    if bool(model.training) != was_training:
        raise RuntimeError("DEV evaluation failed to restore model training mode")
    if _model_state_and_grad_digest(model) != pre_state_digest:
        raise RuntimeError("DEV evaluation mutated model parameters, buffers, or gradients")
    if not _rng_snapshots_equal(pre_rng, capture_cpu_rng_snapshot()):
        raise RuntimeError("DEV evaluation mutated CPU RNG state")

    if set(grouped.keys()) != expected_videos or len(grouped) != EXPECTED_DEV_VIDEOS:
        raise ValueError("dev coverage mismatch")

    per_video_macro = {vid: sum(vals) / len(vals) for vid, vals in grouped.items()}
    per_video_counts = {vid: len(vals) for vid, vals in grouped.items()}
    macro = sum(per_video_macro.values()) / float(EXPECTED_DEV_VIDEOS)
    no_motion_macro = sum(sum(vals) / len(vals) for vals in grouped_no_motion.values()) / float(EXPECTED_DEV_VIDEOS)
    direction_mean = sum(direction_values) / float(len(direction_values))
    magnitude_mean = sum(magnitude_values) / float(len(magnitude_values))
    if not all(math.isfinite(v) for v in (macro, no_motion_macro, direction_mean, magnitude_mean)):
        raise ValueError("nonfinite dev aggregate scalar")

    return {
        "metric_name": METRIC_NAME,
        "metric_status": METRIC_STATUS,
        "beta": BETA,
        "macro": float(macro),
        "no_motion_macro": float(no_motion_macro),
        "direction_mean": float(direction_mean),
        "magnitude_mean": float(magnitude_mean),
        "video_count": len(grouped),
        "pair_count": EXPECTED_DEV_PAIRS,
        "per_video_macro": per_video_macro,
        "per_video_counts": per_video_counts,
        "per_horizon_macro": {h: (sum(v) / len(v)) for h, v in horizon_values.items()},
        "per_horizon_counts": {h: len(v) for h, v in horizon_values.items()},
        "dev_forward_digests_sha256": digest_chain.hexdigest(),
    }


def evaluate(
    model,
    items: list[dict],
    expected: set[tuple],
    *,
    expected_digests: dict[tuple, str] | None = None,
    forbidden_video_ids: set[str] | None = None,
) -> float:
    return float(
        evaluate_with_diagnostics(
            model,
            items,
            expected,
            expected_digests=expected_digests,
            forbidden_video_ids=forbidden_video_ids,
        )["macro"]
    )
