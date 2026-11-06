"""R003 objectives. Separate from the R002 runner. No dataset I/O."""
import torch
import torch.nn.functional as F

DTYPE = torch.float32
NORMALIZE_EPS = 1e-12
EVAL_CUTOFF = 1e-8
COVERAGE_MIN = 40


def weighted_mse(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    target = target.to(DTYPE)
    pred = pred.to(DTYPE)
    return 0.5 * ((target - pred) ** 2).mean(dim=0).sum()


def example_cosine(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    target = target.to(DTYPE)
    pred = pred.to(DTYPE)
    return (F.normalize(target, dim=1, eps=NORMALIZE_EPS) * F.normalize(pred, dim=1, eps=NORMALIZE_EPS)).sum(dim=1)


def arm_loss(arm: str, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    mse = weighted_mse(target, pred)
    if arm == "L0":
        raise ValueError("L0 stays in the pinned R002 loss; this path does not reimplement it")
    cos = example_cosine(target, pred)
    d = target.shape[1]
    if arm == "A1":
        term = torch.clamp(cos, min=0).mean()
    elif arm == "A2":
        term = (1 - cos).mean()
    else:
        raise ValueError(arm)
    return mse + 0.5 * d * term


def evaluate_video(target: torch.Tensor, pred: torch.Tensor, example_ids: list[int]) -> dict:
    if not torch.isfinite(target).all() or not torch.isfinite(pred).all():
        raise ValueError("nonfinite record")
    if target.shape != pred.shape or len(example_ids) != target.shape[0]:
        raise ValueError("missing or mismatched record")
    t_norm = torch.linalg.vector_norm(target, dim=1)
    p_norm = torch.linalg.vector_norm(pred, dim=1)
    defined = (t_norm >= EVAL_CUTOFF) & (p_norm >= EVAL_CUTOFF)
    mse = ((target - pred) ** 2).mean()
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("duplicate example id")
    scores = {}
    if bool(defined.any()):
        cos = example_cosine(target[defined], pred[defined])
        scores = {example_ids[i]: float(v) for i, v in zip([i for i, flag in enumerate(defined.tolist()) if flag], cos.tolist())}
    return {
        "status": "UNDEFINED" if not scores else "DEFINED",
        "mse": float(mse),
        "per_id_cosine": scores,
        "supplied_ids": list(example_ids),
    }


def require_same_ids(expected: list[int], left: dict, right: dict) -> None:
    got = set(left["per_id_cosine"]) | set()
    # Missing record is not undefined cosine. Both arms must be called with the same IDs.
    if left.get("supplied_ids") != expected or right.get("supplied_ids") != expected:
        raise ValueError("missing id")
    _ = got


def paired_delta(left: dict, right: dict) -> tuple[float, float]:
    common = set(left["per_id_cosine"]) & set(right["per_id_cosine"])
    if not common:
        raise ValueError("empty common support")
    delta = sum(right["per_id_cosine"][i] - left["per_id_cosine"][i] for i in common) / len(common)
    return delta, left["mse"]


def paired_decision(left: dict, right: dict) -> str:
    common = set(left["per_id_cosine"]) & set(right["per_id_cosine"])
    if not common:
        return "INCONCLUSIVE"
    cosine_delta, mse_ref = paired_delta(left, right)
    if mse_ref == 0 and right["mse"] > 0:
        return "MSE_GUARDRAIL_FAIL"
    if mse_ref > 0 and right["mse"] > mse_ref * 1.10:
        return "MSE_GUARDRAIL_FAIL"
    if abs(cosine_delta) < 0.02:
        return "BELOW_THRESHOLD"
    return "POSITIVE" if cosine_delta > 0 else "NEGATIVE"


def aggregate_seed(videos: list[dict]) -> dict:
    def finite(value):
        return value is not None and value == value and value not in (float("inf"), float("-inf"))
    for video in videos:
        if ("left_mse" in video and not finite(video.get("left_mse"))) or ("right_mse" in video and not finite(video.get("right_mse"))):
            raise ValueError("nonfinite mse")
    deltas = []
    for video in videos:
        if video["left_ids"] != video["right_ids"]:
            raise ValueError("missing id")
        common = set(video["left_defined"]) & set(video["right_defined"])
        if not common:
            continue
        deltas.append(sum(video["right_defined"][i] - video["left_defined"][i] for i in common) / len(common))
    if len(deltas) < 40:
        return {"status": "INCONCLUSIVE", "common_videos": len(deltas), "mse_guardrail": "ABSENT"}
    delta = sum(deltas) / len(deltas)
    status = "BELOW_THRESHOLD" if abs(delta) < 0.02 else ("POSITIVE" if delta > 0 else "NEGATIVE")
    paired = [video for video in videos if set(video["left_defined"]) & set(video["right_defined"])]
    if any(video.get("left_mse") is None or video.get("right_mse") is None for video in paired):
        guard = "ABSENT"
    else:
        left_mean = sum(video["left_mse"] for video in paired) / len(paired)
        right_mean = sum(video["right_mse"] for video in paired) / len(paired)
        guard = "FAIL" if (left_mean == 0 and right_mean > 0) or (left_mean > 0 and right_mean > left_mean * 1.10) else "PASS"
    return {"status": status, "common_videos": len(deltas), "delta": delta, "mse_guardrail": guard}


def three_seed_decision(seed_rows: list[dict]) -> str:
    if len(seed_rows) != 3:
        return "INCONCLUSIVE"
    if any(row["common_videos"] < 40 for row in seed_rows):
        return "INCONCLUSIVE"
    signs = [row["status"] for row in seed_rows]
    if any(sign not in {"POSITIVE", "NEGATIVE"} for sign in signs):
        return "MIXED"
    if len(set(signs)) == 1 and any(row.get("mse_guardrail") != "PASS" for row in seed_rows):
        return "MSE_GUARDRAIL_FAIL"
    if len(set(signs)) == 1:
        return "SUPPORTED_DIRECTIONAL_EFFECT"
    return "MIXED"
