"""Paper 3 diagnostic evaluator. Synthetic records only. No dataset I/O."""
from __future__ import annotations

import math


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-8 or nb < 1e-8:
        return float("nan")
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def directional_gap(pred: list[float], true: list[float], distractor: list[float]) -> float:
    return cosine(pred, true) - cosine(pred, distractor)


def h1_status(gaps: list[float], mean_change_gap: float, coverage: int, total: int = 50) -> str:
    defined = [g for g in gaps if g == g]
    if len(defined) < 40 or coverage < 40:
        return "INCONCLUSIVE"
    median = sorted(defined)[len(defined) // 2]
    if median - mean_change_gap >= 0.02 and median >= 0.02:
        return "PASS"
    return "FAIL"


def paired_permutation(scores: list[float], pairs: list[tuple[int, int]], rng) -> str:
    if len(scores) < 40 or len(pairs) != len(scores):
        return "INCONCLUSIVE"
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    gaps = [scores[i] - scores[j] for i, j in shuffled]
    if any(g != g for g in gaps):
        return "INVALID"
    median = sorted(gaps)[len(gaps) // 2]
    return "PASS" if median >= 0.02 else "FAIL"


def permutation_null(true_scores: list[float], shuffled_scores: list[float]) -> str:
    if len(true_scores) != len(shuffled_scores) or len(true_scores) < 40:
        return "INCONCLUSIVE"
    if any(x != x or y != y for x, y in zip(true_scores, shuffled_scores)):
        return "INVALID"
    gap = sorted(a - b for a, b in zip(true_scores, shuffled_scores))[len(true_scores) // 2]
    return "PASS" if gap >= 0.02 else "FAIL"


def h2_status(teacher: float, free: float, motion: float, copy_motion: float = 0.0) -> str:
    if teacher != teacher or free != free:
        return "INCONCLUSIVE"
    if motion < 0.02 or motion <= copy_motion:
        return "FAIL"
    if teacher - free >= 0.02:
        return "PASS"
    return "FAIL"
