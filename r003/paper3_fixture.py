"""Synthetic fixtures for the Paper 3 diagnostic. No dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper3_diag import directional_gap, h1_status, h2_status


def main() -> int:
    assert directional_gap([1, 0], [1, 0], [0, 1]) > 0.9
    assert directional_gap([0, 0], [1, 0], [0, 1]) != directional_gap([0, 0], [1, 0], [0, 1])
    gaps = [0.05] * 40 + [float("nan")] * 10
    assert h1_status(gaps, 0.0, 40) == "PASS"
    assert h1_status([0.01] * 40, 0.0, 40) == "FAIL"
    assert h1_status([0.05] * 39, 0.0, 39) == "INCONCLUSIVE"
    assert h2_status(0.10, 0.07, 0.2) == "PASS"
    assert h2_status(0.10, 0.09, 0.2) == "FAIL"
    assert h2_status(0.10, 0.07, 0.0) == "FAIL"
    import random
    from paper3_diag import paired_permutation, permutation_null
    assert h1_status([0.05] * 40, 0.10, 40) == "FAIL"
    from paper3_diag import video_record_status
    assert video_record_status({"video_id": 1, "horizon": 4, "pred_delta": [1, 0], "true_delta": [1, 0]}) == "DEFINED"
    assert video_record_status({"video_id": 1, "horizon": 4, "pred_delta": [1, 0], "true_delta": [1]}) == "INVALID"
    rng = random.Random(2)
    pairs = [(i, (i + 1) % 40) for i in range(40)]
    first = paired_permutation([0.2] * 40, pairs, random.Random(2))
    second = paired_permutation([0.2] * 40, pairs, random.Random(3))
    assert first in {"PASS", "FAIL"} and second in {"PASS", "FAIL"}
    assert permutation_null([0.1] * 40, [0.0] * 40) == "PASS"
    assert permutation_null([0.0] * 40, [0.0] * 40) == "FAIL"
    print("PAPER3_FIXTURE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
