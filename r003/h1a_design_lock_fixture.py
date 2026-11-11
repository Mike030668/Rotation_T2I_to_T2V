"""Execution-free H1a protocol checks. No model. No dataset."""


def caption_family_status(families: list[str]) -> str:
    if not families or any(not item for item in families):
        return "INCONCLUSIVE"
    return "DEFINED"


def negative_pair_ok(video_id: str, negative_id: str, caption: str, negative_caption: str, horizon: int, negative_horizon: int) -> bool:
    return video_id != negative_id and caption != negative_caption and horizon == negative_horizon


def cohort_excludes(new_ids: list[str], scored_ids: list[str]) -> bool:
    return set(new_ids).isdisjoint(scored_ids)


def main() -> int:
    assert caption_family_status(["", "a"]) == "INCONCLUSIVE"
    assert caption_family_status(["cat", "dog"]) == "DEFINED"
    assert negative_pair_ok("v1", "v2", "cat", "dog", 4, 4)
    assert not negative_pair_ok("v1", "v1", "cat", "dog", 4, 4)
    assert not negative_pair_ok("v1", "v2", "cat", "cat", 4, 4)
    assert not negative_pair_ok("v1", "v2", "cat", "dog", 4, 2)
    assert cohort_excludes(["a", "b"], ["c"])
    assert not cohort_excludes(["a", "b"], ["b"])
    print("H1A_DESIGN_LOCK_FIXTURE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
