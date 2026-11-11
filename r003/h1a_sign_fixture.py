"""Sign-contract fixture. No model. No dataset."""


def delta(anchor: list[float], later: list[float]) -> list[float]:
    return [a - b for a, b in zip(anchor, later)]


def main() -> int:
    image = delta([2.0, 0.0], [1.0, 0.0])
    pred = delta([2.0, 0.0], [1.0, 0.0])
    assert image == pred == [1.0, 0.0]
    # The rejected mixed sign is not the contract.
    mixed = [b - a for a, b in zip([2.0, 0.0], [1.0, 0.0])]
    assert mixed == [-1.0, 0.0]
    assert mixed != image
    print("H1A_SIGN_FIXTURE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
