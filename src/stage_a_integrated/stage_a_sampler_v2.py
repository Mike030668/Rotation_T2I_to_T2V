"""Video/epoch keyed sampler. Selected indices are original frame indices."""
import hashlib
import json


def keyed_seed(video_id: str, epoch: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{epoch}:{video_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def sample(labels, video_id: str, epoch: int, seed: int = 17):
    n = len(labels)
    if n < 2:
        raise SystemExit("too short")
    state = keyed_seed(video_id, epoch, seed)
    k = 2 if n == 2 else 2 + (state % min(3, n - 1))
    chosen = {0, n - 1}
    pool = list(range(1, n - 1))
    for index in pool:
        if len(chosen) >= k:
            break
        state = (state * 6364136223846793005 + 1) & ((1 << 64) - 1)
        if state % max(1, len(pool)) == index % max(1, len(pool)):
            chosen.add(index)
    while len(chosen) < min(k, n):
        state = (state * 6364136223846793005 + 1) & ((1 << 64) - 1)
        chosen.add(1 + state % (n - 2))
    indices = tuple(sorted(chosen))
    history = {}
    for pos, current in enumerate(indices[:-1]):
        history[current] = indices[: pos + 1]
    return indices, history


def main() -> int:
    labels = [0, 2, 9, 10, 40]
    a, hist_a = sample(labels, "4", 0)
    b, _ = sample(labels, "4", 1)
    c, _ = sample(labels, "8", 0)
    if a == b or a == c:
        raise SystemExit("sampler did not vary")
    for current, prefix in hist_a.items():
        if any(index > current for index in prefix):
            raise SystemExit("future in history")
    print(json.dumps({
        "epoch0": a,
        "epoch1": b,
        "other_video": c,
        "history_ok": True,
        "optimizer": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
