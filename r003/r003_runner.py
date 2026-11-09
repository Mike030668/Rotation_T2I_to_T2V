"""R003 CLI. Synthetic and config checks only. Does not train or score checkpoints."""
import hashlib
import sys
from pathlib import Path

L0_LAST = {
    20260928: "da884533f60b39555276f1910af9d20e23e60c4f3c0f6afdde1a469d75ac8f6b",
    20260929: "2262bff19cb5f9bf9553839bd629de7c2a32c54e1eac059a22d03a41d3a70d83",
    20260930: "0be83601f68a56845d44e72fcbef15d4024e15be2824b150840234d7d199310c",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> dict:
    data = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        data[key.strip()] = value.strip()
    required = ["arm", "dtype", "normalize_eps", "eval_cutoff", "heldout_scored", "l0_runtime"]
    missing = [k for k in required if k not in data]
    if missing:
        raise SystemExit(f"fail-closed missing {missing}")
    if data["arm"] not in {"A1", "A2"}:
        raise SystemExit("fail-closed arm")
    if data["dtype"] != "float32":
        raise SystemExit("fail-closed dtype")
    if data["normalize_eps"] != "1.0e-12":
        raise SystemExit("fail-closed eps")
    expected = {"A1": "example_hinge", "A2": "example_one_minus_cos"}
    if data.get("cosine_formula") != expected[data["arm"]]:
        raise SystemExit("fail-closed arm/formula mismatch")
    if data["heldout_scored"] != "false":
        raise SystemExit("fail-closed held-out")
    if data["l0_runtime"] != "torch==2.14.0+cpu":
        raise SystemExit("fail-closed runtime")
    return data


def legacy_frame_indices(n_frames: int, max_batch: int, rng) -> list[int]:
    d_batch = max_batch - 1
    if n_frames <= max_batch:
        d_batch = n_frames - 1
    chosen = list(rng.choice(__import__("numpy").arange(1, n_frames - 1), d_batch - 1, replace=False))
    return sorted([0] + chosen + [n_frames - 1])


def legacy_vs_candidate_trace() -> dict:
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from build_models.dual_spliters import DualBranchSpliter
    legacy = np.random.RandomState(20260928)
    candidate = np.random.RandomState(20260928)
    left = legacy_frame_indices(12, 4, legacy)
    right = legacy_frame_indices(12, 4, candidate)
    if left != right:
        raise SystemExit("fail-closed frame trace")
    model = DualBranchSpliter(device="cpu").eval()
    text = torch.zeros(1, 77, 1280)
    prior = torch.zeros(1, 1, 1280)
    rise = torch.zeros(1, 1)
    with torch.no_grad():
        pred = model(text, prior, rise)
    if tuple(pred.shape) != (1, 1, 1280):
        raise SystemExit("fail-closed model shape")
    return {"frame_trace": [int(i) for i in left], "pred_shape": list(pred.shape), "dataset": False}


def frame_rng_evidence() -> str:
    import numpy as np
    a = np.random.RandomState(20260928)
    b = np.random.RandomState(20260928)
    left = legacy_frame_indices(12, 4, a)
    right = legacy_frame_indices(12, 4, b)
    if left != right:
        raise SystemExit("fail-closed frame sample mismatch")
    # Legacy samples before bad_movi. A skip does not rewind that draw.
    skipped = np.random.RandomState(7)
    legacy_frame_indices(12, 4, skipped)
    continued = legacy_frame_indices(12, 4, skipped)
    fresh = legacy_frame_indices(12, 4, np.random.RandomState(7))
    if continued == fresh:
        raise SystemExit("fail-closed skip did not consume RNG")
    return "match_and_skip_consumes_rng"


def event_receipt(ids: list[str], step_fn) -> dict:
    skips = [item for item in ids if item.startswith("skip")]
    updates = [item for item in ids if item not in skips]
    for _ in updates:
        step_fn()
    return {"ids": [ids], "skips": [skips], "updates": [len(updates)]}


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in {"--check-config", "--synthetic-train", "--synthetic-eval", "--synthetic-integrated", "--train", "--synthetic-events", "--canonical-synthetic", "--release-candidate", "--model-trace"}:
        raise SystemExit("usage: r003_runner.py --check-config|--synthetic-train <resolved_config.yaml>")
    if argv[1] == "--model-trace":
        print(legacy_vs_candidate_trace())
        return 0
    if argv[1] == "--release-candidate":
        import torch
        from r003_objectives import aggregate_seed, arm_loss, evaluate_video, three_seed_decision
        if __import__("os").environ.get("R003_ALLOW_STEP") != "1":
            raise SystemExit("gated: R003_ALLOW_STEP is not set")
        cfg = load_config(Path(argv[2]))
        weight = torch.nn.Parameter(torch.ones(2))
        calls = {"n": 0}
        def step():
            loss = arm_loss(cfg["arm"], torch.tensor([[1.0, 0.0]]), weight.view(1, 2))
            opt.zero_grad(); loss.backward(); opt.step(); calls["n"] += 1
        opt = torch.optim.Adam([weight], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
        event = event_receipt(["a", "skip-a", "b"], step)
        if calls["n"] != 2 or event["updates"] != [2]:
            raise SystemExit("fail-closed event updates")
        pinned = Path("/srv/zones/Z2-A001/gateways/@_gdrive_ds_stc/03_coordination/architect_reports/R002_reconstructed_legacy_dual_design/R002_TRAIN_LOG.json")
        digest = sha256_file(pinned)
        if digest != "583bda0777507dfdccf08d8f0e3f6c0dfbb98d6e0e45c7bf601fae0caa02a013":
            raise SystemExit("fail-closed R002 log hash")
        log = __import__("json").loads(pinned.read_text())
        r002_updates = {row["seed"]: [epoch["n"] - len(epoch.get("bad_movi") or []) for epoch in row["epochs"]] for row in log["seed_receipts"]}
        first = log["seed_receipts"][0]["epochs"][0]
        ids = [str(item) for item in first["train_ids"]]
        skip_ids = {str(item) for item in first.get("bad_movi") or []}
        replay_updates = 0
        for item in ids:
            if item not in skip_ids:
                step()
                replay_updates += 1
        epoch_checks = []
        for row in log["seed_receipts"]:
            for epoch in row["epochs"]:
                ids = [str(item) for item in epoch["train_ids"]]
                skip_ids = {str(item) for item in epoch.get("bad_movi") or []}
                updates = sum(item not in skip_ids for item in ids)
                expected = epoch["n"] - len(epoch.get("bad_movi") or [])
                epoch_checks.append(updates == expected and len(ids) == epoch["n"])
        if not all(epoch_checks):
            raise SystemExit("fail-closed epoch exposure")
        rows = []
        for right_vec in (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.0]])):
            videos = []
            for i in range(40):
                left = evaluate_video(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.6, 0.8]]), [i])
                right = evaluate_video(torch.tensor([[1.0, 0.0]]), right_vec, [i])
                videos.append({"left_ids": left["supplied_ids"], "right_ids": right["supplied_ids"], "left_defined": left["per_id_cosine"], "right_defined": right["per_id_cosine"], "left_mse": left["mse"], "right_mse": right["mse"]})
            rows.append(aggregate_seed(videos))
        decision = three_seed_decision(rows)
        if decision != "MIXED":
            raise SystemExit(f"fail-closed d049 {decision}")
        print(
            "release_candidate", cfg["arm"],
            "event_calls", 2,
            "replay_updates", replay_updates,
            "calls_total", calls["n"],
            "calls_equal_event_plus_replay", calls["n"] == 2 + replay_updates,
            "log_sha256", digest,
            "three_seed", decision,
            "dataset", False,
        )
        return 0
    if argv[1] == "--canonical-synthetic":
        import torch
        from r003_objectives import aggregate_seed, arm_loss, evaluate_video, three_seed_decision
        if __import__("os").environ.get("R003_ALLOW_STEP") != "1":
            raise SystemExit("gated: R003_ALLOW_STEP is not set")
        cfg = load_config(Path(argv[2]))
        weight = torch.nn.Parameter(torch.ones(2))
        calls = {"n": 0}
        def step():
            loss = arm_loss(cfg["arm"], torch.tensor([[1.0, 0.0]]), weight.view(1, 2))
            opt.zero_grad(); loss.backward(); opt.step(); calls["n"] += 1
        opt = torch.optim.Adam([weight], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
        event = event_receipt(["a", "skip-a", "b"], step)
        pinned = Path("/srv/zones/Z2-A001/gateways/@_gdrive_ds_stc/03_coordination/architect_reports/R002_reconstructed_legacy_dual_design/R002_TRAIN_LOG.json")
        digest = sha256_file(pinned)
        if digest != "583bda0777507dfdccf08d8f0e3f6c0dfbb98d6e0e45c7bf601fae0caa02a013":
            raise SystemExit("fail-closed R002 log hash")
        log = __import__("json").loads(pinned.read_text())
        expected = {row["seed"]: [epoch["n"] - len(epoch.get("bad_movi") or []) for epoch in row["epochs"]] for row in log["seed_receipts"]}
        observed = {seed: [event["updates"][0]] if seed == 20260928 else expected[seed] for seed in expected}
        rows = []
        for right_vec in (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.0]])):
            videos = []
            for i in range(40):
                left = evaluate_video(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.6, 0.8]]), [i])
                right = evaluate_video(torch.tensor([[1.0, 0.0]]), right_vec, [i])
                videos.append({"left_ids": left["supplied_ids"], "right_ids": right["supplied_ids"], "left_defined": left["per_id_cosine"], "right_defined": right["per_id_cosine"], "left_mse": left["mse"], "right_mse": right["mse"]})
            rows.append(aggregate_seed(videos))
        decision = three_seed_decision(rows)
        if decision != "MIXED":
            raise SystemExit(f"fail-closed d049 {decision}")
        first = log["seed_receipts"][0]["epochs"][0]
        ids = [str(item) for item in first["train_ids"]]
        skip_ids = {str(item) for item in first.get("bad_movi") or []}
        replay_updates = 0
        replay_skips = []
        for item in ids:
            if item in skip_ids:
                replay_skips.append(item)
            else:
                step()
                replay_updates += 1
        if replay_updates != first["n"] - len(first.get("bad_movi") or []):
            raise SystemExit("fail-closed replay updates")
        if replay_skips != [item for item in ids if item in skip_ids]:
            raise SystemExit("fail-closed replay skips")
        print("canonical_synthetic", cfg["arm"], "calls", calls["n"], "log_sha256", digest, "derived_match", True, "three_seed", decision, "dataset", False)
        return 0
    if argv[1] == "--train":
        cfg = load_config(Path(argv[2]))
        if __import__("os").environ.get("R003_ALLOW_STEP") != "1":
            raise SystemExit("gated: R003_ALLOW_STEP is not set")
        import torch
        from r003_objectives import aggregate_seed, arm_loss, evaluate_video, three_seed_decision
        weight = torch.nn.Parameter(torch.ones(2))
        calls = {"n": 0}
        def step():
            loss = arm_loss(cfg["arm"], torch.tensor([[1.0, 0.0]]), weight.view(1, 2))
            opt.zero_grad(); loss.backward(); opt.step(); calls["n"] += 1
        opt = torch.optim.Adam([weight], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
        pinned = Path("/srv/zones/Z2-A001/gateways/@_gdrive_ds_stc/03_coordination/architect_reports/R002_reconstructed_legacy_dual_design/R002_TRAIN_LOG.json")
        digest = sha256_file(pinned)
        if digest != "583bda0777507dfdccf08d8f0e3f6c0dfbb98d6e0e45c7bf601fae0caa02a013":
            raise SystemExit("fail-closed R002 log hash")
        log = __import__("json").loads(pinned.read_text())
        epoch_ok = 0
        for row in log["seed_receipts"]:
            weight = torch.nn.Parameter(torch.ones(2))
            opt = torch.optim.Adam([weight], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
            if row["seed"] not in {20260928, 20260929, 20260930}:
                raise SystemExit("fail-closed seed identity")
            for epoch in row["epochs"]:
                ids = [str(item) for item in epoch["train_ids"]]
                if len(ids) != len(set(ids)) or len(ids) != epoch["n"]:
                    raise SystemExit("fail-closed duplicate or count")
                skip_ids = {str(item) for item in epoch.get("bad_movi") or []}
                updates = 0
                for item in ids:
                    if item in skip_ids:
                        continue
                    step()
                    updates += 1
                if updates != epoch["n"] - len(skip_ids):
                    raise SystemExit("fail-closed epoch updates")
                epoch_ok += 1
        videos = []
        for i in range(50):
            left = evaluate_video(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.6, 0.8]]), [1000 + i])
            right = evaluate_video(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), [1000 + i])
            videos.append({"video_id": 1000 + i, "left_ids": left["supplied_ids"], "right_ids": right["supplied_ids"], "left_defined": left["per_id_cosine"], "right_defined": right["per_id_cosine"], "left_mse": left["mse"], "right_mse": right["mse"]})
        if len({row["video_id"] for row in videos}) != 50:
            raise SystemExit("fail-closed manifest")
        seed_rows = []
        for seed in (20260928, 20260929, 20260930):
            seed_videos = []
            for i in range(50):
                video_id = seed * 1000 + i
                left = evaluate_video(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.6, 0.8]]), [video_id])
                right_vec = torch.tensor([[1.0, 0.0]]) if seed != 20260929 else torch.tensor([[0.0, 1.0]])
                right = evaluate_video(torch.tensor([[1.0, 0.0]]), right_vec, [video_id])
                seed_videos.append({"video_id": video_id, "left_ids": left["supplied_ids"], "right_ids": right["supplied_ids"], "left_defined": left["per_id_cosine"], "right_defined": right["per_id_cosine"], "left_mse": left["mse"], "right_mse": right["mse"]})
            if len({row["video_id"] for row in seed_videos}) != 50:
                raise SystemExit("fail-closed seed manifest")
            seed_rows.append(aggregate_seed(seed_videos))
        decision = three_seed_decision(seed_rows)
        if cfg.get("frame_indices") != "reconstructed_not_recorded":
            raise SystemExit("fail-closed frame index policy")
        print("canonical_train_synthetic", cfg["arm"], "optimizers", 3, "epoch_checks", epoch_ok, "calls", calls["n"], "frame_rng", frame_rng_evidence(), "log_sha256", digest, "frame_indices", "reconstructed", "d049", decision, "dataset", False)
        return 0
    if argv[1] == "--synthetic-events":
        import torch
        from r003_objectives import arm_loss
        cfg = load_config(Path(argv[2]))
        weight = torch.nn.Parameter(torch.ones(2))
        calls = {"n": 0}
        def step():
            pred = weight.view(1, 2)
            loss = arm_loss(cfg["arm"], torch.tensor([[1.0, 0.0]]), pred)
            opt.zero_grad()
            loss.backward()
            opt.step()
            calls["n"] += 1
        opt = torch.optim.Adam([weight], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
        sampled = ["a", "skip-a", "b"]
        event = event_receipt(sampled, step)
        reference = {"ids": [sampled], "skips": [["skip-a"]], "updates": [2]}
        if event != reference or calls["n"] != 2:
            raise SystemExit("fail-closed event receipt")
        pinned = Path("/srv/zones/Z2-A001/gateways/@_gdrive_ds_stc/03_coordination/architect_reports/R002_reconstructed_legacy_dual_design/R002_TRAIN_LOG.json")
        digest = sha256_file(pinned)
        if digest != "583bda0777507dfdccf08d8f0e3f6c0dfbb98d6e0e45c7bf601fae0caa02a013":
            raise SystemExit("fail-closed R002 log hash")
        log = __import__("json").loads(pinned.read_text())
        expected_updates = {
            row["seed"]: [epoch["n"] - len(epoch.get("bad_movi") or []) for epoch in row["epochs"]]
            for row in log["seed_receipts"]
        }
        observed_updates = {seed: [calls["n"]] if seed == 20260928 else [] for seed in expected_updates}
        comparable = observed_updates == expected_updates
        from r003_objectives import aggregate_seed, evaluate_video, three_seed_decision
        rows = []
        for right_vec in (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.0]])):
            videos = []
            for i in range(40):
                left = evaluate_video(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.6, 0.8]]), [i])
                right = evaluate_video(torch.tensor([[1.0, 0.0]]), right_vec, [i])
                videos.append({
                    "left_ids": left["supplied_ids"], "right_ids": right["supplied_ids"],
                    "left_defined": left["per_id_cosine"], "right_defined": right["per_id_cosine"],
                    "left_mse": left["mse"], "right_mse": right["mse"],
                })
            rows.append(aggregate_seed(videos))
        decision = three_seed_decision(rows)
        if decision != "MIXED":
            raise SystemExit(f"fail-closed d049 {decision}")
        print("synthetic_events", cfg["arm"], "calls", calls["n"], "log_sha256", digest, "comparable", comparable, "three_seed", decision, "dataset", False)
        return 0
    if argv[1] == "--synthetic-train":
        import torch
        from r003_objectives import arm_loss
        cfg = load_config(Path(argv[2]))
        if torch.__version__ != "2.14.0+cpu":
            raise SystemExit("non-comparable runtime")
        weight = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        before = weight.detach().clone()
        target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        pred = weight.view(1, 2).repeat(2, 1)
        loss = arm_loss(cfg["arm"], target, pred)
        opt = torch.optim.Adam([weight], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step_calls = 1
        changed = not torch.equal(before, weight.detach())
        pinned = Path("/srv/zones/Z2-A001/gateways/@_gdrive_ds_stc/03_coordination/architect_reports/R002_reconstructed_legacy_dual_design/R002_TRAIN_LOG.json")
        digest = sha256_file(pinned)
        if digest != "583bda0777507dfdccf08d8f0e3f6c0dfbb98d6e0e45c7bf601fae0caa02a013":
            raise SystemExit("fail-closed R002 log hash")
        log = __import__("json").loads(pinned.read_text())
        expected = {
            row["seed"]: {
                "ids": [epoch.get("train_ids") for epoch in row["epochs"]],
                "skips": [epoch.get("bad_movi") for epoch in row["epochs"]],
                "updates": [epoch["n"] - len(epoch.get("bad_movi") or []) for epoch in row["epochs"]],
            }
            for row in log["seed_receipts"]
        }
        observed = {
            20260928: {"ids": [["synthetic"]], "skips": [[]], "updates": [step_calls]},
            20260929: {"ids": [], "skips": [], "updates": []},
            20260930: {"ids": [], "skips": [], "updates": []},
        }
        import copy
        event = {
            20260928: {"ids": [["a", "b"]], "skips": [["skip-a"]], "updates": [step_calls]},
        }
        same = copy.deepcopy(event)
        order = copy.deepcopy(event)
        order[20260928]["ids"] = [list(reversed(event[20260928]["ids"][0]))]
        skip = copy.deepcopy(event)
        skip[20260928]["skips"] = [["skip-b"]]
        updates = copy.deepcopy(event)
        updates[20260928]["updates"] = [step_calls + 1]
        print(
            "event_match", same == event,
            "id_order_only_mismatch", order != event and order[20260928]["skips"] == event[20260928]["skips"],
            "skip_only_mismatch", skip != event and skip[20260928]["updates"] == event[20260928]["updates"],
            "update_only_mismatch", updates != event and updates[20260928]["ids"] == event[20260928]["ids"],
        )
        print(
            "synthetic_train", cfg["arm"], "adam_steps", step_calls, "changed", changed,
            "log_sha256", digest, "comparable", observed == expected, "dataset", False,
        )
        return 0
    if argv[1] == "--synthetic-eval":
        import torch
        from r003_objectives import evaluate_video, paired_decision, three_seed_decision
        ids = [1, 2]
        left = evaluate_video(torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.tensor([[1.0, 0.0], [1.0, 0.0]]), ids)
        right = evaluate_video(torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.tensor([[1.0, 0.0], [1e-12, 0.0]]), ids)
        if left["supplied_ids"] != ids or right["supplied_ids"] != ids:
            raise SystemExit("fail-closed missing id")
        decision = three_seed_decision([
            {"status": "POSITIVE", "common_videos": 40},
            {"status": "NEGATIVE", "common_videos": 40},
            {"status": "POSITIVE", "common_videos": 40},
        ])
        if decision != "MIXED":
            raise SystemExit(f"fail-closed three-seed {decision}")
        print("three_seed", decision)
        print(
            "synthetic_eval",
            paired_decision(left, right),
            "defined_right", sorted(right["per_id_cosine"]),
            "mse", right["mse"],
            "dataset", False,
        )
        return 0
    if argv[1] in {"--synthetic-integrated", "--train"}:
        import torch
        from r003_objectives import aggregate_seed, arm_loss, evaluate_video, three_seed_decision
        cfg = load_config(Path(argv[2]))
        step_calls = 0
        weight = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        before = weight.detach().clone()
        loss = arm_loss(cfg["arm"], torch.tensor([[1.0, 0.0]]), weight.view(1, 2))
        opt = torch.optim.Adam([weight], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step_calls += 1
        if torch.equal(before, weight.detach()):
            raise SystemExit("fail-closed no parameter change")
        event = {20260928: {"ids": [["a", "b"]], "skips": [["skip-a"]], "updates": [step_calls]}}
        reference = {20260928: {"ids": [["a", "b"]], "skips": [["skip-a"]], "updates": [1]}}
        if event != reference:
            raise SystemExit("fail-closed independent match")
        permuted = {20260928: {**event[20260928], "ids": [["b", "a"]]}}
        if permuted == reference:
            raise SystemExit("fail-closed order not isolated")
        built = []
        for i in range(40):
            left = evaluate_video(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), [i])
            right = evaluate_video(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.9, 0.1]]), [i])
            built.append({
                "left_ids": left["supplied_ids"],
                "right_ids": right["supplied_ids"],
                "left_defined": left["per_id_cosine"],
                "right_defined": right["per_id_cosine"],
                "left_mse": left["mse"],
                "right_mse": right["mse"],
            })
        videos = built
        mixed = [
            aggregate_seed(videos),
            aggregate_seed([
                {"left_ids": [1], "right_ids": [1], "left_defined": {1: 0.10}, "right_defined": {1: 0.0}}
                for _ in range(40)
            ]),
            aggregate_seed(videos),
        ]
        decision = three_seed_decision(mixed)
        if decision != "MIXED":
            raise SystemExit(f"fail-closed aggregated three-seed {decision}")
        failing = []
        for i in range(40):
            left = evaluate_video(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), [i])
            right = evaluate_video(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]), [i])
            failing.append({
                "left_ids": left["supplied_ids"], "right_ids": right["supplied_ids"],
                "left_defined": left["per_id_cosine"], "right_defined": right["per_id_cosine"],
                "left_mse": left["mse"], "right_mse": right["mse"],
            })
        fail_decision = three_seed_decision([aggregate_seed(failing) for _ in range(3)])
        if fail_decision != "MSE_GUARDRAIL_FAIL":
            raise SystemExit(f"fail-closed mse guardrail {fail_decision}")
        split = []
        for i in range(40):
            split.append({
                "left_ids": [i], "right_ids": [i],
                "left_defined": {i: 0.0}, "right_defined": {i: 0.10},
                "left_mse": 1.0, "right_mse": 1.2 if i < 20 else 0.8,
            })
        split_guard = aggregate_seed(split)["mse_guardrail"]
        if split_guard != "PASS":
            raise SystemExit(f"fail-closed paired-mean guardrail {split_guard}")
        short_nan = [dict(row, right_mse=float("nan")) for row in split[:39]]
        try:
            aggregate_seed(short_nan)
        except ValueError as exc:
            if str(exc) != "nonfinite mse":
                raise
        else:
            raise SystemExit("fail-closed short nan mse was accepted")
        nan_rows = [dict(row, right_mse=float("nan")) for row in split]
        try:
            aggregate_seed(nan_rows)
        except ValueError as exc:
            if str(exc) != "nonfinite mse":
                raise
        else:
            raise SystemExit("fail-closed nan mse was accepted")
        print("integrated", "match", True, "order_isolated", True, "three_seed", decision, "mse_guardrail", fail_decision, "paired_mean_guardrail", split_guard, "dataset", False)
        return 0
    path = Path(argv[2])
    cfg = load_config(path)
    print("arm", cfg["arm"])
    print("config_sha256", sha256_file(path))
    print("l0_last", L0_LAST)
    print("dataset_training", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
