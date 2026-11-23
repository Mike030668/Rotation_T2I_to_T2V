"""One future-authorized synthetic path. Default CLI still raises."""

from __future__ import annotations

from pathlib import Path

from .authorization import ExecutionNotAuthorized, load
from .checkpoints import publish
from .dev import evaluate, load_keys
from .model_adapter import load_class
from .samples import synthetic_item
from .trainer import optimizer_step


def run_synthetic(release_path: str, trust_anchor_path: str, expected: dict, config: dict) -> dict:
    release = load(release_path, trust_anchor_path, expected)
    if release.get("scope") != "stage_a_synthetic_preflight":
        raise ExecutionNotAuthorized("scope is not the synthetic preflight")
    if int(config["max_updates"]) > int(release.get("max_updates")):
        raise ExecutionNotAuthorized("config exceeds the released budget")
    import torch

    model_cls, config_cls = load_class()
    model = model_cls(config_cls())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    keys = load_keys(Path(config["horizons_path"]))
    completed = 0
    best = None
    root = Path(release.get("output_root"))
    while completed < int(config["max_updates"]):
        item = synthetic_item("1", [0, 2, 9, 10, 40], [0, 2], 9, 10, "near")
        completed += optimizer_step(model, [item], opt, float(config["clip"]))
        if completed in config["dev_events"]:
            value = evaluate(model, config["dev_items"], keys)
            model.train()
            payload = f"{completed}:{value}".encode("utf-8")
            publish(root, f"u{completed}", "LAST", payload)
            if best is None or value < best[1]:
                best = (completed, value)
                publish(root, f"u{completed}", "BEST", payload)
    return {"completed_updates": completed, "best_step": None if best is None else best[0], "observed": False}


def main() -> None:
    raise ExecutionNotAuthorized("NOT_RUN")
