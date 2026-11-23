"""One future-authorized synthetic path. Default CLI still raises."""

from __future__ import annotations

from pathlib import Path

from .authorization import ExecutionNotAuthorized, load
from .checkpoints import pack, publish, unpack
from .dev import evaluate, load_keys, load_rows
from .exposure import require_item, require_records
from .model_adapter import load_class
from .plan import dev_items, next_batch
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
    rows = load_rows(Path(config["horizons_path"]))
    keys = load_keys(Path(config["horizons_path"]))
    completed = 0
    epoch = 0
    cursor = 0
    best = None
    macros = {}
    events = []
    exposure = []
    if config.get("resume_bytes"):
        restored = resume(config["resume_bytes"], model, opt)
        completed = int(restored["completed"])
        epoch = int(restored["epoch"])
        cursor = int(restored["cursor"])
        best = restored["best"]
        events = [int(event) for event in restored["events"]]
        macros = {int(step): float(value) for step, value in restored["macros"].items()}
        exposure = list(restored["exposure"])
    root = Path(release.get("output_root"))
    items = dev_items(rows)
    while completed < int(config["max_updates"]):
        if cursor + int(config["batch"]) > 400:
            epoch += 1
            cursor = 0
        batch, cursor, chosen = next_batch(int(config["seed"]), epoch, cursor, int(config["batch"]))
        for item in batch:
            require_item(item)
        completed += optimizer_step(model, batch, opt, float(config["clip"]))
        exposure.append({
            "schema": "exposure-v1",
            "completed": completed,
            "videos": chosen,
            "selected_ordinals": [item["selected_ordinals"] for item in batch],
            "selected_labels": [item["selected_labels"] for item in batch],
            "current_ordinal": [item["current_ordinal"] for item in batch],
            "target_ordinal": [item["target_ordinal"] for item in batch],
            "current": [item["current"] for item in batch],
            "target": [item["target"] for item in batch],
            "horizon": [item["horizon"] for item in batch],
            "epoch": epoch,
            "cursor": cursor,
            "seed": int(config["seed"]),
            "provider": "SYNTHETIC_APPARATUS",
        })
        if completed in config["dev_events"]:
            value = evaluate(model, items, keys)
            model.train()
            macros[completed] = value
            if best is None or value < best[1]:
                best = (completed, value)
            events.append(completed)
            history = [] if best is None else [{"step": best[0], "macro": best[1]}]
            payload = pack(
                {
                    "completed": completed,
                    "epoch": epoch,
                    "cursor": cursor,
                    "macro": value,
                    "best_step": None if best is None else best[0],
                    "best_macro": None if best is None else best[1],
                    "best_history": history,
                    "events": events,
                    "macros": macros,
                },
                _sections(model, opt, completed, epoch, cursor, exposure),
            )
            publish(root, f"u{completed}", "LAST", payload)
            if best is not None and best[0] == completed:
                publish(root, f"u{completed}", "BEST", payload)
    return {
        "completed_updates": completed,
        "best_step": None if best is None else best[0],
        "last_step": completed,
        "macros": macros,
        "events": events,
        "observed": False,
    }


def _require_exposure_fields(batch: list[dict]) -> None:
    required = ("selected_ordinals", "selected_labels", "current_ordinal", "target_ordinal", "current", "target", "horizon", "provider")
    for item in batch:
        if item.get("provider") != "SYNTHETIC_APPARATUS" or any(key not in item for key in required):
            raise ValueError("batch item is not exposure-v1")


def _admit_exposure(records: list, header: dict) -> None:
    required = ("videos", "selected_ordinals", "selected_labels", "current_ordinal", "target_ordinal", "current", "target", "horizon", "epoch", "cursor", "seed", "provider")
    if len(records) != header["completed"]:
        raise ValueError("exposure length does not match completed")
    for index, record in enumerate(records, start=1):
        if record.get("schema") != "exposure-v1" or record.get("completed") != index or record.get("provider") != "SYNTHETIC_APPARATUS":
            raise ValueError("exposure schema or step mismatch")
        if any(key not in record or record[key] in (None, []) for key in required):
            raise ValueError("exposure record is missing required fields")


def _admit(decoded: dict) -> dict:
    import math

    header = decoded["header"]
    events = header["events"]
    macros = header["macros"]
    if events != sorted(set(events)) or any(type(event) is not int for event in events):
        raise ValueError("event journal is not a strictly increasing int sequence")
    if set(events) != {int(step) for step in macros} or any(not str(step).isdigit() or str(int(step)) != str(step) for step in macros):
        raise ValueError("macro keys do not match event ids")
    for step, value in macros.items():
        if not math.isfinite(float(value)):
            raise ValueError("nonfinite restored macro")
    numeric = {int(step): float(value) for step, value in macros.items()}
    if type(header["completed"]) is not int or any(event > header["completed"] for event in events):
        raise ValueError("event is after completed step")
    if events and header["best_step"] is None:
        raise ValueError("best is missing while events exist")
    if header["best_step"] is not None:
        earliest = min(events, key=lambda step: (numeric[step], step))
        if header["best_step"] != earliest or float(header["best_macro"]) != numeric[earliest]:
            raise ValueError("best is not the earliest finite minimum")
    return header


def _numpy_state() -> list:
    import numpy

    algo, keys, pos, has_gauss, cached = numpy.random.get_state()
    return [algo, keys.tolist(), pos, has_gauss, cached]


def resume(payload: bytes, model, opt) -> dict:
    import io
    import json

    import torch

    decoded = unpack(payload)
    header = _admit(decoded)
    require_records(json.loads(decoded["exposure"].decode("utf-8")), header["completed"])
    model.load_state_dict(torch.load(io.BytesIO(decoded["model"]), map_location="cpu", weights_only=True))
    opt.load_state_dict(torch.load(io.BytesIO(decoded["adamw"]), map_location="cpu", weights_only=True))
    import json
    import random

    import numpy

    version, state, gauss = json.loads(decoded["python_rng"].decode("utf-8"))
    random.setstate((version, tuple(state), gauss))
    algo, keys, pos, has_gauss, cached = json.loads(decoded["numpy_rng"].decode("utf-8"))
    numpy.random.set_state((algo, numpy.array(keys, dtype="uint32"), pos, has_gauss, cached))
    torch.set_rng_state(torch.tensor(list(decoded["torch_rng"]), dtype=torch.uint8))
    return {
        "completed": header["completed"],
        "epoch": header["epoch"],
        "cursor": header["cursor"],
        "best": None if header["best_step"] is None else (header["best_step"], header["best_macro"]),
        "events": header["events"],
        "macros": header["macros"],
        "exposure": json.loads(decoded["exposure"].decode("utf-8")),
    }


def _sections(model, opt, completed: int, epoch: int, cursor: int, exposure: list) -> dict[str, bytes]:
    import io
    import json
    import random

    import numpy
    import torch

    model_buf = io.BytesIO()
    torch.save(model.state_dict(), model_buf)
    opt_buf = io.BytesIO()
    torch.save(opt.state_dict(), opt_buf)
    return {
        "model": model_buf.getvalue(),
        "adamw": opt_buf.getvalue(),
        "python_rng": json.dumps(random.getstate()).encode("utf-8"),
        "numpy_rng": json.dumps(_numpy_state()).encode("utf-8"),
        "torch_rng": bytes(torch.get_rng_state().tolist()),
        "sampler": json.dumps({"completed": completed, "epoch": epoch, "cursor": cursor}).encode("utf-8"),
        "exposure": json.dumps(exposure).encode("utf-8"),
    }


def main() -> None:
    raise ExecutionNotAuthorized("NOT_RUN")
