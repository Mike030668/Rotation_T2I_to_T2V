"""Label-time synthetic rows. Nonzero tensors. No corpus read."""

from __future__ import annotations

WIDTH = 1280


def schedule(ids_frames: list[int], current_label: int, target_label: int) -> list[float]:
    span = float(ids_frames[-1] - ids_frames[0])
    if span <= 0:
        raise ValueError("frame labels do not span a positive interval")
    origin = float(ids_frames[0])
    local = (target_label - current_label) / span
    return [local, (target_label - origin) / span, (current_label - origin) / span, (target_label - origin) / span, abs(local)]


def synthetic_item(id_movi: str, ids_frames: list[int], history: list[int], current: int, target: int, horizon: str):
    if len(set(ids_frames)) != len(ids_frames) or ids_frames != sorted(ids_frames):
        raise ValueError("ids_frames must be strictly increasing")
    if target in history or any(frame > current for frame in history):
        raise ValueError("future frame leaked into history")
    ordinal = {label: index for index, label in enumerate(ids_frames)}
    import torch

    image = torch.arange(len(ids_frames) * WIDTH, dtype=torch.float32).reshape(len(ids_frames), WIDTH) / 997.0
    anchor = image[:1]
    proxy = anchor + (image - image[:1])
    hidden = torch.arange(WIDTH, dtype=torch.float32).reshape(1, WIDTH) / 996.0
    current_index = ordinal[current]
    target_index = ordinal[target]
    kwargs = {
        "history_latents": proxy[[ordinal[frame] for frame in history]].unsqueeze(0),
        "history_progress": torch.tensor([[(frame - ids_frames[0]) / float(ids_frames[-1] - ids_frames[0]) for frame in history]]),
        "history_valid_mask": torch.ones(1, len(history), dtype=torch.bool),
        "text_hidden_states": hidden.unsqueeze(0),
        "text_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        "current_latent": proxy[current_index].unsqueeze(0),
        "anchor_latent": proxy[0].unsqueeze(0),
        "transition": torch.tensor([schedule(ids_frames, current, target)], dtype=torch.float32),
        "mode": "hybrid",
    }
    supervision = {
        "target_delta_local": (image[target_index] - image[current_index]).unsqueeze(0),
        "target_delta_anchor": (image[target_index] - image[0]).unsqueeze(0),
        "target_latent_proxy": (proxy[0] + image[target_index] - image[0]).unsqueeze(0),
    }
    return {
        "id_movi": id_movi,
        "ids_frames": list(ids_frames),
        "history": list(history),
        "current": current,
        "target": target,
        "horizon": horizon,
        "model_kwargs": kwargs,
        "supervision": supervision,
    }


def assert_no_target_leak(item: dict) -> None:
    import torch

    before = {key: value.detach().clone() if hasattr(value, "detach") else value for key, value in item["model_kwargs"].items()}
    mutated_frames = list(item["ids_frames"])
    if item["target"] in mutated_frames:
        mutated_frames[mutated_frames.index(item["target"])] = item["target"]
    rebuilt = synthetic_item(item["id_movi"], item["ids_frames"], item["history"], item["current"], item["target"], item["horizon"])
    for key, value in before.items():
        other = rebuilt["model_kwargs"][key]
        if hasattr(value, "equal") and not bool(torch.equal(value, other)):
            raise ValueError("rebuild changed model kwargs without a source mutation")
