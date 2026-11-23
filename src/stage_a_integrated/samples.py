"""Typed sparse-history batch. Synthetic provider does not open the corpus."""

from __future__ import annotations

WIDTH = 1280


def schedule(n_frames: int, current: int, target: int) -> list[float]:
    span = float(n_frames - 1)
    local = (target - current) / span
    return [local, target / span, current / span, target / span, abs(local)]


def synthetic_item(id_movi: str, ids_frames: list[int], history: list[int], current: int, target: int):
    if target in history or any(frame > current for frame in history):
        raise ValueError("future frame leaked into history")
    ordinal = {label: index for index, label in enumerate(ids_frames)}
    if current not in ordinal or target not in ordinal:
        raise ValueError("frame label is not in ids_frames")
    import torch

    proxy = torch.zeros(len(ids_frames), WIDTH)
    hidden = torch.zeros(1, WIDTH)
    return {
        "id_movi": id_movi,
        "ids_frames": list(ids_frames),
        "history": list(history),
        "current": current,
        "target": target,
        "model_kwargs": {
            "history_latents": proxy[[ordinal[frame] for frame in history]].unsqueeze(0),
            "history_progress": torch.tensor([[ordinal[frame] / float(len(ids_frames) - 1) for frame in history]]),
            "history_valid_mask": torch.ones(1, len(history), dtype=torch.bool),
            "text_hidden_states": hidden.unsqueeze(0),
            "text_valid_mask": torch.ones(1, 1, dtype=torch.bool),
            "current_latent": proxy[ordinal[current]].unsqueeze(0),
            "anchor_latent": proxy[0].unsqueeze(0),
            "transition": torch.tensor([schedule(len(ids_frames), ordinal[current], ordinal[target])]),
            "mode": "hybrid",
        },
        "supervision": {
            "target_delta_local": proxy[ordinal[target]].unsqueeze(0),
            "target_delta_anchor": proxy[ordinal[target]].unsqueeze(0),
            "target_latent_proxy": proxy[ordinal[target]].unsqueeze(0),
        },
    }


def assert_no_target_leak(item: dict) -> None:
    if set(item["model_kwargs"]) & set(item["supervision"]):
        raise ValueError("supervision leaked into model kwargs")
