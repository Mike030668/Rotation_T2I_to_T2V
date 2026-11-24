"""Label-time synthetic rows, representation firewall, and target-leak checks. No corpus read."""

from __future__ import annotations

from .forward_binding import (
    FORWARD_BINDING_VERSION,
    REPRESENTATION_FIREWALL,
    WIDTH,
    canonical_forward_kwargs_digest,
    canonical_supervision_digest,
    expected_synthetic_forward_digest,
    expected_synthetic_supervision_digest,
    synthetic_image_row_f32,
    synthetic_prior_anchor_f32,
    synthetic_progress_f32,
    synthetic_schedule_f32,
    synthetic_text_hidden_f32,
    validate_causal_transition_meta,
    validate_supervision,
)


def schedule(ids_frames: list[int], current_label: int, target_label: int) -> list[float]:
    return synthetic_schedule_f32(ids_frames, current_label, target_label)


def _image(id_movi: str, ids_frames: list[int]):
    import torch

    rows = [synthetic_image_row_f32(id_movi, frame) for frame in ids_frames]
    return torch.tensor(rows, dtype=torch.float32)


def _prior_anchor(id_movi: str):
    import torch

    return torch.tensor([synthetic_prior_anchor_f32(id_movi)], dtype=torch.float32)


def synthetic_item(
    id_movi: str,
    ids_frames: list[int],
    history: list[int],
    current: int,
    target: int,
    horizon: str,
    image=None,
    prior_anchor=None,
):
    """Build one synthetic Stage A transition under strict causal representation firewall."""
    ordinal, current_index, target_index = validate_causal_transition_meta(
        id_movi,
        ids_frames,
        history,
        current,
        target,
        horizon,
        anchor_ordinal=0,
        anchor=ids_frames[0],
    )
    import torch

    if image is None:
        image = _image(id_movi, ids_frames)
    else:
        if (
            not isinstance(image, torch.Tensor)
            or image.device.type != "cpu"
            or image.dtype != torch.float32
            or tuple(image.shape) != (len(ids_frames), WIDTH)
            or not bool(torch.isfinite(image).all().item())
        ):
            raise ValueError("image must be a finite CPU float32 tensor with shape [T, 1280]")
        image = image.detach().contiguous().clone()

    if prior_anchor is None:
        prior_anchor = _prior_anchor(id_movi)
    else:
        if (
            not isinstance(prior_anchor, torch.Tensor)
            or prior_anchor.device.type != "cpu"
            or prior_anchor.dtype != torch.float32
            or tuple(prior_anchor.shape) != (1, WIDTH)
            or not bool(torch.isfinite(prior_anchor).all().item())
        ):
            raise ValueError("prior_anchor must be a finite CPU float32 tensor with shape [1, 1280]")
        prior_anchor = prior_anchor.detach().contiguous().clone()

    # Firewall: only observed indices <= current_index may enter proxy/forward inputs.
    e0 = image[0:1]
    hist_indices = [ordinal[frame] for frame in history]
    e_hist = image[hist_indices]
    e_cur = image[current_index : current_index + 1]

    proxy_hist = prior_anchor + (e_hist - e0)
    proxy_cur = prior_anchor + (e_cur - e0)

    kwargs = {
        "history_latents": proxy_hist.unsqueeze(0).clone(),
        "history_progress": torch.tensor([synthetic_progress_f32(ids_frames, history)], dtype=torch.float32),
        "history_valid_mask": torch.ones(1, len(history), dtype=torch.bool),
        "text_hidden_states": torch.tensor([[synthetic_text_hidden_f32()]], dtype=torch.float32),
        "text_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        "current_latent": proxy_cur.clone(),
        "anchor_latent": prior_anchor.clone(),
        "transition": torch.tensor([synthetic_schedule_f32(ids_frames, current, target)], dtype=torch.float32),
        "mode": "hybrid",
    }

    # Future e_target enters supervision ONLY, never forward kwargs.
    e_target = image[target_index : target_index + 1]
    supervision = {
        "target_delta_local": (e_target - e_cur).clone(),
        "target_delta_anchor": (e_target - e0).clone(),
        "target_latent_proxy": (prior_anchor + (e_target - e0)).clone(),
    }

    forward_input_sha256, frozen_kwargs = canonical_forward_kwargs_digest(kwargs)
    validate_supervision(supervision, forbidden_kwargs=kwargs, frozen_kwargs=frozen_kwargs)
    supervision_sha256 = canonical_supervision_digest(supervision)

    return {
        "id_movi": id_movi,
        "ids_frames": list(ids_frames),
        "anchor_ordinal": 0,
        "anchor": ids_frames[0],
        "history": list(history),
        "current": current,
        "target": target,
        "horizon": horizon,
        "forward_binding": FORWARD_BINDING_VERSION,
        "representation_firewall": REPRESENTATION_FIREWALL,
        "forward_input_sha256": forward_input_sha256,
        "supervision_sha256": supervision_sha256,
        "model_kwargs": kwargs,
        "supervision": supervision,
    }


def inject_target_into_forward_kwargs(item: dict, *, field: str = "current_latent", scale: float = 1e-3) -> dict:
    """Create an adversarial target-leaking item on the exact same forward kwargs call."""
    import copy

    leaked = {
        key: (
            {k: v.detach().clone() if hasattr(v, "detach") else v for k, v in value.items()}
            if key in ("model_kwargs", "supervision") and type(value) is dict
            else copy.deepcopy(value)
        )
        for key, value in item.items()
    }
    target_delta = leaked["supervision"]["target_delta_local"]
    if field == "__extra_target_key__":
        leaked["model_kwargs"]["target_delta_local"] = target_delta.clone()
        return leaked
    if field == "current_latent":
        leaked["model_kwargs"]["current_latent"] = leaked["model_kwargs"]["current_latent"] + float(scale) * target_delta
        return leaked
    if field == "history_latents":
        leaked["model_kwargs"]["history_latents"][:, -1, :] = (
            leaked["model_kwargs"]["history_latents"][:, -1, :] + float(scale) * target_delta
        )
        return leaked
    if field == "transition":
        leaked["model_kwargs"]["transition"][:, 4:5] = float(leaked["supervision"]["target_delta_local"].norm().item())
        return leaked
    raise ValueError(f"unsupported target injection field: {field}")


def assert_no_target_leak(item: dict, *, verify_canonical_oracle: bool = True) -> dict:
    """Verify item forward kwargs & supervision match causal oracle and reject target-only leak."""
    import torch

    if item.get("anchor_ordinal") is None or item.get("anchor") is None:
        raise ValueError("item missing non-null anchor grounding fields")
    validate_causal_transition_meta(
        item["id_movi"],
        item["ids_frames"],
        item["history"],
        item["current"],
        item["target"],
        item["horizon"],
        anchor_ordinal=item["anchor_ordinal"],
        anchor=item["anchor"],
    )
    actual_digest, frozen_actual = canonical_forward_kwargs_digest(item["model_kwargs"])
    if "forward_input_sha256" in item and item["forward_input_sha256"] != actual_digest:
        raise ValueError("item forward_input_sha256 disagrees with actual model_kwargs digest")
    validate_supervision(item["supervision"], forbidden_kwargs=item["model_kwargs"], frozen_kwargs=frozen_actual)

    oracle_digest = expected_synthetic_forward_digest(
        item["id_movi"],
        item["ids_frames"],
        item["history"],
        item["current"],
        item["target"],
        item["horizon"],
    )
    actual_sup_digest = canonical_supervision_digest(item["supervision"])
    oracle_sup_digest = expected_synthetic_supervision_digest(
        item["id_movi"],
        item["ids_frames"],
        item["current"],
        item["target"],
    )
    if verify_canonical_oracle:
        if actual_digest != oracle_digest:
            raise ValueError("actual model_kwargs digest disagrees with independent causal oracle")
        if actual_sup_digest != oracle_sup_digest:
            raise ValueError("actual supervision digest disagrees with independent target oracle")
        if "supervision_sha256" in item and item["supervision_sha256"] != oracle_sup_digest:
            raise ValueError("item supervision_sha256 disagrees with target oracle")

    image = _image(item["id_movi"], item["ids_frames"])
    mutated = image.clone()
    mutated[item["ids_frames"].index(item["target"])] += 1.0
    clean = synthetic_item(item["id_movi"], item["ids_frames"], item["history"], item["current"], item["target"], item["horizon"], image)
    dirty = synthetic_item(item["id_movi"], item["ids_frames"], item["history"], item["current"], item["target"], item["horizon"], mutated)
    if clean["forward_input_sha256"] != dirty["forward_input_sha256"]:
        raise ValueError("target-frame mutation changed canonical forward_input_sha256")
    if verify_canonical_oracle and clean["forward_input_sha256"] != actual_digest:
        raise ValueError("item forward_input_sha256 does not match counterfactual clean item")
    for key, value in clean["model_kwargs"].items():
        other = dirty["model_kwargs"][key]
        if hasattr(value, "equal"):
            if not bool(torch.equal(value, other)):
                raise ValueError(f"target-frame mutation changed model input {key}")
        elif value != other:
            raise ValueError(f"target-frame mutation changed non-tensor model input {key}")
    for sup_key in ("target_delta_local", "target_delta_anchor", "target_latent_proxy"):
        if bool(torch.equal(clean["supervision"][sup_key], dirty["supervision"][sup_key])):
            raise ValueError(f"target-frame mutation did not change supervision {sup_key}")
    return {
        "actual_digest": actual_digest,
        "oracle_digest": oracle_digest,
        "actual_supervision_digest": actual_sup_digest,
        "oracle_supervision_digest": oracle_sup_digest,
        "target_mutation_preserved_digest": True,
    }


def assert_causal_input_sensitivity(item: dict) -> dict:
    """Verify causal input mutation changes forward_input_sha256 unlike target-only mutation."""
    base_digest = item["forward_input_sha256"]
    image = _image(item["id_movi"], item["ids_frames"])
    prior_anchor = _prior_anchor(item["id_movi"])
    if item["current"] == item["ids_frames"][0]:
        # At ordinal 0, e_cur == e0 so (e_cur - e0) == 0; causal latent input is prior_anchor z0.
        mutated_anchor = prior_anchor.clone()
        mutated_anchor[0, 0] += 1.0
        changed = synthetic_item(
            item["id_movi"],
            item["ids_frames"],
            item["history"],
            item["current"],
            item["target"],
            item["horizon"],
            image=image,
            prior_anchor=mutated_anchor,
        )
        causal_kind = "prior_anchor_at_origin"
    else:
        causal_mutated = image.clone()
        causal_mutated[item["ids_frames"].index(item["current"])] += 1.0
        changed = synthetic_item(
            item["id_movi"],
            item["ids_frames"],
            item["history"],
            item["current"],
            item["target"],
            item["horizon"],
            image=causal_mutated,
            prior_anchor=prior_anchor,
        )
        causal_kind = "current_frame_latent"
    if changed["forward_input_sha256"] == base_digest:
        raise ValueError("causal input mutation failed to change forward_input_sha256")
    return {
        "base_digest": base_digest,
        "causal_mutated_digest": changed["forward_input_sha256"],
        "causal_input_kind": causal_kind,
    }


def assert_future_unselected_invariance(item: dict) -> dict:
    """Verify mutating future unselected frames (> current, != target) alters neither inputs nor targets."""
    import torch

    unselected_future = [
        frame for frame in item["ids_frames"] if frame > item["current"] and frame != item["target"]
    ]
    if not unselected_future:
        raise ValueError("item has no future unselected frame beyond current/target")

    image = _image(item["id_movi"], item["ids_frames"])
    mutated = image.clone()
    for frame in unselected_future:
        mutated[item["ids_frames"].index(frame)] += 1.0

    clean = synthetic_item(item["id_movi"], item["ids_frames"], item["history"], item["current"], item["target"], item["horizon"], image)
    perturbed = synthetic_item(item["id_movi"], item["ids_frames"], item["history"], item["current"], item["target"], item["horizon"], mutated)
    if clean["forward_input_sha256"] != perturbed["forward_input_sha256"]:
        raise ValueError("future unselected frame mutation altered forward_input_sha256")
    if clean["supervision_sha256"] != perturbed["supervision_sha256"]:
        raise ValueError("future unselected frame mutation altered supervision_sha256")
    for sup_key in ("target_delta_local", "target_delta_anchor", "target_latent_proxy"):
        if not bool(torch.equal(clean["supervision"][sup_key], perturbed["supervision"][sup_key])):
            raise ValueError(f"future unselected frame mutation altered supervision {sup_key}")
    return {
        "unselected_future_frames": unselected_future,
        "forward_input_sha256": clean["forward_input_sha256"],
        "supervision_sha256": clean["supervision_sha256"],
    }
