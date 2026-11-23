from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    delta_huber_weight: float = 1.0
    direction_weight: float = 0.25
    magnitude_weight: float = 0.10
    hybrid_state_weight: float = 0.10
    dual_consistency_weight: float = 0.10
    huber_beta: float = 0.05
    eps: float = 1e-8


def directional_cosine_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Correct per-sample cosine loss across latent features.

    This deliberately avoids the historical ``.T`` pattern that made cosine operate
    across the batch axis instead of the 1280-D feature axis.
    """
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError("pred and target must both be [B,D] with identical shapes")
    cos = F.cosine_similarity(pred, target, dim=-1, eps=eps)
    valid = torch.linalg.vector_norm(target, dim=-1) > eps
    loss = 1.0 - cos
    return torch.where(valid, loss, torch.zeros_like(loss))


def magnitude_log_ratio_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    p = torch.linalg.vector_norm(pred, dim=-1).clamp_min(eps)
    t = torch.linalg.vector_norm(target, dim=-1).clamp_min(eps)
    return (torch.log(p) - torch.log(t)).abs()


class SplitterV3Loss(nn.Module):
    def __init__(self, config: LossConfig = LossConfig()):
        super().__init__()
        self.config = config

    def _huber(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(
            pred,
            target,
            reduction="none",
            beta=self.config.huber_beta,
        ).mean(dim=-1)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        local_delta_huber = self._huber(outputs["delta_local"], batch["target_delta_local"])
        anchor_delta_huber = self._huber(outputs["delta_anchor"], batch["target_delta_anchor"])
        local_direction = directional_cosine_per_sample(
            outputs["delta_local"], batch["target_delta_local"], eps=self.config.eps
        )
        anchor_direction = directional_cosine_per_sample(
            outputs["delta_anchor"], batch["target_delta_anchor"], eps=self.config.eps
        )
        local_magnitude = magnitude_log_ratio_per_sample(
            outputs["delta_local"], batch["target_delta_local"], eps=self.config.eps
        )
        anchor_magnitude = magnitude_log_ratio_per_sample(
            outputs["delta_anchor"], batch["target_delta_anchor"], eps=self.config.eps
        )
        hybrid_state = self._huber(outputs["pred_hybrid"], batch["target_latent_proxy"])
        dual_consistency = self._huber(outputs["pred_local"], outputs["pred_anchor"])

        delta_huber = 0.5 * (local_delta_huber + anchor_delta_huber)
        direction = 0.5 * (local_direction + anchor_direction)
        magnitude = 0.5 * (local_magnitude + anchor_magnitude)

        total_per_sample = (
            self.config.delta_huber_weight * delta_huber
            + self.config.direction_weight * direction
            + self.config.magnitude_weight * magnitude
            + self.config.hybrid_state_weight * hybrid_state
            + self.config.dual_consistency_weight * dual_consistency
        )
        return {
            "total": total_per_sample.mean(),
            "delta_huber": delta_huber.mean(),
            "direction": direction.mean(),
            "magnitude": magnitude.mean(),
            "hybrid_state": hybrid_state.mean(),
            "dual_consistency": dual_consistency.mean(),
            "local_delta_huber": local_delta_huber.mean(),
            "anchor_delta_huber": anchor_delta_huber.mean(),
            "local_direction": local_direction.mean(),
            "anchor_direction": anchor_direction.mean(),
        }
