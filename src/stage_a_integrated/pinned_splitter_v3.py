from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Literal

import torch
import torch.nn as nn

InferenceMode = Literal["local", "anchor", "hybrid"]


@dataclass
class SplitterV3Config:
    """Configuration for Splitter-v3: Dual-Context Delta Transformer.

    The default V3-S configuration is intentionally small: contextual computation
    happens in d_model=256 while external Kandinsky/unCLIP latents remain 1280-D.
    """

    latent_dim: int = 1280
    text_dim: int = 1280
    d_model: int = 256
    num_heads: int = 8
    num_trajectory_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.10
    transition_hidden_dim: int = 128
    fourier_features: int = 16


class ScalarFourierEmbedding(nn.Module):
    """Continuous scalar position embedding for normalized trajectory progress."""

    def __init__(self, d_model: int, num_frequencies: int = 16):
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be >= 1")
        # Geometric frequencies support both local and global progress changes.
        freqs = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", freqs, persistent=False)
        self.proj = nn.Linear(1 + 2 * num_frequencies, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [...]
        phase = 2.0 * torch.pi * x.unsqueeze(-1) * self.frequencies
        feats = torch.cat([x.unsqueeze(-1), torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.proj(feats)


class TransitionEncoder(nn.Module):
    """Encodes the successor of the legacy scalar ``rise`` conditioning.

    External descriptor columns (shape [B, 5]):
      0 local_step          signed current->target normalized step (legacy rise analogue)
      1 anchor_step         signed anchor->target normalized step
      2 current_progress    normalized current position in the source trajectory
      3 target_progress     normalized target position
      4 perceptual_strength optional requested/estimated visual-change strength

    ``abs(local_step)`` and ``sign(local_step)`` are derived internally to avoid
    inconsistent duplicated user inputs.
    """

    INPUT_DIM = 5

    def __init__(self, d_model: int, hidden_dim: int = 128):
        super().__init__()
        engineered_dim = self.INPUT_DIM + 2
        self.net = nn.Sequential(
            nn.Linear(engineered_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        if descriptor.ndim != 2 or descriptor.size(-1) != self.INPUT_DIM:
            raise ValueError(
                f"transition descriptor must have shape [B, {self.INPUT_DIM}], "
                f"got {tuple(descriptor.shape)}"
            )
        local_step = descriptor[:, :1]
        engineered = torch.cat(
            [descriptor, local_step.abs(), torch.sign(local_step)], dim=-1
        )
        return self.norm(self.net(engineered))


class ConditionalCausalBlock(nn.Module):
    """Pre-norm causal self-attention block conditioned by transition control."""

    def __init__(self, d_model: int, num_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        # Two independent FiLM transforms: one for attention, one for FFN.
        self.film_attn = nn.Linear(d_model, 2 * d_model)
        self.film_ffn = nn.Linear(d_model, 2 * d_model)

    @staticmethod
    def _film(x: torch.Tensor, cond: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        gamma, beta = layer(cond).chunk(2, dim=-1)
        # tanh bounds multiplicative modulation at initialization/training time.
        gamma = torch.tanh(gamma).unsqueeze(1)
        beta = beta.unsqueeze(1)
        return x * (1.0 + gamma) + beta

    def forward(
        self,
        x: torch.Tensor,
        transition: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self._film(self.norm1(x), transition, self.film_attn)
        attn_mask = causal_mask
        empty = None
        pad = key_padding_mask
        if key_padding_mask is not None:
            blocked = causal_mask.unsqueeze(0) | key_padding_mask[:, None, :]
            empty = blocked.logical_not().sum(dim=-1) == 0
            if empty.any():
                heads = self.attn.num_heads
                attn_mask = blocked[:, None, :, :].expand(-1, heads, -1, -1).contiguous()
                b_idx, q_idx = empty.nonzero(as_tuple=True)
                attn_mask[b_idx, :, q_idx, q_idx] = False
                attn_mask = attn_mask.reshape(attn_mask.size(0) * heads, causal_mask.size(0), causal_mask.size(0))
                pad = None
        attn_out, _ = self.attn(
            qkv,
            qkv,
            qkv,
            attn_mask=attn_mask,
            key_padding_mask=pad,
            need_weights=False,
        )
        if empty is not None:
            attn_out = torch.where(empty.unsqueeze(-1), torch.zeros_like(attn_out), attn_out)
        x = x + self.dropout(attn_out)
        ffn_in = self._film(self.norm2(x), transition, self.film_ffn)
        x = x + self.dropout(self.ffn(ffn_in))
        return x


class SplitterV3(nn.Module):
    """Dual-Context Delta Transformer for latent transition prediction.

    Design intent:
      * Trajectory branch: causal self-attention over latent states and their
        first/second finite differences.
      * Semantic branch: cross-attention from the current transition query to
        text hidden states.
      * Transition control: successor of legacy ``rise``; conditions attention,
        branch modulation, and explicit residual step scaling.
      * Dual delta heads: local/current-relative and anchor-relative predictions.
      * Hybrid mode: learned interpolation between local and anchor paths.

    This module is deliberately *model-only*: no dataset, optimizer, loss,
    checkpoint, decoder, or rollout policy is embedded here.
    """

    def __init__(self, config: SplitterV3Config = SplitterV3Config()):
        super().__init__()
        self.config = config
        d = config.d_model
        ld = config.latent_dim

        self.state_proj = nn.Linear(ld, d)
        self.delta_proj = nn.Linear(ld, d)
        self.accel_proj = nn.Linear(ld, d)
        self.anchor_proj = nn.Linear(ld, d)
        self.text_proj = nn.Linear(config.text_dim, d)
        self.progress_embed = ScalarFourierEmbedding(d, config.fourier_features)
        self.transition_encoder = TransitionEncoder(d, config.transition_hidden_dim)

        self.trajectory_blocks = nn.ModuleList(
            [
                ConditionalCausalBlock(
                    d_model=d,
                    num_heads=config.num_heads,
                    ff_mult=config.ff_mult,
                    dropout=config.dropout,
                )
                for _ in range(config.num_trajectory_layers)
            ]
        )

        self.semantic_query_norm = nn.LayerNorm(d)
        self.text_norm = nn.LayerNorm(d)
        self.semantic_attn = nn.MultiheadAttention(
            d,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Transition control modulates both context branches after attention.
        self.traj_film = nn.Linear(d, 2 * d)
        self.sem_film = nn.Linear(d, 2 * d)

        self.fusion = nn.Sequential(
            nn.Linear(5 * d, 2 * d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
        )

        self.writer_norm = nn.LayerNorm(d)
        self.writer = nn.Sequential(
            nn.Linear(d, config.ff_mult * d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_mult * d, d),
        )

        self.local_direction_head = nn.Linear(d, ld)
        self.anchor_direction_head = nn.Linear(d, ld)
        self.hybrid_gate = nn.Linear(d, 1)

    @staticmethod
    def _to_2d_latent(x: torch.Tensor, name: str) -> torch.Tensor:
        if x.ndim == 3 and x.size(1) == 1:
            x = x[:, 0, :]
        if x.ndim != 2:
            raise ValueError(f"{name} must have shape [B,D] or [B,1,D], got {tuple(x.shape)}")
        return x

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        # Boolean mask: True means blocked.
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    @staticmethod
    def _apply_film(x: torch.Tensor, cond: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        gamma, beta = layer(cond).chunk(2, dim=-1)
        return x * (1.0 + torch.tanh(gamma)) + beta

    def build_trajectory_tokens(
        self,
        history_latents: torch.Tensor,
        history_progress: torch.Tensor,
        history_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if history_latents.ndim != 3:
            raise ValueError(
                f"history_latents must have shape [B,L,D], got {tuple(history_latents.shape)}"
            )
        if history_progress.ndim != 2 or history_progress.shape[:2] != history_latents.shape[:2]:
            raise ValueError(
                "history_progress must have shape [B,L] matching history_latents"
            )
        if history_latents.size(-1) != self.config.latent_dim:
            raise ValueError(
                f"history latent dim must be {self.config.latent_dim}, "
                f"got {history_latents.size(-1)}"
            )

        states = history_latents
        if history_valid_mask is not None:
            states = torch.where(history_valid_mask.bool().unsqueeze(-1), history_latents, torch.zeros_like(history_latents))
        delta = torch.zeros_like(states)
        if states.size(1) > 1:
            delta[:, 1:] = states[:, 1:] - states[:, :-1]
            if history_valid_mask is not None:
                keep = (history_valid_mask[:, 1:] & history_valid_mask[:, :-1]).unsqueeze(-1)
                delta[:, 1:] = torch.where(keep, delta[:, 1:], torch.zeros_like(delta[:, 1:]))

        accel = torch.zeros_like(states)
        if states.size(1) > 2:
            accel[:, 2:] = delta[:, 2:] - delta[:, 1:-1]
            if history_valid_mask is not None:
                keep = (history_valid_mask[:, 2:] & history_valid_mask[:, 1:-1] & history_valid_mask[:, :-2]).unsqueeze(-1)
                accel[:, 2:] = torch.where(keep, accel[:, 2:], torch.zeros_like(accel[:, 2:]))

        return (
            self.state_proj(states)
            + self.delta_proj(delta)
            + self.accel_proj(accel)
            + self.progress_embed(history_progress)
        )

    def encode_trajectory(
        self,
        history_latents: torch.Tensor,
        history_progress: torch.Tensor,
        transition_embedding: torch.Tensor,
        history_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.build_trajectory_tokens(history_latents, history_progress, history_valid_mask)
        _, length, _ = x.shape
        causal = self._causal_mask(length, x.device)

        key_padding_mask = None
        if history_valid_mask is not None:
            if history_valid_mask.shape != history_progress.shape:
                raise ValueError("history_valid_mask must have shape [B,L]")
            # MultiheadAttention expects True for ignored/padded positions.
            key_padding_mask = ~history_valid_mask.bool()

        for block in self.trajectory_blocks:
            x = block(
                x,
                transition=transition_embedding,
                causal_mask=causal,
                key_padding_mask=key_padding_mask,
            )
        return x

    @staticmethod
    def _last_valid_token(sequence: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if valid_mask is None:
            return sequence[:, -1]
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("history_valid_mask is all False")
        last = valid_mask.size(1) - 1 - valid_mask.long().flip(1).argmax(dim=1)
        return sequence[torch.arange(sequence.size(0), device=sequence.device), last]

    def forward(
        self,
        *,
        text_hidden_states: torch.Tensor,
        anchor_latent: torch.Tensor,
        current_latent: torch.Tensor,
        history_latents: torch.Tensor,
        history_progress: torch.Tensor,
        transition: torch.Tensor,
        mode: InferenceMode = "hybrid",
        history_valid_mask: Optional[torch.Tensor] = None,
        text_valid_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if mode not in ("local", "anchor", "hybrid"):
            raise ValueError(f"unknown mode: {mode}")

        anchor = self._to_2d_latent(anchor_latent, "anchor_latent")
        current = self._to_2d_latent(current_latent, "current_latent")
        if anchor.shape != current.shape:
            raise ValueError("anchor_latent and current_latent must have the same shape")
        if text_hidden_states.ndim != 3 or text_hidden_states.size(-1) != self.config.text_dim:
            raise ValueError(
                f"text_hidden_states must have shape [B,T,{self.config.text_dim}]"
            )
        if history_latents.size(0) != current.size(0):
            raise ValueError("history batch size must match current_latent")

        step = self.transition_encoder(transition)
        trajectory_seq = self.encode_trajectory(
            history_latents,
            history_progress,
            transition_embedding=step,
            history_valid_mask=history_valid_mask,
        )
        traj_context = self._last_valid_token(trajectory_seq, history_valid_mask)
        traj_context = self._apply_film(traj_context, step, self.traj_film)

        current_h = self.state_proj(current)
        anchor_h = self.anchor_proj(anchor)
        text_h = self.text_norm(self.text_proj(text_hidden_states))
        semantic_query = self.semantic_query_norm(current_h + step).unsqueeze(1)

        text_key_padding_mask = None
        if text_valid_mask is not None:
            if text_valid_mask.shape != text_hidden_states.shape[:2]:
                raise ValueError("text_valid_mask must have shape [B,T_text]")
            text_key_padding_mask = ~text_valid_mask.bool()

        semantic_context, semantic_weights = self.semantic_attn(
            semantic_query,
            text_h,
            text_h,
            key_padding_mask=text_key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False if return_attention else True,
        )
        semantic_context = semantic_context[:, 0]
        semantic_context = self._apply_film(semantic_context, step, self.sem_film)
        if text_valid_mask is not None:
            semantic_context = torch.where(text_valid_mask.any(dim=1).unsqueeze(-1), semantic_context, torch.zeros_like(semantic_context))
        self._last_semantic_context = semantic_context.detach()

        fused = self.fusion(
            torch.cat(
                [current_h, anchor_h, traj_context, semantic_context, step], dim=-1
            )
        )
        writer_state = fused + self.writer(self.writer_norm(fused))

        local_direction = self.local_direction_head(writer_state)
        anchor_direction = self.anchor_direction_head(writer_state)

        local_step = transition[:, 0:1]
        anchor_step = transition[:, 1:2]
        delta_local = local_step * local_direction
        delta_anchor = anchor_step * anchor_direction

        pred_local = current + delta_local
        pred_anchor = anchor + delta_anchor
        gate = torch.sigmoid(self.hybrid_gate(writer_state))
        pred_hybrid = gate * pred_local + (1.0 - gate) * pred_anchor

        if mode == "local":
            pred = pred_local
        elif mode == "anchor":
            pred = pred_anchor
        else:
            pred = pred_hybrid

        out: Dict[str, torch.Tensor] = {
            "pred": pred,
            "pred_local": pred_local,
            "pred_anchor": pred_anchor,
            "pred_hybrid": pred_hybrid,
            "delta_local": delta_local,
            "delta_anchor": delta_anchor,
            "local_direction": local_direction,
            "anchor_direction": anchor_direction,
            "hybrid_gate": gate,
            "trajectory_context": traj_context,
            "semantic_context": semantic_context,
            "transition_embedding": step,
            "writer_state": writer_state,
        }
        if return_attention:
            out["semantic_attention_weights"] = semantic_weights
        return out


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = SplitterV3Config()
    model = SplitterV3(cfg)
    print(f"SplitterV3 trainable parameters: {count_trainable_parameters(model):,}")
