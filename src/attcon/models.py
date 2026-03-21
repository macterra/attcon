from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .data import TaskConfig


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass
class ModelConfig:
    hidden_size: int = 32
    cue_embedding_dim: int = 8
    scene_embedding_dim: int = 16
    temperature: float = 0.6

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        return cls(**values)


class BaseAttentionModel(nn.Module):
    def __init__(self, task_config: TaskConfig, model_config: ModelConfig):
        super().__init__()
        self.task_config = task_config
        self.model_config = model_config
        self.num_cells = task_config.num_cells
        self.visible_dim = task_config.visible_feature_dim
        self.hidden_dim = task_config.hidden_feature_dim
        self.cue_embedding = nn.Embedding(task_config.num_types, model_config.cue_embedding_dim)
        self.visible_encoder = _mlp(
            self.visible_dim,
            model_config.scene_embedding_dim,
            model_config.scene_embedding_dim,
        )
        self.scene_summary = _mlp(
            self.num_cells * model_config.scene_embedding_dim + model_config.cue_embedding_dim,
            max(model_config.hidden_size, 64),
            model_config.hidden_size,
        )
        self.task_head = _mlp(
            self.hidden_dim + model_config.hidden_size,
            max(model_config.hidden_size, 64),
            task_config.digit_vocab_size,
        )

    def split_scene(self, scene: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        visible = scene[..., : self.visible_dim]
        hidden = scene[..., self.visible_dim :]
        return visible, hidden

    def initial_state(self, scene: torch.Tensor, cue: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        visible, hidden = self.split_scene(scene)
        cue_emb = self.cue_embedding(cue)
        visible_emb = self.visible_encoder(visible)
        flat_scene = visible_emb.reshape(scene.shape[0], -1)
        init_state = self.scene_summary(torch.cat([flat_scene, cue_emb], dim=-1))
        return init_state, hidden

    def _compute_loss_proxy(
        self,
        logits: torch.Tensor,
        target: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values.detach().unsqueeze(-1)
        if target is None:
            loss_proxy = (-confidence).detach()
        else:
            loss_proxy = F.cross_entropy(logits, target, reduction="none").detach().unsqueeze(-1)
        return loss_proxy, confidence


class StaticAttentionBaseline(BaseAttentionModel):
    def __init__(self, task_config: TaskConfig, model_config: ModelConfig):
        super().__init__(task_config, model_config)
        self.attention_head = nn.Linear(model_config.hidden_size, self.num_cells)

    def forward(
        self,
        scene: torch.Tensor,
        cue: torch.Tensor,
        target: torch.Tensor | None = None,
        num_steps: int | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        steps = num_steps or self.task_config.num_steps
        static_state, hidden_features = self.initial_state(scene, cue)
        attention_logits = self.attention_head(static_state) / self.model_config.temperature
        attention = torch.softmax(attention_logits, dim=-1)
        glimpse = torch.sum(attention.unsqueeze(-1) * hidden_features, dim=1)
        logits = self.task_head(torch.cat([glimpse, static_state], dim=-1))
        loss_proxy, confidence = self._compute_loss_proxy(logits, target)

        repeated_attention = attention.unsqueeze(1).repeat(1, steps, 1)
        repeated_logits = logits.unsqueeze(1).repeat(1, steps, 1)
        repeated_confidence = confidence.unsqueeze(1).repeat(1, steps, 1)
        repeated_loss = loss_proxy.unsqueeze(1).repeat(1, steps, 1)
        repeated_state = static_state.unsqueeze(1).repeat(1, steps, 1)

        return {
            "logits": logits,
            "logits_seq": repeated_logits,
            "attention_seq": repeated_attention,
            "confidence_seq": repeated_confidence,
            "loss_seq": repeated_loss,
            "controller_state_seq": repeated_state,
        }


class RecurrentAttentionController(BaseAttentionModel):
    def __init__(self, task_config: TaskConfig, model_config: ModelConfig):
        super().__init__(task_config, model_config)
        self.controller = nn.GRUCell(self.summary_dim, model_config.hidden_size)
        self.summary_adapter = nn.Sequential(
            nn.Linear(self.summary_dim, model_config.hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(model_config.hidden_size, self.num_cells)

    @property
    def summary_dim(self) -> int:
        return self.num_cells + self.hidden_dim + 2 + self.model_config.cue_embedding_dim

    def _build_summary(
        self,
        previous_attention: torch.Tensor,
        previous_glimpse: torch.Tensor,
        previous_loss: torch.Tensor,
        previous_confidence: torch.Tensor,
        cue_embedding: torch.Tensor,
        *,
        ablation: dict[str, bool] | None = None,
    ) -> torch.Tensor:
        if ablation is None:
            ablation = {}
        if ablation.get("zero_prev_attention"):
            previous_attention = torch.zeros_like(previous_attention)
        if ablation.get("zero_prev_glimpse"):
            previous_glimpse = torch.zeros_like(previous_glimpse)
        if ablation.get("zero_prev_loss"):
            previous_loss = torch.zeros_like(previous_loss)
        if ablation.get("zero_prev_confidence"):
            previous_confidence = torch.zeros_like(previous_confidence)
        return torch.cat(
            [
                previous_attention,
                previous_glimpse,
                previous_loss,
                previous_confidence,
                cue_embedding,
            ],
            dim=-1,
        )

    def forward(
        self,
        scene: torch.Tensor,
        cue: torch.Tensor,
        target: torch.Tensor | None = None,
        num_steps: int | None = None,
        *,
        ablation: dict[str, bool] | None = None,
    ) -> dict[str, torch.Tensor]:
        steps = num_steps or self.task_config.num_steps
        if ablation is None:
            ablation = {}

        if ablation.get("shuffle_cue"):
            perm = torch.randperm(cue.shape[0], device=cue.device)
            cue_for_policy = cue[perm]
        else:
            cue_for_policy = cue

        hidden_state, hidden_features = self.initial_state(scene, cue_for_policy)
        cue_emb = self.cue_embedding(cue_for_policy)
        previous_attention = torch.zeros(scene.shape[0], self.num_cells, device=scene.device)
        previous_glimpse = torch.zeros(scene.shape[0], self.hidden_dim, device=scene.device)
        previous_loss = torch.zeros(scene.shape[0], 1, device=scene.device)
        previous_confidence = torch.zeros(scene.shape[0], 1, device=scene.device)

        attention_seq = []
        logits_seq = []
        confidence_seq = []
        loss_seq = []
        controller_state_seq = [hidden_state]

        for step_idx in range(steps):
            if step_idx > 0:
                summary = self._build_summary(
                    previous_attention,
                    previous_glimpse,
                    previous_loss,
                    previous_confidence,
                    cue_emb,
                    ablation=ablation,
                )
                if ablation.get("feedforward_summary"):
                    hidden_state = self.summary_adapter(summary)
                elif not ablation.get("freeze_recurrence"):
                    hidden_state = torch.tanh(
                        self.controller(summary, hidden_state) + self.summary_adapter(summary)
                    )

            attention_logits = self.policy_head(hidden_state) / self.model_config.temperature
            attention = torch.softmax(attention_logits, dim=-1)
            glimpse = torch.sum(attention.unsqueeze(-1) * hidden_features, dim=1)
            step_logits = self.task_head(torch.cat([glimpse, hidden_state], dim=-1))
            step_loss, step_confidence = self._compute_loss_proxy(step_logits, target)

            attention_seq.append(attention)
            logits_seq.append(step_logits)
            confidence_seq.append(step_confidence)
            loss_seq.append(step_loss)
            controller_state_seq.append(hidden_state)

            previous_attention = attention.detach()
            previous_glimpse = glimpse
            previous_loss = step_loss
            previous_confidence = step_confidence

        stacked_logits = torch.stack(logits_seq, dim=1)
        return {
            "logits": stacked_logits[:, -1],
            "logits_seq": stacked_logits,
            "attention_seq": torch.stack(attention_seq, dim=1),
            "confidence_seq": torch.stack(confidence_seq, dim=1),
            "loss_seq": torch.stack(loss_seq, dim=1),
            "controller_state_seq": torch.stack(controller_state_seq[:-1], dim=1),
        }
