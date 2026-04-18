from __future__ import annotations

"""Model definitions for static attention and recurrent attention control."""

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
    """Architecture hyperparameters shared by both models."""

    hidden_size: int = 32
    cue_embedding_dim: int = 8
    scene_embedding_dim: int = 16
    temperature: float = 0.6

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        return cls(**values)


class BaseAttentionModel(nn.Module):
    """Shared scene encoding and cue-conditioned observation logic.

    Both models see the same scene encoding. The key difference is whether future
    attention depends only on the current scene summary or on recurrent state that
    carries forward previous observations and feedback.
    """

    def __init__(self, task_config: TaskConfig, model_config: ModelConfig):
        super().__init__()
        self.task_config = task_config
        self.model_config = model_config
        self.num_cells = task_config.num_cells
        self.num_types = task_config.num_types
        self.digit_vocab_size = task_config.digit_vocab_size
        self.visible_dim = task_config.visible_feature_dim
        self.hidden_dim = task_config.hidden_feature_dim
        self.observation_dim = 1 + self.digit_vocab_size
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
            self.observation_dim + model_config.hidden_size,
            max(model_config.hidden_size, 64),
            task_config.digit_vocab_size,
        )

    def split_scene(self, scene: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split visible cell-type features from hidden task-relevant features."""

        visible = scene[..., : self.visible_dim]
        hidden = scene[..., self.visible_dim :]
        return visible, hidden

    def initial_state(self, scene: torch.Tensor, cue: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the visible scene and cue into the initial controller state."""

        visible, hidden = self.split_scene(scene)
        cue_emb = self.cue_embedding(cue)
        visible_emb = self.visible_encoder(visible)
        flat_scene = visible_emb.reshape(scene.shape[0], -1)
        init_state = self.scene_summary(torch.cat([flat_scene, cue_emb], dim=-1))
        return init_state, hidden

    def observe_glimpse(self, hidden_glimpse: torch.Tensor, cue: torch.Tensor) -> torch.Tensor:
        """Collapse the hidden glimpse into cue-relevant target evidence plus digit features."""

        cue_one_hot = F.one_hot(cue, num_classes=self.num_types).float()
        target_flags = hidden_glimpse[..., : self.num_types]
        digit_features = hidden_glimpse[..., self.num_types :]
        matched_target = (target_flags * cue_one_hot).sum(dim=-1, keepdim=True)
        return torch.cat([matched_target, digit_features], dim=-1)

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
    """Cue-conditioned attention baseline with no recurrent control loop."""

    def __init__(self, task_config: TaskConfig, model_config: ModelConfig):
        super().__init__(task_config, model_config)
        self.attention_head = nn.Linear(model_config.hidden_size, self.num_cells)

    def forward(
        self,
        scene: torch.Tensor,
        cue: torch.Tensor,
        cue_seq: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        target_pos: torch.Tensor | None = None,
        target_pos_seq: torch.Tensor | None = None,
        num_steps: int | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        steps = num_steps or self.task_config.num_steps
        cue_seq = cue.unsqueeze(1).repeat(1, steps) if cue_seq is None else cue_seq
        logits_seq = []
        attention_seq = []
        confidence_seq = []
        loss_seq = []
        controller_state_seq = []
        observation_seq = []

        for step_idx in range(steps):
            step_cue = cue_seq[:, step_idx]
            static_state, hidden_features = self.initial_state(scene, step_cue)
            attention_logits = self.attention_head(static_state) / self.model_config.temperature
            attention = torch.softmax(attention_logits, dim=-1)
            hidden_glimpse = torch.sum(attention.unsqueeze(-1) * hidden_features, dim=1)
            observed_glimpse = self.observe_glimpse(hidden_glimpse, step_cue)
            logits = self.task_head(torch.cat([observed_glimpse, static_state], dim=-1))
            loss_proxy, confidence = self._compute_loss_proxy(logits, target)

            logits_seq.append(logits)
            attention_seq.append(attention)
            confidence_seq.append(confidence)
            loss_seq.append(loss_proxy)
            controller_state_seq.append(static_state)
            observation_seq.append(observed_glimpse)

        stacked_logits = torch.stack(logits_seq, dim=1)

        return {
            "logits": stacked_logits[:, -1],
            "logits_seq": stacked_logits,
            "attention_seq": torch.stack(attention_seq, dim=1),
            "observation_seq": torch.stack(observation_seq, dim=1),
            "confidence_seq": torch.stack(confidence_seq, dim=1),
            "loss_seq": torch.stack(loss_seq, dim=1),
            "controller_state_seq": torch.stack(controller_state_seq, dim=1),
        }


class RecurrentAttentionController(BaseAttentionModel):
    """Recurrent controller that reallocates attention from prior internal summaries."""

    def __init__(self, task_config: TaskConfig, model_config: ModelConfig):
        super().__init__(task_config, model_config)
        self.controller = nn.GRUCell(self.summary_dim, model_config.hidden_size)
        self.summary_adapter = nn.Sequential(
            nn.Linear(self.summary_dim, model_config.hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(model_config.hidden_size, self.num_cells)
        self.self_model_head = nn.Linear(model_config.hidden_size + self.num_cells, self.num_cells)
        self.target_found_head = nn.Linear(model_config.hidden_size + self.num_cells + 1, 1)
        self.relevant_region_head = nn.Linear(model_config.hidden_size + self.num_cells + 1, 1)
        self.unresolved_search_head = nn.Linear(model_config.hidden_size + self.num_cells + 1, 1)
        self.allocation_error_head = nn.Linear(model_config.hidden_size + 2 * self.num_cells + 1, 1)

    @property
    def summary_dim(self) -> int:
        return 2 * self.num_cells + self.observation_dim + 2 + self.model_config.cue_embedding_dim

    def _build_summary(
        self,
        previous_attention: torch.Tensor,
        inspection_state: torch.Tensor,
        previous_observation: torch.Tensor,
        previous_loss: torch.Tensor,
        previous_confidence: torch.Tensor,
        cue_embedding: torch.Tensor,
        *,
        ablation: dict[str, bool] | None = None,
    ) -> torch.Tensor:
        """Assemble the feedback vector used to choose the next attention allocation."""

        if ablation is None:
            ablation = {}
        if ablation.get("zero_prev_attention"):
            previous_attention = torch.zeros_like(previous_attention)
        if ablation.get("zero_prev_glimpse"):
            previous_observation = torch.zeros_like(previous_observation)
        if ablation.get("zero_prev_loss"):
            previous_loss = torch.zeros_like(previous_loss)
        if ablation.get("zero_prev_confidence"):
            previous_confidence = torch.zeros_like(previous_confidence)
        return torch.cat(
            [
                previous_attention,
                inspection_state,
                previous_observation,
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
        cue_seq: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        target_pos: torch.Tensor | None = None,
        target_pos_seq: torch.Tensor | None = None,
        num_steps: int | None = None,
        *,
        ablation: dict[str, bool] | None = None,
        intervention: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        steps = num_steps or self.task_config.num_steps
        if ablation is None:
            ablation = {}
        if intervention is None:
            intervention = {}
        cue_seq = cue.unsqueeze(1).repeat(1, steps) if cue_seq is None else cue_seq
        if target_pos_seq is None and target_pos is not None:
            target_pos_seq = target_pos.unsqueeze(1).repeat(1, steps)

        if ablation.get("shuffle_cue"):
            perm = torch.randperm(cue_seq.shape[0], device=cue_seq.device)
            cue_seq_for_policy = cue_seq[perm]
        else:
            cue_seq_for_policy = cue_seq

        hidden_state, hidden_features = self.initial_state(scene, cue_seq_for_policy[:, 0])
        previous_attention = torch.zeros(scene.shape[0], self.num_cells, device=scene.device)
        inspection_state = torch.zeros(scene.shape[0], self.num_cells, device=scene.device)
        found_state = torch.zeros(scene.shape[0], 1, device=scene.device)
        previous_observation = torch.zeros(scene.shape[0], self.observation_dim, device=scene.device)
        previous_loss = torch.zeros(scene.shape[0], 1, device=scene.device)
        previous_confidence = torch.zeros(scene.shape[0], 1, device=scene.device)

        attention_seq = []
        logits_seq = []
        observation_seq = []
        confidence_seq = []
        loss_seq = []
        controller_state_seq = [hidden_state]
        inspection_seq = []
        self_model_logits_seq = []
        target_found_logits_seq = []
        relevant_region_logits_seq = []
        unresolved_search_logits_seq = []
        allocation_error_logits_seq = []
        found_state_seq = []
        relevant_region_seq = []
        unresolved_search_seq = []
        allocation_error_seq = []

        for step_idx in range(steps):
            step_cue = cue_seq_for_policy[:, step_idx]
            cue_emb = self.cue_embedding(step_cue)
            if step_idx > 0:
                # The next fixation is chosen from the previous fixation, what it observed,
                # and detached task feedback rather than from the raw scene alone.
                summary = self._build_summary(
                    previous_attention,
                    inspection_state,
                    previous_observation,
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

            if step_idx == intervention.get("step"):
                if "state_override" in intervention:
                    hidden_state = intervention["state_override"]
                if "delta" in intervention:
                    hidden_state = hidden_state + intervention["delta"]

            self_model_logits = self.self_model_head(torch.cat([hidden_state, inspection_state], dim=-1))
            attention_logits = self.policy_head(hidden_state) / self.model_config.temperature
            attention = torch.softmax(attention_logits, dim=-1)
            hidden_glimpse = torch.sum(attention.unsqueeze(-1) * hidden_features, dim=1)
            observed_glimpse = self.observe_glimpse(hidden_glimpse, step_cue)
            step_logits = self.task_head(torch.cat([observed_glimpse, hidden_state], dim=-1))
            step_loss, step_confidence = self._compute_loss_proxy(step_logits, target)
            attended_cell = attention.argmax(dim=-1)
            attended_one_hot = F.one_hot(attended_cell, num_classes=self.num_cells).float()
            inspection_state_post = torch.maximum(inspection_state, attended_one_hot)
            found_state_post = torch.maximum(found_state, observed_glimpse[:, :1].detach())
            report_features = torch.cat([hidden_state, inspection_state, found_state], dim=-1)
            target_found_logits = self.target_found_head(
                report_features
            )
            relevant_region_logits = self.relevant_region_head(report_features)
            unresolved_search_logits = self.unresolved_search_head(report_features)

            if target_pos_seq is not None:
                step_target_pos = target_pos_seq[:, step_idx]
                target_inspected = inspection_state.gather(1, step_target_pos.unsqueeze(-1))
                unresolved_search = 1.0 - target_inspected
                if step_idx > 0:
                    previous_attended_cell = previous_attention.argmax(dim=-1)
                    allocation_error = (
                        (previous_attended_cell != step_target_pos)
                        & (target_inspected.squeeze(-1) < 0.5)
                        & (previous_observation[:, 0] < 0.5)
                    ).float().unsqueeze(-1)
                else:
                    allocation_error = torch.zeros(scene.shape[0], 1, device=scene.device)
            else:
                target_inspected = torch.zeros(scene.shape[0], 1, device=scene.device)
                unresolved_search = torch.zeros(scene.shape[0], 1, device=scene.device)
                allocation_error = torch.zeros(scene.shape[0], 1, device=scene.device)

            allocation_error_logits = self.allocation_error_head(
                torch.cat(
                    [hidden_state, inspection_state, previous_attention, previous_observation[:, :1]],
                    dim=-1,
                )
            )

            attention_seq.append(attention)
            logits_seq.append(step_logits)
            observation_seq.append(observed_glimpse)
            confidence_seq.append(step_confidence)
            loss_seq.append(step_loss)
            controller_state_seq.append(hidden_state)
            inspection_seq.append(inspection_state)
            self_model_logits_seq.append(self_model_logits)
            target_found_logits_seq.append(target_found_logits)
            relevant_region_logits_seq.append(relevant_region_logits)
            unresolved_search_logits_seq.append(unresolved_search_logits)
            allocation_error_logits_seq.append(allocation_error_logits)
            found_state_seq.append(found_state_post)
            relevant_region_seq.append(target_inspected)
            unresolved_search_seq.append(unresolved_search)
            allocation_error_seq.append(allocation_error)

            previous_attention = attention.detach()
            inspection_state = inspection_state_post
            found_state = found_state_post
            previous_observation = observed_glimpse
            previous_loss = step_loss
            previous_confidence = step_confidence

        stacked_logits = torch.stack(logits_seq, dim=1)
        return {
            "logits": stacked_logits[:, -1],
            "logits_seq": stacked_logits,
            "attention_seq": torch.stack(attention_seq, dim=1),
            "observation_seq": torch.stack(observation_seq, dim=1),
            "confidence_seq": torch.stack(confidence_seq, dim=1),
            "loss_seq": torch.stack(loss_seq, dim=1),
            "controller_state_seq": torch.stack(controller_state_seq[:-1], dim=1),
            "inspection_seq": torch.stack(inspection_seq, dim=1),
            "self_model_logits_seq": torch.stack(self_model_logits_seq, dim=1),
            "self_model_seq": torch.sigmoid(torch.stack(self_model_logits_seq, dim=1)),
            "found_state_seq": torch.stack(found_state_seq, dim=1),
            "target_found_logits_seq": torch.stack(target_found_logits_seq, dim=1),
            "target_found_seq": torch.sigmoid(torch.stack(target_found_logits_seq, dim=1)),
            "relevant_region_logits_seq": torch.stack(relevant_region_logits_seq, dim=1),
            "relevant_region_seq": torch.stack(relevant_region_seq, dim=1),
            "unresolved_search_logits_seq": torch.stack(unresolved_search_logits_seq, dim=1),
            "unresolved_search_seq": torch.stack(unresolved_search_seq, dim=1),
            "allocation_error_logits_seq": torch.stack(allocation_error_logits_seq, dim=1),
            "allocation_error_seq": torch.stack(allocation_error_seq, dim=1),
        }
