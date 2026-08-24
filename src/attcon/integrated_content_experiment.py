from __future__ import annotations

"""Models and causal measurements for the Stage 8 integrated-content pilot."""

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import torch
from torch import nn

from attcon.integrated_content import (
    IntegratedContentConfig,
    IntegratedContentExample,
    TARGET_STATUSES,
    UNKNOWN_ANSWER,
)


@dataclass(frozen=True)
class IntegratedContentTensors:
    initial_features: torch.Tensor
    content_ids: torch.Tensor
    query: torch.Tensor
    transition: torch.Tensor
    location: torch.Tensor
    feature_type: torch.Tensor
    value: torch.Tensor
    access_answer: torch.Tensor
    status: torch.Tensor
    lure_location: torch.Tensor
    lure_feature_type: torch.Tensor
    lure_value: torch.Tensor

    def __len__(self) -> int:
        return self.initial_features.shape[0]

    def to(self, device: str | torch.device) -> IntegratedContentTensors:
        return IntegratedContentTensors(
            **{name: value.to(device) for name, value in self.__dict__.items()}
        )

    def subset(self, indices: torch.Tensor) -> IntegratedContentTensors:
        return IntegratedContentTensors(
            **{name: value[indices] for name, value in self.__dict__.items()}
        )


@dataclass(frozen=True)
class IntegratedPrediction:
    location: torch.Tensor
    feature_type: torch.Tensor
    value: torch.Tensor
    access_answer: torch.Tensor
    binding_state: torch.Tensor
    access_initial_state: torch.Tensor
    access_state: torch.Tensor


def initial_feature_dim(config: IntegratedContentConfig) -> int:
    return config.num_cells + config.feature_type_vocab_size + config.value_vocab_size


def tensorize_integrated_content_examples(
    examples: Iterable[IntegratedContentExample],
    config: IntegratedContentConfig,
) -> IntegratedContentTensors:
    cases = list(examples)
    if not cases:
        raise ValueError("at least one integrated-content example is required")
    features = torch.zeros(
        len(cases), config.num_objects, initial_feature_dim(config)
    )
    content_ids = torch.zeros(
        len(cases), config.num_objects, config.content_id_vocab_size
    )
    query = torch.zeros(len(cases), config.content_id_vocab_size)
    transition = torch.zeros(len(cases), 4)
    status_to_index = {status: index for index, status in enumerate(TARGET_STATUSES)}
    for row, example in enumerate(cases):
        query[row, example.binding_cue_content_id] = 1.0
        for column, obj in enumerate(example.objects):
            content_ids[row, column, obj.content_id] = 1.0
            features[row, column, obj.location] = 1.0
            features[
                row,
                column,
                config.num_cells + obj.feature_type,
            ] = 1.0
            features[
                row,
                column,
                config.num_cells + config.feature_type_vocab_size + obj.initial_value,
            ] = 1.0
        target = example.target
        transition[row, 0] = float(target.current_observation_value is not None)
        transition[row, 1] = float(target.access_cache_value is not None)
        transition[row, 2] = float(target.attended_before)
        transition[row, 3] = float(
            target.current_observation_value is not None
            and target.current_observation_value != target.initial_value
        )
    return IntegratedContentTensors(
        initial_features=features,
        content_ids=content_ids,
        query=query,
        transition=transition,
        location=torch.tensor([case.target.location for case in cases]),
        feature_type=torch.tensor([case.target.feature_type for case in cases]),
        value=torch.tensor([case.target.initial_value for case in cases]),
        access_answer=torch.tensor(
            [
                config.value_vocab_size
                if case.expected_access_answer == UNKNOWN_ANSWER
                else case.expected_access_answer
                for case in cases
            ]
        ),
        status=torch.tensor([status_to_index[case.target_status] for case in cases]),
        lure_location=torch.tensor(
            [case.false_binding_lure.location for case in cases]
        ),
        lure_feature_type=torch.tensor(
            [case.false_binding_lure.feature_type for case in cases]
        ),
        lure_value=torch.tensor([case.false_binding_lure.value for case in cases]),
    )


class IntegratedContentModel(nn.Module):
    """Solve binding and later access through shared or separated content states.

    All modes have identical parameters. ``shared`` averages two encoders into one state
    consumed by both branches. ``split`` keeps the encoder states separate, and ``pooled``
    destroys object identity before applying the otherwise shared architecture.
    """

    def __init__(
        self,
        config: IntegratedContentConfig,
        hidden_size: int = 64,
        *,
        mode: Literal["shared", "split", "pooled"] = "shared",
    ) -> None:
        super().__init__()
        if mode not in {"shared", "split", "pooled"}:
            raise ValueError(f"unknown integrated-content mode: {mode}")
        self.config = config
        self.mode = mode
        feature_dim = initial_feature_dim(config)
        self.binding_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_size), nn.Tanh()
        )
        self.access_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_size), nn.Tanh()
        )
        self.transition = nn.GRUCell(4, hidden_size)
        self.location_head = nn.Linear(hidden_size, config.num_cells)
        self.feature_type_head = nn.Linear(
            hidden_size, config.feature_type_vocab_size
        )
        self.value_head = nn.Linear(hidden_size, config.value_vocab_size)
        self.access_head = nn.Linear(hidden_size, config.value_vocab_size + 1)

    def initial_states(
        self,
        initial_features: torch.Tensor,
        content_ids: torch.Tensor,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "pooled":
            selected = initial_features.mean(dim=1)
        else:
            matches = torch.einsum("bok,bk->bo", content_ids, query)
            selected = torch.einsum("bo,bof->bf", matches, initial_features)
        binding = self.binding_encoder(selected)
        access = self.access_encoder(selected)
        if self.mode in {"shared", "pooled"}:
            shared = (binding + access) / 2.0
            return shared, shared
        return binding, access

    def forward(
        self,
        initial_features: torch.Tensor,
        content_ids: torch.Tensor,
        query: torch.Tensor,
        transition: torch.Tensor,
        *,
        binding_state_override: torch.Tensor | None = None,
    ) -> IntegratedPrediction:
        binding_state, access_initial_state = self.initial_states(
            initial_features, content_ids, query
        )
        if binding_state_override is not None:
            binding_state = binding_state_override
            if self.mode in {"shared", "pooled"}:
                access_initial_state = binding_state_override
        access_state = self.transition(transition, access_initial_state)
        return IntegratedPrediction(
            location=self.location_head(binding_state),
            feature_type=self.feature_type_head(binding_state),
            value=self.value_head(binding_state),
            access_answer=self.access_head(access_state),
            binding_state=binding_state,
            access_initial_state=access_initial_state,
            access_state=access_state,
        )


class NeutralRoutingContentModel(IntegratedContentModel):
    """Dual content lanes with an optional learned binding-to-access route.

    The learned and blocked variants have exactly the same parameters. The routing gate is
    initialized near closed; no loss directly supervises its value or representational overlap.
    """

    def __init__(
        self,
        config: IntegratedContentConfig,
        hidden_size: int = 64,
        *,
        routing: Literal["learned", "blocked"] = "learned",
        initial_routing_weight: float = 0.05,
        private_access_dropout: float = 0.0,
        private_access_dropout_rescale: bool = True,
    ) -> None:
        if routing not in {"learned", "blocked"}:
            raise ValueError(f"unknown routing condition: {routing}")
        if not 0.0 < initial_routing_weight < 1.0:
            raise ValueError("initial_routing_weight must be strictly between zero and one")
        if not 0.0 <= private_access_dropout < 1.0:
            raise ValueError("private_access_dropout must be in [0, 1)")
        super().__init__(config, hidden_size, mode="split")
        self.routing = routing
        self.private_access_dropout = private_access_dropout
        self.private_access_dropout_rescale = private_access_dropout_rescale
        initial_logit = torch.logit(torch.tensor(float(initial_routing_weight)))
        self.routing_logit = nn.Parameter(initial_logit)

    def routing_weight(self) -> torch.Tensor:
        if self.routing == "blocked":
            return torch.zeros((), device=self.routing_logit.device)
        return torch.sigmoid(self.routing_logit)

    def forward(
        self,
        initial_features: torch.Tensor,
        content_ids: torch.Tensor,
        query: torch.Tensor,
        transition: torch.Tensor,
        *,
        binding_state_override: torch.Tensor | None = None,
    ) -> IntegratedPrediction:
        binding_state, private_access_state = self.initial_states(
            initial_features, content_ids, query
        )
        if binding_state_override is not None:
            binding_state = binding_state_override
        if self.training and self.private_access_dropout:
            keep = torch.rand(
                private_access_state.shape[0],
                1,
                device=private_access_state.device,
            ).ge(self.private_access_dropout)
            private_access_state = (
                private_access_state
                * keep
            )
            if self.private_access_dropout_rescale:
                private_access_state = private_access_state / (
                    1.0 - self.private_access_dropout
                )
        weight = self.routing_weight()
        access_initial_state = (
            weight * binding_state + (1.0 - weight) * private_access_state
        )
        access_state = self.transition(transition, access_initial_state)
        return IntegratedPrediction(
            location=self.location_head(binding_state),
            feature_type=self.feature_type_head(binding_state),
            value=self.value_head(binding_state),
            access_answer=self.access_head(access_state),
            binding_state=binding_state,
            access_initial_state=access_initial_state,
            access_state=access_state,
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _loss(
    prediction: IntegratedPrediction, batch: IntegratedContentTensors
) -> torch.Tensor:
    return sum(
        nn.functional.cross_entropy(getattr(prediction, name), getattr(batch, name))
        for name in ("location", "feature_type", "value", "access_answer")
    )


def train_integrated_content_model(
    model: IntegratedContentModel,
    train: IntegratedContentTensors,
    *,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 3e-3,
    seed: int = 827,
    device: str = "cpu",
) -> list[float]:
    torch.manual_seed(seed)
    model.to(device)
    data = train.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(data), generator=generator)
        total = 0.0
        for start in range(0, len(data), batch_size):
            indices = order[start : start + batch_size].to(device)
            batch = data.subset(indices)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch.initial_features,
                batch.content_ids,
                batch.query,
                batch.transition,
            )
            loss = _loss(prediction, batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / len(data))
    return losses


@torch.no_grad()
def evaluate_integrated_content_model(
    model: IntegratedContentModel,
    data: IntegratedContentTensors,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    model.eval()
    batch = data.to(device)
    output = model(
        batch.initial_features, batch.content_ids, batch.query, batch.transition
    )
    location = output.location.argmax(dim=-1)
    feature_type = output.feature_type.argmax(dim=-1)
    value = output.value.argmax(dim=-1)
    access = output.access_answer.argmax(dim=-1)
    binding_joint = (
        location.eq(batch.location)
        & feature_type.eq(batch.feature_type)
        & value.eq(batch.value)
    )
    access_correct = access.eq(batch.access_answer)
    lure = (
        location.eq(batch.lure_location)
        & feature_type.eq(batch.lure_feature_type)
        & value.eq(batch.lure_value)
    )
    by_status = {}
    for index, status in enumerate(TARGET_STATUSES):
        mask = batch.status.eq(index)
        by_status[status] = access_correct[mask].float().mean().item()
    return {
        "count": len(data),
        "binding_joint_accuracy": binding_joint.float().mean().item(),
        "false_binding_lure_rejection": (~lure).float().mean().item(),
        "access_accuracy": access_correct.float().mean().item(),
        "binding_and_access_joint_accuracy": (
            binding_joint & access_correct
        ).float().mean().item(),
        "access_accuracy_by_status": by_status,
    }


def _donor_indices(batch: IntegratedContentTensors) -> torch.Tensor:
    donors = torch.arange(len(batch))
    for status_index in range(len(TARGET_STATUSES)):
        rows = torch.nonzero(batch.status.eq(status_index), as_tuple=False).flatten()
        if len(rows) > 1:
            donors[rows] = rows.roll(1)
    return donors


@torch.no_grad()
def binding_state_swap_metrics(
    model: IntegratedContentModel,
    data: IntegratedContentTensors,
    *,
    device: str = "cpu",
) -> dict[str, float]:
    """Swap the binding branch's content state between same-status held-out cases."""

    model.eval()
    batch = data.to(device)
    donors = _donor_indices(batch).to(device)
    binding_state, _ = model.initial_states(
        batch.initial_features, batch.content_ids, batch.query
    )
    swapped = model(
        batch.initial_features,
        batch.content_ids,
        batch.query,
        batch.transition,
        binding_state_override=binding_state[donors],
    )
    binding_follows = (
        swapped.location.argmax(dim=-1).eq(batch.location[donors])
        & swapped.feature_type.argmax(dim=-1).eq(batch.feature_type[donors])
        & swapped.value.argmax(dim=-1).eq(batch.value[donors])
    )
    accessible = ~batch.status.eq(0)
    access_prediction = swapped.access_answer.argmax(dim=-1)
    access_follows = access_prediction.eq(batch.value[donors]) & accessible
    access_retains = access_prediction.eq(batch.value) & accessible
    both_follow = binding_follows & access_follows
    return {
        "binding_donor_follow_rate": binding_follows.float().mean().item(),
        "accessible_access_donor_follow_rate": access_follows[accessible]
        .float()
        .mean()
        .item(),
        "accessible_access_receiver_retention_rate": access_retains[accessible]
        .float()
        .mean()
        .item(),
        "accessible_joint_donor_follow_rate": both_follow[accessible]
        .float()
        .mean()
        .item(),
    }


@torch.no_grad()
def value_direction_intervention_metrics(
    model: IntegratedContentModel,
    fit_data: IntegratedContentTensors,
    test_data: IntegratedContentTensors,
    *,
    alpha: float = 1.0,
    permute_fit_labels: bool = False,
    seed: int = 883,
    device: str = "cpu",
) -> dict[str, float]:
    """Fit value centroids on one split and intervene along them on another.

    The intervention changes only the fitted value-content direction. Location and feature
    type are held at the receiver labels, and access status is unchanged. A permuted-label
    call provides an identically computed null direction.
    """

    model.eval()
    fit = fit_data.to(device)
    test = test_data.to(device)
    fit_states, _ = model.initial_states(
        fit.initial_features, fit.content_ids, fit.query
    )
    fit_labels = fit.value.clone()
    if permute_fit_labels:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(len(fit_labels), generator=generator).to(device)
        fit_labels = fit_labels[order]
    centroids = []
    for value in range(model.config.value_vocab_size):
        mask = fit_labels.eq(value)
        if not mask.any():
            raise ValueError(f"fit split has no examples for value {value}")
        centroids.append(fit_states[mask].mean(dim=0))
    centroid_matrix = torch.stack(centroids)

    test_states, _ = model.initial_states(
        test.initial_features, test.content_ids, test.query
    )
    donor_values = (test.value + 1) % model.config.value_vocab_size
    directions = centroid_matrix[donor_values] - centroid_matrix[test.value]
    edited_states = test_states + float(alpha) * directions
    edited = model(
        test.initial_features,
        test.content_ids,
        test.query,
        test.transition,
        binding_state_override=edited_states,
    )
    binding_value = edited.value.argmax(dim=-1)
    access_value = edited.access_answer.argmax(dim=-1)
    location_stable = edited.location.argmax(dim=-1).eq(test.location)
    type_stable = edited.feature_type.argmax(dim=-1).eq(test.feature_type)
    binding_follows = binding_value.eq(donor_values)
    accessible = ~test.status.eq(0)
    access_follows = access_value.eq(donor_values)
    unavailable_retains_unknown = access_value[test.status.eq(0)].eq(
        model.config.value_vocab_size
    )
    return {
        "binding_value_donor_follow_rate": binding_follows.float().mean().item(),
        "binding_other_fields_stability": (location_stable & type_stable)
        .float()
        .mean()
        .item(),
        "accessible_access_donor_follow_rate": access_follows[accessible]
        .float()
        .mean()
        .item(),
        "accessible_binding_access_joint_donor_follow_rate": (
            binding_follows & access_follows
        )[accessible]
        .float()
        .mean()
        .item(),
        "unavailable_unknown_retention_rate": unavailable_retains_unknown
        .float()
        .mean()
        .item(),
        "mean_direction_norm": directions.norm(dim=-1).mean().item(),
        "alpha": float(alpha),
    }
