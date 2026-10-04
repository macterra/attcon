"""Explicit deterministic Controller for an inspect-then-answer task.

Planning and answering receive modeled distributions, never simulator truth or
reporter text. The physical execution harness has a separate evaluator boundary.
"""
from dataclasses import dataclass
import torch
from .functional_model import buffer_contents
from .functional_controls import advance_world, last_forecast


@dataclass
class DecisionWorld:
    replay: torch.Tensor
    direction: torch.Tensor
    recovery: torch.Tensor
    controlled: torch.Tensor


def rounded_probabilities(values):
    """Use the existing interface's Python five-decimal rounding exactly."""
    return torch.tensor([round(v, 5) for v in values.flatten().tolist()],
                        dtype=torch.float64).reshape(values.shape)


def joint_confidence(values):
    """Minimum of the two attribute peaks; both must exceed the answer cutoff."""
    return torch.minimum(values[..., :4].max(-1).values,
                         values[..., 4:].max(-1).values)


def plan_command(visual, predicted_recovery, query):
    """Choose the command with maximal prospective confidence at the queried slot.

    visual: [batch,position,8]; recovery: [batch,command,position].
    Exact score ties choose the smallest command index. No physical-wiring input.
    """
    rows = torch.arange(len(query))
    future = buffer_contents(visual[:, None], predicted_recovery)
    scores = joint_confidence(future)[rows, :, query]
    return scores.argmax(-1), scores


def answer_query(visual, modeled_recovery, query):
    """Emit both labels or one explicit abstention, using only modeled readout.

    The threshold 0.6 is inherited from the interface but is now a Controller
    choice. Labels use color/shape indices, with (-1,-1) denoting abstention.
    """
    rows = torch.arange(len(query))
    distributions = buffer_contents(visual, modeled_recovery)[rows, query]
    rounded = rounded_probabilities(distributions)
    labels = torch.stack((rounded[:, :4].argmax(-1), rounded[:, 4:].argmax(-1)), -1)
    answered = joint_confidence(rounded) >= .6
    response = torch.where(answered[:, None], labels, torch.full_like(labels, -1))
    return {'distributions': distributions, 'rounded_distributions': rounded,
            'labels': labels, 'answered': answered, 'response': response}


@torch.no_grad()
def execute_decision(model, hidden, world, command, quality=.8):
    """Execute the chosen command in the world, then observe it and update A.

    Actual allocation/acquisition observations, rather than prospective values,
    enter the estimator. Physical recovery remains evaluator-only. No input is
    mutated and no reporting component is called.
    """
    replay, recovery, observed, allocation = advance_world(
        world.replay, world.direction, world.recovery, world.controlled, command, quality)
    sequence, updated_hidden = model(observed[:, None], hidden.clone())
    return last_forecast(sequence), updated_hidden, {
        'command': command.clone(), 'observed': observed, 'allocation': allocation,
        'physical_recovery': recovery, 'physical_replay': replay}
