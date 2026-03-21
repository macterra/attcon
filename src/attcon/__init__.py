from .data import Batch, TaskConfig, generate_batch
from .models import RecurrentAttentionController, StaticAttentionBaseline


def train_experiment(*args, **kwargs):
    from .train import train_experiment as _train_experiment

    return _train_experiment(*args, **kwargs)


def run_ablations(*args, **kwargs):
    from .eval import run_ablations as _run_ablations

    return _run_ablations(*args, **kwargs)

__all__ = [
    "Batch",
    "TaskConfig",
    "generate_batch",
    "StaticAttentionBaseline",
    "RecurrentAttentionController",
    "train_experiment",
    "run_ablations",
]
