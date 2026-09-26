"""Context-cluster uncertainty for paired-history reporting measurements."""

import torch

from attcon.history_reporting import HistoryBatch, HistoryConfig


METRICS = ("seen_value_accuracy", "unseen_unknown_accuracy", "paired_accuracy", "balanced_accuracy")


def context_scores(predicted: torch.Tensor, data: HistoryBatch, config: HistoryConfig) -> torch.Tensor:
    correct = predicted == data.report_labels(config)
    scores = []
    for group in data.group.unique(sorted=True):
        indices = torch.where(data.group == group)[0]
        order = (data.value[indices] * 2 + (~data.seen[indices]).long()).argsort()
        indices = indices[order]
        if len(indices) != 2 * config.values:
            raise ValueError("expected one seen/unseen pair per value in each context")
        if not torch.equal(data.value[indices], torch.arange(config.values).repeat_interleave(2)) or not torch.equal(
            data.seen[indices], torch.tensor([True, False]).repeat(config.values)
        ):
            raise ValueError("context has missing or duplicated content/status pairs")
        seen = correct[indices][data.seen[indices]].float().mean()
        unseen = correct[indices][~data.seen[indices]].float().mean()
        paired = correct[indices].reshape(-1, 2).all(1).float().mean()
        scores.append(torch.stack((seen, unseen, paired, (seen + unseen) / 2)))
    return torch.stack(scores)


def context_bootstrap(scores: torch.Tensor, *, seed: int, draws: int = 2000) -> dict:
    if scores.ndim != 2 or scores.shape[1] != len(METRICS) or len(scores) < 2 or draws < 2:
        raise ValueError("need >=2 contexts, four metrics, and >=2 bootstrap draws")
    indices = torch.randint(len(scores), (draws, len(scores)), generator=torch.Generator().manual_seed(seed))
    distribution = scores[indices].mean(1)
    quantiles = distribution.quantile(torch.tensor([0.025, 0.975]), dim=0)
    return {metric: {"mean": scores[:, column].mean().item(), "low": quantiles[0, column].item(),
                     "high": quantiles[1, column].item()} for column, metric in enumerate(METRICS)}
