"""External verified-information reporters, fitted after control learning."""
import torch
from torch import nn
from torch.nn import functional as F
from attcon.active_inspection import all_states, verified_labels, rollout
from attcon.regulation import Readout, WIDTH
from attcon.history_reporting import pad_features


@torch.no_grad()
def measurement_data(agent, data):
    states = all_states(agent, data)
    features = {'state': pad_features(torch.cat(states), WIDTH),
        'action': pad_features(torch.cat([agent.answer(state) for state in states]), WIDTH)}
    labels = torch.cat([verified_labels(data, stage) for stage in (0, 1, 2)])
    visited = rollout(agent, data)['visited'].flatten()
    return features, labels, visited


def report_scores(predicted, labels, mask=None):
    mask = torch.ones_like(labels, dtype=torch.bool) if mask is None else mask
    verified, unverified = mask & (labels != 6), mask & (labels == 6)
    known = (predicted[verified] == labels[verified]).float().mean().item() if verified.any() else None
    unknown = (predicted[unverified] == 6).float().mean().item() if unverified.any() else None
    return {'verified_count': int(verified.sum()), 'unverified_count': int(unverified.sum()),
        'verified_value_accuracy': known, 'unverified_accuracy': unknown,
        'balanced_accuracy': (known + unknown) / 2 if known is not None and unknown is not None else None,
        'overall_accuracy': (predicted[mask] == labels[mask]).float().mean().item() if mask.any() else None}


def fit_report(features, labels, validation_features, validation_labels, seed, steps=200):
    best, best_score, candidates = None, None, []
    for l2 in (0., .001):
        torch.manual_seed(seed)
        model = Readout(features, classes=7, hidden=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        for _ in range(steps):
            logits = model(features)
            penalty = sum(layer.weight.square().sum() for layer in model.network if isinstance(layer, nn.Linear))
            loss = F.cross_entropy(logits, labels) + .5 * l2 * penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval().requires_grad_(False)
        with torch.no_grad():
            metrics = report_scores(model(validation_features).argmax(-1), validation_labels)
        record = {'l2': l2, 'validation': metrics}
        candidates.append(record)
        score = (min(metrics['verified_value_accuracy'], metrics['unverified_accuracy']), metrics['balanced_accuracy'])
        if best_score is None or score > best_score:
            best, best_score = (model, record), score
    return best[0], {'selected': best[1], 'candidates': candidates}
