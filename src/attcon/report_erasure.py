"""Selective state lesions; failures do not prove that content is inaccessible."""

import torch


def erasure_delta(states: torch.Tensor, center: torch.Tensor, basis: torch.Tensor, strength: float,
                  reference_basis: torch.Tensor | None = None) -> torch.Tensor:
    if not 0 <= strength <= 1:
        raise ValueError("erasure strength must be in [0, 1]")
    delta = strength * ((center - states) @ basis) @ basis.T
    if reference_basis is not None:
        reference = strength * ((center - states) @ reference_basis) @ reference_basis.T
        delta = delta * (reference.norm(dim=1, keepdim=True) / delta.norm(dim=1, keepdim=True).clamp_min(1e-12))
    return delta


def loss_report_metrics(choice: torch.Tensor, report: torch.Tensor, target: torch.Tensor,
                        unknown: int, mask: torch.Tensor) -> dict:
    count = mask.sum().item()
    if not count:
        return {"count": 0}
    wrong_choice = choice != target
    wrong_value = (report != target) & (report != unknown)
    choice_errors = mask & wrong_choice
    return {
        "count": count,
        "choice_accuracy": (choice[mask] == target[mask]).float().mean().item(),
        "true_value_report_accuracy": (report[mask] == target[mask]).float().mean().item(),
        "unavailable_report_rate": (report[mask] == unknown).float().mean().item(),
        "incorrect_value_report_rate": wrong_value[mask].float().mean().item(),
        "choice_error_count": choice_errors.sum().item(),
        "unavailable_on_choice_error": (report[choice_errors] == unknown).float().mean().item() if choice_errors.any() else None,
        "true_value_report_on_choice_error": (report[choice_errors] == target[choice_errors]).float().mean().item() if choice_errors.any() else None,
        "incorrect_value_report_on_choice_error": wrong_value[choice_errors].float().mean().item() if choice_errors.any() else None,
    }
