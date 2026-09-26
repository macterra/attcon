from __future__ import annotations

"""Run validation-selected availability reporting on fresh action-trained agents."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from attcon.history_reporting import (
    HistoryAgent, HistoryConfig, action_metrics, content_basis, fit_probe,
    intervention_metrics, make_history_splits, pad_features, report_metrics,
    split_checks, train_agent,
)
from attcon.report_calibration import ActionScoreReporter, select_entropy_reporter, select_readout
from paired_history_reporting import THRESHOLDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1801)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--train-groups", type=int, default=1024)
    parser.add_argument("--probe-steps", type=int, default=500)
    parser.add_argument("--null-fits", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.train_groups, args.probe_steps) < 1 or args.null_fits < 2:
        parser.error("positive training sizes/steps and at least two null fits required")
    out = args.out or ROOT / "audits" / f"paired_history_calibration_seed{args.seed}.json"
    settings = {**vars(args), "out": str(out), "norm_matched_controls": True}
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = HistoryConfig()
    sizes = {"agent_train": args.train_groups, "report_fit": 128, "validation": 64, "test": 128}
    splits = make_history_splits(sizes, seed=args.seed, config=config)
    checks = split_checks(splits, config)
    if not all(checks.values()):
        raise RuntimeError(f"invalid dataset: {checks}")
    torch.manual_seed(args.seed)
    agent = HistoryAgent(config)
    untrained = HistoryAgent(config).eval().requires_grad_(False)
    untrained.load_state_dict(agent.state_dict())
    losses = train_agent(agent, splits["agent_train"], seed=args.seed, epochs=args.epochs)
    print(f"seed {args.seed}: agent loss {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
    with torch.no_grad():
        states = {name: agent.state(data.events) for name, data in splits.items()}
        action_features = {name: pad_features(agent.choice(state), config.hidden) for name, state in states.items()}
        observations = {name: pad_features(data.events[:, -1], config.hidden) for name, data in splits.items()}
        untrained_states = {name: untrained.state(data.events) for name, data in splits.items()}
    fit, validation = splits["report_fit"], splits["validation"]
    labels = fit.report_labels(config)
    # Candidate selection cannot see test tensors or labels.
    selected_state = select_readout(states["report_fit"], labels, states["validation"], validation,
                                    config, seed=args.seed + 1, steps=args.probe_steps)
    selected_action = select_readout(action_features["report_fit"], labels, action_features["validation"],
                                     validation, config, seed=args.seed + 1, steps=args.probe_steps)
    selected_entropy = select_entropy_reporter(agent.choice, states["validation"], validation, config)
    state_reporters = {
        "state": selected_state.model,
        "action_logits": ActionScoreReporter(agent.choice, selected_action.model, config.hidden),
        "entropy": selected_entropy.model,
        "uncalibrated_state": fit_probe(states["report_fit"], labels, config.values + 1,
                                         seed=args.seed + 1, steps=250),
        "uncalibrated_action_logits": ActionScoreReporter(agent.choice,
            fit_probe(action_features["report_fit"], labels, config.values + 1, seed=args.seed + 1, steps=250), config.hidden),
    }
    observation_probe = fit_probe(observations["report_fit"], labels, config.values + 1, seed=args.seed + 1)
    untrained_probe = fit_probe(untrained_states["report_fit"], labels, config.values + 1, seed=args.seed + 1)
    query_probe = fit_probe(states["report_fit"], fit.query, config.keys, seed=args.seed + 2)
    basis = content_basis(states["report_fit"], fit, config)
    permuted = content_basis(states["report_fit"], fit, config, permute_seed=args.seed + 3)
    random = torch.linalg.qr(torch.randn(config.hidden, config.values - 1,
                                       generator=torch.Generator().manual_seed(args.seed + 4))).Q
    directions = {"content": basis, "random": random, "permuted": permuted,
                  "norm_matched_random": random, "norm_matched_permuted": permuted}
    evaluations = {}
    for name in ("validation", "test"):
        data = splits[name]
        reports = {key: report_metrics(probe, states[name], data, config) for key, probe in state_reporters.items()}
        reports["observation"] = report_metrics(observation_probe, observations[name], data, config)
        reports["untrained_state"] = report_metrics(untrained_probe, untrained_states[name], data, config)
        causal = {
            reporter_name: {
                direction_name: intervention_metrics(agent, reporter, query_probe, states[name], data,
                    direction, config, norm_reference_basis=basis if direction_name.startswith("norm_matched") else None)
                for direction_name, direction in directions.items()
            } for reporter_name, reporter in state_reporters.items()
        }
        evaluations[name] = {"action": action_metrics(agent, data), "reports": reports,
                             "interventions": causal["state"], "interventions_by_reporter": causal}
    nulls, null_selections = [], []
    for index in range(args.null_fits):
        seed = args.seed + 100 + index
        permutation = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))
        selected = select_readout(states["report_fit"], labels[permutation], states["validation"],
                                   validation, config, seed=seed, steps=args.probe_steps)
        nulls.append(report_metrics(selected.model, states["test"], splits["test"], config)["balanced_accuracy"])
        null_selections.append(selected.selected)
        if (index + 1) % 5 == 0:
            print(f"seed {args.seed}: {index + 1}/{args.null_fits} selection-matched nulls complete", flush=True)
    p95 = torch.tensor(nulls).quantile(0.95).item()
    test = evaluations["test"]
    report = test["reports"]["state"]
    causal = test["interventions"]["content"]
    observed = {
        "seen_choice_accuracy": test["action"]["seen_accuracy"],
        "seen_report_accuracy": report["seen_value_accuracy"],
        "unseen_report_accuracy": report["unseen_unknown_accuracy"],
        "paired_report_accuracy": report["paired_accuracy"],
        "report_advantage_over_observation": report["balanced_accuracy"] - test["reports"]["observation"]["balanced_accuracy"],
        "report_advantage_over_null_p95": report["balanced_accuracy"] - p95,
        "joint_donor_follow": causal["joint_donor_follow"],
        "joint_advantage_over_random": causal["joint_donor_follow"] - test["interventions"]["random"]["joint_donor_follow"],
        "joint_advantage_over_permuted": causal["joint_donor_follow"] - test["interventions"]["permuted"]["joint_donor_follow"],
        **{name: causal[name] for name in ("report_access_stability", "query_identity_stability", "query_identity_accuracy_before", "eligible_fraction")},
    }
    thresholds = dict(THRESHOLDS)
    for control in ("random", "permuted"):
        name = f"joint_advantage_over_norm_matched_{control}"
        observed[name] = causal["joint_donor_follow"] - test["interventions"][f"norm_matched_{control}"]["joint_donor_follow"]
        thresholds[name] = 0.25
    gates = {name: observed[name] >= threshold for name, threshold in thresholds.items()}
    selections = {name: {"selected": selection.selected, "candidates": selection.candidates}
                  for name, selection in (("state", selected_state), ("action_logits", selected_action), ("entropy", selected_entropy))}
    checkpoint = ROOT / "outputs" / "paired_history_reporting" / f"{out.stem}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"agent": agent.state_dict(), "config": asdict(config), "settings": settings,
                "selections": selections, "reporters": {name: model.state_dict() for name, model in state_reporters.items()},
                "query_probe": query_probe.state_dict(), "content_basis": basis, "random_basis": random,
                "permuted_basis": permuted}, checkpoint)
    source_files = ("src/attcon/history_reporting.py", "src/attcon/report_calibration.py",
                    "scripts/paired_history_calibration.py", "scripts/paired_history_reporting.py",
                    "docs/REPORTING_CALIBRATION.md")
    result = {
        "audit": "paired_history_availability_calibration_v1", "protocol": "docs/REPORTING_CALIBRATION.md",
        "status": "bounded_calibrated_memory_report_coupling" if all(gates.values()) else "reporting_gates_not_met",
        "settings": settings, "config": asdict(config), "torch_version": torch.__version__,
        "split_group_counts": sizes, "split_case_counts": {name: len(data) for name, data in splits.items()},
        "dataset_checks": checks,
        "dataset_sha256": {name: hashlib.sha256(b''.join(tensor.numpy().tobytes() for tensor in data.__dict__.values())).hexdigest()
                           for name, data in splits.items()},
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_files},
        "agent_losses": losses, "action_by_split": {name: action_metrics(agent, data) for name, data in splits.items()},
        "selections": selections, "evaluations": evaluations,
        "report_probe_parameters": {"state": sum(p.numel() for p in selected_state.model.parameters()),
                                    "action_logits": sum(p.numel() for p in selected_action.model.parameters())},
        "permuted_report_null": {"balanced_accuracies": nulls, "p95": p95, "selected_candidates": null_selections},
        "thresholds": thresholds, "observed": observed, "gates": gates,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "claim_boundary": "Validation-selected supervised reports of a frozen value-choice agent. Test data do not select normalization, regularization, or entropy thresholds. Same task and GRU architecture; no spontaneous reporting, regulatory self-model, higher-order state, or Stage 8 claim.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"seed": args.seed, "status": result["status"], "observed": observed, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
