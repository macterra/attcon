from __future__ import annotations

"""Paired delay training with independently fitted reporting instruments."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from attcon.history_reporting import content_basis, fit_probe, intervention_metrics, pad_features, report_metrics, split_checks
from attcon.report_calibration import ActionScoreReporter, select_entropy_reporter
from attcon.report_delay import insert_delay
from attcon.report_sufficiency import batch_fingerprint, history_oracle, make_sufficiency_splits
from attcon.regulation import WIDTH, StateReporter, agent_for, choice_metrics, mixed_states, select_report, train_delay_agent
from paired_history_reporting import THRESHOLDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--recipe", choices=("fixed", "variable"), default="variable")
    parser.add_argument("--architecture", choices=("gru", "rnn_matched"), default="gru")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--probe-steps", type=int, default=300)
    parser.add_argument("--null-fits", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.epochs, args.probe_steps) < 1 or args.null_fits < 2:
        parser.error("positive steps and at least two null fits required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    splits = make_sufficiency_splits(args.seed, 512)
    torch.manual_seed(args.seed)
    agent, config = agent_for(args.architecture)
    training = train_delay_agent(agent, splits["agent_train"], seed=args.seed, recipe=args.recipe, epochs=args.epochs)
    print(f"{args.recipe} {args.architecture} {args.seed}: agent ready", flush=True)
    offsets = {"agent_train": 11, "report_fit": 13, "validation": 17, "test": 19}
    states, delays = {}, {}
    for name, data in splits.items():
        states[name], delays[name] = mixed_states(agent, data, args.seed + offsets[name])
    checks = split_checks(splits, config)
    checks["history_oracle_matches_labels"] = all(torch.equal(history_oracle(data.events, config), data.report_labels(config)) for data in splits.values())
    if not all(checks.values()):
        raise ValueError(f"invalid data: {checks}")
    with torch.no_grad():
        features = {"state": {name: pad_features(state, WIDTH) for name, state in states.items()},
                    "action_logits": {name: pad_features(agent.choice(state), WIDTH) for name, state in states.items()},
                    "observation": {name: pad_features(data.events[:, -1], WIDTH) for name, data in splits.items()}}
    fit, validation = splits["report_fit"], splits["validation"]
    labels = fit.report_labels(config)
    selections = {name: select_report(data["report_fit"], labels, data["validation"], validation,
                  config, seed=args.seed + 1, steps=args.probe_steps) for name, data in features.items()}
    entropy = select_entropy_reporter(agent.choice, states["validation"], validation, config)
    reporters = {"state": StateReporter(selections["state"].model),
                 "action_logits": ActionScoreReporter(agent.choice, selections["action_logits"].model, WIDTH),
                 "entropy": entropy.model}
    query = fit_probe(states["report_fit"], fit.query, config.keys, seed=args.seed + 2)
    basis = content_basis(states["report_fit"], fit, config)
    permuted = content_basis(states["report_fit"], fit, config, permute_seed=args.seed + 3)
    random = torch.linalg.qr(torch.randn(config.hidden, config.values - 1, generator=torch.Generator().manual_seed(args.seed + 4))).Q
    directions = {"content": basis, "random": random, "permuted": permuted,
                  "norm_matched_random": random, "norm_matched_permuted": permuted}
    evaluations = {}
    for name in ("validation", "test"):
        data = splits[name]
        reports = {key: report_metrics(reporter, states[name], data, config) for key, reporter in reporters.items()}
        reports["observation"] = report_metrics(selections["observation"].model, features["observation"][name], data, config)
        interventions = {key: intervention_metrics(agent, reporters["state"], query, states[name], data, direction, config,
                        norm_reference_basis=basis if key.startswith("norm_matched") else None) for key, direction in directions.items()}
        evaluations[name] = {"action": choice_metrics(agent, states[name], data), "reports": reports, "interventions": interventions}
    delay_slices = {}
    for delay in (0, 1, 3, 6, 9):
        data = insert_delay(splits["test"], delay)
        with torch.no_grad():
            state = agent.state(data.events)
        if not torch.equal(history_oracle(data.events, config), data.report_labels(config)):
            raise ValueError("delay changed available information")
        delay_slices[str(delay)] = {"action": choice_metrics(agent, state, data),
            "reports": {name: report_metrics(reporter, state, data, config) for name, reporter in reporters.items()}, "oracle_accuracy": 1.0}
    nulls, null_selections = [], []
    for index in range(args.null_fits):
        seed = args.seed + 100 + index
        permutation = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))
        selected = select_report(features["state"]["report_fit"], labels[permutation], features["state"]["validation"],
                                 validation, config, seed=seed, steps=args.probe_steps)
        nulls.append(report_metrics(selected.model, features["state"]["test"], splits["test"], config)["balanced_accuracy"])
        null_selections.append(selected.selected)
        if (index + 1) % 5 == 0:
            print(f"{args.recipe} {args.architecture} {args.seed}: nulls {index + 1}/{args.null_fits}", flush=True)
    p95 = torch.tensor(nulls).quantile(0.95).item()
    test = evaluations["test"]
    report, causal = test["reports"]["state"], test["interventions"]["content"]
    observed = {"seen_choice_accuracy": test["action"]["seen_accuracy"], "seen_report_accuracy": report["seen_value_accuracy"],
        "unseen_report_accuracy": report["unseen_unknown_accuracy"], "paired_report_accuracy": report["paired_accuracy"],
        "report_advantage_over_observation": report["balanced_accuracy"] - test["reports"]["observation"]["balanced_accuracy"],
        "report_advantage_over_null_p95": report["balanced_accuracy"] - p95,
        **{name: causal[name] for name in ("joint_donor_follow", "report_access_stability", "query_identity_stability", "query_identity_accuracy_before", "eligible_fraction")}}
    thresholds = dict(THRESHOLDS)
    for key in ("random", "permuted", "norm_matched_random", "norm_matched_permuted"):
        observed[f"joint_advantage_over_{key}"] = causal["joint_donor_follow"] - test["interventions"][key]["joint_donor_follow"]
        thresholds[f"joint_advantage_over_{key}"] = 0.25
    gates = {key: observed[key] >= threshold for key, threshold in thresholds.items()}
    settings = {**vars(args), "out": str(args.out), "train_groups": 1024, "fit_groups": 512, "norm_matched_controls": True}
    fingerprints = {name: batch_fingerprint(data) for name, data in splits.items()}
    records = {name: {"selected": selected.selected, "candidates": selected.candidates} for name, selected in selections.items()}
    records["entropy"] = {"selected": entropy.selected, "candidates": entropy.candidates}
    checkpoint = ROOT / "outputs/regulation" / f"{args.out.stem}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"agent": agent.state_dict(), "config": asdict(config), "settings": settings, "training": training,
        "dataset_sha256": fingerprints, "mixed_delay_offsets": offsets,
        "readouts": {name: selected.model.state_dict() for name, selected in selections.items()}, "selections": records,
        "query_probe": query.state_dict(), "content_basis": basis, "random_basis": random, "permuted_basis": permuted,
        "seen_state_mean": states["report_fit"][fit.seen].mean(0)}, checkpoint)
    source_paths = ("src/attcon/regulation.py", "scripts/train_regulation.py", "docs/REGULATION_PROTOCOL.md",
                    "src/attcon/history_reporting.py", "src/attcon/report_sufficiency.py", "src/attcon/report_calibration.py")
    result = {"audit": "delay_regulation_reporting_v1", "protocol": "docs/REGULATION_PROTOCOL.md",
        "status": "bounded_reporting_supported" if all(gates.values()) else "reporting_gates_not_met",
        "settings": settings, "config": asdict(config), "torch_version": str(torch.__version__), "training": training,
        "split_group_counts": {name: len(data.group.unique()) for name, data in splits.items()},
        "dataset_checks": checks, "dataset_sha256": fingerprints,
        "delay_assignment_sha256": {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in delays.items()},
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths},
        "agent_parameters": sum(p.numel() for p in agent.parameters()),
        "report_probe_parameters": {name: sum(p.numel() for p in selected.model.parameters()) for name, selected in selections.items()},
        "evaluations": evaluations, "delay_slices": delay_slices, "selections": records,
        "permuted_report_null": {"balanced_accuracies": nulls, "p95": p95, "selected_candidates": null_selections},
        "thresholds": thresholds, "observed": observed, "gates": gates, "checkpoint": str(checkpoint.relative_to(ROOT)),
        "claim_boundary": "Task-only recurrent training with external supervised reporters. Variable training uses more recurrent steps at equal optimizer updates. Reporting gates concern mixed trained delays; delay 9 is extrapolation. No spontaneous introspection or Stage 8 upgrade."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "observed": observed}, indent=2))


if __name__ == "__main__":
    main()
