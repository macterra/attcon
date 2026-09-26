from __future__ import annotations

"""Run the preregistered action-trained paired-history reporting pilot."""

import argparse
from dataclasses import asdict
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


THRESHOLDS = {
    "seen_choice_accuracy": 0.85,
    "seen_report_accuracy": 0.85,
    "unseen_report_accuracy": 0.85,
    "paired_report_accuracy": 0.75,
    "report_advantage_over_observation": 0.25,
    "report_advantage_over_null_p95": 0.20,
    "joint_donor_follow": 0.70,
    "joint_advantage_over_random": 0.25,
    "joint_advantage_over_permuted": 0.25,
    "report_access_stability": 0.90,
    "query_identity_stability": 0.90,
    "query_identity_accuracy_before": 0.85,
    "eligible_fraction": 0.75,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train-groups", type=int, default=256)
    parser.add_argument("--probe-steps", type=int, default=250)
    parser.add_argument("--null-fits", type=int, default=20)
    parser.add_argument("--norm-matched-controls", action="store_true")
    parser.add_argument("--out", default="audits/paired_history_reporting_seed1729.json")
    args = parser.parse_args()
    if args.epochs < 1 or args.probe_steps < 1 or args.null_fits < 2:
        parser.error("positive epochs/probe steps and at least two null fits required")
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
    # Same initial weights, so the untrained control isolates training effects.
    untrained.load_state_dict(agent.state_dict())
    losses = train_agent(agent, splits["agent_train"], seed=args.seed, epochs=args.epochs)
    print(f"agent trained: loss {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
    features = {name: {} for name in ("state", "observation", "action_logits", "untrained_state")}
    with torch.no_grad():
        for name, batch in splits.items():
            state = agent.state(batch.events)
            features["state"][name] = state
            features["observation"][name] = pad_features(batch.events[:, -1], config.hidden)
            features["action_logits"][name] = pad_features(agent.choice(state), config.hidden)
            features["untrained_state"][name] = untrained.state(batch.events)
    fit = splits["report_fit"]
    labels = fit.report_labels(config)
    reporters = {
        name: fit_probe(data["report_fit"], labels, config.values + 1,
                        seed=args.seed + 1, steps=args.probe_steps)
        for name, data in features.items()
    }
    query_probe = fit_probe(features["state"]["report_fit"], fit.query, config.keys,
                            seed=args.seed + 2, steps=args.probe_steps)
    basis = content_basis(features["state"]["report_fit"], fit, config)
    permuted_basis = content_basis(features["state"]["report_fit"], fit, config, permute_seed=args.seed + 3)
    generator = torch.Generator().manual_seed(args.seed + 4)
    random_basis = torch.linalg.qr(torch.randn(config.hidden, config.values - 1, generator=generator)).Q
    evaluations = {}
    for split in ("validation", "test"):
        batch = splits[split]
        evaluations[split] = {
            "action": action_metrics(agent, batch),
            "reports": {
                name: report_metrics(reporters[name], data[split], batch, config)
                for name, data in features.items()
            },
            "interventions": {
                name: intervention_metrics(agent, reporters["state"], query_probe,
                    features["state"][split], batch, directions, config)
                for name, directions in (("content", basis), ("random", random_basis), ("permuted", permuted_basis))
            },
        }
        if args.norm_matched_controls:
            for name, directions in (("norm_matched_random", random_basis), ("norm_matched_permuted", permuted_basis)):
                evaluations[split]["interventions"][name] = intervention_metrics(
                    agent, reporters["state"], query_probe, features["state"][split],
                    batch, directions, config, norm_reference_basis=basis,
                )
    nulls = []
    for index in range(args.null_fits):
        seed = args.seed + 100 + index
        permutation = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))
        probe = fit_probe(features["state"]["report_fit"], labels[permutation], config.values + 1,
                          seed=seed, steps=args.probe_steps)
        nulls.append(report_metrics(probe, features["state"]["test"], splits["test"], config)["balanced_accuracy"])
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
    gates = {name: observed[name] >= threshold for name, threshold in THRESHOLDS.items()}
    thresholds = dict(THRESHOLDS)
    if args.norm_matched_controls:
        for control in ("random", "permuted"):
            name = f"joint_advantage_over_norm_matched_{control}"
            observed[name] = causal["joint_donor_follow"] - test["interventions"][f"norm_matched_{control}"]["joint_donor_follow"]
            thresholds[name] = 0.25
            gates[name] = observed[name] >= thresholds[name]
    checkpoint = ROOT / "outputs" / "paired_history_reporting" / f"{Path(args.out).stem}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": asdict(config), "args": vars(args), "agent": agent.state_dict(),
        "reporters": {name: probe.state_dict() for name, probe in reporters.items()},
        "query_probe": query_probe.state_dict(), "content_basis": basis,
        "random_basis": random_basis, "permuted_basis": permuted_basis,
    }, checkpoint)
    result = {
        "audit": "paired_history_reporting_v2" if args.norm_matched_controls else "paired_history_reporting_v1", "protocol": "docs/REPORTING_PLAN.md",
        "status": "bounded_memory_report_coupling" if all(gates.values()) else "pilot_gates_not_met",
        "settings": vars(args), "config": asdict(config), "torch_version": torch.__version__,
        "split_group_counts": sizes, "split_case_counts": {name: len(batch) for name, batch in splits.items()},
        "dataset_checks": checks, "agent_losses": losses,
        "action_by_split": {name: action_metrics(agent, batch) for name, batch in splits.items()},
        "report_probe_parameters": {name: sum(p.numel() for p in probe.parameters()) for name, probe in reporters.items()},
        "evaluations": evaluations,
        "permuted_report_null": {"balanced_accuracies": nulls, "p95": p95},
        "thresholds": thresholds, "observed": observed, "gates": gates,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "claim_boundary": "Single-seed frozen-state decoding and content-intervention assay. Agent trained only on value choice, including unanswerable trials. Reporter is supervised; recurrent memory is provided. No spontaneous reporting, regulatory self-model, higher-order state, independent theory-family convergence, or Stage 8 claim.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "observed": observed, "gates": gates, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
