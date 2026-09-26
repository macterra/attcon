from __future__ import annotations

"""Run matched nonlinear reporters under the six-cycle frozen protocol."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from attcon.history_reporting import HistoryConfig, action_metrics, content_basis, fit_probe, intervention_metrics, pad_features, report_metrics, split_checks, train_agent
from attcon.report_calibration import ActionScoreReporter
from attcon.report_sufficiency import WIDTH, SequenceAgent, StateInputReporter, batch_fingerprint, history_oracle, make_sufficiency_splits, select_nonlinear
from paired_history_reporting import THRESHOLDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--architecture", choices=("gru", "rnn"), default="gru")
    parser.add_argument("--fit-groups", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--probe-steps", type=int, default=300)
    parser.add_argument("--null-fits", type=int, default=20)
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.probe_steps < 1 or args.null_fits < 2:
        parser.error("positive training steps and at least two null fits required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = HistoryConfig()
    out = args.out or ROOT / "audits" / f"report_sufficiency_{args.architecture}_fit{args.fit_groups}_seed{args.seed}.json"
    settings = {**vars(args), "reuse": str(args.reuse) if args.reuse else None, "out": str(out),
                "train_groups": 1024, "norm_matched_controls": True}
    splits = make_sufficiency_splits(args.seed, args.fit_groups)
    fingerprints = {name: batch_fingerprint(data) for name, data in splits.items()}
    checks = split_checks(splits, config)
    checks["history_oracle_matches_labels"] = all(torch.equal(history_oracle(data.events, config), data.report_labels(config)) for data in splits.values())
    if not all(checks.values()):
        raise RuntimeError(f"invalid data: {checks}")
    torch.manual_seed(args.seed)
    agent = SequenceAgent(config, args.architecture)
    untrained = SequenceAgent(config, args.architecture)
    untrained.load_state_dict(agent.state_dict())
    untrained.eval().requires_grad_(False)
    if args.reuse:
        saved = torch.load(args.reuse, map_location="cpu", weights_only=True)
        for field in ("seed", "architecture", "epochs", "train_groups"):
            if saved["settings"][field] != settings[field]:
                raise ValueError(f"checkpoint mismatch: {field}")
        if saved["config"] != asdict(config):
            raise ValueError("checkpoint configuration mismatch")
        for name in ("agent_train", "validation", "test"):
            if saved["dataset_sha256"][name] != fingerprints[name]:
                raise ValueError(f"checkpoint data mismatch: {name}")
        agent.load_state_dict(saved["agent"])
        agent.eval().requires_grad_(False)
        losses = saved["agent_losses"]
    else:
        losses = train_agent(agent, splits["agent_train"], seed=args.seed, epochs=args.epochs)
    print(f"{args.architecture} seed {args.seed}, fit {args.fit_groups}: agent ready", flush=True)
    with torch.no_grad():
        states = {name: agent.state(data.events) for name, data in splits.items()}
        features = {
            "state": {name: pad_features(state, WIDTH) for name, state in states.items()},
            "action_logits": {name: pad_features(agent.choice(state), WIDTH) for name, state in states.items()},
            "untrained_state": {name: pad_features(untrained.state(data.events), WIDTH) for name, data in splits.items()},
            "full_history": {name: pad_features(data.events.flatten(1), WIDTH) for name, data in splits.items()},
            "observation": {name: pad_features(data.events[:, -1], WIDTH) for name, data in splits.items()},
        }
    fit, validation = splits["report_fit"], splits["validation"]
    labels = fit.report_labels(config)
    selections = {name: select_nonlinear(data["report_fit"], labels, data["validation"], validation,
                    config, seed=args.seed + 1, steps=args.probe_steps) for name, data in features.items()}
    reporters = {"state": StateInputReporter(selections["state"].model),
                 "action_logits": ActionScoreReporter(agent.choice, selections["action_logits"].model, WIDTH)}
    query = fit_probe(states["report_fit"], fit.query, config.keys, seed=args.seed + 2)
    basis = content_basis(states["report_fit"], fit, config)
    permuted = content_basis(states["report_fit"], fit, config, permute_seed=args.seed + 3)
    random = torch.linalg.qr(torch.randn(config.hidden, config.values - 1, generator=torch.Generator().manual_seed(args.seed + 4))).Q
    directions = {"content": basis, "random": random, "permuted": permuted,
                  "norm_matched_random": random, "norm_matched_permuted": permuted}
    evaluations = {}
    for name in ("validation", "test"):
        batch = splits[name]
        causal = {reporter_name: {
            key: intervention_metrics(agent, reporter, query, states[name], batch, direction, config,
                                       norm_reference_basis=basis if key.startswith("norm_matched") else None)
            for key, direction in directions.items()} for reporter_name, reporter in reporters.items()}
        evaluations[name] = {
            "action": action_metrics(agent, batch),
            "reports": {key: report_metrics(selection.model, features[key][name], batch, config) for key, selection in selections.items()},
            "history_oracle_accuracy": (history_oracle(batch.events, config) == batch.report_labels(config)).float().mean().item(),
            "interventions": causal["state"], "interventions_by_reporter": causal,
        }
    nulls, null_selections = [], []
    for index in range(args.null_fits):
        seed = args.seed + 100 + index
        permutation = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))
        selection = select_nonlinear(features["state"]["report_fit"], labels[permutation], features["state"]["validation"],
                                     validation, config, seed=seed, steps=args.probe_steps)
        nulls.append(report_metrics(selection.model, features["state"]["test"], splits["test"], config)["balanced_accuracy"])
        null_selections.append(selection.selected)
        if (index + 1) % 5 == 0:
            print(f"seed {args.seed}: nulls {index + 1}/{args.null_fits}", flush=True)
    p95 = torch.tensor(nulls).quantile(0.95).item()
    test = evaluations["test"]
    report, causal = test["reports"]["state"], test["interventions"]["content"]
    observed = {
        "seen_choice_accuracy": test["action"]["seen_accuracy"],
        "seen_report_accuracy": report["seen_value_accuracy"],
        "unseen_report_accuracy": report["unseen_unknown_accuracy"],
        "paired_report_accuracy": report["paired_accuracy"],
        "report_advantage_over_observation": report["balanced_accuracy"] - test["reports"]["observation"]["balanced_accuracy"],
        "report_advantage_over_null_p95": report["balanced_accuracy"] - p95,
        "joint_donor_follow": causal["joint_donor_follow"],
        **{key: causal[key] for key in ("report_access_stability", "query_identity_stability", "query_identity_accuracy_before", "eligible_fraction")},
    }
    thresholds = dict(THRESHOLDS)
    for key in ("random", "permuted", "norm_matched_random", "norm_matched_permuted"):
        metric = f"joint_advantage_over_{key}"
        observed[metric] = causal["joint_donor_follow"] - test["interventions"][key]["joint_donor_follow"]
        thresholds[metric] = 0.25
    gates = {name: observed[name] >= threshold for name, threshold in thresholds.items()}
    checkpoint = ROOT / "outputs" / "paired_history_reporting" / f"{out.stem}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    selection_records = {name: {"selected": selection.selected, "candidates": selection.candidates} for name, selection in selections.items()}
    torch.save({"agent": agent.state_dict(), "config": asdict(config), "settings": settings, "dataset_sha256": fingerprints,
                "agent_losses": losses, "readouts": {name: selection.model.state_dict() for name, selection in selections.items()},
                "query_probe": query.state_dict(), "content_basis": basis, "random_basis": random, "permuted_basis": permuted,
                "seen_state_mean": states["report_fit"][fit.seen].mean(0), "selections": selection_records}, checkpoint)
    source_paths = ("src/attcon/report_sufficiency.py", "src/attcon/history_reporting.py", "src/attcon/report_calibration.py",
                    "scripts/report_sufficiency.py", "scripts/paired_history_reporting.py", "docs/REPORTING_SIX_CYCLE_PROTOCOL.md")
    result = {
        "audit": "nonlinear_report_sufficiency_v1", "protocol": "docs/REPORTING_SIX_CYCLE_PROTOCOL.md",
        "status": "bounded_nonlinear_report_support" if all(gates.values()) else "reporting_gates_not_met",
        "settings": settings, "config": asdict(config), "torch_version": str(torch.__version__),
        "split_group_counts": {name: len(data.group.unique()) for name, data in splits.items()},
        "split_case_counts": {name: len(data) for name, data in splits.items()}, "dataset_checks": checks,
        "dataset_sha256": fingerprints, "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths},
        "agent_losses": losses, "action_by_split": {name: action_metrics(agent, data) for name, data in splits.items()},
        "agent_parameters": sum(p.numel() for p in agent.parameters()),
        "report_probe_parameters": {name: sum(p.numel() for p in selection.model.parameters()) for name, selection in selections.items()},
        "selections": selection_records, "evaluations": evaluations,
        "permuted_report_null": {"balanced_accuracies": nulls, "p95": p95, "selected_candidates": null_selections},
        "thresholds": thresholds, "observed": observed, "gates": gates, "checkpoint": str(checkpoint.relative_to(ROOT)),
        "claim_boundary": "Matched supervised nonlinear reporters of frozen recurrent state. Same synthetic task; history oracle is an information upper bound, not introspection. No spontaneous reporting, regulatory self-model, or Stage 8 upgrade.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"seed": args.seed, "status": result["status"], "observed": observed}, indent=2))


if __name__ == "__main__":
    main()
