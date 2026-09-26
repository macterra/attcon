"""Validate paired training controls and summarize delay robustness."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from attcon.history_reporting import pad_features
from attcon.regulation import WIDTH, load_checkpoint, mixed_states
from attcon.report_sufficiency import batch_fingerprint, make_sufficiency_splits
from attcon.report_uncertainty import context_bootstrap, context_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", nargs="+", type=Path, required=True)
    parser.add_argument("--variable", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    def read(paths):
        data = [json.loads(path.read_text()) for path in paths]
        if len({r["settings"]["seed"] for r in data}) != len(data):
            raise ValueError("duplicate seeds")
        return {r["settings"]["seed"]: r for r in data}
    fixed, variable = read(args.fixed), read(args.variable)
    if set(fixed) != set(variable):
        raise ValueError("unpaired seeds")
    pairs = {}
    for seed in sorted(fixed):
        before, after = fixed[seed], variable[seed]
        if before["settings"]["recipe"] != "fixed" or after["settings"]["recipe"] != "variable":
            raise ValueError("wrong training conditions")
        for field in ("architecture", "epochs", "probe_steps", "null_fits", "train_groups", "fit_groups"):
            if before["settings"][field] != after["settings"][field]:
                raise ValueError(f"mismatched {field}")
        for field in ("dataset_sha256", "delay_assignment_sha256", "source_sha256", "thresholds", "config"):
            if before[field] != after[field]:
                raise ValueError(f"mismatched {field}")
        for field in ("initial_sha256", "updates"):
            if before["training"][field] != after["training"][field]:
                raise ValueError(f"training control mismatch: {field}")
        scores = []
        for run in (before, after):
            saved, agent, config, readouts = load_checkpoint(str(ROOT / run["checkpoint"]))
            data = make_sufficiency_splits(seed, 512)["test"]
            if batch_fingerprint(data) != run["dataset_sha256"]["test"]:
                raise ValueError("test reconstruction mismatch")
            states, _ = mixed_states(agent, data, seed + saved["mixed_delay_offsets"]["test"])
            with torch.no_grad():
                predicted = readouts["state"](pad_features(states, WIDTH)).argmax(-1)
            scores.append(context_scores(predicted, data, config))
        pairs[str(seed)] = {"controls_valid": True,
            "recurrent_step_ratio": after["training"]["recurrent_example_steps"] / before["training"]["recurrent_example_steps"],
            "mixed_report_change_intervals": context_bootstrap(scores[1] - scores[0], seed=seed + 1000),
            "delay_choice_changes": {delay: after["delay_slices"][delay]["action"]["seen_accuracy"] - before["delay_slices"][delay]["action"]["seen_accuracy"] for delay in before["delay_slices"]},
            "fixed_delay_slices": before["delay_slices"], "variable_delay_slices": after["delay_slices"]}
    result = {"audit": "paired_delay_training", "fixed_sources": list(map(str, args.fixed)),
              "variable_sources": list(map(str, args.variable)), "pairs": pairs,
              "boundary": "Same initialization, optimizer updates, task contexts, and reporter-delay assignments. Variable training processes more recurrent steps. Context intervals are conditional, not simultaneous or between-seed uncertainty. No reporting/Stage 8 upgrade."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pairs": len(pairs), "out": str(args.out)}))


if __name__ == "__main__":
    main()
