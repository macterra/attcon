"""Paired, fixed-agent comparison of prespecified reporting-data conditions."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from attcon.history_reporting import HistoryConfig, pad_features
from attcon.report_sufficiency import WIDTH, batch_fingerprint, load_sufficiency_checkpoint, make_sufficiency_splits
from attcon.report_uncertainty import context_bootstrap, context_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small", nargs="+", type=Path, required=True)
    parser.add_argument("--large", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    def read(paths):
        runs = [json.loads(path.read_text()) for path in paths]
        if len({r["settings"]["seed"] for r in runs}) != len(runs):
            raise ValueError("duplicate seeds")
        return {run["settings"]["seed"]: run for run in runs}
    small, large = read(args.small), read(args.large)
    if set(small) != set(large):
        raise ValueError("seed sets must match")
    comparisons = {}
    for seed in sorted(small):
        before, after = small[seed], large[seed]
        for field in ("seed", "architecture", "epochs", "probe_steps", "null_fits", "train_groups"):
            if before["settings"][field] != after["settings"][field]:
                raise ValueError(f"mismatched {field}")
        if before["settings"]["fit_groups"] >= after["settings"]["fit_groups"]:
            raise ValueError("fitting-set size must increase")
        if before["thresholds"] != after["thresholds"] or before["source_sha256"] != after["source_sha256"]:
            raise ValueError("threshold or source mismatch")
        for split in ("agent_train", "validation", "test"):
            if before["dataset_sha256"][split] != after["dataset_sha256"][split]:
                raise ValueError(f"changed {split} data")
        old, agent, old_readouts, _ = load_sufficiency_checkpoint(str(ROOT / before["checkpoint"]))
        new, other, new_readouts, _ = load_sufficiency_checkpoint(str(ROOT / after["checkpoint"]))
        if not all(torch.equal(value, new["agent"][key]) for key, value in old["agent"].items()):
            raise ValueError("agent weights changed")
        data = make_sufficiency_splits(seed, after["settings"]["fit_groups"])["test"]
        if batch_fingerprint(data) != after["dataset_sha256"]["test"]:
            raise ValueError("test reconstruction mismatch")
        with torch.no_grad():
            features = pad_features(agent.state(data.events), WIDTH)
            old_predictions = old_readouts["state"](features).argmax(-1)
            new_predictions = new_readouts["state"](features).argmax(-1)
        differences = context_scores(new_predictions, data, HistoryConfig()) - context_scores(old_predictions, data, HistoryConfig())
        comparisons[str(seed)] = {
            "agent_weights_identical": True, "training_validation_test_identical": True,
            "primary_paired_change_intervals": context_bootstrap(differences, seed=seed + 900),
            "balanced_report_changes": {name: after["evaluations"]["test"]["reports"][name]["balanced_accuracy"]
                - before["evaluations"]["test"]["reports"][name]["balanced_accuracy"]
                for name in before["evaluations"]["test"]["reports"]},
        }
    result = {"audit": "reporter_data_comparison", "small_sources": list(map(str, args.small)),
              "large_sources": list(map(str, args.large)), "comparisons": comparisons,
              "boundary": "Prespecified repeated-test comparison on identical frozen agents and evaluation contexts. Paired context-bootstrap intervals are descriptive, conditional on these systems, and do not change gates or establish introspection."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
