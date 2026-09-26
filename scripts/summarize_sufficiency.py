"""Validate replicated sufficiency audits and add context-cluster intervals."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from attcon.history_reporting import HistoryConfig, pad_features
from attcon.report_sufficiency import WIDTH, batch_fingerprint, load_sufficiency_checkpoint, make_sufficiency_splits
from attcon.report_uncertainty import context_bootstrap, context_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / "scripts/summarize_history_reporting.py"),
                    *map(str, args.artifacts), "--out", str(args.out), "--audit-name", args.out.stem],
                   check=True, stdout=subprocess.DEVNULL)
    summary = json.loads(args.out.read_text())
    torch.set_num_threads(1)
    intervals = {}
    for source in args.artifacts:
        run = json.loads(source.read_text())
        saved, agent, readouts, _ = load_sufficiency_checkpoint(str(ROOT / run["checkpoint"]))
        data = make_sufficiency_splits(run["settings"]["seed"], run["settings"]["fit_groups"])["test"]
        if batch_fingerprint(data) != run["dataset_sha256"]["test"]:
            raise ValueError("test reconstruction mismatch")
        with torch.no_grad():
            predicted = readouts["state"](pad_features(agent.state(data.events), WIDTH)).argmax(-1)
        interval = context_bootstrap(context_scores(predicted, data, HistoryConfig(**saved["config"])),
                                     seed=run["settings"]["seed"] + 800)
        for name, values in interval.items():
            if abs(values["mean"] - run["evaluations"]["test"]["reports"]["state"][name]) > 1e-6:
                raise ValueError(f"saved report not reproduced: {name}")
        intervals[str(run["settings"]["seed"])] = interval
    summary["context_bootstrap_95_intervals"] = intervals
    summary["uncertainty_boundary"] = "2000 context-cluster resamples per seed, conditional on the fitted agent/reporter. These are descriptive percentile intervals, not simultaneous intervals, seed-level uncertainty, or new gate thresholds."
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "ranges": summary["observed_ranges"], "gates": summary["gate_pass_counts"]}, indent=2))


if __name__ == "__main__":
    main()
