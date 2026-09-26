"""Evaluate extra blank delays without training or retuning any reporter."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from attcon.history_reporting import HistoryConfig, action_metrics, pad_features, report_metrics
from attcon.report_calibration import ActionScoreReporter
from attcon.report_delay import insert_delay
from attcon.report_sufficiency import WIDTH, StateInputReporter, batch_fingerprint, history_oracle, load_sufficiency_checkpoint, make_sufficiency_splits
from attcon.report_uncertainty import context_bootstrap, context_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "audits/report_delay_stress.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    runs = []
    for source in args.artifacts:
        original = json.loads(source.read_text())
        saved, agent, readouts, query = load_sufficiency_checkpoint(str(ROOT / original["checkpoint"]))
        config = HistoryConfig(**saved["config"])
        seed = saved["settings"]["seed"]
        data = make_sufficiency_splits(seed, saved["settings"]["fit_groups"])["test"]
        if batch_fingerprint(data) != original["dataset_sha256"]["test"]:
            raise ValueError("test reconstruction mismatch")
        if saved["settings"]["fit_groups"] != 512:
            raise ValueError("the frozen delay protocol requires 512 fitting groups")
        reporters = {"state": StateInputReporter(readouts["state"]),
                     "action_logits": ActionScoreReporter(agent.choice, readouts["action_logits"], WIDTH)}
        conditions = {}
        baseline_scores = None
        with torch.no_grad():
            for delay in (0, 1, 3, 6):
                batch = insert_delay(data, delay)
                state = agent.state(batch.events)
                oracle = history_oracle(batch.events, config)
                if not torch.equal(oracle, batch.report_labels(config)):
                    raise ValueError("delay changed available information")
                reports = {name: report_metrics(reporter, state, batch, config) for name, reporter in reporters.items()}
                scores = context_scores(reporters["state"](state).argmax(-1), batch, config)
                action = action_metrics(agent, batch)
                if delay == 0:
                    for name, values in reports.items():
                        if values != original["evaluations"]["test"]["reports"][name]:
                            raise ValueError("baseline report not reproduced")
                    baseline_scores = scores
                conditions[str(delay)] = {
                    "action": action, "reports": reports, "oracle_accuracy": 1.0,
                    "query_accuracy": (query(state).argmax(-1) == data.query).float().mean().item(),
                    "primary_paired_change_intervals": context_bootstrap(scores - baseline_scores, seed=seed + 1000),
                    "basic_gates": {
                        "seen_choice": action["seen_accuracy"] >= 0.85,
                        "seen_report": reports["state"]["seen_value_accuracy"] >= 0.85,
                        "unavailable_report": reports["state"]["unseen_unknown_accuracy"] >= 0.85,
                        "paired_report": reports["state"]["paired_accuracy"] >= 0.75,
                    },
                }
        runs.append({"source": str(source), "seed": seed, "architecture": saved["settings"]["architecture"],
                     "checkpoint_sha256": hashlib.sha256((ROOT / original["checkpoint"]).read_bytes()).hexdigest(),
                     "baseline_reproduced": True, "conditions": conditions})
    result = {"audit": "frozen_reporting_delay_stress", "protocol": "docs/REPORTING_SIX_CYCLE_PROTOCOL.md",
              "runs": runs, "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                  for p in ("src/attcon/report_delay.py", "scripts/report_delay_stress.py")},
              "boundary": "No refitting or test selection. Blank events preserve observable history information but shift recurrent dynamics. Basic gates are descriptive stress checks, not the full 15-gate assay. Context intervals are conditional, not simultaneous; no Stage 8 upgrade."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"runs": len(runs), "out": str(args.out)}))


if __name__ == "__main__":
    main()
