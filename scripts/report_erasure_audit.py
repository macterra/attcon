"""Diagnostic content-subspace lesions with matched-norm and restoration controls."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from attcon.history_reporting import HistoryConfig
from attcon.report_erasure import erasure_delta, loss_report_metrics
from attcon.report_sufficiency import StateInputReporter, batch_fingerprint, load_sufficiency_checkpoint, make_sufficiency_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "audits/report_erasure_audit.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    runs = []
    for source in args.artifacts:
        original = json.loads(source.read_text())
        saved, agent, readouts, query = load_sufficiency_checkpoint(str(ROOT / original["checkpoint"]))
        settings, config = saved["settings"], HistoryConfig(**saved["config"])
        if settings["architecture"] != "gru" or settings["fit_groups"] != 512:
            raise ValueError("the protocol requires the 512-group GRU condition")
        data = make_sufficiency_splits(settings["seed"], settings["fit_groups"])["test"]
        if batch_fingerprint(data) != original["dataset_sha256"]["test"]:
            raise ValueError("test reconstruction mismatch")
        reporter = StateInputReporter(readouts["state"])
        batch = data.subset(torch.where(data.seen)[0])
        with torch.no_grad():
            state = agent.state(batch.events)
            base_choice = agent.choice(state).argmax(-1)
            base_report = reporter(state).argmax(-1)
            base_query = query(state).argmax(-1)
            eligible = (base_choice == batch.value) & (base_report == batch.value)
            all_cases = torch.ones(len(batch), dtype=torch.bool)
            baseline = loss_report_metrics(base_choice, base_report, batch.value, config.values, all_cases)
            if abs(baseline["true_value_report_accuracy"] - original["evaluations"]["test"]["reports"]["state"]["seen_value_accuracy"]) > 1e-6:
                raise ValueError("baseline report not reproduced")
            basis = saved["content_basis"]
            center = saved["seen_state_mean"]
            conditions = {}
            checks = {"baseline_reproduced": True, "norms_matched": True, "states_restored": True,
                      "choice_restored": True, "report_restored": True}
            for strength in (0.0, 0.25, 0.5, 1.0):
                reference = erasure_delta(state, center, basis, strength)
                outcomes = {}
                for name, directions in (("content", basis), ("random", saved["random_basis"]), ("permuted", saved["permuted_basis"])):
                    delta = erasure_delta(state, center, directions, strength, None if name == "content" else basis)
                    changed = state + delta
                    choice = agent.choice(changed).argmax(-1)
                    report = reporter(changed).argmax(-1)
                    restored = changed - delta
                    norm_error = (delta.norm(dim=1) - reference.norm(dim=1)).abs().max().item()
                    restoration_error = (restored - state).abs().max().item()
                    choice_agreement = (agent.choice(restored).argmax(-1) == base_choice).float().mean().item()
                    report_agreement = (reporter(restored).argmax(-1) == base_report).float().mean().item()
                    checks["norms_matched"] &= norm_error < 1e-5
                    checks["states_restored"] &= restoration_error < 1e-6
                    checks["choice_restored"] &= choice_agreement == 1.0
                    checks["report_restored"] &= report_agreement == 1.0
                    outcomes[name] = {
                        "all_seen": loss_report_metrics(choice, report, batch.value, config.values, all_cases),
                        "baseline_correct_cohort": loss_report_metrics(choice, report, batch.value, config.values, eligible),
                        "query_stability": (query(changed).argmax(-1) == base_query).float().mean().item(),
                        "mean_delta_norm": delta.norm(dim=1).mean().item(), "max_norm_match_error": norm_error,
                        "max_restoration_error": restoration_error,
                        "restored_choice_agreement": choice_agreement, "restored_report_agreement": report_agreement,
                    }
                conditions[str(strength)] = outcomes
        runs.append({"source": str(source), "seed": settings["seed"], "baseline": baseline,
                     "baseline_correct_fraction": eligible.float().mean().item(), "checks": checks, "conditions": conditions,
                     "checkpoint_sha256": hashlib.sha256((ROOT / original["checkpoint"]).read_bytes()).hexdigest()})
    result = {"audit": "report_content_erasure_diagnostic", "protocol": "docs/REPORTING_SIX_CYCLE_PROTOCOL.md",
              "runs": runs, "all_controls_valid": all(all(run["checks"].values()) for run in runs),
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                  for p in ("src/attcon/report_erasure.py", "scripts/report_erasure_audit.py")},
              "boundary": "Off-distribution state lesions of a supervised content subspace, on historically seen content. Choice failure does not prove unavailable internal content; correct reports despite choice failure indicate residual usable information. Incorrect-value and unavailable reports are separate outcomes. No access-erasure support gate or Stage 8 upgrade."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"runs": len(runs), "all_controls_valid": result["all_controls_valid"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
