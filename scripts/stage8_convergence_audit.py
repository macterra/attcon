from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITS = ROOT / "audits"


def _load(name: str) -> dict:
    return json.loads((AUDITS / name).read_text())


def build_stage8_convergence_audit() -> dict:
    full = _load("post_rehab_full_eval_tune_prob_035_summary.json")
    cross_arch = _load("cross_architecture_rnn_summary.json")
    perturbation = _load("perturbational_multiseed_with_rnn.json")
    branch_c_cross = _load("branch_c_binding_cross_model.json")
    branch_c_seeds = _load("branch_c_binding_multiseed.json")
    branch_d_cross = _load("branch_d_access_cross_model.json")
    branch_d_seeds = _load("branch_d_access_multiseed.json")
    branch_e = _load("branch_e_higher_order_pilot.json")
    branch_f = _load("branch_f_broadcast_multiseed.json")
    integrated_directional = _load(
        "stage8_integrated_content_directional_multiseed.json"
    )
    task_induced_routing = _load("stage8_task_induced_routing_duration.json")

    rnn_perturbation = next(
        checkpoint
        for checkpoint in perturbation["checkpoints"]
        if checkpoint["label"] == "rnn_tune_prob_035"
    )
    evidence = {
        "attention_control_foundation": {
            "stage3_checkpoint_family_robust": full["stage3"][
                "checkpoint_family_verdict"
            ]
            == "robust",
            "stage3_cross_architecture_replication": cross_arch["claims"][
                "stage3_explicit_attention_modeling_robust"
            ]["replicates"],
        },
        "access_report_family": {
            "stage6a_capacity_audited": full["stage6a"]["supported"]
            and full["stage6a"]["capacity_audit_passed"],
            "stage6a_cross_architecture_replication": cross_arch["claims"][
                "stage6A_report_probes"
            ]["replicates"],
            "branch_d_multi_seed": branch_d_seeds["multi_seed_supported"],
            "branch_d_cross_model_surface": branch_d_cross[
                "cross_model_surface_supported"
            ],
            "branch_d_boundary": branch_d_cross["support_boundary"],
        },
        "non_reportability_family": {
            "perturbation_seed_robust": perturbation[
                "robust_across_perturbation_seeds"
            ],
            "perturbation_checkpoint_replicated": perturbation[
                "replicates_across_checkpoints"
            ],
            "perturbation_rnn_supported_fraction": rnn_perturbation[
                "supported_fraction"
            ],
            "branch_c_multi_seed": branch_c_seeds["multi_seed_supported"],
            "branch_c_cross_model_surface": branch_c_cross[
                "cross_model_supported"
            ],
        },
        "engineering_only_families_excluded": {
            "branch_e_engineering_gates_pass": branch_e[
                "all_engineering_gates_pass"
            ],
            "branch_e_theoretical_support": branch_e["branch_e_supported"],
            "branch_f_engineering_seed_robust": branch_f[
                "multi_seed_engineering_support"
            ],
            "branch_f_theoretical_support": branch_f["branch_f_supported"],
        },
        "integrated_same_content_assay": {
            "multi_seed_directional_engineering_support": integrated_directional[
                "multi_seed_directional_engineering_support"
            ],
            "minimum_directional_metrics": integrated_directional["summary"][
                "minimum_metrics"
            ],
            "stage8_same_content_gate_satisfied": integrated_directional[
                "stage8_same_content_gate_satisfied"
            ],
            "support_boundary": integrated_directional["support_boundary"],
            "multi_seed_task_induced_routing_support": task_induced_routing[
                "duration_robust_support"
            ],
            "task_induced_support_rate": task_induced_routing["summary"][
                "support_rate"
            ],
            "minimum_task_induced_metrics": task_induced_routing["summary"][
                "minimum_metrics"
            ],
            "task_induced_support_boundary": task_induced_routing[
                "support_boundary"
            ],
        },
        "comparators": {
            "base_comparators_fail_as_intended": full[
                "comparators_failed_as_intended"
            ],
            "negative_controls_fail_as_intended": full["negative_controls"][
                "failed_as_intended"
            ],
            "cross_architecture_replicated_claim_fraction": cross_arch[
                "replicated_count"
            ]
            / cross_arch["total_comparable"],
        },
    }
    gates = {
        "robust_attention_control_foundation": {
            "status": "pass",
            "reason": "Stage 3 is seed/checkpoint-family robust and replicates on the RNN.",
        },
        "access_report_family": {
            "status": "partial",
            "reason": (
                "Stage 6A is capacity-audited and cross-architecture; Branch D is seed/cross-model "
                "strong bounded. No genuinely different benchmark replication exists."
            ),
        },
        "non_reportability_family": {
            "status": "partial",
            "reason": (
                "Perturbational evidence is seed/checkpoint/RNN replicated and Branch C is "
                "seed/cross-model strong bounded, but both lack a genuinely different benchmark."
            ),
        },
        "same_internal_content_across_families": {
            "status": "partial",
            "reason": (
                "The integrated assay tracks one identity through binding and access and shows "
                "seed-robust, disjoint-split directional causal overlap against permuted and "
                "split-state nulls. A learned route also emerges under severe private-lane "
                "robustness pressure, but not ordinary joint supervision. Directional transfer "
                "replicates over three fresh seeds, while the full routing gate passes only two; "
                "this remains short of independent qualifying theory families."
            ),
        },
        "comparators_fail_as_predicted": {
            "status": "pass",
            "reason": "Base negative controls and first-class comparators fail as intended.",
        },
        "different_architecture_replication": {
            "status": "partial",
            "reason": (
                "Six of seven claims replicate from GRU to ungated RNN, but these remain closely "
                "related recurrent architectures and cue-switch adaptation drops."
            ),
        },
        "different_benchmark_replication": {
            "status": "fail",
            "reason": (
                "Surface-vocabulary/cardinality variants preserve their synthetic task structure "
                "and do not satisfy the roadmap's different-benchmark requirement."
            ),
        },
        "claim_framed_as_evidence_not_proof": {
            "status": "pass",
            "reason": (
                "Repository claims explicitly distinguish bounded/engineering evidence from "
                "proof of consciousness."
            ),
        },
    }
    counts = {
        status: sum(gate["status"] == status for gate in gates.values())
        for status in ("pass", "partial", "fail")
    }
    return {
        "audit": "stage8_convergence_current",
        "evidence": evidence,
        "gates": gates,
        "gate_counts": counts,
        "stage8_supported": all(
            gate["status"] == "pass" for gate in gates.values()
        ),
        "verdict": (
            "not_met: three gates pass, four are partial, and one fails. Same-content causal "
            "overlap now has a seed-robust engineered assay, but qualifying independent-family "
            "overlap and a genuinely different benchmark remain open."
        ),
        "next_decisive_experiment": (
            "Remove forced state sharing and test whether a neutral jointly trained controller "
            "develops the same directional overlap beyond a matched split-state null, then "
            "replicate the package on a structurally different task."
        ),
    }


def main() -> None:
    result = build_stage8_convergence_audit()
    output = AUDITS / "stage8_convergence_current.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
