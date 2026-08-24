from __future__ import annotations

"""Supersede the rescaled-dropout routing interpretation with corrected occlusion tests."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITS = ROOT / "audits"


def build_correction() -> dict:
    rescaled = json.loads(
        (AUDITS / "stage8_task_induced_routing_duration.json").read_text()
    )
    corrected_from_scratch = json.loads(
        (AUDITS / "stage8_task_induced_routing_no_rescale.json").read_text()
    )
    corrected_curriculum = json.loads(
        (AUDITS / "stage8_task_induced_routing_curriculum_no_rescale.json").read_text()
    )
    corrected = (corrected_from_scratch, corrected_curriculum)
    return {
        "audit": "stage8_task_induced_routing_correction",
        "status": "task_induced_routing_unsupported",
        "supersedes_support_interpretation_of": [
            "audits/stage8_task_induced_routing_sweep.json",
            "audits/stage8_task_induced_routing_multiseed.json",
            "audits/stage8_task_induced_routing_duration.json",
        ],
        "confound": (
            "Whole-lane dropout used inverted-dropout scaling. At 0.95 dropout, the private "
            "access state was multiplied by 20 whenever present during training but had normal "
            "magnitude at evaluation. The apparent routing effect therefore does not isolate "
            "robustness to lane absence."
        ),
        "rescaled_dropout_result_retained_as_diagnostic_only": {
            "support_rate_90_epochs": rescaled["summary"]["support_rate"],
            "minimum_joint_directional_follow": rescaled["summary"][
                "minimum_metrics"
            ]["learned_joint_directional_follow"],
            "valid_for_task_induced_support": False,
        },
        "corrected_tests": {
            "from_scratch": {
                "artifact": "audits/stage8_task_induced_routing_no_rescale.json",
                "all_gates_pass": corrected_from_scratch[
                    "all_routing_gates_pass"
                ],
                "observed": corrected_from_scratch["observed"],
            },
            "viability_first_curriculum": {
                "artifact": "audits/stage8_task_induced_routing_curriculum_no_rescale.json",
                "all_gates_pass": corrected_curriculum["all_routing_gates_pass"],
                "pre_pressure_learned_task_joint_accuracy": corrected_curriculum[
                    "models"
                ]["learned"]["pre_pressure_heldout"][
                    "binding_and_access_joint_accuracy"
                ],
                "pre_pressure_blocked_task_joint_accuracy": corrected_curriculum[
                    "models"
                ]["blocked"]["pre_pressure_heldout"][
                    "binding_and_access_joint_accuracy"
                ],
                "observed": corrected_curriculum["observed"],
            },
        },
        "corrected_summary": {
            "all_corrected_gates_pass": all(
                run["all_routing_gates_pass"] for run in corrected
            ),
            "maximum_joint_directional_follow": max(
                run["observed"]["learned_joint_directional_follow"]
                for run in corrected
            ),
            "minimum_learned_task_joint_accuracy": min(
                run["observed"]["learned_task_joint_accuracy"]
                for run in corrected
            ),
            "minimum_blocked_task_joint_accuracy": min(
                run["observed"]["blocked_task_joint_accuracy"]
                for run in corrected
            ),
        },
        "task_induced_routing_supported": False,
        "stage8_same_content_gate_satisfied": False,
        "support_boundary": (
            "Corrected lane occlusion does not induce ordinary-condition causal overlap. The "
            "seed-robust directional result in the explicitly shared architecture remains an "
            "engineering assay result only."
        ),
        "next_experiment": (
            "use a naturally coupled task or resource constraint with identical train/evaluation "
            "state scaling, and require emergence against the neutral dual-lane control"
        ),
    }


def main() -> None:
    result = build_correction()
    output = AUDITS / "stage8_task_induced_routing_correction.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
