"""Descriptive figure of the retained first diagnostic, not an inference test."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path("audits/attention_relational_report_v1/distribution_geometry_v1")
RESULT = BASE / "first_result.json"
EXPECTED = "1186e955ca9fc2e5228381d3e4dbe0610cf9dc73cc987397f0ac1fda3eb7307c"


def main():
    raw = RESULT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED
    data = json.loads(raw)
    all_values = {source: [] for source in ("A", "ordinary", "analytic")}
    A_peak_subsets = {"Equal peaks": [], "Different peaks": []}
    examples = []
    for position in data["positions"]:
        for group in position["groups"]:
            identity = group["common_identity"]
            if identity is None or identity["origin_category"] != "source_conditional":
                continue
            source = identity["actual_source"]
            for comparison in group["comparisons"]:
                if comparison["state"] != "comparable":
                    continue
                metric = comparison["metrics"]
                all_values[source].append(metric["normalized_total_variation"])
                if source == "A":
                    key = "Equal peaks" if metric["complete_peak_sets_equal"] else "Different peaks"
                    A_peak_subsets[key].append(metric["normalized_total_variation"])
            if position["id"].endswith("_A_A_only_baseline") and group["packet_candidate_node_and_queried_role"]:
                case = position["id"].split("_t12_")[0]
                examples.append((case, group["role"], group["score_rule_reconstruction"]))
    assert len(examples) == 12
    assert all(len(values) == 1008 for values in all_values.values())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    def ecdf(axis, values, label, color):
        values = sorted(values)
        axis.step(values, [(i + 1) / len(values) for i in range(len(values))], where="post", label=f"{label} (n={len(values)})", color=color)
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Normalized total variation", ylabel="Fraction of retained comparisons")
        axis.grid(alpha=.2)
    for source, color in (("A", "#0072B2"), ("ordinary", "#D55E00"), ("analytic", "#009E73")):
        ecdf(axes[0], all_values[source], source, color)
    axes[0].set_title("Conditional distributions across commands")
    axes[0].legend(fontsize=8)
    for key, color in (("Equal peaks", "#0072B2"), ("Different peaks", "#CC79A7")):
        ecdf(axes[1], A_peak_subsets[key], key, color)
    axes[1].set_title("A: equal peaks can hide changed distributions")
    axes[1].legend(fontsize=8)
    for i, color in enumerate(("#0072B2", "#D55E00", "#009E73")):
        axes[2].plot(range(12), [e[2]["raw_scores"][i] for e in examples], marker="o", markersize=3, label="k" + str(i), color=color)
    axes[2].set_xticks(range(12), [e[0].replace("s", "") + "/" + e[1][0] for e in examples], rotation=90, fontsize=7)
    axes[2].set(ylim=(0, 1.02), ylabel="Raw maximum probability (native score)", title="All twelve fixed baseline A queries")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=.2)
    fig.suptitle("Native distribution and processing-score diagnostic", fontsize=14)
    fig.text(.5, .01, "Repeated views/variants of twelve development cases; no independent-sample inference. Scores are not calibrated confidence or experience.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .93))
    for suffix in ("png", "pdf"):
        fig.savefig(BASE / ("first_figure." + suffix), dpi=160)
    binding = {"schema": "attention_model_distribution_figure_binding_v1",
               "input": {"path": str(RESULT), "sha256": EXPECTED},
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "conditional_comparison_counts": {k: len(v) for k, v in all_values.items()},
               "A_subset_counts": {k: len(v) for k, v in A_peak_subsets.items()},
               "fixed_baseline_cases": [{"id": e[0], "queried_role": e[1]} for e in examples],
               "outputs": [{"path": str(BASE / ("first_figure." + suffix)), "sha256": hashlib.sha256((BASE / ("first_figure." + suffix)).read_bytes()).hexdigest()} for suffix in ("png", "pdf")],
               "descriptive_post_result_visualization": True}
    with (BASE / "first_figure_binding.json").open("x") as stream:
        json.dump(binding, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
