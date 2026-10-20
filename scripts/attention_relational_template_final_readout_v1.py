"""Summarize the retained fixed analysis; never grade text or repair a first.

Run from the repository root. The exclusive output records missing semantic
measurements explicitly. Source judgments remain in the bound full inventory.
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path("audits/attention_relational_report_v1/template_attribution_v1")


def bound(path, expected=None):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"Changed input: {path}")
    return json.loads(raw), {"path": str(path), "sha256": digest}


def main():
    inventory, inventory_ref = bound(
        BASE / "binding_inventory/first_inventory.json",
        "9438c05913719a4e94d7e78341d3a9a629226e40aac323322de49850d4afaa02",
    )
    accounting, accounting_ref = bound(
        BASE / "complete_readout_accounting_review/first_review.json",
        "b6e73c8e596832e291c4b5ddfa54a7fec3bbec9f63c6b3daee19d69b7d4dfc49",
    )
    numbers, numbers_ref = bound(BASE / "property_pairs/first_summary.json")
    disposition, disposition_ref = bound(
        BASE / "semantic_closure_v1/root_disposition.json"
    )
    assert accounting["accounting_consistent"] and not accounting["discrepancies"]
    assert accounting["fatal"] is None
    assert len(inventory["positions"]) == 552
    assert len(inventory["pairs"]) == 684
    assert len(disposition["cases"]) == 11
    assert disposition["established_source_endpoints_added"] == 0

    streams = {name: Counter() for name in ("reviewer_0", "reviewer_1")}
    cases = defaultdict(Counter)
    strata = defaultdict(Counter)
    scopes = Counter()
    candidates = []
    positions = []
    binding_assessments = Counter()
    for position in inventory["positions"]:
        case = position["case"]
        case_key = f"{case['encoder_seed']}/{case['world_index']}"
        stratum = f"{case['source']}/{case['view']}"
        for counts in (cases[case_key], strata[stratum]):
            counts["positions"] += 1
            counts["original_reference_established"] += position["reference_established"]
        stream_rows = {}
        for name, row in position["first_streams"].items():
            streams[name]["positions"] += 1
            streams[name]["valid_chunk_positions"] += row["chunk_structurally_valid"]
            streams[name]["retained_established_reference_positions"] += (
                row["chunk_structurally_valid"] and position["reference_established"]
            )
            stream_rows[name] = {
                "chunk_number": row["chunk_number"],
                "chunk_structurally_valid": row["chunk_structurally_valid"],
                "review_state": row["review_state"],
                "bindings": [],
            }
            for binding in row["bindings"] or []:
                composite = binding["composite"]
                assessment = binding["first_binding_review"]["assessment"]
                supported = binding["all_recorded_support_facts_true"]
                streams[name][f"{composite}_bindings"] += 1
                streams[name][f"{composite}_binding_{assessment}"] += 1
                streams[name][f"{composite}_recorded_support_candidates"] += supported
                binding_assessments[f"{name}/{composite}/{assessment}"] += 1
                if name == "reviewer_0":
                    scopes[f"{composite}/{binding['original_scope']}/{binding['original_composite_status']}"] += 1
                item = {
                    "binding_path": binding["binding_path"],
                    "composite": composite,
                    "original_status": binding["original_composite_status"],
                    "scope": binding["original_scope"],
                    "time": binding["original_time"],
                    "first_binding_assessment": assessment,
                    "all_recorded_support_facts_true": supported,
                }
                stream_rows[name]["bindings"].append(item)
                if supported:
                    content_sources = sorted({
                        (operand.get("atomic_identity") or {}).get("actual_source", "unmapped")
                        for evidence in binding["selected_evidence"]
                        for operand in evidence["operands"]
                        if operand["first_review_property"]["semantic_role"] == "content_operand"
                    })
                    candidates.append({
                        "id": position["id"], "stream": name, **item,
                        "selected_content_operand_actual_sources": content_sources,
                        "semantic_source_endpoint_established": False,
                        "operand_detail_location": inventory_ref,
                    })
                    for counts in (cases[case_key], strata[stratum]):
                        counts[f"{name}_{composite}_recorded_support_candidates"] += 1
        assert not position["both_first_streams_recorded_support_candidates"]
        positions.append({
            "id": position["id"], "case": case,
            "reference_established": position["reference_established"],
            "original_composite_statuses": position["original_composite_statuses"],
            "first_streams": stream_rows,
            "joint_recorded_support_candidates": [],
            "source_attributed_COD": "unestablished",
            "source_attributed_CA": "unestablished",
            "unestablished_is_feature_absence": False,
        })
    assert len({p["id"] for p in positions}) == 552
    assert len(candidates) == accounting["descriptive_counts"]["stream_recorded_support_candidates"]
    assert sum(len(r["bindings"]) for p in positions for r in p["first_streams"].values()) == accounting["descriptive_counts"]["stream_binding_inventories"]
    by_id = {p["id"]: p for p in positions}
    pairs = []
    pair_counts = defaultdict(Counter)
    for pair in inventory["pairs"]:
        original = pair["original_pair"]
        first = by_id[original["first_id"] + "_template"]
        second = by_id[original["second_id"] + "_template"]
        assert not pair["first_recorded_support_candidates"]
        assert not pair["second_recorded_support_candidates"]
        pair_counts[original["kind"]]["pairs"] += 1
        pair_counts[original["kind"]].update(pair["numerical_property_counts"])
        pairs.append({
            "original_pair": original,
            "first_template_id": first["id"],
            "second_template_id": second["id"],
            "numerical_property_counts": pair["numerical_property_counts"],
            "numerical_by_actual_source": pair["numerical_by_actual_source"],
            "literal_text_equal": pair["literal_text_equal"],
            "source_attributed_semantic_tracking": "unestablished",
            "source_attributed_semantic_null": "unestablished",
            "source_attributed_semantic_restoration": "unestablished",
            "reason": "No jointly supported source-attributed semantic binding is established at the endpoints; numerical equality/change is a separate result.",
        })
    for kind, totals in pair_counts.items():
        numerical = numbers["summary_by_pair_kind"][kind]
        assert totals["pairs"] == numerical["pair_count"]
        assert {k: v for k, v in totals.items() if k != "pairs"} == numerical["property_counts"]
    for counts in streams.values():
        counts["unestablished_reference_or_missing_review_positions"] = (
            552 - counts["retained_established_reference_positions"]
        )
    output = {
        "schema": "attention_relational_template_final_fixed_readout_v1",
        "analysis_type": "exposed_development_readout_not_a_new_semantic_grader",
        "inputs": [inventory_ref, accounting_ref, numbers_ref, disposition_ref],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixed_analysis_disposition_complete": True,
        "semantic_measurement_qualified": False,
        "new_system_observations": 0,
        "goal_achieved": False,
        "summary": {
            "positions": 552, "pairs": 684,
            "original_reference_established": 148,
            "original_reference_unestablished": 404,
            "original_status_counts": inventory["summary"]["original_composite_statuses"],
            "streams": streams,
            "distinct_one_stream_candidate_positions": len({c["id"] for c in candidates}),
            "joint_recorded_support_candidate_bindings": 0,
            "established_source_attributed_COD_positions": 0,
            "established_source_attributed_CA_positions": 0,
            "unestablished_semantic_pairs": 684,
            "post_first_advisory_supported_core_readings": 4,
            "post_first_advisory_unresolved_core_readings": 7,
            "root_unresolved_semantic_boundary_cases": 11,
        },
        "original_binding_scopes_from_complete_reviewer_0_inventory": scopes,
        "binding_assessment_counts": binding_assessments,
        "per_case": dict(sorted(cases.items())),
        "per_source_view": dict(sorted(strata.items())),
        "per_pair_kind": dict(sorted(pair_counts.items())),
        "one_stream_recorded_support_candidates": candidates,
        "positions": positions,
        "pairs": pairs,
        "numerical_summary": numbers,
        "interpretation": [
            "Zero established endpoints is measurement yield, not zero underlying feature prevalence or evidence against consciousness.",
            "Four and five first-stream candidates are reported separately; their union of nine is not a population upper bound.",
            "404 missing original references and the missing second chunk remain unknown; no valid-only denominator replaces 552.",
            "The four positive post-first core readings and the opposing/unresolved readings all remain visible; none replaces a first judgment.",
            "The numerical summary's independent-review-pending flag is historical; the later retained independent review completed separately.",
            "All 684 semantic pair results remain unestablished; numerical nulls/restoration are real and do not fill missing meaning.",
            "No appearance scope, private experience, task necessity, A-superiority, hidden status variable or mandatory human criterion is added.",
            "This closes the fixed data's analysis with unresolved measurements; it does not qualify a reporting instrument or confirm the theory.",
        ],
    }
    output_path = BASE / "final_readout/first_readout.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"path": str(output_path), "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(), "summary": output["summary"], "per_case": output["per_case"]}, sort_keys=True))


if __name__ == "__main__":
    main()
