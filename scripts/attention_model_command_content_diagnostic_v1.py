"""Read-only development census of command-conditioned native peak contents."""

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("audits/attention_relational_report_v1")
BANK = ROOT / "template_attribution_v1/property_pairs/first_pairs.json"
EXPECTED = "c944d239cd52863fefa9331fc91800d048362d38d2fd7ee2659dc3b41d5706a5"
OUT = ROOT / "command_content_diagnostic_v1/first_result.json"


def signature(identity):
    # Explicit within-state command contrast, not a temporal/source substitution.
    projected = {**identity, "context": {**identity["context"], "command": None}}
    return json.dumps(projected, sort_keys=True, separators=(",", ":"))


def census(bank):
    positions, examples, total = [], [], Counter()
    by_source = defaultdict(Counter)
    for identifier, table in sorted(bank["property_tables"].items()):
        panels = {}
        grouped = defaultdict(dict)
        for key, prop in table.items():
            identity = prop["identity"]
            context = identity["context"]
            panels.setdefault(context["panel"], [])
            if identity["kind"] != "role_peaks" or context["view"] != "candidate":
                continue
            group = (context["panel"], context["node"], context["role"])
            command = context["command"]
            assert command in ("k0", "k1", "k2")
            assert command not in grouped[group]
            grouped[group][command] = {"bank_key": key, **prop}
        groups = []
        for (panel, node, role), commands in sorted(grouped.items()):
            assert set(commands) == {"k0", "k1", "k2"}
            comparisons = []
            for first, second in itertools.combinations(sorted(commands), 2):
                a, b = commands[first], commands[second]
                comparable = signature(a["identity"]) == signature(b["identity"])
                equal = a["value"] == b["value"] if comparable else None
                raw_a, raw_b = a["raw_distribution_content_sha256"], b["raw_distribution_content_sha256"]
                raw_changed = raw_a != raw_b if comparable and raw_a and raw_b else None
                state = "unmatched" if not comparable else "equal_peaks" if equal else "different_peaks"
                source = a["identity"]["actual_source"] if comparable else "unmatched"
                comparisons.append({
                    "first_command": first, "second_command": second,
                    "state": state, "complete_peak_sets_equal": equal,
                    "raw_distribution_changed": raw_changed,
                    "peak_null_with_raw_change": bool(equal and raw_changed),
                })
                for counts in (total, by_source[source]):
                    counts["command_pairs"] += 1
                    counts[state] += 1
                    counts["peak_null_with_raw_change"] += bool(equal and raw_changed)
            group = {"panel": panel, "node": node, "role": role,
                     "commands": commands, "comparisons": comparisons}
            groups.append(group)
            panels[panel].append(len(groups) - 1)
            if identifier.endswith("_A_A_only_baseline") and role == "red" and all(
                prop["identity"]["origin_category"] == "source_conditional"
                and prop["identity"]["actual_source"] == "A"
                for prop in commands.values()
            ):
                examples.append({"id": identifier, "group": group})
        positions.append({"id": identifier, "panels": [
            {"panel": panel, "group_indices": indices,
             "conditional_content_available": bool(indices)}
            for panel, indices in sorted(panels.items())
        ], "groups": groups})
    return {
        "positions": positions, "fixed_twelve_case_examples": examples,
        "summary": {
            "positions": len(positions),
            "panels": sum(len(p["panels"]) for p in positions),
            "panels_without_conditional_content": sum(
                not panel["conditional_content_available"] for p in positions for panel in p["panels"]),
            "three_command_groups": sum(len(p["groups"]) for p in positions),
            "command_pairs": dict(total),
            "by_actual_source": {s: dict(v) for s, v in sorted(by_source.items())},
        },
    }


def main():
    raw = BANK.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED
    result = census(json.loads(raw))
    assert result["summary"]["positions"] == 552
    assert len(result["fixed_twelve_case_examples"]) == 12
    result.update({
        "schema": "attention_model_command_content_diagnostic_v1",
        "bank": {"path": str(BANK), "sha256": EXPECTED},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256(Path("docs/ATTENTION_MODEL_COMMAND_CONTENT_DIAGNOSTIC_V1.md").read_bytes()).hexdigest(),
        "analysis_type": "informed_archived_development_diagnostic",
        "new_api_calls": 0, "new_model_calls": 0, "new_systems": 0,
        "semantic_or_consciousness_verdict": None,
        "goal_achieved": False,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"path": str(OUT), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(), "summary": result["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
