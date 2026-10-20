"""Magnitude/geometry of existing H fields; no semantic or consciousness grader."""

import copy
import hashlib
import itertools
import json
import math
import struct
from collections import defaultdict
from pathlib import Path


BASE = Path("audits/attention_relational_report_v1")
INPUTS = {
    "census": (BASE / "command_content_diagnostic_v1/first_result.json", "2fc69de904878ddd7ff7432ff7fb3d064dca4dc3a5d14ccd1d4faf0e5e8ff8b6"),
    "exports": (BASE / "run/export/first_exports.json", "4f5f0bc2fa78b3b24b3580f52e306f20df0f3011d079b8139a2bc1f029050837"),
    "pairs": (BASE / "template_attribution_v1/binding_inventory/first_inventory.json", "9438c05913719a4e94d7e78341d3a9a629226e40aac323322de49850d4afaa02"),
}
CONTRACT = Path("docs/ATTENTION_MODEL_DISTRIBUTION_GEOMETRY_V1.md")
OUT = BASE / "distribution_geometry_v1/first_result.json"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def resolve(value, path):
    for key in path:
        if isinstance(value, list):
            assert type(key) is int and 0 <= key < len(value)
        else:
            assert type(value) is dict and type(key) is str and key in value
        value = value[key]
    return value


def geometry(grid):
    assert type(grid) is list and len(grid) == 4
    assert all(type(row) is list and len(row) == 4 for row in grid)
    values = [value for row in grid for value in row]
    assert all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values)
    assert list(struct.unpack("<16f", struct.pack("<16f", *values))) == values
    mass = math.fsum(values)
    assert mass > 0 and abs(mass - 1) <= 1e-6 + 1e-12
    p = [v / mass for v in values]
    row = math.fsum((i // 4) * v for i, v in enumerate(p))
    col = math.fsum((i % 4) * v for i, v in enumerate(p))
    spread = math.fsum((((i // 4) - row) ** 2 + ((i % 4) - col) ** 2) * v for i, v in enumerate(p))
    return {
        "mass": mass, "mass_deviation": mass - 1,
        "raw_maximum_probability": max(values),
        "complete_peak_set": [[i // 4, i % 4] for i, v in enumerate(values) if v == max(values)],
        "normalized_entropy_nats": -math.fsum(v * math.log(v) for v in p if v),
        "normalized_centroid": [row, col], "normalized_variance_trace": spread,
    }


def contrast(first, second):
    a, b = geometry(first), geometry(second)
    raw_a = [v for row in first for v in row]
    raw_b = [v for row in second for v in row]
    p, q = [v / a["mass"] for v in raw_a], [v / b["mass"] for v in raw_b]
    row_p = [math.fsum(p[i * 4:i * 4 + 4]) for i in range(4)]
    row_q = [math.fsum(q[i * 4:i * 4 + 4]) for i in range(4)]
    col_p = [math.fsum(p[i::4]) for i in range(4)]
    col_q = [math.fsum(q[i::4]) for i in range(4)]
    shift = [b["normalized_centroid"][i] - a["normalized_centroid"][i] for i in range(2)]
    return {
        "raw_half_L1": math.fsum(abs(x - y) for x, y in zip(raw_a, raw_b)) / 2,
        "maximum_absolute_cell_difference": max(abs(x - y) for x, y in zip(raw_a, raw_b)),
        "normalized_total_variation": math.fsum(abs(x - y) for x, y in zip(p, q)) / 2,
        "row_marginal_total_variation": math.fsum(abs(x - y) for x, y in zip(row_p, row_q)) / 2,
        "column_marginal_total_variation": math.fsum(abs(x - y) for x, y in zip(col_p, col_q)) / 2,
        "centroid_displacement": shift, "centroid_L1_displacement": math.fsum(abs(x) for x in shift),
        "entropy_delta_nats": b["normalized_entropy_nats"] - a["normalized_entropy_nats"],
        "variance_trace_delta": b["normalized_variance_trace"] - a["normalized_variance_trace"],
        "raw_maximum_probability_delta": b["raw_maximum_probability"] - a["raw_maximum_probability"],
        "complete_peak_sets_equal": a["complete_peak_set"] == b["complete_peak_set"],
        "raw_grids_equal": first == second,
    }


def score_rule(commands):
    scores = [commands[c]["geometry"]["raw_maximum_probability"] for c in ("k0", "k1", "k2")]
    largest = max(scores)
    winners = ["k" + str(i) for i, v in enumerate(scores) if v == largest]
    ordered = sorted(scores, reverse=True)
    return {"raw_scores": scores, "tied_winners": winners, "first_index_choice": winners[0],
            "top_two_margin": ordered[0] - ordered[1],
            "scores_differ": len(set(scores)) > 1,
            "all_command_peak_sets_equal": all(commands[c]["geometry"]["complete_peak_set"] == commands["k0"]["geometry"]["complete_peak_set"] for c in commands)}


def identity_without_command(identity):
    result = copy.deepcopy(identity)
    result["context"]["command"] = None
    return result


def quantiles(values):
    data = sorted(values)
    if not data:
        return {"count": 0}
    def at(q):
        rank = (len(data) - 1) * q
        low, high = math.floor(rank), math.ceil(rank)
        return data[low] + (rank - low) * (data[high] - data[low])
    return {"count": len(data), "min": data[0], "q25": at(.25), "median": at(.5), "q75": at(.75), "max": data[-1]}


def summarize(positions):
    strata = defaultdict(lambda: defaultdict(list))
    selection = defaultdict(lambda: defaultdict(int))
    for position in positions:
        for group in position["groups"]:
            identity = group["common_identity"]
            key = identity["actual_source"] + "/" + identity["origin_category"]
            if group["packet_candidate_node_and_queried_role"]:
                selection[key]["candidate_query_groups"] += 1
                rule = group["score_rule_reconstruction"]
                selection[key]["unique_winner"] += len(rule["tied_winners"]) == 1
                selection[key]["all_peak_sets_equal"] += rule["all_command_peak_sets_equal"]
                selection[key]["equal_peak_sets_scores_differ"] += rule["all_command_peak_sets_equal"] and rule["scores_differ"]
                selection[key]["choice_" + rule["first_index_choice"]] += 1
            for pair in group["comparisons"]:
                if pair["state"] != "comparable":
                    continue
                metrics = pair["metrics"]
                subsets = ["all", "equal_peaks" if metrics["complete_peak_sets_equal"] else "different_peaks"]
                for subset in subsets:
                    for name, value in metrics.items():
                        if type(value) in (int, float):
                            strata[key + "/" + subset][name].append(value)
    return {"distribution_metrics": {k: {m: quantiles(v) for m, v in sorted(metrics.items())} for k, metrics in sorted(strata.items())},
            "candidate_query_score_rule": {k: dict(v) for k, v in sorted(selection.items())}}


def analyze(census, exports, pair_inventory):
    packets = {p["id"]: p["content"] for p in exports["packets"]}
    assert len(packets) == len(exports["packets"])
    positions = []
    for original in census["positions"]:
        packet = packets[original["id"]]
        groups = []
        for retained in original["groups"]:
            commands = {}
            for command, prop in retained["commands"].items():
                identity = prop["identity"]
                context = identity["context"]
                assert context == {"panel": retained["panel"], "node": retained["node"], "role": retained["role"], "command": command, "view": "candidate"}
                assert identity["represented_time"] == 13 and identity["decision_time"] == packet["decision_time"] == 12
                path = prop["raw_distribution_packet_path"]
                panel_index = next(i for i, p in enumerate(packet["panels"]) if p["panel"] == context["panel"])
                nodes = packet["panels"][panel_index]["candidates"][command]
                node_index = next(i for i, n in enumerate(nodes) if n["node"] == context["node"])
                assert path == ["panels", panel_index, "candidates", command, node_index, "distributions", context["role"]]
                grid = resolve(packet, path)
                assert digest(grid) == prop["raw_distribution_content_sha256"]
                measurement = geometry(grid)
                assert measurement["complete_peak_set"] == prop["value"]
                commands[command] = {"grid": grid, "geometry": measurement, "bank_key": prop["bank_key"],
                                     "raw_distribution_packet_path": path, "raw_distribution_content_sha256": digest(grid)}
            comparisons = []
            for first, second in itertools.combinations(sorted(commands), 2):
                a = retained["commands"][first]["identity"]
                b = retained["commands"][second]["identity"]
                comparable = identity_without_command(a) == identity_without_command(b)
                comparisons.append({"first_command": first, "second_command": second,
                                    "state": "comparable" if comparable else "unmatched",
                                    "metrics": contrast(commands[first]["grid"], commands[second]["grid"]) if comparable else None})
            assert len(commands) == 3
            groups.append({"panel": retained["panel"], "node": retained["node"], "role": retained["role"],
                           "common_identity": identity_without_command(retained["commands"]["k0"]["identity"]),
                           "commands": commands, "comparisons": comparisons,
                           "packet_candidate_node_and_queried_role": retained["node"] == packet["candidate_node"] and retained["role"] == packet["queried_role"],
                           "score_rule_reconstruction": score_rule(commands)})
        positions.append({"id": original["id"], "panels": original["panels"], "groups": groups})
    indexed = {p["id"]: p for p in positions}
    pairs = []
    for original in pair_inventory["pairs"]:
        pair = original["original_pair"]
        a, b = indexed[pair["first_id"]], indexed[pair["second_id"]]
        def query_groups(position):
            return {g["panel"]: g for g in position["groups"] if g["packet_candidate_node_and_queried_role"]}
        first, second = query_groups(a), query_groups(b)
        comparisons = []
        for panel in sorted(set(first) | set(second)):
            x, y = first.get(panel), second.get(panel)
            comparable = x is not None and y is not None and x["common_identity"] == y["common_identity"]
            comparisons.append({"panel": panel, "actual_source": x["common_identity"]["actual_source"] if comparable else None,
                "state": "comparable" if comparable else "unmatched_or_unavailable",
                "scores_changed": x["score_rule_reconstruction"]["raw_scores"] != y["score_rule_reconstruction"]["raw_scores"] if comparable else None,
                "reconstructed_choice_changed": x["score_rule_reconstruction"]["first_index_choice"] != y["score_rule_reconstruction"]["first_index_choice"] if comparable else None,
                "command_peak_sets_equal": all(x["commands"][c]["geometry"]["complete_peak_set"] == y["commands"][c]["geometry"]["complete_peak_set"] for c in ("k0", "k1", "k2")) if comparable else None})
        pairs.append({"original_pair": pair, "literal_template_equal": original["literal_text_equal"], "candidate_query_panels": comparisons})
    return {"positions": positions, "original_pairs": pairs, "summary": summarize(positions)}


def main():
    inputs, bindings = {}, {}
    for name, (path, expected) in INPUTS.items():
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, str(path)
        inputs[name] = json.loads(raw)
        bindings[name] = {"path": str(path), "sha256": expected}
    result = analyze(inputs["census"], inputs["exports"], inputs["pairs"])
    assert len(result["positions"]) == 552 and len(result["original_pairs"]) == 684
    assert sum(len(p["groups"]) for p in result["positions"]) == 6336
    result.update({"schema": "attention_model_distribution_geometry_v1", "inputs": bindings,
                   "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
                   "analysis_type": "informed_development_not_new_system_evidence",
                   "source_grids_modified": False, "normalization_for_geometry_only": True,
                   "new_model_or_API_calls": 0, "semantic_verdict": None, "goal_achieved": False})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(OUT), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(), "positions": len(result["positions"]), "pairs": len(result["original_pairs"]), "score_rule_summary": result["summary"]["candidate_query_score_rule"]}, sort_keys=True))


if __name__ == "__main__":
    main()
