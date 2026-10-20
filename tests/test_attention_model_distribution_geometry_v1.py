import copy
import importlib.util
import math
from pathlib import Path
import struct
import unittest


PATH = Path(__file__).resolve().parents[1] / "scripts/attention_model_distribution_geometry_v1.py"
SPEC = importlib.util.spec_from_file_location("distribution_geometry", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def grid(cells):
    value = [[0.0] * 4 for _ in range(4)]
    for row, column, mass in cells:
        value[row][column] = mass
    return value


class DistributionGeometryTests(unittest.TestCase):
    def test_point_displacement_has_exact_geometry(self):
        result = MODULE.contrast(grid([(0, 0, 1.0)]), grid([(0, 3, 1.0)]))
        self.assertEqual(result["normalized_total_variation"], 1.0)
        self.assertEqual(result["centroid_displacement"], [0.0, 3.0])
        self.assertEqual(result["centroid_L1_displacement"], 3.0)
        self.assertEqual(result["row_marginal_total_variation"], 0.0)
        self.assertEqual(result["column_marginal_total_variation"], 1.0)
        self.assertEqual(result["entropy_delta_nats"], 0.0)

    def test_equal_peak_hides_distribution_and_score_difference(self):
        a, b = grid([(0, 0, .75), (1, 1, .25)]), grid([(0, 0, .625), (1, 1, .375)])
        result = MODULE.contrast(a, b)
        self.assertTrue(result["complete_peak_sets_equal"])
        self.assertFalse(result["raw_grids_equal"])
        self.assertEqual(result["normalized_total_variation"], .125)
        self.assertEqual(result["raw_maximum_probability_delta"], -.125)
        self.assertEqual(result["centroid_displacement"], [.125, .125])

    def test_equal_centroids_entropy_and_marginals_do_not_hide_shape_change(self):
        a = grid([(1, 1, .5), (2, 2, .5)])
        b = grid([(1, 2, .5), (2, 1, .5)])
        result = MODULE.contrast(a, b)
        self.assertEqual(result["normalized_total_variation"], 1.0)
        for metric in ("centroid_L1_displacement", "row_marginal_total_variation", "column_marginal_total_variation", "entropy_delta_nats", "variance_trace_delta"):
            self.assertEqual(result[metric], 0.0)

    def test_roundoff_mass_normalization_is_explicit_and_preserves_raw_score(self):
        base = grid([(0, 0, .25), (0, 1, .25), (1, 0, .25), (1, 1, .25)])
        next_float = struct.unpack("<f", struct.pack("<I", 0x3e800001))[0]
        larger = grid([(0, 0, next_float), (0, 1, next_float), (1, 0, next_float), (1, 1, next_float)])
        saved = copy.deepcopy(larger)
        measurement = MODULE.geometry(larger)
        self.assertGreater(measurement["mass_deviation"], 0)
        self.assertEqual(measurement["raw_maximum_probability"], next_float)
        result = MODULE.contrast(base, larger)
        self.assertGreater(result["raw_half_L1"], 0)
        self.assertEqual(result["normalized_total_variation"], 0)
        self.assertEqual(larger, saved)

    def test_first_index_score_ties_and_margin(self):
        a, b = grid([(0, 0, .75), (1, 1, .25)]), grid([(0, 0, .625), (1, 1, .375)])
        commands = {c: {"geometry": MODULE.geometry(g)} for c, g in (("k0", a), ("k1", b), ("k2", a))}
        result = MODULE.score_rule(commands)
        self.assertEqual(result["tied_winners"], ["k0", "k2"])
        self.assertEqual(result["first_index_choice"], "k0")
        self.assertEqual(result["top_two_margin"], 0)
        self.assertTrue(result["all_command_peak_sets_equal"])
        self.assertTrue(result["scores_differ"])
        commands["k0"] = {"geometry": MODULE.geometry(b)}
        result = MODULE.score_rule(commands)
        self.assertEqual(result["first_index_choice"], "k2")
        self.assertEqual(result["top_two_margin"], .125)

    def test_invalid_mass_nonfinite_and_non_float32_are_rejected(self):
        for invalid in (grid([]), grid([(0, 0, float("nan"))]), grid([(0, 0, 1.1)]), grid([(0, 0, .7), (0, 1, .3)])):
            with self.assertRaises(AssertionError):
                MODULE.geometry(invalid)
        self.assertEqual(MODULE.quantiles([0, 1, 2, 3]), {"count": 4, "min": 0, "q25": .75, "median": 1.5, "q75": 2.25, "max": 3})
        self.assertEqual(MODULE.quantiles([]), {"count": 0})

    def test_panel_unavailable_at_both_pair_endpoints_is_retained(self):
        panel = {"panel": "p0", "conditional_content_available": False, "group_indices": []}
        census = {"positions": [{"id": name, "panels": [panel], "groups": []} for name in ("a", "b")]}
        exports = {"packets": [{"id": name, "content": {"candidate_node": "n0", "queried_role": "red"}} for name in ("a", "b")]}
        pairs = {"pairs": [{"original_pair": {"first_id": "a", "second_id": "b", "kind": "toy"}, "literal_text_equal": True}]}
        result = MODULE.analyze(census, exports, pairs)
        retained = result["original_pairs"][0]["candidate_query_panels"]
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]["state"], "unmatched_or_unavailable")
        self.assertIsNone(retained[0]["reconstructed_choice_changed"])

    def test_cross_source_command_alternatives_have_no_qualified_score_rule(self):
        field = grid([(0, 0, 1.0)])
        node = {"node": "n0", "distributions": {"red": field}}
        props = {}
        for cmd in ("k0", "k1", "k2"):
            props[cmd] = {"identity": {
                "actual_source": "A" if cmd == "k0" else "W",
                "origin_category": "source_conditional", "represented_time": 13, "decision_time": 12,
                "context": {"panel": "p0", "node": "n0", "role": "red", "command": cmd, "view": "candidate"}},
                "raw_distribution_packet_path": ["panels", 0, "candidates", cmd, 0, "distributions", "red"],
                "raw_distribution_content_sha256": MODULE.digest(field), "value": [[0, 0]], "bank_key": cmd}
        census = {"positions": [{"id": "a", "panels": [{"panel": "p0", "conditional_content_available": True}],
                                  "groups": [{"panel": "p0", "node": "n0", "role": "red", "commands": props}]}]}
        packet = {"candidate_node": "n0", "queried_role": "red", "decision_time": 12,
                  "panels": [{"panel": "p0", "candidates": {c: [node] for c in props}}]}
        result = MODULE.analyze(census, {"packets": [{"id": "a", "content": packet}]}, {"pairs": []})
        self.assertEqual(result["positions"][0]["groups"][0]["score_rule_reconstruction"]["state"], "unmatched")
        self.assertEqual(result["summary"]["candidate_query_score_rule"]["unmatched"]["unmatched_score_rule"], 1)
        self.assertEqual(set(result["summary"]["distribution_metrics"]), {"W/source_conditional/all", "W/source_conditional/equal_peaks"})


if __name__ == "__main__":
    unittest.main()
