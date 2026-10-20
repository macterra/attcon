import copy
import importlib.util
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/attention_model_command_content_diagnostic_v1.py"
SPEC = importlib.util.spec_from_file_location("command_diagnostic", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def bank(source="A"):
    properties = {}
    for index, command in enumerate(("k0", "k1", "k2")):
        properties[command] = {
            "identity": {
                "kind": "role_peaks", "actual_source": source,
                "origin_category": "source_conditional",
                "represented_time": 13,
                "temporal_coordinates": [{"path": ["feedback_through"], "value": 11}],
                "context": {"panel": "p0", "node": "n0", "role": "red",
                            "view": "candidate", "command": command},
            },
            "value": [[0, 0]] if index == 0 else [[1, 1]],
            "raw_distribution_content_sha256": str(index),
        }
    return {"property_tables": {"toy_A_A_only_baseline": properties}}


class CommandContentDiagnosticTests(unittest.TestCase):
    def test_complete_sets_distinguish_movement_from_true_peak_null(self):
        result = MODULE.census(bank())
        counts = result["summary"]["command_pairs"]
        self.assertEqual(counts["different_peaks"], 2)
        self.assertEqual(counts["equal_peaks"], 1)
        self.assertEqual(counts["peak_null_with_raw_change"], 1)
        self.assertEqual(len(result["positions"][0]["groups"][0]["commands"]), 3)

    def test_temporal_mismatch_cannot_be_counted_as_invariance(self):
        data = bank()
        for prop in data["property_tables"]["toy_A_A_only_baseline"].values():
            prop["value"] = [[0, 0]]
        data["property_tables"]["toy_A_A_only_baseline"]["k1"]["identity"]["temporal_coordinates"][0]["value"] = 10
        counts = MODULE.census(data)["summary"]["command_pairs"]
        self.assertEqual(counts["unmatched"], 2)
        self.assertEqual(counts["equal_peaks"], 1)

    def test_panel_name_does_not_supply_actual_A_source(self):
        data = bank("W")
        before = copy.deepcopy(data)
        result = MODULE.census(data)
        self.assertEqual(set(result["summary"]["by_actual_source"]), {"W"})
        self.assertEqual(result["fixed_twelve_case_examples"], [])
        self.assertEqual(data, before)

    def test_sensor_and_missing_commands_remain_unavailable_or_incomplete(self):
        data = bank()
        original = data["property_tables"]["toy_A_A_only_baseline"]
        current = copy.deepcopy(original["k0"])
        current["identity"]["context"].update(view="current", command=None)
        sensor = {"property_tables": {"sensor": {"current": current}}}
        result = MODULE.census(sensor)
        self.assertEqual(result["summary"]["panels_without_conditional_content"], 1)
        self.assertEqual(result["summary"]["three_command_groups"], 0)
        del original["k1"]
        with self.assertRaises(AssertionError):
            MODULE.census(data)


if __name__ == "__main__":
    unittest.main()
