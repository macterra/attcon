"""Exact configuration/inventory review binding, no API or experiment data."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_attention_relational_reports_v1 import validate_review_inventory


class ReviewBindingTests(unittest.TestCase):
    def test_changed_model_schema_or_opportunity_inventory_requires_new_review(self):
        manifest = {'settings': {'model': 'fixed-fixture', 'output_schema': None},
                    'requests_path': 'fixture.json', 'request_count': 1, 'request_ids': ['x']}
        review = {'reviewed_'+k: copy.deepcopy(v) for k, v in manifest.items()}
        validate_review_inventory(manifest, review)
        variants = []
        for key, value in [('requests_path', 'other.json'), ('request_count', 2), ('request_ids', ['y'])]:
            changed = copy.deepcopy(manifest); changed[key] = value; variants.append(changed)
        for key, value in [('model', 'different-fixture'), ('output_schema', {'type': 'string'})]:
            changed = copy.deepcopy(manifest); changed['settings'][key] = value; variants.append(changed)
        for changed in variants:
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    validate_review_inventory(changed, review)


if __name__ == '__main__':
    unittest.main()
