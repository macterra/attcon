"""One handcrafted API-schema equivalence and unchanged-auditor fixture.

No provider call, archive, world, model, fitted parameter or report is loaded.
This checks the narrow amendment and deterministic validation, not provider
acceptance or free-prose fidelity. Root coordinates its first execution.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import attention_relational_api_schema_v2 as amendment
import attention_relational_report_v1 as report


class TypedAPISchemaAmendmentFixture(unittest.TestCase):
    def test_exact_22_type_additions_and_unchanged_literal_duplicate_rejection(self):
        source_path = ROOT / 'audits/attention_relational_report_v1/run/export/response_schema.json'
        raw = source_path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
            '1623d63a88ffeb806f9e4536c35ea39def3991e785718db3a8366dc98e3bc4eb')
        source = json.loads(raw)
        self.assertEqual(source, report.REPORT_RESPONSE_SCHEMA)
        original = deepcopy(source)
        amended = amendment.build_typed_response_schema(source)
        self.assertEqual(source, original)
        additions = []

        def compare(old, new, path=()):
            self.assertIs(type(new), type(old))
            if type(old) is dict:
                extra = set(new) - set(old)
                self.assertFalse(set(old) - set(new))
                self.assertLessEqual(extra, {'type'})
                if extra:
                    self.assertNotIn('type', old)
                    self.assertTrue('enum' in old or 'const' in old)
                    values = old.get('enum', [old.get('const')])
                    expected = ['string', 'null'] if None in values else 'string'
                    self.assertEqual(new['type'], expected)
                    additions.append((path, deepcopy(expected)))
                for key, value in old.items():
                    compare(value, new[key], (*path, key))
            elif type(old) is list:
                self.assertEqual(len(old), len(new))
                for index, (left, right) in enumerate(zip(old, new)):
                    compare(left, right, (*path, index))
            else:
                self.assertEqual(old, new)

        compare(source, amended)
        self.assertEqual(len(additions), 22)
        self.assertEqual(sum(kind == 'string' for _, kind in additions), 19)
        self.assertEqual(sum(kind == ['string', 'null'] for _, kind in additions), 3)
        reversed_schema = deepcopy(amended)
        for path, _ in additions:
            leaf = reversed_schema
            for key in path:
                leaf = leaf[key]
            del leaf['type']
        self.assertEqual(reversed_schema, source)
        self.assertEqual(json.dumps(reversed_schema, sort_keys=True),
                         json.dumps(source, sort_keys=True))
        altered_contract = deepcopy(source)
        altered_contract['properties']['schema']['const'] = 'different_contract'
        with self.assertRaisesRegex(ValueError, 'exact frozen v1 schema'):
            amendment.build_typed_response_schema(altered_contract)

        one_hot = [[0.] * 4 for _ in range(4)]
        one_hot[1][2] = 1.
        fields = [[deepcopy(one_hot), deepcopy(one_hot)] for _ in range(4)]
        raw_map = struct.pack('<' + str(report.CURRENT_MAP_BYTES // 4) + 'f',
                             *([0.] * (report.CURRENT_MAP_BYTES // 4)))
        panel = report.SourcePanel('A', fields, [deepcopy(fields) for _ in range(3)],
                                   raw_map, raw_map * 3, {})
        content = report.export_packet([panel], 3, 0)['content']
        valid = report.deterministic_template(content)
        self.assertTrue(report.audit_report(content, valid)['listed_machine_facts_pass'])
        # These are contract-invalid literals/types, not valid but wrong estimates.
        for key, value in (('kind', 'different_kind'), ('panel', 'p9'),
                           ('view', 'different_view'), ('command', 'k9'),
                           ('command', 0), ('command', False), ('node', 'n9'),
                           ('role', 'green')):
            invalid = deepcopy(valid)
            invalid['propositions'][0][key] = value
            with self.subTest(key=key, value=value):
                self.assertFalse(report.audit_report(content, invalid)['listed_machine_facts_pass'])
        invalid = deepcopy(valid)
        invalid['schema'] = 'different_contract'
        self.assertFalse(report.audit_report(content, invalid)['listed_machine_facts_pass'])
        duplicate_cells = deepcopy(valid)
        duplicate_cells['propositions'][0]['cells'] = [[1, 2], [1, 2]]
        self.assertIn('invalid_peak_cells', [row['category'] for row in
            report.audit_report(content, duplicate_cells)['errors']])
        duplicate_peak = deepcopy(valid)
        duplicate_peak['propositions'].append(deepcopy(valid['propositions'][0]))
        self.assertIn('duplicate_or_contradictory_claim', [row['category'] for row in
            report.audit_report(content, duplicate_peak)['errors']])
        node = content['panels'][0]['current'][0]
        relation = dict(kind='relation', panel='p0', view='current', command=None,
                        node='n0', relation='red_blue_peak_relation',
                        **node['red_blue_peak_relation'])
        scalar = dict(kind='scalar', panel='p0', view='current', command=None,
                      node='n0', role='red', name='peak_probability', value=1.)
        optional = deepcopy(valid)
        optional['propositions'] += [relation, scalar]
        self.assertTrue(report.audit_report(content, optional)['listed_machine_facts_pass'])
        for position, key, value in ((-2, 'relation', 'different_relation'),
                                      (-1, 'name', 'different_scalar')):
            invalid = deepcopy(optional)
            invalid['propositions'][position][key] = value
            self.assertFalse(report.audit_report(content, invalid)['listed_machine_facts_pass'])
        optional['propositions'] += [deepcopy(relation), deepcopy(scalar)]
        audited = report.audit_report(content, optional)
        self.assertFalse(audited['listed_machine_facts_pass'])
        self.assertEqual(sum(row['category'] == 'duplicate_or_contradictory_claim'
                             for row in audited['errors']), 2)


if __name__ == '__main__':
    unittest.main()
