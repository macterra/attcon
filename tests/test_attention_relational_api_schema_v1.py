"""One prospective stdlib schema/uniqueness fixture; never makes API calls.

The local walk checks the declared API subset constraints addressed by this
amendment. It is not provider validation or a universal JSON-schema validator.
No archive, model, renderer, world, outcome or study fixture is loaded.
"""
from copy import deepcopy
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import attention_relational_report_v1 as report


class APISchemaAmendmentFixture(unittest.TestCase):
    def test_subset_walk_and_strict_machine_uniqueness(self):
        schema=report.REPORT_RESPONSE_SCHEMA
        self.assertEqual(schema['type'],'object')
        self.assertNotIn('anyOf',schema)
        visited={'objects':0,'arrays':0,'unions':0}
        def walk(node):
            self.assertIs(type(node),dict)
            self.assertFalse(set(node)&{'uniqueItems','allOf','not','dependentRequired',
                'dependentSchemas','if','then','else'})
            if node.get('type')=='object':
                visited['objects']+=1
                self.assertIs(node['additionalProperties'],False)
                self.assertEqual(set(node['required']),set(node['properties']))
                self.assertEqual(len(node['required']),len(set(node['required'])))
                for child in node['properties'].values():
                    walk(child)
            if node.get('type')=='array':
                visited['arrays']+=1
                self.assertLessEqual(set(node),{'type','items','minItems','maxItems'})
                for name in ('minItems','maxItems'):
                    if name in node:
                        self.assertIs(type(node[name]),int)
                        self.assertGreaterEqual(node[name],0)
                walk(node['items'])
            if 'anyOf' in node:
                visited['unions']+=1
                self.assertGreater(len(node['anyOf']),0)
                for child in node['anyOf']:
                    walk(child)
        walk(schema)
        self.assertEqual(visited,dict(objects=4,arrays=5,unions=1))
        peak_branch=schema['properties']['propositions']['items']['anyOf'][0]
        self.assertEqual(peak_branch['properties']['cells']['minItems'],1)
        self.assertEqual(peak_branch['properties']['cells']['maxItems'],16)

        one_hot=[[0.]*4 for _ in range(4)];one_hot[1][2]=1.
        fields=[[deepcopy(one_hot),deepcopy(one_hot)] for _ in range(4)]
        raw=struct.pack('<'+str(report.CURRENT_MAP_BYTES//4)+'f',
            *([0.]*(report.CURRENT_MAP_BYTES//4)))
        panel=report.SourcePanel('A',fields,[deepcopy(fields) for _ in range(3)],raw,raw*3,{})
        content=report.export_packet([panel],3,0)['content']
        valid=report.deterministic_template(content)
        self.assertTrue(report.audit_report(content,valid)['listed_machine_facts_pass'])
        duplicate_cells=deepcopy(valid)
        duplicate_cells['propositions'][0]['cells']=[[1,2],[1,2]]
        audited=report.audit_report(content,duplicate_cells)
        self.assertIn('invalid_peak_cells',[row['category'] for row in audited['errors']])
        duplicate_peak=deepcopy(valid)
        duplicate_peak['propositions'].append(deepcopy(valid['propositions'][0]))
        self.assertIn('duplicate_or_contradictory_claim',
            [row['category'] for row in report.audit_report(content,duplicate_peak)['errors']])
        node=content['panels'][0]['current'][0]
        relation=dict(kind='relation',panel='p0',view='current',command=None,node='n0',
            relation='red_blue_peak_relation',**node['red_blue_peak_relation'])
        scalar=dict(kind='scalar',panel='p0',view='current',command=None,node='n0',role='red',
            name='peak_probability',value=1.)
        optional=deepcopy(valid);optional['propositions'] += [relation,scalar]
        self.assertTrue(report.audit_report(content,optional)['listed_machine_facts_pass'])
        optional['propositions'] += [deepcopy(relation),deepcopy(scalar)]
        audited=report.audit_report(content,optional)
        self.assertFalse(audited['listed_machine_facts_pass'])
        self.assertEqual(sum(row['category']=='duplicate_or_contradictory_claim'
            for row in audited['errors']),2)


if __name__=='__main__':
    unittest.main()
