"""Bounded prospective software fixtures; no retained study data or model calls.

Arrays below are literal artificial distributions/native byte markers, not H
outputs or simulated worlds. Torch is imported only inside the tiny archive-table
fixture, with CPU4 deterministic settings. No loader deserialization is exercised.
First execution requires separately published source review and authorization.
"""
from copy import deepcopy
from dataclasses import replace
import builtins
import json
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import attention_relational_report_v1 as report


def grid(cells=((1,2),)):
    value=[[0.]*4 for _ in range(4)]
    for r,c in cells:
        value[r][c]=1./len(cells)
    return value


def fields():
    return [[grid(((1,2),(2,1))),grid(((1,3),))] for _ in range(4)]


def panel(source='A',variant='baseline',marker=0.):
    current=fields()
    candidates=[deepcopy(current) for _ in range(3)] if source!='sensor' else None
    raw=struct.pack('<'+str(report.CURRENT_MAP_BYTES//4)+'f',*([marker]*(report.CURRENT_MAP_BYTES//4)))
    return report.SourcePanel(source,current,candidates,raw,
        raw*3 if candidates is not None else None,dict(variant=variant,private_source_label=source))


def packet(source='A'):
    return report.export_packet([panel(source)],3,0,'blue')


class RelationalFactsTests(unittest.TestCase):
    def test_full_precision_tied_peaks_and_transparent_arithmetic(self):
        summary=report.localization_summary(grid(((1,2),(2,1))))
        self.assertEqual(summary['peaks'],[[1,2],[2,1]])
        self.assertEqual(summary['peak_probability'],.5)
        self.assertEqual(summary['expected_row'],1.5)
        self.assertEqual(summary['expected_column'],1.5)
        self.assertAlmostEqual(summary['entropy_nats'],0.6931471805599453,places=15)
        relation=report.peak_relation(summary['peaks'],[[1,3]])
        self.assertEqual(relation,dict(horizontal=['left_of'],vertical=['same_row','below'],
            coincident_possible=False,coincident_all=False))
        with self.assertRaisesRegex(ValueError,'lossless float32'):
            report.validate_panel(replace(panel(),current_fields=[[[[.1]*4]*4]*2]*4))
        with self.assertRaisesRegex(ValueError,'mass'):
            report.validate_panel(replace(panel(),current_fields=[[[[.125]*4]*4]*2]*4))

    def test_sources_neutralized_with_honest_timing(self):
        first=packet('A');second=packet('ordinary');third=packet('analytic')
        self.assertEqual(first['content_sha256'],second['content_sha256'])
        self.assertEqual(second['content_sha256'],third['content_sha256'])
        self.assertNotEqual(first['provenance'],second['provenance'])
        public=first['content'];self.assertEqual(public['queried_role'],'blue')
        self.assertEqual(public['panels'][0]['current'][3]['origin'],
            dict(kind='history_estimate',represented_time=12,feedback_through=11))
        self.assertEqual(public['panels'][0]['current'][0]['origin']['record_time'],11)
        self.assertEqual(public['panels'][0]['candidates']['k0'][0]['origin']['kind'],'command_estimate')
        self.assertNotIn('private_source_label',report.json_bytes(public).decode())

    def test_sensor_current_only_and_synthetic_W_origin(self):
        sensor=packet('sensor')['content']
        self.assertIsNone(sensor['panels'][0]['candidates'])
        self.assertEqual(len(report.required_propositions(sensor)),8)
        self.assertEqual(sensor['panels'][0]['current'][0]['origin']['record_time'],0)
        changed=report.export_packet([panel('W','W_changed')],3,0)['content']
        origin=changed['panels'][0]['current'][0]['origin']
        self.assertEqual(origin['kind'],'counterfactual_native_snapshot')
        self.assertEqual(origin['represented_time'],12)
        self.assertEqual(origin['archived_origin_time'],13)
        self.assertFalse(origin['executed_branch_claim'])
        inherited=changed['panels'][0]['candidates']['k0'][1]['origin']['inherited_origin']
        self.assertEqual(inherited,origin)

    def test_complete_literal_template_bounded_and_no_prose_credit(self):
        content=report.export_packet([panel(),panel('W')],3,0)['content']
        template=report.deterministic_template(content)
        self.assertEqual(set(template),{'schema','text','propositions'})
        self.assertEqual(len(template['propositions']),64)
        self.assertLessEqual(len(template['text'].split()),220)
        self.assertIn('p1/candidate/k2/n3/blue=',template['text'])
        self.assertIn('feedback_through',template['text'])
        audit=report.audit_report(content,template)
        self.assertTrue(audit['listed_machine_facts_pass'])
        self.assertFalse(audit['main_endpoint_pass'])
        self.assertFalse(audit['appearance_validated'])

    def test_all_omissions_contradictions_and_unlicensed_claims_retained(self):
        content=packet()['content'];value=report.deterministic_template(content)
        missing=value['propositions'].pop(0)
        value['propositions'][0]['cells']=[[0,0]]
        value['propositions']+= [dict(kind='appearance',value='felt vivid'),
            dict(kind='physical_truth'),dict(kind='source_attribution')]
        timed=deepcopy(missing);timed['time']=13;value['propositions'].append(timed)
        wrong=deepcopy(missing);wrong['node']='n4';value['propositions'].append(wrong)
        wrong=deepcopy(missing);wrong['role']='green';value['propositions'].append(wrong)
        audit=report.audit_report(content,value)
        categories=[row['category'] for row in audit['errors']]
        for category in ('localization_contradiction','omitted_required_peak_set','unsupported_appearance',
                'unlicensed_physical_truth','incorrect_source_attribution','timing_error','address_error','role_error'):
            self.assertIn(category,categories)
        self.assertEqual(audit['error_count'],len(audit['errors']))
        self.assertEqual(categories.count('omitted_required_peak_set'),2)

    def test_peak_types_order_uniqueness_and_unhashable_malformed_values(self):
        content=packet()['content'];base=report.deterministic_template(content)
        for cells in ([[True,2],[2,1]],[[1.,2],[2,1]],[[2,1],[1,2]],[[1,2],[1,2]]):
            value=deepcopy(base);value['propositions'][0]['cells']=cells
            audit=report.audit_report(content,value)
            self.assertIn('invalid_peak_cells',[row['category'] for row in audit['errors']])
        for key,bad in [('kind',[]),('panel',{}),('view',[]),('command',[]),('node',[]),('role',[])]:
            value=deepcopy(base);value['propositions'][0][key]=bad
            self.assertFalse(report.audit_report(content,value)['listed_machine_facts_pass'])

    def test_optional_relations_scalars_duplicate_and_absolute_tolerance(self):
        content=packet()['content'];base=report.deterministic_template(content)
        node=content['panels'][0]['current'][0]
        relation=dict(kind='relation',panel='p0',view='current',command=None,node='n0',
            relation='red_blue_peak_relation',**node['red_blue_peak_relation'])
        scalar=dict(kind='scalar',panel='p0',view='current',command=None,node='n0',role='red',
            name='peak_probability',value=.5+0.5e-6)
        value=deepcopy(base);value['propositions'] += [relation,scalar]
        self.assertTrue(report.audit_report(content,value)['listed_machine_facts_pass'])
        value['propositions'] += [deepcopy(relation),deepcopy(scalar)]
        self.assertEqual(sum(row['category']=='duplicate_or_contradictory_claim'
            for row in report.audit_report(content,value)['errors']),2)
        value=deepcopy(base);scalar['value']=.5+2e-6;value['propositions'].append(scalar)
        self.assertFalse(report.audit_report(content,value)['listed_machine_facts_pass'])
        scalar['value']=float('nan')
        audit=report.audit_report(content,value)
        self.assertIn('nonfinite_type',report.json_bytes(audit).decode())
        relation['coincident_possible']=0
        value=deepcopy(base);value['propositions'].append(relation)
        self.assertFalse(report.audit_report(content,value)['listed_machine_facts_pass'])

    def test_restoration_null_pair_and_W_change_at_identical_A(self):
        baseline=report.export_packet([panel(),panel('W')],3,0)
        restored=report.export_packet([panel('A','current_restored'),panel('W','W_restored')],3,0)
        self.assertTrue(report.compare_packets(baseline,restored)['report_input_identical'])
        changed_w=panel('W','W_changed',1.)
        changed=report.export_packet([panel(),changed_w],3,0)
        comparison=report.compare_packets(baseline,changed)
        self.assertTrue(comparison['panel_pairs'][0]['native_current_identical'])
        self.assertTrue(comparison['panel_pairs'][0]['current_fields_identical'])
        self.assertFalse(comparison['panel_pairs'][1]['native_current_identical'])
        self.assertEqual(comparison['panel_pairs'][1]['changed_peak_sets'],[])
        self.assertFalse(comparison['causal_or_phenomenal_attribution_established'])

    def test_binding_failure_precedes_torch_import_or_deserialization(self):
        imported=[];original=builtins.__import__
        def spy(name,*args,**kwargs):
            imported.append(name);return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=spy),patch.object(report,'validate_input_bindings',
                side_effect=ValueError('handcrafted missing commitment')):
            with self.assertRaisesRegex(ValueError,'missing commitment'):
                report.build_reporting_packets({}, {})
        self.assertNotIn('torch',imported)
        with self.assertRaisesRegex(ValueError,'closure incomplete'):
            report.validate_input_bindings({},dict(schema=report.INPUT_SCHEMA,study_source_hashes={}))
        with self.assertRaisesRegex(ValueError,'full preservation commit'):
            report.committed_bytes('not-a-commit',{})


class ArchiveTableFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        cls.torch=torch

    def artificial_bundle(self):
        t=self.torch;b=4
        f=t.tensor(fields(),dtype=t.float32)
        family=t.zeros(b,4,3,24,4,4)
        family_fields=t.zeros(b,4,3,2,4,4)
        for band in range(3):
            family[:,:,band]=float(band+1)
            family_fields[:,:,band,:,0,band]=1.
        current=f.repeat(b,16,1,1,1,1)
        selected=f[0].repeat(b,14,1,1,1)
        next_fields=f[0].repeat(b,15,3,1,1,1)
        selected_maps=t.zeros(b,14,24,4,4)
        next_maps=t.zeros(b,15,3,24,4,4)
        commands=t.zeros(b,15,dtype=t.int64);commands[:,11]=3;commands[:,12]=t.arange(b)
        episode=dict(split='heldout',encoder_seed=1301,V=t.zeros(b,4,24,4,4),command_addresses=commands,
            command_symbols=t.zeros(b,15,dtype=t.int64),query_roles=t.zeros(b,15,dtype=t.int64),
            evaluator_mappings=t.tensor([4,1,2]).repeat(b,4,1))
        exogenous=current[:,12].clone()
        for world in range(b):
            exogenous[world,world]=family_fields[world,world,1]
        shared=dict(actual_operator_family=family,actual_family_fields=family_fields,
            actual_completed_fields=current,actual_controls=dict(exogenous_role_fields=exogenous[:,None].repeat(1,3,1,1,1,1),
                restored_role_fields=current[:,12:15].clone(),exogenous_local_logits=t.zeros(b,3,4,3,4,4),
                restored_local_logits=t.zeros(b,3,4,3,4,4)))
        tensors={};checks={};calls={};updates={};prefixes=[]
        for source in report.SOURCE_KEYS:
            tensors[source]=dict(selected_current_maps=selected_maps.clone(),selected_current_fields=selected.clone(),
                selected_next_maps=next_maps.clone(),selected_next_fields=next_fields.clone(),donors={})
            checks[source]={};calls[source]=[dict(kind='primary',decision_time=12)]
            updates[source]=[]
            for schedule in ('current','prospective'):
                ledger=dict(selected_current_maps=selected_maps[:,:3].clone()+1,
                    selected_current_fields=selected[:,:3].clone(),selected_next_maps=next_maps[:,:3].clone()+2,
                    policy_candidate_fields=next_fields[:,:3].clone(),restored_selected_current_maps=selected_maps[:,:3].clone(),
                    restored_selected_next_maps=next_maps[:,:3].clone(),restored_policy_candidate_fields=next_fields[:,:3].clone())
                if source=='analytic':
                    ledger.update(donor_survivors_at_patched_address=t.ones(b,3,6,dtype=t.bool),
                        donor_revealed_symbols_at_patched_address=t.zeros(b,3,3,dtype=t.bool))
                else:
                    ledger['donor_hidden_at_patched_address']=t.zeros(b,3,48)
                tensors[source]['donors'][schedule]=ledger
                for suffix in ('current','next','policy_fields','state'):
                    checks[source]['restore_'+schedule+'_'+suffix+'_exact']=t.ones(b,3,dtype=t.bool)
                for suffix in ('nonselected_state','selected_state_transfer','nonselected_current','nonselected_next',
                        'nonselected_decoder','next_inheritance'):
                    checks[source]['patch_'+schedule+'_'+suffix+'_exact']=t.ones(b,3,dtype=t.bool)
                calls[source]+=[dict(kind=kind+'_'+schedule,decision_time=12) for kind in ('donor','restored')]
                updates[source]+=[dict(kind='donor_replay',donor_schedule=schedule,donor_decision_time=12,
                    command_index=i) for i in range(11)]
        for schedule in ('current','prospective'):
            prefixes.append(dict(time=12,schedule=schedule,permitted_feedback_command_indices=list(range(11)),
                executor_command_indices=list(range(12)),patched_addresses=[3]*b if schedule=='current' else list(range(b)),
                updates=[dict(command_index=i,completed_map_hash='handcrafted') for i in range(11)],
                mapping_indices=[[0]*4 for _ in range(b)],pre_reveal_mask=[True]*12+[False],
                concealed_completion_hash='handcrafted'))
        control_rows=[]
        for world in range(b):
            actual=report._actual_maps(episode,shared,t,world,12)
            changed=actual.clone();changed[world]=family[world,world,1]
            def h(item):
                return report.sha256_bytes(b'torch.float32:(24, 4, 4):'+report._tensor_bytes(item,t,(24,4,4)))
            control_rows.append(dict(time=12,world_index=world,saved_w_hashes=[h(item) for item in actual],
                changed_w_hashes=[h(item) for item in changed],restoration='exact_saved_snapshot'))
        return dict(episode=episode,evaluation=dict(tensors=dict(shared=shared,sources=tensors),
            metadata=dict(donor_provenance=prefixes,source_call_visibility=calls,source_update_visibility=updates,
                actual_control_provenance=control_rows),software_checks=dict(sources=checks)))

    def test_archive_table_index_copy_donor_restoration_without_H(self):
        bundle=self.artificial_bundle()
        sources,w,sensor,previous,candidate=report._case_panels(bundle,0,1301,'handcrafted',self.torch)
        self.assertEqual((previous,candidate),(3,0))
        # Opaque symbol0 selects operator4 => family2, whose literal peak is (0,2).
        self.assertEqual(report.localization_summary(w['baseline'].candidate_fields[0][0][0])['peaks'],[[0,2]])
        self.assertIsNone(sensor.candidate_fields)
        baseline=report.export_packet([sources['A']['baseline']],previous,candidate)
        donor=report.export_packet([sources['A']['current_donor']],previous,candidate)
        self.assertFalse(report.compare_packets(baseline,donor)['panel_pairs'][0]['native_current_identical'])
        for variant in ('current_restored','prospective_restored','W_changed','W_restored'):
            value=report.export_packet([sources['A'][variant]],previous,candidate)
            self.assertTrue(report.compare_packets(baseline,value)['report_input_identical'])
        self.assertEqual(len(sources['A']['current_donor'].audit_metadata['donor_evidence']['source_update_receipts']),11)

    def test_archive_restoration_mismatch_or_false_gate_rejects(self):
        bundle=self.artificial_bundle()
        bundle['evaluation']['tensors']['sources']['A']['donors']['current']['restored_selected_current_maps'][0,0,0,0,0]=3.
        with self.assertRaisesRegex(ValueError,'restoration map/field'):
            report._case_panels(bundle,0,1301,'handcrafted',self.torch)
        bundle=self.artificial_bundle()
        bundle['evaluation']['software_checks']['sources']['A']['restore_current_state_exact'][0,0]=False
        with self.assertRaisesRegex(ValueError,'named donor/restoration gate'):
            report._case_panels(bundle,0,1301,'handcrafted',self.torch)
        bundle=self.artificial_bundle()
        bundle['evaluation']['metadata']['actual_control_provenance'][0]['changed_w_hashes'][0]='false'
        with self.assertRaisesRegex(ValueError,'control map/receipt hash'):
            report._case_panels(bundle,0,1301,'handcrafted',self.torch)


if __name__=='__main__':
    unittest.main()
