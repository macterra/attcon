"""Unexecuted handcrafted engine fixtures: no native artifacts, renderer or fit.

The sole fit_sources invocation is mocked control flow: scalar fake sources and
a no-op optimizer. It never trains the registered architecture or updates any
parameter. Production WORLDS and the fixed 300-update loop remain unchanged.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import attention_content_model_v1_experiment as engine
from attention_content_model_v1 import (
    FitNormalizer, MAPPINGS, PredictionBatch, SourceState,
    build_learned_sources,
)
from attention_content_model_v1_data import actual_episode, donor_prefix
from spatial_native_task_v1 import ChannelStandardizer, LocalOccupancyHead


def setUpModule():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def fixture_episode(split='dev'):
    # Numerical maps only; these are neither rendered worlds nor encoder outputs.
    v = torch.arange(2*4*384, dtype=torch.float32).reshape(2,4,24,4,4)/3072
    addresses = torch.tensor([list(range(4))*3+[3,3,1],
                              list(range(4))*3+[0,2,2]], dtype=torch.long)
    symbols = torch.tensor([[symbol for symbol in range(3) for _ in range(4)]+[0,1,2],
                            [symbol for symbol in range(3) for _ in range(4)]+[2,0,1]], dtype=torch.long)
    indices = torch.tensor([[0,1,2,3],[4,5,0,1]],dtype=torch.long)
    mappings = torch.tensor(MAPPINGS)[indices]
    episode = actual_episode(v,addresses,symbols,mappings)
    episode.update({'V':v,'encoder_seed':1301,'split':split,
        'command_addresses':addresses,'command_symbols':symbols,
        'evaluator_mapping_indices':indices,'evaluator_mappings':mappings,
        'query_roles':torch.tensor([[-1]*12+[0,1,0],[-1]*12+[1,0,1]]),
        'evaluator_cells':torch.tensor([[[[0,0],[3,3]]]*4]*2),
        'pixels':torch.zeros(2,4,3,16,16),'world_indices':torch.arange(2),
        'world_namespace':'handcrafted_software_fixture',
        'render_nuisance':{'renderer_used':False}})
    return episode


def fresh_head():
    head = LocalOccupancyHead()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.net[0].weight[0,0] = 1
        head.net[0].weight[1,1] = 1
        head.net[2].weight[1,0] = 1
        head.net[2].weight[2,1] = 1
    return head


class NumericalControlStub:
    """Toy W-derived outputs; not a fitted/frozen native encoder."""
    def controls(self,item):
        logits = item.mean((-2,-1))[:,:8]
        return {'category_logits':logits,'category_probabilities':logits.softmax(-1),
                'orientation_logits':item.mean((-3,-2,-1))[:,None]}


def independent_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(f'{value.dtype}:{tuple(value.shape)}:'.encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class ExperimentFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episode = fixture_episode()
        normalizer = FitNormalizer.fit(cls.episode['V'],split='fit')
        attention,ordinary,_ = build_learned_sources(normalizer,1301)
        cls.models = {'A':attention,'ordinary':ordinary}
        cls.before = {name:deepcopy(model.state_dict()) for name,model in cls.models.items()}
        cls.partial = {}
        with patch.object(engine,'WORLDS',2):
            cls.output = engine.evaluate_sources(cls.models,cls.episode,fresh_head(),
                ChannelStandardizer((0.,)*24,(1.,)*24),NumericalControlStub(),'dev',
                partial=cls.partial)

    def test_named_hard_gate_inventory_shapes_and_no_fitted_mutation(self):
        checks = self.output['software_checks']
        shared = {'sham_selected_exact_v','sham_nonselected_unchanged','same_w_driver_exact',
            'same_w_driver_fields_exact','same_w_driver_action_exact','actual_restoration_exact',
            'actual_restoration_fields_exact','actual_restoration_action_exact',
            'actual_address_fields_exact','actual_address_action_exact'}
        self.assertEqual(set(checks['shared']),shared|{'actual_family_reconstruction_exact',
                                                     'actual_compaction_reconstruction_exact'})
        for name,value in checks['shared'].items():
            width = {'actual_family_reconstruction_exact':60,
                     'actual_compaction_reconstruction_exact':61}.get(name,3)
            self.assertEqual(tuple(value.shape),(2,width))
        widths = {'initial_current_exact_v':1,'current_nonselected_copy_exact':14,
            'next_noncandidate_inheritance_exact':15,'observation_history_exact':15,
            'last_revealed_snapshot_exact':15,'prediction_inputs_unchanged_exact':15,
            'prediction_state_unchanged_exact':15,'selected_policy_fork_exact':3,
            'selected_policy_native_action_exact':3,'confidence_choice_exact':3,
            'address_current_equivariance':3,'address_next_equivariance':3}
        for schedule in ('current','prospective'):
            for item in ('current','next','policy','policy_fields','state'):
                widths[f'restore_{schedule}_{item}_exact'] = 3
            for item in ('nonselected_state','selected_state_transfer','nonselected_decoder',
                         'next_inheritance','nonselected_current','nonselected_next'):
                widths[f'patch_{schedule}_{item}_exact'] = 3
        for source,gates in checks['sources'].items():
            expected = dict(widths)
            if source != 'analytic':
                expected['fitted_weights_unchanged'] = None
            if source == 'A':
                expected.update(alias_current_exact=15,alias_next_exact=15)
            self.assertEqual(set(gates),set(expected))
            for name,value in gates.items():
                self.assertEqual(tuple(value.shape),(1,) if expected[name] is None else (2,expected[name]))
        for scope in (checks['shared'],*checks['sources'].values()):
            for value in scope.values():
                self.assertEqual(value.dtype,torch.bool)
                self.assertEqual(value.device.type,'cpu')
                self.assertTrue(bool(value.all()))
        self.assertTrue(self.output['metadata']['software_invariants_pass'])
        for name,model in self.models.items():
            for key,value in model.state_dict().items():
                self.assertTrue(torch.equal(value,self.before[name][key]))

    def test_compaction_success_and_false_gate_retains_first_raw_evidence(self):
        with patch.object(engine,'WORLDS',2):
            compact = engine.compact_episode(self.episode)
            bad = dict(self.episode)
            bad['actual_forks'] = bad['actual_forks'].clone()
            bad['actual_forks'][0,4,1,2,0,0,0] += .25
            retained = engine.compact_episode(bad)
        self.assertEqual(compact['actual_compaction_status'],'exact_reconstruction_bound')
        self.assertNotIn('actual_forks',compact)
        self.assertTrue(torch.equal(compact['actual_reconstruction_exact'],
            self.output['software_checks']['shared']['actual_compaction_reconstruction_exact']))
        self.assertEqual(tuple(retained['actual_reconstruction_exact'].shape),(2,61))
        self.assertFalse(bool(retained['actual_reconstruction_exact'].all()))
        self.assertEqual(retained['actual_compaction_status'],'failed_reconstruction_raw_first_evidence_retained')
        self.assertIs(retained['actual_forks'],bad['actual_forks'])
        self.assertIs(retained['actual_completed_w'],bad['actual_completed_w'])

    def test_actual_prediction_and_update_manifests_bind_candidates_and_latents(self):
        metadata = self.output['metadata']; episode = self.episode
        for source in engine.SOURCES:
            inventory = metadata['source_call_inventory'][source]
            self.assertEqual(inventory,{'prediction_calls':45 if source=='A' else 30,
                'primary_update_calls':14,'donor_update_calls':72,
                'address_local_decoder_probe_calls':27})
            tensors = self.output['tensors']['sources'][source]
            primary = [c for c in metadata['source_call_visibility'][source] if c['kind']=='primary']
            self.assertEqual(len(primary),15)
            for time,call in enumerate(primary):
                self.assertEqual(call['candidate_address_hash'],independent_hash(episode['command_addresses'][:,time]))
                self.assertEqual(call['last_revealed_W_hash'],independent_hash(episode['actual_completed_w'][:,max(0,time-1)]))
                self.assertEqual(call['update_command_indices'],list(range(max(0,time-1))))
                latent = ({'hidden':tensors['pre_reveal_hidden'][:,time]} if source!='analytic'
                    else {'survivors':tensors['pre_reveal_survivors'][:,time],
                          'revealed_symbols':tensors['pre_reveal_revealed_symbols'][:,time]})
                self.assertEqual(call['state_hashes'],{key:independent_hash(value) for key,value in latent.items()})
                self.assertEqual(call['post_call_state_hashes'],call['state_hashes'])
                self.assertFalse(call['fork_outcomes_visible'])
                self.assertTrue(call['completed'])
            updates = metadata['source_update_visibility'][source]
            ordinary = [c for c in updates if c['kind']=='primary_feedback']
            self.assertEqual([c['command_index'] for c in ordinary],list(range(14)))
            for call in ordinary:
                completion = call['completed_at']
                self.assertEqual(call['completed_W_hash'],independent_hash(episode['actual_completed_w'][:,completion]))
                self.assertEqual(call['post_update_command_indices'],list(range(completion)))
            for time in (12,13,14):
                for schedule in ('current','prospective'):
                    donor_calls = [c for c in updates if c['kind']=='donor_replay'
                        and c['donor_decision_time']==time and c['donor_schedule']==schedule]
                    self.assertEqual([c['command_index'] for c in donor_calls],list(range(time-1)))
                    address = episode['command_addresses'][:,time-1 if schedule=='current' else time]
                    prefix = donor_prefix(episode['V'],episode,time,address)
                    for call in donor_calls:
                        self.assertEqual(call['completed_W_hash'],independent_hash(prefix['actual_completed_w'][:,call['completed_at']]))
            self.assertEqual(tensors['terminal_revealed_through'],14)
            self.assertEqual(tensors['terminal_update_command_indices'],tuple(range(14)))

    def test_analytic_seen_masks_and_complete_donor_restoration_exports(self):
        tensors = self.output['tensors']['sources']['analytic']
        expected_seen = torch.zeros(2,4,3,dtype=torch.bool)
        for time in range(14):
            expected_seen[torch.arange(2),self.episode['command_addresses'][:,time],
                          self.episode['command_symbols'][:,time]] = True
        self.assertTrue(torch.equal(tensors['terminal_revealed_symbols'],expected_seen))
        self.assertEqual(tuple(tensors['pre_reveal_survivors'].shape),(2,15,4,6))
        self.assertEqual(tuple(tensors['pre_reveal_revealed_symbols'].shape),(2,15,4,3))
        for source in engine.SOURCES:
            raw = self.output['tensors']['sources'][source]
            for schedule,ledger in raw['donors'].items():
                self.assertEqual(tuple(ledger['selected_current_maps'].shape),(2,3,24,4,4))
                self.assertEqual(tuple(ledger['selected_next_maps'].shape),(2,3,3,24,4,4))
                self.assertTrue(torch.equal(ledger['restored_selected_current_maps'],raw['selected_current_maps'][:,11:14]))
                self.assertTrue(torch.equal(ledger['restored_selected_next_maps'],raw['selected_next_maps'][:,12:15]))
                self.assertTrue(torch.equal(ledger['restored_policy_symbol'],raw['policy_symbol']))
                if source=='analytic':
                    self.assertEqual(tuple(ledger['donor_survivors_at_patched_address'].shape),(2,3,6))
                    self.assertEqual(tuple(ledger['donor_revealed_symbols_at_patched_address'].shape),(2,3,3))
                self.assertEqual(self.output['metadata']['donor_summaries'][source][schedule]['all_predetermined_pairs'],6)
                self.assertFalse(self.output['metadata']['donor_summaries'][source][schedule]['changed_only_subset_used'])

    def test_bootstrap_full_and_inherited_metrics_use_saved_paired_world_draws(self):
        metadata = self.output['metadata']; shared = self.output['tensors']['shared']
        expected_indices = torch.randint(2,(256,2),generator=torch.Generator().manual_seed(350000000+1301))
        self.assertTrue(torch.equal(shared['bootstrap_indices'],expected_indices))
        for metric in ('next_inherited_mse','current_full_mse','next_full_mse',
            'current_full_field_l1','next_full_field_l1','current_cell_disagreement',
            'next_cell_disagreement','current_full_coarse_disagreement',
            'next_full_red_cell_disagreement','policy_red_accuracy','policy_blue_accuracy'):
            self.assertIn(metric,metadata['bootstrap_primary_metric_names'])
            self.assertIn('A_minus_ordinary/'+metric,metadata['bootstrap'])
        values = {}
        for source in engine.SOURCES:
            errors = self.output['tensors']['sources'][source]['next_full_normalized_mse_by_address']
            totals = []
            for world in range(2):
                parts = [errors[world,t,s,a] for t in range(15) for s in range(3)
                         for a in range(4) if a!=int(self.episode['command_addresses'][world,t])]
                totals.append(torch.stack(parts).mean())
            values[source] = torch.stack(totals).double()
        difference = values['A']-values['ordinary']
        entry = metadata['bootstrap']['A_minus_ordinary/next_inherited_mse']
        self.assertAlmostEqual(entry['mean'],float(difference.mean()),places=7)
        expected = torch.quantile(difference[expected_indices].mean(-1),torch.tensor([.025,.975],dtype=torch.double))
        self.assertTrue(torch.allclose(torch.tensor(entry['interval'],dtype=torch.double),expected,atol=1e-7,rtol=1e-6))

    def test_all_predetermined_donor_field_scalar_and_factual_diagnostics_retained(self):
        for source in engine.SOURCES:
            for schedule,ledger in self.output['tensors']['sources'][source]['donors'].items():
                self.assertEqual(tuple(ledger['coincident_schedule_addresses'].shape),(2,3))
                self.assertTrue(bool(ledger['coincident_schedule_addresses'].any()))
                for label in ('current','next'):
                    for key in ('full_field_l1_change_by_address','full_entropy_change',
                        'full_maxima_change','full_top_two_margin_change','full_expected_columns_change',
                        'full_cell_disagreement_with_fixed_W','full_coarse_disagreement_with_fixed_W',
                        'full_normalized_mse_to_fixed_W_by_address','factual_cell_disagreement',
                        'factual_estimated_cell_correct','factual_actual_cell_correct'):
                        value = ledger[label+'_'+key]
                        self.assertEqual(tuple(value.shape[:2]),(2,3))
                summary = self.output['metadata']['donor_summaries'][source][schedule]
                value = ledger['next_full_entropy_change'].double().abs().mean()
                self.assertAlmostEqual(summary['next_full_entropy_change_mean_abs_all_pairs'],float(value))


class FakeSource(torch.nn.Module):
    """Counter/gradient plumbing only: zero derivative and no learned decoder."""
    def __init__(self,normalizer,source_key):
        super().__init__(); self.normalizer=normalizer; self.source_key=source_key
        self.scalar=torch.nn.Parameter(torch.tensor(0.))
    def initial_state(self,batch):
        return SourceState(torch.zeros(batch,4,48),0,(),self.source_key)
    def predict(self,observation,state,address):
        current=observation.v+self.scalar*0
        return PredictionBatch(current,current[:,None].expand(-1,3,-1,-1,-1,-1),
                               observation.decision_time,self.source_key)
    def update(self,state,v,feedback):
        return replace(state,revealed_through=feedback.completed_at,
                       update_command_indices=tuple(range(feedback.completed_at)))


class NoOpOptimizer:
    """Never modifies parameters; preserves only synthetic call counters."""
    def __init__(self,parameters,**kwargs):
        self.parameters=list(parameters); self.calls=0
    def zero_grad(self,**kwargs):
        for value in self.parameters:
            value.grad=None
    def step(self):
        self.calls+=1
    def state_dict(self):
        return {'fixture_no_parameter_updates':True,'step_calls':self.calls}


class FailurePreservationFixtures(unittest.TestCase):
    def test_completed_source_calls_actions_and_secondary_survive_next_source_failure(self):
        episode=fixture_episode(); partial={}
        attention,ordinary,_=build_learned_sources(FitNormalizer.fit(episode['V'],split='fit'),1301)
        original=engine.AnalyticMappingSource.predict
        def injected(source,observation,state,address):
            if observation.decision_time==1:
                raise RuntimeError('fixture injected analytic prediction failure')
            return original(source,observation,state,address)
        with patch.object(engine,'WORLDS',2),patch.object(engine.AnalyticMappingSource,'predict',injected):
            with self.assertRaisesRegex(RuntimeError,'fixture injected analytic'):
                engine.evaluate_sources({'A':attention,'ordinary':ordinary},episode,fresh_head(),
                    ChannelStandardizer((0.,)*24,(1.,)*24),NumericalControlStub(),'dev',partial=partial)
        retained=partial['engine_evaluation']
        for source in ('A','ordinary'):
            self.assertEqual(len(retained['source_update_visibility'][source]),86)
            self.assertEqual(len(retained['source_decoder_visibility'][source]),27)
            self.assertTrue(all(call['completed'] for call in retained['source_update_visibility'][source]))
            self.assertTrue(all(call['completed'] for call in retained['source_decoder_visibility'][source]))
            self.assertIn(source,retained['source_tensors'])
        self.assertEqual(len(retained['actual_actions']),9)
        self.assertEqual(len(retained['analytic_secondary']['expectation_of_H_fields']),1)
        self.assertEqual(len(retained['in_progress_calls']),2)
        self.assertNotIn('completed',retained['in_progress_calls'][-1])

    def test_mocked_handoff_preserves_completed_A_before_ordinary_constructor_failure(self):
        episode=fixture_episode('fit'); partial={}; constructors=[]; sources={}
        def fake_build(normalizer,seed):
            sources.update({name:FakeSource(normalizer,name) for name in ('A','ordinary')})
            return sources['A'],sources['ordinary'],None
        def optimizer_factory(parameters,**kwargs):
            constructors.append(True)
            if len(constructors)==2:
                raise RuntimeError('fixture injected ordinary optimizer constructor failure')
            return NoOpOptimizer(parameters,**kwargs)
        with patch.object(engine,'WORLDS',2),patch.object(engine,'build_learned_sources',fake_build),\
             patch.object(torch.optim,'Adam',optimizer_factory):
            with self.assertRaisesRegex(RuntimeError,'fixture injected ordinary'):
                engine.fit_sources(1301,episode,partial=partial)
        retained=partial['engine_fit']['source_states']
        self.assertEqual(retained['A']['status'],'completed')
        self.assertEqual(retained['A']['completed_updates'],300)
        self.assertEqual(retained['A']['optimizer']['step_calls'],300)
        self.assertEqual(retained['ordinary']['completed_updates'],0)
        self.assertIn('checkpoint',retained['ordinary'])
        self.assertEqual(len(partial['engine_fit']['training_log']),300)
        self.assertEqual(float(sources['A'].scalar),0.)
        sources['A'].scalar.data.add_(1.)
        self.assertEqual(float(retained['A']['checkpoint']['scalar']),0.)


if __name__=='__main__':
    unittest.main()
