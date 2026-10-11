"""Prospective bounded correction fixtures; source review precedes execution.

Handcrafted tensors/fresh weights and mocked provenance only. No archive,
registered world, renderer, optimizer, backward, fitting or actual Git call.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import attention_content_model_v1_reference as reference
import run_attention_content_model_v1_corrected as boundary
import verify_attention_content_model_v1_corrected as corrected
from attention_content_model_v1 import (
    AnalyticMappingSource, CommandBatch, FitNormalizer, LearnedContentModel, ObservationBatch,
)


def setUpModule():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def toy_observation(time=0):
    v=torch.arange(2*4*24*16,dtype=torch.float32).reshape(2,4,24,4,4)/2048
    previous=CommandBatch(torch.tensor([0,3]),torch.tensor([1,2])) if time else None
    return ObservationBatch(v,v.clone() if not time else v*.75,time,max(0,time-1),
        previous,torch.ones(2,4,dtype=torch.bool))


def toy_decoy(obs,state,candidate):
    relevant=torch.zeros(2,4,dtype=torch.bool);rows=torch.arange(2)
    relevant[rows,candidate]=True
    if obs.decision_time:
        relevant[rows,obs.previous_command.address]=True
    v=obs.v.clone();v[~relevant]=-v[~relevant]+.125
    if hasattr(state,'hidden'):
        h=state.hidden.clone();h[~relevant]+=1.
        state=replace(state,hidden=h)
    else:
        survivors=state.survivors.clone();survivors[~relevant]=True
        state=replace(state,survivors=survivors)
    return v,state


def toy_models():
    norm=FitNormalizer((0.,)*24,(1.,)*24)
    return (LearnedContentModel(norm,91,'A'),AnalyticMappingSource())


def minimal_context():
    return dict(amendment_path='fixture/amendment.json',amendment_sha256='a'*64,
        amendment_commit='b'*40,reference_source_commit='c'*40,
        corrected_verifier_sha256='d'*64,corrected_runner_sha256='e'*64,
        registered_verifier_sha256=reference.ORIGINAL_VERIFIER_SHA,immutable={'fixture_only':True})


def minimal_report(context,phase='dev'):
    return dict(schema=reference.REPORT_SCHEMA,version=1,phase=phase,status=reference.PASS_STATUS,
        verifier_sha256=context['corrected_verifier_sha256'],registered_verifier_sha256=reference.ORIGINAL_VERIFIER_SHA,
        reference_provenance=context,hard_invariants_pass=True,error_count=0,errors=[],check_count=1,
        checks=[dict(name='handcrafted_fixture_only',passed=True)],no_fit_replay=True,no_optimizer=True,
        no_reporter=True,qualia_established=False,presentation_qualified=False,retry_or_repair_performed=False)


class FakeGitProcess:
    def __init__(self,raw):
        self.stdout=io.BytesIO(raw);self.stderr=io.BytesIO(b'')

    def wait(self):
        return 0


class TimeZeroReferenceFixtures(unittest.TestCase):
    def test_coherent_time_zero_decoy_exact_selected_next_for_both_sources(self):
        with torch.no_grad():
            for model in toy_models():
                obs=toy_observation();candidate=torch.tensor([1,2]);state=model.initial_state(2)
                before=obs.v.clone();snapshot=obs.last_revealed_w.clone()
                baseline=model.predict(obs,state,candidate)
                v,decoy_state=toy_decoy(obs,state,candidate)
                decoy_obs=corrected.coherent_decoy_observation(obs,v)
                self.assertTrue(torch.equal(decoy_obs.v,decoy_obs.last_revealed_w))
                decoy_obs.validate()
                decoy=model.predict(decoy_obs,decoy_state,candidate)
                rows=torch.arange(2)[:,None];symbols=torch.arange(3)[None,:]
                self.assertTrue(torch.equal(decoy.next[rows,symbols,candidate[:,None]],
                    baseline.next[rows,symbols,candidate[:,None]]))
                self.assertTrue(torch.equal(decoy.current,v))
                self.assertTrue(torch.equal(obs.v,before));self.assertTrue(torch.equal(obs.last_revealed_w,snapshot))

    def test_original_inconsistent_time_zero_decoy_rejected_before_decoder(self):
        for model in toy_models():
            obs=toy_observation();candidate=torch.tensor([1,2]);state=model.initial_state(2)
            v,decoy_state=toy_decoy(obs,state,candidate)
            with patch.object(model,'decode',side_effect=AssertionError('decoder must not be reached')) as decoder:
                with self.assertRaisesRegex(ValueError,'initial state is exactly observed V'):
                    model.predict(replace(obs,v=v),decoy_state,candidate)
            decoder.assert_not_called()

    def test_later_decoy_preserves_last_snapshot_and_full_prediction_for_both_sources(self):
        with torch.no_grad():
            for model in toy_models():
                obs=toy_observation(2);candidate=torch.tensor([1,2])
                state=replace(model.initial_state(2),revealed_through=1,update_command_indices=(0,))
                baseline=model.predict(obs,state,candidate)
                v,decoy_state=toy_decoy(obs,state,candidate)
                decoy_obs=corrected.coherent_decoy_observation(obs,v)
                self.assertIs(decoy_obs.last_revealed_w,obs.last_revealed_w)
                self.assertIs(decoy_obs.previous_command,obs.previous_command)
                decoy_obs.validate();decoy=model.predict(decoy_obs,decoy_state,candidate)
                self.assertTrue(torch.equal(decoy.current,baseline.current))
                self.assertTrue(torch.equal(decoy.next,baseline.next))


class ExactProvenanceFixtures(unittest.TestCase):
    def test_original_prefix_requires_all_evidence_not_only_names_and_pass_flags(self):
        original=[dict(name='first',passed=True,actual_shape=[2,3],actual_dtype='torch.float32',
            maximum_absolute_difference=0.,expected_sha256='a'*64),
            dict(name='second',passed=True,actual=2,expected=2)]
        context={'immutable':{'first_failed_check_count':2,
            'first_failed_evidence_sha256':reference.evidence_hash(original)}}
        checks=[dict(name='reference_amendment/supplement',passed=True),*deepcopy(original)]
        with patch.object(reference,'first_failed_report',return_value={'checks':original}):
            matched,mismatches=reference.original_prefix_evidence(checks,context)
            self.assertTrue(matched);self.assertEqual(mismatches,[])
            changed=deepcopy(checks);changed[1]['maximum_absolute_difference']=1e-9
            matched,mismatches=reference.original_prefix_evidence(changed,context)
            self.assertFalse(matched);self.assertEqual(mismatches[0]['name'],'first')
            self.assertEqual(mismatches[0]['original'],original[0])
            self.assertEqual(mismatches[0]['corrected'],changed[1])
            matched,mismatches=reference.original_prefix_evidence(checks[:-1],context)
            self.assertFalse(matched);self.assertIsNone(mismatches[0]['corrected'])

    def test_committed_binding_rejects_wrong_bytes_missing_and_short_commit(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);name='fixture.py';raw=b'handcrafted software bytes\n'
            (root/name).write_bytes(raw);sha=hashlib.sha256(raw).hexdigest()
            with patch.object(reference,'ROOT',root),patch.object(reference.subprocess,'Popen',return_value=FakeGitProcess(b'wrong')):
                with self.assertRaisesRegex(ValueError,'committed reference binding failed'):
                    reference.bind('a'*40,{name:sha})
            with patch.object(reference,'ROOT',root),patch.object(reference.subprocess,'Popen') as git:
                with self.assertRaises(FileNotFoundError):reference.bind('a'*40,{'missing.py':sha})
                with self.assertRaisesRegex(ValueError,'full lowercase'):reference.bind('short',{})
                with self.assertRaisesRegex(ValueError,'noncanonical'):reference.bind('a'*40,{'./fixture.py':sha})
                git.assert_not_called()

    def test_exact_creation_validation_and_exclusive_immutable_amendment(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);immutable={'handcrafted_context_only':True,
                'original_source_hashes':{}}
            for name in reference.NEW_CLOSURE:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(Path(reference.__file__).read_bytes() if name.endswith('/attention_content_model_v1_reference.py')
                    else ('handcrafted source '+name).encode())
            hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in reference.NEW_CLOSURE}
            approval=dict(scope=reference.APPROVAL_SCOPE,approved=True,approved_source_sha256=hashes,
                original_registration_sha256=reference.REGISTRATION_SHA,first_failed_report_raw_sha256=reference.FIRST_RAW_SHA)
            (root/'approval.json').write_text(json.dumps(approval))
            with patch.object(reference,'ROOT',root),patch.object(reference,'immutable_context',return_value=immutable),\
                    patch.object(reference,'bind') as bind,patch.object(reference,'ancestor') as ancestor:
                reference.create_amendment('a'*40,'approval.json','amendment.json')
                raw=(root/'amendment.json').read_bytes()
                context=reference.validate_amendment('amendment.json','b'*40,reference.CORRECTED_VERIFIER,
                    reference.REGISTRATION_PATH,reference.REGISTRATION_COMMIT,reference.FIT_COMMIT)
                self.assertEqual(context['corrected_verifier_sha256'],hashes[reference.CORRECTED_VERIFIER])
                self.assertEqual(context['amendment_sha256'],hashlib.sha256(raw).hexdigest())
                self.assertTrue(bind.called);self.assertTrue(ancestor.called)
                with self.assertRaises(FileExistsError):
                    reference.create_amendment('a'*40,'approval.json','amendment.json')
                self.assertEqual((root/'amendment.json').read_bytes(),raw)
                value=json.loads(raw);value['source_hashes'][reference.CORRECTED_VERIFIER]='0'*64
                (root/'amendment.json').write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError,'exact independent approval'):
                    reference.validate_amendment('amendment.json','b'*40,reference.CORRECTED_VERIFIER,
                        reference.REGISTRATION_PATH,reference.REGISTRATION_COMMIT,reference.FIT_COMMIT)
                approval['approved']=False;(root/'approval.json').write_text(json.dumps(approval))
                with self.assertRaisesRegex(ValueError,'exact new bytes'):
                    reference.create_amendment('a'*40,'approval.json','second.json')
                self.assertFalse((root/'second.json').exists())

    def test_corrected_report_rejects_original_or_arbitrary_hash_false_gate_and_bad_provenance(self):
        context=minimal_context();report=minimal_report(context)
        self.assertTrue(reference.report_identity_matches(report,context,'dev'))
        mutations=[('schema','attention_content_model_v1_independent_phase_reconstruction'),
            ('verifier_sha256',reference.ORIGINAL_VERIFIER_SHA),('verifier_sha256','0'*64),
            ('registered_verifier_sha256','0'*64),('hard_invariants_pass',False),
            ('error_count',1),('errors',['fixture failure']),('exception','original failure'),
            ('serialization_error','serialization failure'),('no_reporter',False),
            ('presentation_qualified',True),('retry_or_repair_performed',True)]
        for key,value in mutations:
            changed=deepcopy(report);changed[key]=value
            self.assertFalse(reference.report_identity_matches(changed,context,'dev'),key)
        changed=deepcopy(report);changed['reference_provenance']['amendment_sha256']='0'*64
        self.assertFalse(reference.report_identity_matches(changed,context,'dev'))
        changed=deepcopy(report);changed['checks'][0]['passed']=False
        self.assertFalse(reference.report_identity_matches(changed,context,'dev'))

    def test_boundary_rejects_fit_register_and_amendment_failure_before_claim_or_torch(self):
        with patch.object(boundary,'validate_amendment') as amendment,patch.object(boundary,'write_exclusive') as write,\
                patch.object(torch,'load',side_effect=AssertionError('no archive load')) as load:
            for phase in ('fit','register'):
                with self.assertRaisesRegex(ValueError,'rejects fit/register'):
                    boundary.execute_phase(SimpleNamespace(phase=phase))
            with self.assertRaisesRegex(ValueError,'rejects fit/register'):boundary.register(SimpleNamespace())
            amendment.assert_not_called();write.assert_not_called();load.assert_not_called()
        args=SimpleNamespace(phase='dev',reference_amendment=Path('fixture.json'),reference_commit='a'*40,
            registration=Path(reference.REGISTRATION_PATH),registration_commit=reference.REGISTRATION_COMMIT,
            fit_commit=reference.FIT_COMMIT)
        with patch.object(boundary,'validate_amendment',side_effect=ValueError('unapproved fixture amendment')),\
                patch.object(boundary,'write_exclusive') as write,patch.object(torch,'load') as load:
            with self.assertRaisesRegex(ValueError,'unapproved fixture amendment'):boundary.execute_phase(args)
            write.assert_not_called();load.assert_not_called()

    def test_boundary_requires_corrected_fit_full_pass_before_phase_claim(self):
        context=minimal_context()
        args=SimpleNamespace(phase='dev',reference_amendment=Path('fixture.json'),reference_commit='a'*40,
            registration=Path(reference.REGISTRATION_PATH),registration_commit=reference.REGISTRATION_COMMIT,
            fit_commit=reference.FIT_COMMIT,fit_reference_manifest=Path('fixture_manifest.json'),fit_reference_commit='b'*40)
        with patch.object(boundary,'validate_amendment',return_value=context),\
                patch.object(boundary,'digest',return_value=context['corrected_runner_sha256']),\
                patch.object(boundary,'validate_corrected_fit',side_effect=ValueError('fixture fit audit failed')),\
                patch.object(boundary,'write_exclusive') as write,patch.object(torch,'load') as load:
            with self.assertRaisesRegex(ValueError,'fixture fit audit failed'):boundary.execute_phase(args)
            write.assert_not_called();load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
