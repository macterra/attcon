"""Unexecuted bounded reconstruction fixtures; no artifacts, worlds or fitting.

Use handcrafted tensors/fresh weights only. Source publication and separate
review/release must precede the first test execution. Git responses are mocked;
no Git command, archive load, renderer, optimizer or Reporter is used.
"""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import verify_attention_content_model_v1 as verifier
from attention_content_model_v1 import FitNormalizer
from spatial_native_task_v1 import LocalOccupancyHead,role_fields


def setUpModule():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    verifier.torch=torch


def coordinates():
    return dict(mean=torch.linspace(-.2,.2,24,dtype=torch.float64).tolist(),
                scale=torch.linspace(.5,1.5,24,dtype=torch.float64).tolist())


def fixture_maps(batch=2):
    # Numerical inputs only; never rendered or extracted task worlds.
    return torch.arange(batch*24*16,dtype=torch.float32).reshape(batch,24,4,4)/1024


class FakeGitProcess:
    def __init__(self,raw):
        self.stdout=io.BytesIO(raw)
        self.stderr=io.BytesIO(b'')

    def wait(self):
        return 0


class IndependentReconstructionFixtures(unittest.TestCase):
    def test_explicit_gru_equations_against_fresh_grucell(self):
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.manual_seed(71)
            cell=torch.nn.GRUCell(771,48)
            weights={'gru.'+key:value for key,value in cell.state_dict().items()}
            v=fixture_maps();completed=v.flip(-1)*.75
            h=torch.linspace(-.2,.2,96).reshape(2,48);symbols=torch.tensor([0,2])
            norm=coordinates();mean=torch.tensor(norm['mean'])[:,None,None];scale=torch.tensor(norm['scale'])[:,None,None]
            features=torch.cat((((v-mean)/scale).flatten(1),((completed-mean)/scale).flatten(1),
                torch.nn.functional.one_hot(symbols,3).float()),1)
            expected=cell(features,h)
            actual=verifier.gru_reference(h,v,completed,symbols,weights,norm)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
            self.assertEqual(actual.shape,(2,48))

    def test_explicit_residual_decoder_against_fresh_linear_modules(self):
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.manual_seed(72)
            decoder=torch.nn.Sequential(torch.nn.Linear(435,96),torch.nn.ReLU(),torch.nn.Linear(96,384))
            weights={'decoder.'+key:value for key,value in decoder.state_dict().items()}
            v=fixture_maps();h=torch.linspace(-.1,.1,96).reshape(2,48);symbols=torch.tensor([1,2])
            norm=coordinates();mean=torch.tensor(norm['mean'])[:,None,None];scale=torch.tensor(norm['scale'])[:,None,None]
            normalized=(v-mean)/scale
            features=torch.cat((normalized.flatten(1),h,torch.nn.functional.one_hot(symbols,3).float()),1)
            expected=(normalized+decoder(features).reshape(2,24,4,4))*scale+mean
            actual=verifier.decoder_reference(v,h,symbols,weights,norm)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)

    def test_fit_only_moment_coordinates_constant_channel_and_mse(self):
        v=fixture_maps(2)[:,None].expand(-1,4,-1,-1,-1).clone();v[:,:,0]=2.
        actual=verifier.normalizer_reference(v)
        rows=v.double().permute(0,1,3,4,2).reshape(-1,24)
        expected_mean=rows.sum(0)/rows.shape[0]
        expected_scale=(rows.square().sum(0)/rows.shape[0]-expected_mean.square()).clamp_min(0).sqrt()
        expected_scale[0]=1.
        torch.testing.assert_close(torch.tensor(actual['mean'],dtype=torch.float64),expected_mean,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(torch.tensor(actual['scale'],dtype=torch.float64),expected_scale,atol=1e-12,rtol=1e-12)
        self.assertEqual(actual['mean'][0],2.);self.assertEqual(actual['scale'][0],1.)
        core=FitNormalizer.fit(v,split='fit')
        torch.testing.assert_close(torch.tensor(core.mean),torch.tensor(actual['mean']),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(torch.tensor(core.scale),torch.tensor(actual['scale']),atol=1e-6,rtol=1e-5)
        self.assertTrue(torch.equal(verifier.mse(v,v,actual),torch.zeros(2,4,dtype=torch.float64)))
        shifted=v+torch.tensor(actual['scale'],dtype=torch.float32)[None,None,:,None,None]
        torch.testing.assert_close(verifier.mse(shifted,v,actual),torch.ones(2,4,dtype=torch.float64),atol=1e-6,rtol=1e-5)
        with self.assertRaisesRegex(ValueError,'fit split only'):
            FitNormalizer.fit(v,split='heldout')

    def test_block_operators_from_literal_means_and_no_input_mutation(self):
        v=torch.arange(24*16,dtype=torch.float32).reshape(1,24,4,4);saved=v.clone()
        base=torch.arange(24,dtype=torch.float32)[None,:,None,None]*16
        expected2=base+torch.tensor([[2.5,2.5,4.5,4.5],[2.5,2.5,4.5,4.5],
            [10.5,10.5,12.5,12.5],[10.5,10.5,12.5,12.5]])
        expected4=(base+7.5).expand_as(v)
        self.assertTrue(torch.equal(verifier.block_reference(v,1),v))
        self.assertTrue(torch.equal(verifier.block_reference(v,2),expected2))
        self.assertTrue(torch.equal(verifier.block_reference(v,4),expected4))
        family=verifier.family_reference(v)
        self.assertEqual(family.shape,(1,3,24,4,4))
        self.assertTrue(torch.equal(family[:,1],expected2));self.assertTrue(torch.equal(v,saved))

    def test_native_head_itemwise_equations_and_manual_fields(self):
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.manual_seed(73);head=LocalOccupancyHead();norm=coordinates()
            native=dict(native_head_fit=dict(weights=head.state_dict(),fit_only_standardizer=norm))
            maps=fixture_maps(6).reshape(2,3,24,4,4)
            mean=torch.tensor(norm['mean'])[:,None,None];scale=torch.tensor(norm['scale'])[:,None,None]
            expected=torch.stack([head((item-mean)/scale) for item in maps.reshape(-1,24,4,4)]).reshape(2,3,3,4,4)
            actual=verifier.head_reference(maps,native)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
            fields=verifier.fields_reference(actual)
            expected_fields=torch.stack([role_fields(item) for item in actual.reshape(-1,3,4,4)]).reshape(2,3,2,4,4)
            torch.testing.assert_close(fields,expected_fields,atol=1e-6,rtol=1e-5)
            uniform=verifier.fields_reference(torch.tensor([0.,3.,-2.])[:,None,None].expand(3,4,4))
            self.assertTrue(torch.equal(uniform,torch.full((2,4,4),1/16)))

    def test_literal_content_metrics_and_exact_uniform_ties(self):
        estimated=torch.full((2,2,4,4),1/16);actual=estimated.clone()
        estimated[0]=0.;actual[0]=0.
        estimated[0,0,0,0]=1.;estimated[0,1,0,3]=1.
        actual[0,0,0,1]=1.;actual[0,1,0,3]=1.
        truth=torch.tensor([[[0,0],[0,3]],[[1,0],[2,3]]])
        metrics=verifier.field_metrics(estimated,actual,truth)
        self.assertTrue(torch.equal(metrics['cell_disagreement'],torch.tensor([[True,False],[False,False]])))
        self.assertTrue(torch.equal(metrics['estimated_cell_correct'],torch.tensor([[True,True],[False,False]])))
        self.assertTrue(torch.equal(metrics['actual_cell_correct'],torch.tensor([[False,True],[False,False]])))
        self.assertEqual(float(metrics['field_l1'][0]),1/16)
        self.assertTrue(torch.equal(metrics['max_tie_count'],torch.tensor([[1,1],[16,16]])))
        self.assertTrue(torch.equal(metrics['top_two_margin'][1],torch.zeros(2)))
        self.assertTrue(torch.equal(verifier.coarse_reference(estimated),torch.tensor([1,0])))
        self.assertTrue(torch.equal(estimated[1].flatten(-2).argmax(-1),torch.tensor([0,0])))
        torch.testing.assert_close(metrics['entropy'][1],torch.log(torch.tensor(16.,dtype=torch.float64)).expand(2),atol=1e-12,rtol=1e-12)

    def test_independent_paired_quantile_interpolation(self):
        indices=torch.arange(256*4).reshape(256,4)%4
        indices[1::2]=torch.tensor([0,0,1,2]);indices[2::3]=torch.tensor([2,3,3,3])
        metrics={name:{'endpoint':torch.tensor(values,dtype=torch.float64)} for name,values in
            [('A',[.1,.2,.3,.4]),('ordinary',[.05,.1,.15,.2]),('analytic',[.4,.3,.2,.1])]}
        result=verifier.bootstrap_reference(metrics,indices)
        self.assertEqual(result['draws'],256)
        for name in verifier.SOURCES:
            value=metrics[name]['endpoint'];expected=torch.quantile(value[indices].mean(-1),torch.tensor([.025,.975],dtype=torch.float64))
            torch.testing.assert_close(torch.tensor(result[name+'/endpoint']['interval'],dtype=torch.float64),expected,atol=1e-12,rtol=1e-12)
        difference=metrics['A']['endpoint']-metrics['ordinary']['endpoint']
        expected=torch.quantile(difference[indices].mean(-1),torch.tensor([.025,.975],dtype=torch.float64))
        torch.testing.assert_close(torch.tensor(result['A_minus_ordinary/endpoint']['interval'],dtype=torch.float64),expected,atol=1e-12,rtol=1e-12)

    def test_false_and_missing_committed_binding_fail_closed_without_git(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);file=root/'fixture.py';file.write_bytes(b'fixture source\n')
            expected=hashlib.sha256(file.read_bytes()).hexdigest();audit=verifier.Audit()
            with patch.object(verifier,'ROOT',root),patch.object(verifier.subprocess,'Popen',return_value=FakeGitProcess(b'different bytes')) as fake:
                verifier.committed_bindings(audit,root,'a'*40,{'fixture.py':expected},'fixture_binding')
            fake.assert_called_once();self.assertEqual(audit.errors,['fixture_binding/fixture.py'])
            self.assertFalse(audit.checks[0]['passed']);self.assertEqual(audit.checks[0]['actual_sha256'],expected)
            missing=verifier.Audit()
            with patch.object(verifier,'ROOT',root),patch.object(verifier.subprocess,'Popen',return_value=FakeGitProcess(file.read_bytes())):
                verifier.committed_bindings(missing,root,'a'*40,{'missing.py':expected},'missing_binding')
            self.assertEqual(missing.errors,['missing_binding/missing.py']);self.assertIsNone(missing.checks[0]['actual_sha256'])
            with self.assertRaisesRegex(ValueError,'full lowercase hex'):
                verifier.committed_bindings(verifier.Audit(),root,'short',{},'bad_commit')

    def test_exact_comparison_does_not_absorb_discrete_failure_or_wrong_dtype(self):
        audit=verifier.Audit()
        audit.compare('fixture_cells',torch.tensor([0,1]),torch.tensor([0,2]),exact=True)
        audit.compare('fixture_dtype',torch.tensor([True]),torch.tensor([1]),exact=True)
        audit.check('fixture_gate',False,expected='named invariant retained')
        self.assertEqual(audit.errors,['fixture_cells','fixture_dtype','fixture_gate'])
        self.assertEqual(audit.checks[0]['unequal_elements'],1)
        self.assertEqual(audit.checks[0]['atol'],0.)

    def test_missing_registration_retains_first_failure_and_exclusive_report(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);report=root/'fixture-report.json';missing=root/'absent-registration.json'
            argv=['fixture_verifier','--phase','fit','--registration',str(missing),'--registration-commit','a'*40,
                '--phase-commit','b'*40,'--preservation-checkout',str(root),'--report',str(report),'--ack-first-phase-preserved']
            with patch.object(sys,'argv',argv),patch.object(verifier,'native_acceptance') as native,patch.object(torch,'load') as loader,redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    verifier.main()
                self.assertEqual(stopped.exception.code,1);native.assert_not_called();loader.assert_not_called()
            first=report.read_bytes();value=json.loads(first)
            self.assertFalse(value['hard_invariants_pass']);self.assertEqual(value['status'],'failed_first_phase_reconstruction')
            self.assertIn('verification_exception',value['errors']);self.assertIn('FileNotFoundError',value['exception'])
            with patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()):
                with self.assertRaises(FileExistsError):
                    verifier.main()
            self.assertEqual(report.read_bytes(),first)

    def test_nonfinite_primitive_evidence_retains_original_failure_and_all_checks(self):
        # Synthetic CLI failure only: no registration, artifacts or numerical run.
        def fail_with_nonfinite(args,audit):
            audit.check('fixture_bad_count',False,actual=float('nan'),expected=300)
            audit.check('fixture_bad_bound',False,actual=float('inf'),expected=1)
            audit.check('fixture_finite_record',True,actual=7,expected=7)
            raise ValueError('fixture original verification failure')
        with TemporaryDirectory() as directory:
            root=Path(directory);report=root/'fixture-nonfinite.json'
            argv=['fixture_verifier','--phase','fit','--registration',str(root/'unused.json'),'--registration-commit','a'*40,
                '--phase-commit','b'*40,'--preservation-checkout',str(root),'--report',str(report),'--ack-first-phase-preserved']
            with patch.object(sys,'argv',argv),patch.object(verifier,'run',side_effect=fail_with_nonfinite),patch.object(torch,'load') as loader,redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    verifier.main()
                self.assertEqual(stopped.exception.code,1);loader.assert_not_called()
            raw=report.read_bytes();value=json.loads(raw)
            self.assertEqual(value['status'],'failed_first_phase_reconstruction');self.assertFalse(value['hard_invariants_pass'])
            self.assertEqual(value['check_count'],3);self.assertEqual(len(value['checks']),3)
            self.assertEqual([row['name'] for row in value['checks']],['fixture_bad_count','fixture_bad_bound','fixture_finite_record'])
            self.assertEqual([row['passed'] for row in value['checks']],[False,False,True])
            self.assertEqual(value['checks'][0]['actual'],{'type':'float','value':'nan','finite':False})
            self.assertEqual(value['checks'][1]['actual'],{'type':'float','value':'inf','finite':False})
            self.assertIn('fixture original verification failure',value['exception'])
            self.assertIn('ValueError',value['serialization_error'])
            self.assertIn('report_serialization_failure',value['errors'])
            json.dumps(value,allow_nan=False)
            with patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()):
                with self.assertRaises(FileExistsError):
                    verifier.main()
            self.assertEqual(report.read_bytes(),raw)

    def test_optimizer_step_fractional_deviation_rejected_without_relaxing_numerical_tolerance(self):
        self.assertTrue(verifier.optimizer_step_is_exact_300(torch.tensor(300.)))
        self.assertTrue(verifier.optimizer_step_is_exact_300(300))
        fractional=torch.tensor(300.0001)
        ordinary_numeric=verifier.Audit();ordinary_numeric.compare('fixture_continuous_numeric',fractional,300.)
        self.assertTrue(ordinary_numeric.checks[0]['passed'])
        self.assertFalse(verifier.optimizer_step_is_exact_300(fractional))
        for value in (299,301,float('nan'),float('inf'),'300',torch.tensor([300.,300.])):
            self.assertFalse(verifier.optimizer_step_is_exact_300(value))
        self.assertEqual((verifier.ATOL,verifier.RTOL),(1e-6,1e-5))

    def test_writer_fault_preserves_partial_first_path_and_separate_full_evidence(self):
        class FaultingStream:
            def __init__(self,raw):
                self.raw=raw
            def write(self,text):
                self.raw.write(text[:20]);self.raw.flush()
                raise OSError('fixture first writer fault')
            def flush(self):
                self.raw.flush()
            def close(self):
                self.raw.close()
        with TemporaryDirectory() as directory:
            path=Path(directory)/'fixture-partial.json';audit=verifier.Audit()
            audit.check('fixture_original_failed_gate',False,actual=2,expected=1)
            report=dict(status='failed_first_phase_reconstruction',phase='fit',hard_invariants_pass=False,
                check_count=1,error_count=1,checks=audit.checks,errors=audit.errors,exception='fixture original exception')
            with path.open('x') as raw:
                result,written=verifier.write_first_report(FaultingStream(raw),path,report,audit)
            self.assertFalse(written);self.assertFalse(result['hard_invariants_pass'])
            partial=path.read_bytes();self.assertEqual(len(partial),20)
            sidecar=path.with_name(path.name+'.writer_failure.json');value=json.loads(sidecar.read_text())
            self.assertEqual(value['status'],'failed_first_report_retention')
            self.assertIn('fixture first writer fault',value['original_writer_error'])
            self.assertEqual(value['original_verification_exception'],'fixture original exception')
            self.assertEqual(value['first_report']['checks'][0]['name'],'fixture_original_failed_gate')
            self.assertIn('report_writer_failure',value['first_report']['errors'])
            self.assertFalse(value['retry_or_primary_overwrite_performed'])
            self.assertEqual(path.read_bytes(),partial)


if __name__=='__main__':
    unittest.main()
