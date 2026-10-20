"""Invented calculator checks; no exporter import, archive or output reads."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

PATH = Path(__file__).with_name('reconstruct.py')
SPEC = importlib.util.spec_from_file_location('independent_source_calculator', PATH)
CALC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALC)


class IndependentReconstructionTests(unittest.TestCase):
    def setUp(self):
        CALC.ERRORS.clear(); CALC.COUNTS.clear()

    def test_import_is_stdlib_and_root_resolves(self):
        code = 'import runpy,sys;runpy.run_path(' + repr(str(PATH)) + ");assert 'torch' not in sys.modules;assert 'same_address_model_report_source_v1' not in sys.modules"
        subprocess.run([sys.executable, '-c', code], check=True)
        self.assertTrue((CALC.ROOT / 'scripts/attention_content_model_v1.py').is_file())

    def test_complete_predetermined_inventory(self):
        rows,pairs=CALC.design()
        self.assertEqual(len(rows),72);self.assertEqual(len(pairs),84)
        self.assertEqual(len({r['id'] for r in rows}),72)
        self.assertEqual([sum(r['source']==s for r in rows) for s in ('A','ordinary','analytic','W')],[20,20,20,12])
        self.assertEqual({r['queried_role'] for r in rows if r['world_index']==1},{'blue'})
        self.assertEqual({r['candidate_node'] for r in rows},{'n3'})

    def test_raw_grid_location_and_score_ties(self):
        uniform=[[1/16]*4 for _ in range(4)]
        result=CALC.location(uniform)
        self.assertEqual(result['expected_row'],1.5);self.assertEqual(result['expected_column'],1.5)
        self.assertEqual(len(result['peaks']),16)
        score=CALC.score_values([uniform]*3)
        self.assertEqual(score['tied_winners'],['k0','k1','k2']);self.assertEqual(score['first_index_choice'],'k0')
        self.assertEqual(score['top_two_margin'],0)
        alternate=[[0.0]*4 for _ in range(4)];alternate[3][3]=1.0
        self.assertEqual(CALC.score_values([uniform,alternate,uniform])['first_index_choice'],'k1')

    def test_only_derived_roundoff_is_tolerated(self):
        CALC.compare({'derived':{'entropy_nats':1.0+1e-15,'peak_probability':.25}},
                     {'derived':{'entropy_nats':1.0,'peak_probability':.25}},'tiny roundoff')
        self.assertEqual(CALC.ERRORS,[])
        CALC.compare({'derived':{'peak_probability':.25+1e-15}},
                     {'derived':{'peak_probability':.25}},'raw maximum must be exact')
        self.assertEqual(len(CALC.ERRORS),1)

    def test_public_json_tuple_boundary_does_not_weaken_private_types(self):
        expected={'source_receipts':{'baseline_state':{'update_command_indices':(0,1,2)}}}
        observed={'source_receipts':{'baseline_state':{'update_command_indices':[0,1,2]}}}
        CALC.compare(observed,CALC.json_boundary(expected),'persisted public JSON')
        self.assertEqual(CALC.ERRORS,[])
        CALC.compare([0,1,2],(0,1,2),'private tuple/list must remain distinct')
        self.assertEqual(len(CALC.ERRORS),1)

    def test_float32_bytes_and_selected_private_rows_are_exact(self):
        import torch
        a=torch.zeros(96,4);b=a.clone();b[0,0]=-0.0
        self.assertNotEqual(CALC.thash(a),CALC.thash(b))
        CALC.compare(a,b,'signed zero');self.assertEqual(len(CALC.ERRORS),1)
        self.assertEqual(CALC.selected({'maps':a})['maps'].shape,(4,4))

    def test_heterogeneous_actual_symbol_current_identity(self):
        sys.path.insert(0,str(CALC.ROOT/'scripts'))
        import torch
        from attention_content_model_v1 import ObservationBatch,CommandBatch,SourceState,_assemble
        v=torch.zeros(96,4,24,4,4);w11=torch.ones_like(v)*11
        previous=torch.full((96,),3,dtype=torch.long);symbols=torch.full((96,),2,dtype=torch.long)
        symbols[:4]=torch.tensor([0,1,1,0]);candidate=torch.arange(96)%4;candidate[:4]=3
        obs=ObservationBatch(v,w11,12,11,CommandBatch(previous,symbols),torch.ones(96,4,dtype=torch.bool))
        state=SourceState(torch.zeros(96,4,48),11,tuple(range(11)),'A')
        def decoder(address,symbol):return v[torch.arange(96),address]+symbol[:,None,None,None].float()
        prediction=_assemble(obs,state,candidate,decoder,'A')
        self.assertTrue(torch.equal(prediction.current[:4,3],prediction.next[torch.arange(4),symbols[:4],3]))
        self.assertFalse(torch.equal(prediction.current[:4,3],prediction.next[:4,2,3]))
        self.assertEqual(CALC.origin('W',False,True)['fixed_command'],'k0')
        self.assertEqual(CALC.origin('W',False,True)['archived_origin_time'],13)


if __name__ == '__main__':unittest.main()
