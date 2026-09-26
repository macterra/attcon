import importlib.util,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reproduce',ROOT/'scripts/reproduce_final.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ReproductionPlanTests(unittest.TestCase):
    def test_plan_covers_every_registered_cell_and_correction(self):
        stages=dict(module.command_plan())
        self.assertEqual(len(stages['fitted']),12)
        self.assertEqual(len(stages['exploration']),6)
        self.assertEqual(len(stages['reporting']),12)
        self.assertEqual(len(stages['identity_correction']),12)
        self.assertEqual(len(stages['stress']),12)
        self.assertIn('--replay',stages['final_audit'][0])
        outputs=[args[args.index('--out')+1] for jobs in stages.values() for args in jobs if '--out' in args]
        self.assertEqual(len(outputs),len(set(outputs)))
if __name__=='__main__':unittest.main()
