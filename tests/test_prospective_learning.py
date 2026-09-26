from pathlib import Path
import sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from attcon.prospective import make_splits
from attcon.prospective_learning import train_fitted,evaluate

class ProspectiveLearningTests(unittest.TestCase):
    def test_fitted_learning_and_evaluation_both_tasks(self):
        torch.set_num_threads(1)
        for task in ('serial','routing'):
            splits=make_splits(3,task)
            fit=splits['train'].subset(torch.arange(108));val=splits['validation'].subset(torch.arange(108))
            models={};records={}
            for family in ('state','blind','cue'):
                models[family],records[family]=train_fitted(fit,val,3,'gru',family,epochs=2)
                self.assertNotEqual(records[family]['initial_sha256'],records[family]['selected_sha256'])
                self.assertEqual(records[family]['updates'],2)
            self.assertEqual(len({v['initial_sha256'] for v in records.values()}),1)
            result=evaluate(models,val,3)
            self.assertEqual(set(result['costs']),{'0.1','0.25','0.4'})
            self.assertTrue(all('gain_over_fair_cue' in v['gates'] for v in result['costs'].values()))
if __name__=='__main__':unittest.main()
