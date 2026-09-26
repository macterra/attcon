import importlib.util,sys,unittest,io,tarfile,tempfile,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('final_audit',ROOT/'scripts/final_project_audit.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class FinalAuditTests(unittest.TestCase):
    def test_matrix_has_every_registered_cell(self):
        self.assertEqual(len(module.paths_for('fitted')),12)
        self.assertEqual(len(module.paths_for('exploration')),6)
        self.assertEqual(len(module.paths_for('identity')),12)
        self.assertEqual(len(set(module.paths_for('stress'))),12)

    def test_metric_replay_detects_drift(self):
        module.assert_close({'x':[.1,True,None]},{'x':[.1,True,None]})
        with self.assertRaises(ValueError):module.assert_close({'x':.11},{'x':.1})

    def archive(self,path,name,payload=b'checkpoint'):
        with tarfile.open(path,'w:gz') as archive:
            info=tarfile.TarInfo(name);info.size=len(payload);archive.addfile(info,io.BytesIO(payload))

    def test_archive_payloads_require_exact_safe_members_and_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'archive.tar.gz';name='outputs/prospective/test.pt'
            self.archive(path,name)
            expected={name:hashlib.sha256(b'checkpoint').hexdigest()}
            self.assertEqual(module.safe_archive_payloads(path,expected)[name],b'checkpoint')
            with self.assertRaises(ValueError):module.safe_archive_payloads(path,{name:'wrong'})
            unsafe='../outside.pt';self.archive(path,unsafe)
            with self.assertRaisesRegex(ValueError,'unsafe'):module.safe_archive_payloads(path,{unsafe:expected[name]})
if __name__=='__main__':unittest.main()
