"""Public CI tests use an explicitly fake network, never claim model accuracy."""
from pathlib import Path
import io
import json
import os
import unittest

os.environ['MOF_SKIP_WARM']='1'
from server.app import create_app
from server.preprocessing import prepare

ROOT=Path(__file__).resolve().parents[1]
CATALOG=json.loads((ROOT/'app/examples/catalog.json').read_text(encoding='utf-8'))['examples']


class FakePredictor:
    max_atoms=200
    def __init__(self):self.paths=[]
    def status(self):return {'ready':True,'loading':False,'error':None,'max_atoms':200,'execution':'server_cpu','weights_sent_to_browser':False}
    def predict(self,path,descriptors):
        self.paths.append(path)
        assert path.is_file()
        return {'predictions':{'CO2':[{'pressure_bar':p,'uptake_mmol_g':float(i+1)} for i,p in enumerate([.01,.05,.1,.5,2.5])],
                              'CH4':[{'pressure_bar':p,'uptake_mmol_g':float(i+1)} for i,p in enumerate([.05,.5,.9,2.5,4.5])]},
                'structure':{'atoms':3,'periodic_edges':6,'elements':['C','O']},'timing':{'graph_seconds':0.,'inference_seconds':0.,'total_seconds':0.}}


class APIContractTests(unittest.TestCase):
    def setUp(self):
        self.fake=FakePredictor();self.client=create_app(self.fake,warm=False).test_client();self.e=CATALOG[0]

    def test_catalog_and_cif(self):
        entries=self.client.get('/api/examples').get_json()['examples'];self.assertEqual(len(entries),18)
        self.assertTrue({'hmof-5060237','hmof-5036458'}.isdisjoint(e['mof_id'] for e in entries))
        self.assertNotIn('reference_output',entries[0]);self.assertNotIn('graph',entries[0])
        with self.client.get('/api/examples/'+self.e['mof_id']+'/cif') as response:self.assertEqual(response.status_code,200)
        self.assertEqual(self.client.get('/api/examples/not-real/cif').status_code,404)

    def test_no_weight_or_internal_routes(self):
        for route in ['/model/mof_fusion.onnx','/private/model.onnx','/app/assets/best_dual_gas_alchemi_50k.pt','/server/model_manifest.json','/static/../private/model.onnx','/.env']:
            self.assertEqual(self.client.get(route).status_code,404,route)
        with self.client.get('/') as response:self.assertIn('서버 추론',response.data.decode())
        with self.client.get('/app.js') as response:self.assertNotIn('ort.wasm',response.data.decode())

    def test_removed_demo_entries_are_not_available_in_app(self):
        for mof_id in ['hmof-5060237','hmof-5036458']:
            self.assertEqual(self.client.get('/api/examples/'+mof_id+'/cif').status_code,404)
            self.assertEqual(self.client.post('/api/predict/example',json={'mof_id':mof_id}).status_code,400)
        self.assertFalse(self.fake.paths)

    def test_removed_demo_structure_keeps_research_reference_on_upload(self):
        e=CATALOG[11]
        data={k:str(v) for k,v in e['descriptors'].items()}
        data['cif']=(io.BytesIO((ROOT/'app/examples'/e['cif']).read_bytes()),e['mof_id']+'.cif')
        response=self.client.post('/api/predict',data=data,content_type='multipart/form-data')
        self.assertEqual(response.status_code,200)
        self.assertTrue(response.get_json()['comparable'])
        self.assertEqual(len(response.get_json()['reference']),10)

    def test_manual_status_button_is_removed(self):
        with self.client.get('/') as response:
            self.assertNotIn('id="check-server"',response.data.decode())

    def test_example_reference(self):
        response=self.client.post('/api/predict/example',json={'mof_id':self.e['mof_id']});self.assertEqual(response.status_code,200)
        result=response.get_json();self.assertTrue(result['comparable']);self.assertEqual(len(result['reference']),10)
        self.assertFalse(result['weights_sent_to_browser'])
        self.assertNotIn('model_path',result)

    def test_modified_reference_not_truth(self):
        d={**self.e['descriptors'],'surface_area_m2g':4000}
        result=self.client.post('/api/predict/example',json={'mof_id':self.e['mof_id'],'descriptors':d}).get_json()
        self.assertFalse(result['comparable']);self.assertTrue(result['modified'])

    def test_upload_cleanup_and_hash_reference(self):
        d={k:str(v) for k,v in self.e['descriptors'].items()};d['cif']=(io.BytesIO((ROOT/'app/examples'/self.e['cif']).read_bytes()),'uploaded.cif')
        r=self.client.post('/api/predict',data=d,content_type='multipart/form-data')
        self.assertEqual(r.status_code,200);self.assertTrue(r.get_json()['comparable']);self.assertFalse(self.fake.paths[-1].exists())

    def test_invalid_descriptors(self):
        for changed in [{'void_fraction':0},{'void_fraction':1.1},{'lcd':-1},{'pld':1000},{'surface_area_m2g':'nan'},{'surface_area_m2g':0}]:
            r=self.client.post('/api/predict/example',json={'mof_id':self.e['mof_id'],'descriptors':{**self.e['descriptors'],**changed}})
            self.assertEqual(r.status_code,400)

    def test_wrong_file(self):
        d={k:str(v) for k,v in self.e['descriptors'].items()};d['cif']=(io.BytesIO(b'bad'),'test.txt')
        self.assertEqual(self.client.post('/api/predict',data=d).status_code,400)

    def test_cors_denied(self):
        self.assertEqual(self.client.post('/api/predict/example',json={'mof_id':self.e['mof_id']},headers={'Origin':'https://unapproved.invalid'}).status_code,403)

    def test_malformed_json(self):
        for payload in [[],[1],{'mof_id':[]},None]:
            self.assertEqual(self.client.post('/api/predict/example',json=payload).status_code,400)


class CIFPreprocessingTests(unittest.TestCase):
    def test_real_20_preprocessing(self):
        for e in CATALOG:
            with self.subTest(mof=e['mof_id']):
                inputs,structure=prepare(ROOT/'app/examples'/e['cif'],e['descriptors'],200)
                self.assertEqual(structure['atoms'],e['atoms']);self.assertEqual(structure['periodic_edges'],e['periodic_edges'])
                self.assertEqual(inputs['descriptor'].shape,(1,30));self.assertEqual(inputs['x_cont'].shape[1],13)

    def test_atom_limit(self):
        with self.assertRaises(ValueError):prepare(ROOT/'app/examples'/CATALOG[0]['cif'],CATALOG[0]['descriptors'],10)


if __name__=='__main__':unittest.main()
