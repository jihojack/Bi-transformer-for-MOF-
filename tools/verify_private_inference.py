"""Private-only numerical verification. Requires an externally supplied model."""
from pathlib import Path
import io
import json
import os
import sys
os.environ['MOF_SKIP_WARM']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from server.app import create_app
from server.runtime import Predictor


def main():
    runtime=Predictor();runtime.load()
    client=create_app(runtime,warm=False).test_client()
    cases=json.loads((ROOT/'tools/fixtures/manual_descriptor_cases.json').read_text())['cases']
    catalog={e['mof_id']:e for e in json.loads((ROOT/'app/examples/catalog.json').read_text())['examples']}
    differences=[]
    for c in cases:
        response=client.post('/api/predict/example',json={'mof_id':c['mof_id'],'descriptors':c['descriptors']})
        assert response.status_code==200,response.get_json()
        payload=response.get_json();actual=payload['values']
        error=max(abs(a-b) for a,b in zip(actual,c['expected']))
        assert error<1e-4,(c['mof_id'],c['case'],error)
        assert payload['comparable']==(c['case']=='original')
        differences.append({'mof_id':c['mof_id'],'case':c['case'],'max_absolute_difference_mmol_g':error})
    e=next(iter(catalog.values()))
    data={k:str(v) for k,v in e['descriptors'].items()};data['cif']=(io.BytesIO((ROOT/'app/examples'/e['cif']).read_bytes()),'new_upload.cif')
    result=client.post('/api/predict',data=data,content_type='multipart/form-data')
    assert result.status_code==200,result.get_json()
    upload_diff=max(abs(a-b) for a,b in zip(result.get_json()['values'],e['reference_output']))
    assert upload_diff<1e-4
    assert all(client.get(route).status_code==404 for route in ['/model/mof_fusion.onnx','/app/assets/best_dual_gas_alchemi_50k.pt','/private/model.onnx'])
    report={'private_model_required':True,'cases':len(differences),'upload_case':1,'max_absolute_difference_mmol_g':max(d['max_absolute_difference_mmol_g'] for d in differences),
            'reference':'Original frozen PyTorch inference and original sklearn preprocessing','tolerance_mmol_g':1e-4,'private_model_sha256':runtime.expected_hash,'per_case':differences}
    output=ROOT/'docs/server_parity_validation.json';output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('PASS: 35 actual server predictions, direct CIF upload, GCMC comparison eligibility and private route blocking.')
    print('Max difference mmol/g:',report['max_absolute_difference_mmol_g'])


if __name__=='__main__':main()
