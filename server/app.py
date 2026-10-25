"""Flask API and optional same-origin UI. Only predictions leave the server."""
from collections import defaultdict,deque
from pathlib import Path
import csv
import hashlib
import io
import json
import logging
import math
import os
import tempfile
import threading
import time

from flask import Flask,abort,jsonify,request,send_file
from werkzeug.exceptions import RequestEntityTooLarge
from .runtime import Predictor,PRESSURES

ROOT=Path(__file__).resolve().parents[1]
KEYS=['surface_area_m2g','void_fraction','lcd','pld']


def validate_descriptors(data):
    try:d={k:float(data[k]) for k in KEYS}
    except (KeyError,ValueError,TypeError):raise ValueError('네 가지 기술자를 모두 숫자로 입력하세요.')
    if not all(math.isfinite(v) for v in d.values()):raise ValueError('유한한 수치만 입력하세요.')
    if d['surface_area_m2g']<=0 or not 0<d['void_fraction']<=1 or d['pld']<=0 or d['lcd']<=0 or d['pld']>d['lcd']:
        raise ValueError('표면적·LCD·PLD > 0, 공극률 0 초과 1 이하, PLD ≤ LCD 조건을 확인하세요.')
    return d


def create_app(predictor=None,warm=True):
    app=Flask(__name__,static_folder=None)
    app.config['MAX_CONTENT_LENGTH']=2_100_000
    runtime=predictor or Predictor()
    examples=json.loads((ROOT/'app/examples/catalog.json').read_text(encoding='utf-8'))['examples']
    lookup={e['mof_id']:e for e in examples}
    labels=json.loads((ROOT/'app/examples/reference-gcmc.json').read_text(encoding='utf-8'))['values']
    allowed={s.strip().rstrip('/') for s in os.getenv('MOF_ALLOWED_ORIGINS','').split(',') if s.strip()}
    gate=threading.BoundedSemaphore(1);rate_lock=threading.Lock();rate=defaultdict(deque)
    limit=int(os.getenv('MOF_REQUESTS_PER_MINUTE','60'))
    app.extensions['mof_predictor']=runtime

    @app.before_request
    def guard():
        origin=request.headers.get('Origin','').rstrip('/')
        same=origin==request.host_url.rstrip('/')
        if request.path.startswith('/api/') and origin and not same and origin not in allowed:return jsonify(error='허용되지 않은 웹 출처입니다.'),403
        if request.method=='OPTIONS':return '',204
        if request.method=='POST' and request.path.startswith('/api/predict'):
            now=time.monotonic();key=request.remote_addr or 'unknown'
            with rate_lock:
                # Bound both per-IP history and unique-address memory.
                if len(rate)>4096:
                    for address in list(rate):
                        if not rate[address] or now-rate[address][-1]>60:del rate[address]
                    if len(rate)>4096:return jsonify(error='요청이 많습니다. 잠시 후 다시 시도하세요.'),429
                q=rate[key]
                while q and now-q[0]>60:q.popleft()
                if len(q)>=limit:return jsonify(error='분당 요청 한도를 초과했습니다.'),429
                q.append(now)

    @app.after_request
    def headers(response):
        origin=request.headers.get('Origin','').rstrip('/')
        if origin in allowed:
            response.headers['Access-Control-Allow-Origin']=origin;response.headers['Vary']='Origin'
            response.headers['Access-Control-Allow-Methods']='GET, POST, OPTIONS';response.headers['Access-Control-Allow-Headers']='Content-Type'
        response.headers['Cache-Control']='no-store';response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self' https:; img-src 'self' data:; object-src 'none'; base-uri 'self'; frame-ancestors 'self'"
        return response

    @app.get('/')
    def index():return send_file(ROOT/'frontend/index.html')

    @app.get('/<filename>')
    def frontend_file(filename):
        if filename not in {'app.js','styles.css','config.js'}:abort(404)
        return send_file(ROOT/'frontend'/filename)

    @app.get('/healthz')
    def health():return jsonify(ready=runtime.status()['ready']),200 if runtime.status()['ready'] else 503

    @app.get('/api/status')
    def status():return jsonify(runtime.status())

    @app.get('/api/examples')
    def example_list():
        return jsonify(examples=[{k:e[k] for k in ['mof_id','descriptors','atoms','periodic_edges','elements','cif_sha256']} for e in examples])

    @app.get('/api/examples/<mof_id>/cif')
    def cif_example(mof_id):
        if mof_id not in lookup:abort(404)
        return send_file(ROOT/'app/examples'/lookup[mof_id]['cif'],as_attachment=True,download_name=mof_id+'.cif')

    def run_prediction(path,descriptors,filename,known=None):
        if not gate.acquire(blocking=False):return jsonify(error='서버가 다른 예측을 처리 중입니다. 잠시 후 재시도하세요.'),429,{'Retry-After':'2'}
        try:
            result=runtime.predict(path,descriptors)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            matched=known or next((e for e in examples if e['cif_sha256']==digest),None)
            modified=bool(matched) and descriptors!=matched['descriptors']
            values=[r['uptake_mmol_g'] for g in ['CO2','CH4'] for r in result['predictions'][g]]
            reference=labels[matched['mof_id']] if matched else None
            result.update(filename=filename,mof_key=matched['mof_id'] if matched else 'upload-'+digest[:12],descriptors=descriptors,cif_sha256=digest,
                          values=values,reference=reference,reference_kind='GCMC simulation; not experimental measurement' if reference else None,
                          comparable=bool(reference) and not modified,modified=modified,execution='server_cpu',weights_sent_to_browser=False)
            return jsonify(result)
        except ValueError as e:return jsonify(error=str(e)),422
        except Exception:
            app.logger.error('Prediction failed; request data and private paths are not logged')
            return jsonify(error='서버 예측을 완료하지 못했습니다. 모델 준비 상태와 구조 지원 범위를 확인하세요.'),503
        finally:gate.release()

    @app.post('/api/predict/example')
    def predict_example():
        data=request.get_json(silent=True)
        if not isinstance(data,dict) or not isinstance(data.get('mof_id'),str):return jsonify(error='mof_id가 있는 JSON 객체를 보내세요.'),400
        e=lookup.get(data['mof_id'])
        if e is None:return jsonify(error='알 수 없는 예제입니다.'),400
        try:d=validate_descriptors(data.get('descriptors',e['descriptors']))
        except ValueError as exc:return jsonify(error=str(exc)),400
        return run_prediction(ROOT/'app/examples'/e['cif'],d,e['mof_id']+'.cif',e)

    @app.post('/api/predict')
    def predict_upload():
        f=request.files.get('cif')
        if f is None or not f.filename or not f.filename.lower().endswith('.cif'):return jsonify(error='.cif 파일을 선택하세요.'),400
        try:d=validate_descriptors(request.form)
        except ValueError as exc:return jsonify(error=str(exc)),400
        content=f.read(2_000_001)
        if not 0<len(content)<=2_000_000:return jsonify(error='CIF는 0바이트 초과 2 MB 이하만 지원합니다.'),400
        if b'\0' in content:return jsonify(error='텍스트 CIF 파일이 아닙니다.'),400
        with tempfile.TemporaryDirectory(prefix='mof-request-') as directory:
            path=Path(directory)/'input.cif';path.write_bytes(content)
            return run_prediction(path,d,Path(f.filename.replace('\\','/')).name[:160])

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):return jsonify(error='요청 파일이 너무 큽니다. CIF는 2 MB 이하를 사용하세요.'),413

    if warm:
        def start_model():
            try:runtime.load()
            except Exception:logging.getLogger(__name__).error('Private model unavailable; configure server model path and SHA-256')
        threading.Thread(target=start_model,daemon=True).start()
    return app


app=create_app(warm=os.getenv('MOF_SKIP_WARM')!='1')
if __name__=='__main__':app.run(host=os.getenv('MOF_BIND','127.0.0.1'),port=int(os.getenv('PORT','7866')),debug=False)
