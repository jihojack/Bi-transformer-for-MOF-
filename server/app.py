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
from .diagnostics import emit

ROOT=Path(__file__).resolve().parents[1]
KEYS=['surface_area_m2g','void_fraction','lcd','pld']


def validate_descriptors(data):
    try:d={k:float(data[k]) for k in KEYS}
    except (KeyError,ValueError,TypeError):raise ValueError('네 가지 기술자를 모두 숫자로 입력하세요.')
    if not all(math.isfinite(v) for v in d.values()):raise ValueError('유한한 수치만 입력하세요.')
    if d['surface_area_m2g']<=0 or not 0<d['void_fraction']<=1 or d['pld']<=0 or d['lcd']<=0 or d['pld']>d['lcd']:
        raise ValueError('표면적·LCD·PLD > 0, 공극률 0 초과 1 이하, PLD ≤ LCD 조건을 확인하세요.')
    return d


def create_app(predictor=None,warm=True,defer_warm=False):
    app=Flask(__name__,static_folder=None)
    app.config['MAX_CONTENT_LENGTH']=2_100_000
    runtime=predictor or Predictor()
    catalog=json.loads((ROOT/'app/examples/catalog.json').read_text(encoding='utf-8'))['examples']
    exclusions=set(json.loads((ROOT/'app/examples/demo_selection.json').read_text(encoding='utf-8'))['excluded_mof_ids'])
    examples=[e for e in catalog if e['mof_id'] not in exclusions]
    lookup={e['mof_id']:e for e in examples}
    labels=json.loads((ROOT/'app/examples/reference-gcmc.json').read_text(encoding='utf-8'))['values']
    allowed={s.strip().rstrip('/') for s in os.getenv('MOF_ALLOWED_ORIGINS','').split(',') if s.strip()}
    gate=threading.BoundedSemaphore(1);rate_lock=threading.Lock();rate=defaultdict(deque)
    limit=int(os.getenv('MOF_REQUESTS_PER_MINUTE','60'))
    app.extensions['mof_predictor']=runtime
    app.extensions['mof_loader_thread']=None
    app.extensions['mof_warm_enabled']=warm
    app.extensions['mof_health_state']=None
    runtime_pid=os.getpid()
    start_lock=threading.Lock()
    emit('application_created',warm_enabled=warm,deferred_warm=defer_warm,max_atoms=runtime.max_atoms)

    def start_model_for_process():
        nonlocal runtime,runtime_pid,start_lock
        pid=os.getpid()
        # A preload/fork copies Python state but NOT a running loader thread.
        # Recreate both runtime and locks before touching any inherited lock.
        if runtime_pid!=pid:
            previous_pid=runtime_pid
            if not isinstance(runtime,Predictor):raise RuntimeError('Injected predictor cannot be inherited across processes')
            runtime=Predictor(path=runtime.path,expected_hash=runtime.expected_hash,max_atoms=runtime.max_atoms)
            runtime_pid=pid;start_lock=threading.Lock()
            app.extensions['mof_predictor']=runtime
            app.extensions['mof_loader_thread']=None
            app.extensions['mof_health_state']=None
            emit('worker_runtime_created',previous_process_id=previous_pid)
        if not warm:return
        with start_lock:
            loader=app.extensions['mof_loader_thread']
            state=runtime.status()
            if state['ready'] or state.get('error') or (loader and loader.is_alive()):return
            def start_model():
                emit('model_loader_thread_started')
                try:runtime.load()
                except Exception:logging.getLogger(__name__).error('Private model unavailable; configure server model path and SHA-256')
            loader=threading.Thread(target=start_model,name='mof-model-loader',daemon=True)
            app.extensions['mof_loader_thread']=loader;loader.start()
    app.extensions['mof_start_model']=start_model_for_process

    def runtime_status():
        snapshot=runtime.status()
        loader=app.extensions['mof_loader_thread']
        snapshot.update(warm_enabled=warm,loader_thread_alive=bool(loader and loader.is_alive()))
        return snapshot

    @app.before_request
    def guard():
        # Fallback when a host overrides Gunicorn's config/hooks. A request is
        # handled in the worker, never in the preloading master process.
        start_model_for_process()
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
    def health():
        snapshot=runtime_status()
        state=(snapshot['ready'],snapshot.get('model_stage'),snapshot.get('error_code'))
        if state!=app.extensions['mof_health_state']:
            emit('readiness_changed',ready=state[0],stage=state[1],error_code=state[2],
                 loader_thread_alive=snapshot['loader_thread_alive'],warm_enabled=warm,
                 loader_process_id=snapshot.get('loader_process_id'),load_elapsed_seconds=snapshot.get('load_elapsed_seconds'))
            app.extensions['mof_health_state']=state
        return jsonify({key:snapshot.get(key) for key in ['ready','loading','model_stage','error_code','load_elapsed_seconds','loader_thread_alive']}),200 if snapshot['ready'] else 503

    @app.get('/api/status')
    def status():return jsonify(runtime_status())

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
            # App curation does not remove original validation/reference data.
            matched=known or next((e for e in catalog if e['cif_sha256']==digest),None)
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

    if warm and not defer_warm:start_model_for_process()
    elif warm:emit('model_warm_start_deferred_to_worker')
    else:emit('model_warm_start_disabled')
    return app


# Importing this WSGI module must NEVER start an ONNX loader in a master process.
app=create_app(warm=os.getenv('MOF_SKIP_WARM')!='1',defer_warm=True)
if __name__=='__main__':
    app.extensions['mof_start_model']()
    app.run(host=os.getenv('MOF_BIND','127.0.0.1'),port=int(os.getenv('PORT','7866')),debug=False)
