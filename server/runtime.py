"""Private ONNX model loading; no model-download HTTP routes."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import threading
import time

import numpy as np
import onnxruntime as ort
from .preprocessing import prepare
from .diagnostics import emit

BASE=Path(__file__).resolve().parent
MANIFEST=json.loads((BASE/'model_manifest.json').read_text(encoding='utf-8'))
PRESSURES={'CO2':[.01,.05,.1,.5,2.5],'CH4':[.05,.5,.9,2.5,4.5]}


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class Predictor:
    def __init__(self,path=None,expected_hash=None,max_atoms=None):
        self.path=Path(path or os.getenv('MOF_MODEL_PATH',str(Path(tempfile.gettempdir())/'mof-private/model.onnx'))).resolve()
        self.expected_hash=expected_hash or os.getenv('MOF_MODEL_SHA256') or MANIFEST['onnx']['sha256']
        self.max_atoms=max_atoms or int(os.getenv('MOF_MAX_ATOMS','200'))
        self.session=None;self.loading=False;self.failed=False;self.lock=threading.Lock()
        self.stage='not_started';self.error_code=None;self.load_started_at=None;self.load_finished_at=None;self.load_pid=None
        self.stage_started_at=time.monotonic()

    def elapsed(self):
        if self.load_started_at is None:return 0.
        return round((self.load_finished_at or time.monotonic())-self.load_started_at,3)

    def mark_stage(self,stage):
        self.stage=stage;self.stage_started_at=time.monotonic()
        emit('model_stage',stage=stage,elapsed_seconds=self.elapsed())

    def status(self):
        wrong_process=self.load_pid is not None and self.load_pid!=os.getpid()
        return {'ready':self.session is not None and not wrong_process,'loading':self.loading and not wrong_process,'error':'서버 모델을 준비하지 못했습니다. 관리자에게 문의하세요.' if self.failed or wrong_process else None,
                'execution':'server_cpu','device':'Server CPU','max_atoms':self.max_atoms,'weights_sent_to_browser':False,
                'model_stage':self.stage,'error_code':'model_process_mismatch' if wrong_process else self.error_code,'load_elapsed_seconds':self.elapsed(),
                'stage_elapsed_seconds':round(time.monotonic()-self.stage_started_at,3),'process_id':os.getpid(),'loader_process_id':self.load_pid}

    def load(self):
        if self.load_pid is not None and self.load_pid!=os.getpid():
            raise RuntimeError('Recreate predictor in the worker before loading; inherited locks/sessions are unsafe')
        with self.lock:
            if self.session is not None:return
            self.loading=True;self.failed=False;self.error_code=None
            self.load_started_at=time.monotonic();self.load_finished_at=None;self.load_pid=os.getpid()
            stop=threading.Event()
            def progress():
                while not stop.wait(15):
                    emit('model_loading_progress',stage=self.stage,elapsed_seconds=self.elapsed(),
                         stage_elapsed_seconds=round(time.monotonic()-self.stage_started_at,3))
            monitor=threading.Thread(target=progress,name='mof-startup-progress',daemon=True);monitor.start()
            try:
                self.mark_stage('checking_model_file')
                if not self.path.is_file():
                    self.error_code='model_file_missing';raise RuntimeError('Private model file not configured')
                self.mark_stage('verifying_model_hash')
                if not self.expected_hash or file_hash(self.path)!=self.expected_hash:
                    self.error_code='model_hash_mismatch';raise RuntimeError('Private model SHA-256 mismatch')
                emit('model_hash_verified',bytes=self.path.stat().st_size,elapsed_seconds=self.elapsed())
                options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.log_severity_level=3
                self.mark_stage('creating_onnx_session')
                emit('onnx_session_options',runtime_version=ort.__version__,intra_threads=1,inter_threads=1,provider='CPUExecutionProvider')
                self.session=ort.InferenceSession(str(self.path),sess_options=options,providers=['CPUExecutionProvider'])
                self.failed=False;self.load_finished_at=time.monotonic();self.mark_stage('ready')
            except Exception as exc:
                failure_stage=self.stage
                self.error_code=self.error_code or ('model_out_of_memory' if isinstance(exc,MemoryError) else 'onnx_session_init_failed' if failure_stage=='creating_onnx_session' else 'model_init_failed')
                self.failed=True;self.load_finished_at=time.monotonic();self.stage='failed';self.stage_started_at=time.monotonic()
                emit('model_load_failed',stage=failure_stage,error_code=self.error_code,error_type=type(exc).__name__,elapsed_seconds=self.elapsed())
                raise
            finally:
                self.loading=False;stop.set()

    def predict(self,path,descriptors):
        if self.session is None or self.load_pid!=os.getpid():self.load()
        start=time.perf_counter();tensors,structure=prepare(path,descriptors,self.max_atoms);graph_seconds=time.perf_counter()-start
        tick=time.perf_counter();values=self.session.run(None,{i.name:tensors[i.name] for i in self.session.get_inputs()})[0].reshape(-1)
        if values.shape!=(10,) or not np.isfinite(values).all():raise RuntimeError('Invalid network output')
        if any(np.any(np.diff(values[a:a+5]) < -1e-6) for a in [0,5]):raise RuntimeError('Monotonicity check failed')
        predictions={gas:[{'pressure_bar':p,'uptake_mmol_g':float(values[offset+i])} for i,p in enumerate(pressures)] for gas,offset,pressures in [('CO2',0,PRESSURES['CO2']),('CH4',5,PRESSURES['CH4'])]}
        return {'predictions':predictions,'structure':structure,'timing':{'graph_seconds':graph_seconds,'inference_seconds':time.perf_counter()-tick,'total_seconds':time.perf_counter()-start}}
