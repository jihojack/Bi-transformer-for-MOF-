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

    def status(self):
        return {'ready':self.session is not None,'loading':self.loading,'error':'서버 모델을 준비하지 못했습니다. 관리자에게 문의하세요.' if self.failed else None,
                'execution':'server_cpu','device':'Server CPU','max_atoms':self.max_atoms,'weights_sent_to_browser':False}

    def load(self):
        with self.lock:
            if self.session is not None:return
            self.loading=True
            try:
                if not self.path.is_file():raise RuntimeError('Private model file not configured')
                if not self.expected_hash or file_hash(self.path)!=self.expected_hash:raise RuntimeError('Private model SHA-256 mismatch')
                options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.log_severity_level=3
                self.session=ort.InferenceSession(str(self.path),sess_options=options,providers=['CPUExecutionProvider'])
                self.failed=False
            except Exception:
                self.failed=True;raise
            finally:self.loading=False

    def predict(self,path,descriptors):
        if self.session is None:self.load()
        start=time.perf_counter();tensors,structure=prepare(path,descriptors,self.max_atoms);graph_seconds=time.perf_counter()-start
        tick=time.perf_counter();values=self.session.run(None,{i.name:tensors[i.name] for i in self.session.get_inputs()})[0].reshape(-1)
        if values.shape!=(10,) or not np.isfinite(values).all():raise RuntimeError('Invalid network output')
        if any(np.any(np.diff(values[a:a+5]) < -1e-6) for a in [0,5]):raise RuntimeError('Monotonicity check failed')
        predictions={gas:[{'pressure_bar':p,'uptake_mmol_g':float(values[offset+i])} for i,p in enumerate(pressures)] for gas,offset,pressures in [('CO2',0,PRESSURES['CO2']),('CH4',5,PRESSURES['CH4'])]}
        return {'predictions':predictions,'structure':structure,'timing':{'graph_seconds':graph_seconds,'inference_seconds':time.perf_counter()-tick,'total_seconds':time.perf_counter()-start}}
