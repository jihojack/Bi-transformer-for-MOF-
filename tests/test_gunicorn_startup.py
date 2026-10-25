"""Linux CI: real Gunicorn startup, no private weights, no fake healthy response."""
import json
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform=='linux','Gunicorn deployment smoke test runs in Linux CI')
class GunicornStartupTests(unittest.TestCase):
    def run_synthetic_ready(self,preload,hook):
        # A tiny untrained Identity graph exercises the REAL Linux ONNX loader
        # and Gunicorn fork lifecycle. It is NOT a research accuracy test.
        import onnx
        from onnx import helper,TensorProto
        with tempfile.TemporaryDirectory(prefix='mof-prefork-test-') as directory:
            model=Path(directory)/'identity.onnx'
            graph=helper.make_graph([helper.make_node('Identity',['input'],['output'])],'startup-regression',
                [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1])],
                [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1])])
            proto=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)]);proto.ir_version=10
            onnx.save(proto,model)
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1',0));port=reservation.getsockname()[1]
            env=os.environ.copy();env.pop('MOF_SKIP_WARM',None)
            env['MOF_MODEL_PATH']=str(model);env['MOF_MODEL_SHA256']=hashlib.sha256(model.read_bytes()).hexdigest()
            env['PYTHONUNBUFFERED']='1'
            config=str(ROOT/'gunicorn.conf.py') if hook else '/dev/null'
            command=[sys.executable,'-m','gunicorn','--config',config,'--workers','1','--threads','2','--timeout','180',
                     '--bind',f'127.0.0.1:{port}','--capture-output','--error-logfile','-']
            if preload:command.append('--preload')
            command.append('server.app:app')
            log=Path(directory)/'startup.log'
            with log.open('wb') as output:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
                master_pid=process.pid
                try:
                    deadline=time.monotonic()+30;state=None
                    while time.monotonic()<deadline:
                        if process.poll() is not None:self.fail('Gunicorn exited during synthetic startup')
                        try:
                            with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/status',timeout=1) as response:state=json.load(response)
                            if state['ready']:break
                        except (OSError,ValueError):pass
                        time.sleep(.1)
                    self.assertIsNotNone(state);self.assertTrue(state['ready'],state)
                    self.assertEqual(state['model_stage'],'ready')
                    self.assertEqual(state['process_id'],state['loader_process_id'])
                    self.assertNotEqual(state['loader_process_id'],master_pid)
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=2) as response:self.assertEqual(response.status,200)
                finally:
                    process.terminate()
                    try:process.wait(timeout=8)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)
            records=[json.loads(line.split('[mof_startup] ',1)[1]) for line in log.read_text().splitlines() if line.startswith('[mof_startup] ')]
            loaders=[r for r in records if r['event']=='model_loader_thread_started']
            self.assertEqual(len(loaders),1);self.assertNotEqual(loaders[0]['pid'],master_pid)
            if preload:self.assertTrue(any(r['event']=='application_created' and r['pid']==master_pid for r in records))

    def test_preloaded_master_does_not_own_model(self):
        self.run_synthetic_ready(preload=True,hook=True)

    def test_normal_worker_owns_model(self):
        self.run_synthetic_ready(preload=False,hook=True)

    def test_preloaded_app_recovers_without_gunicorn_hook(self):
        self.run_synthetic_ready(preload=True,hook=False)

    def test_missing_model_is_reported_by_actual_worker(self):
        with tempfile.TemporaryDirectory(prefix='mof-gunicorn-test-') as directory:
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1',0));port=reservation.getsockname()[1]
            env=os.environ.copy();env.pop('MOF_SKIP_WARM',None)
            env['MOF_MODEL_PATH']=str(Path(directory)/'not-mounted.onnx')
            env['PYTHONUNBUFFERED']='1'
            # Keep the same worker/thread model used on Render. The missing
            # private file must fail safely, not leave readiness silently stuck.
            command=[sys.executable,'-m','gunicorn','--workers','1','--threads','2','--timeout','180',
                     '--bind',f'127.0.0.1:{port}','--capture-output','--error-logfile','-','server.app:app']
            log=Path(directory)/'startup.log'
            with log.open('wb') as output:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
                try:
                    deadline=time.monotonic()+25;status=None
                    while time.monotonic()<deadline:
                        if process.poll() is not None:self.fail('Gunicorn exited before status was available')
                        try:
                            with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/status',timeout=1) as response:
                                status=json.load(response)
                            if status.get('model_stage')=='failed':break
                        except (OSError,ValueError):pass
                        time.sleep(.1)
                    self.assertIsNotNone(status)
                    self.assertEqual(status.get('error_code'),'model_file_missing')
                    self.assertFalse(status['ready']);self.assertTrue(status['warm_enabled'])
                    self.assertEqual(status['process_id'],status['loader_process_id'])
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=2)
                    self.assertEqual(error.exception.code,503);error.exception.close()
                finally:
                    process.terminate()
                    try:process.wait(timeout=8)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)
            text=log.read_text(encoding='utf-8',errors='replace')
            self.assertIn('model_load_failed',text);self.assertNotIn(env['MOF_MODEL_PATH'],text)


if __name__=='__main__':unittest.main()
