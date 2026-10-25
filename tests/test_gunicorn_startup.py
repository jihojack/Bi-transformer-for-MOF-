"""Linux CI: real Gunicorn startup, no private weights, no fake healthy response."""
import json
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
