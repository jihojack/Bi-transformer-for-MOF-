"""Exercise warm startup and safe diagnostics without distributing model weights."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

os.environ['MOF_SKIP_WARM']='1'
from server.app import create_app
from server.runtime import Predictor
from tools import download_private_model


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(prefix='mof-startup-test-')
        self.addCleanup(self.directory.cleanup)
        self.model=Path(self.directory.name)/'private-model.onnx'
        self.payload=b'Unit-test placeholder; not a neural network'
        self.model.write_bytes(self.payload)
        self.digest=hashlib.sha256(self.payload).hexdigest()

    def predictor(self,path=None,sha=None):
        return Predictor(path=path or self.model,expected_hash=sha or self.digest)

    def test_initial_status_does_not_expose_path(self):
        state=self.predictor().status()
        self.assertFalse(state['ready']);self.assertEqual(state['model_stage'],'not_started')
        self.assertNotIn(str(self.model),json.dumps(state))

    def test_missing_file_fails_with_safe_stage(self):
        runtime=self.predictor(path=Path(self.directory.name)/'absent.onnx')
        output=io.StringIO()
        with contextlib.redirect_stdout(output),self.assertRaises(RuntimeError):runtime.load()
        self.assertEqual(runtime.status()['error_code'],'model_file_missing')
        self.assertEqual(runtime.status()['model_stage'],'failed')
        self.assertFalse(runtime.status()['loading']);self.assertNotIn(self.directory.name,output.getvalue())

    def test_hash_mismatch_never_creates_session(self):
        runtime=self.predictor(sha='0'*64)
        with patch('server.runtime.ort.InferenceSession') as native,self.assertRaises(RuntimeError):runtime.load()
        native.assert_not_called();self.assertEqual(runtime.status()['error_code'],'model_hash_mismatch')

    def test_success_is_idempotent_and_preserves_options(self):
        runtime=self.predictor();sentinel=object()
        with patch('server.runtime.ort.InferenceSession',return_value=sentinel) as native:
            runtime.load();runtime.load()
        self.assertEqual(native.call_count,1);self.assertIs(runtime.session,sentinel)
        self.assertEqual(runtime.stage,'ready');self.assertFalse(runtime.loading)
        options=native.call_args.kwargs['sess_options']
        self.assertEqual(options.intra_op_num_threads,1);self.assertEqual(options.inter_op_num_threads,1)
        self.assertEqual(native.call_args.kwargs['providers'],['CPUExecutionProvider'])

    def test_session_failure_does_not_log_exception_secrets(self):
        runtime=self.predictor();output=io.StringIO()
        secret='https://private.example/model?token=DO_NOT_LOG_THIS'
        with contextlib.redirect_stdout(output),patch('server.runtime.ort.InferenceSession',side_effect=RuntimeError(secret)):
            with self.assertRaises(RuntimeError):runtime.load()
        self.assertEqual(runtime.error_code,'onnx_session_init_failed')
        self.assertNotIn(secret,output.getvalue());self.assertNotIn(secret,json.dumps(runtime.status()))
        self.assertIn('creating_onnx_session',output.getvalue())

    def test_real_background_path_health_waits_for_model(self):
        entered=threading.Event();release=threading.Event();runtime=self.predictor()
        def native(*args,**kwargs):
            entered.set()
            if not release.wait(5):raise RuntimeError('Unit-test event timed out')
            return object()
        with patch('server.runtime.ort.InferenceSession',side_effect=native):
            app=create_app(runtime,warm=True);client=app.test_client()
            try:
                self.assertTrue(entered.wait(3))
                status=client.get('/api/status').get_json()
                self.assertTrue(status['loader_thread_alive']);self.assertTrue(status['loading'])
                self.assertEqual(status['model_stage'],'creating_onnx_session')
                self.assertEqual(client.get('/healthz').status_code,503)
            finally:
                release.set();app.extensions['mof_loader_thread'].join(3)
        self.assertEqual(client.get('/healthz').status_code,200)
        self.assertFalse(client.get('/api/status').get_json()['loader_thread_alive'])

    def test_disabled_warm_is_visible_and_not_healthy(self):
        client=create_app(self.predictor(),warm=False).test_client()
        self.assertFalse(client.get('/api/status').get_json()['warm_enabled'])
        self.assertEqual(client.get('/healthz').status_code,503)

    def test_deferred_warm_only_starts_after_worker_hook_or_request(self):
        runtime=self.predictor()
        with patch('server.runtime.ort.InferenceSession',return_value=object()) as native:
            app=create_app(runtime,warm=True,defer_warm=True)
            self.assertEqual(runtime.stage,'not_started');native.assert_not_called()
            self.assertIsNone(app.extensions['mof_loader_thread'])
            # This same callback is invoked by Gunicorn's post_worker_init hook.
            app.extensions['mof_start_model']()
            app.extensions['mof_loader_thread'].join(3)
            self.assertEqual(app.test_client().get('/healthz').status_code,200)
            self.assertEqual(native.call_count,1)

    def test_request_fallback_starts_deferred_model_without_hook(self):
        runtime=self.predictor()
        with patch('server.runtime.ort.InferenceSession',return_value=object()) as native:
            app=create_app(runtime,warm=True,defer_warm=True)
            app.test_client().get('/api/status')
            app.extensions['mof_loader_thread'].join(3)
            self.assertEqual(app.test_client().get('/healthz').status_code,200)
            self.assertEqual(native.call_count,1)

    def test_forked_worker_recreates_runtime_without_touching_inherited_lock(self):
        runtime=self.predictor();runtime.load_pid=os.getpid();runtime.loading=True
        runtime.stage='verifying_model_hash';runtime.lock.acquire()
        app=create_app(runtime,warm=True,defer_warm=True)
        child_pid=os.getpid()+1000
        try:
            with patch('server.app.os.getpid',return_value=child_pid),patch('server.runtime.ort.InferenceSession',return_value=object()):
                app.extensions['mof_start_model']()
                app.extensions['mof_loader_thread'].join(3)
                fresh=app.extensions['mof_predictor']
                self.assertIsNot(fresh,runtime);self.assertTrue(runtime.lock.locked())
                self.assertEqual(fresh.status()['loader_process_id'],child_pid)
                self.assertTrue(fresh.status()['ready'])
        finally:runtime.lock.release()

    def test_runtime_rejects_session_from_different_process(self):
        runtime=self.predictor();runtime.session=object();runtime.load_pid=os.getpid()+1000
        self.assertFalse(runtime.status()['ready'])
        self.assertEqual(runtime.status()['error_code'],'model_process_mismatch')
        with self.assertRaises(RuntimeError):runtime.load()

    def test_existing_private_download_is_verified_without_network_or_leaks(self):
        output=io.StringIO()
        settings={'MOF_MODEL_PATH':str(self.model),'MOF_MODEL_SHA256':self.digest,
                  'MOF_MODEL_URL':'https://private.example/model?signature=SECRET_SIGNATURE',
                  'MOF_MODEL_BEARER_TOKEN':'SECRET_BEARER'}
        with patch.dict(os.environ,settings),contextlib.redirect_stdout(output),patch('urllib.request.build_opener') as network:
            download_private_model.main()
        network.assert_not_called()
        self.assertIn('private_model_verified',output.getvalue())
        for sensitive in [self.directory.name,'private.example','SECRET_SIGNATURE','SECRET_BEARER']:
            self.assertNotIn(sensitive,output.getvalue())


if __name__=='__main__':unittest.main()
