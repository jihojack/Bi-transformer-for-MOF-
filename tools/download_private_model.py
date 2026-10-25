"""Fetch a private model at server startup, never in client JavaScript or logs."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import tempfile
import urllib.parse
import urllib.request
import urllib.error

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from server.diagnostics import emit
manifest=json.loads((ROOT/'server/model_manifest.json').read_text())


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        parsed=urllib.parse.urlparse(newurl)
        if parsed.scheme!='https':raise RuntimeError('Non-HTTPS redirect rejected')
        result=super().redirect_request(req,fp,code,msg,headers,newurl)
        if urllib.parse.urlparse(req.full_url).netloc!=parsed.netloc:result.remove_header('Authorization')
        return result


def main():
    started=time.monotonic()
    emit('private_model_preparation_started')
    target=Path(os.getenv('MOF_MODEL_PATH',str(Path(tempfile.gettempdir())/'mof-private/model.onnx'))).resolve()
    expected=os.getenv('MOF_MODEL_SHA256') or manifest['onnx']['sha256']
    if target.is_file():
        emit('existing_private_model_hash_check_started')
        if digest(target)!=expected:raise RuntimeError('Existing private model hash mismatch; not overwritten')
        emit('private_model_verified',bytes=target.stat().st_size,elapsed_seconds=round(time.monotonic()-started,3));return
    url=os.getenv('MOF_MODEL_URL','')
    parsed=urllib.parse.urlparse(url)
    if parsed.scheme!='https' or not parsed.netloc or parsed.username or parsed.password:raise RuntimeError('Configure a private HTTPS model endpoint or mount the model file')
    emit('private_model_download_started')
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='model-',suffix='.partial',dir=target.parent)
    partial=Path(name)
    try:
        headers={'User-Agent':'MOF-private-inference'}
        token=os.getenv('MOF_MODEL_BEARER_TOKEN')
        if token:headers['Authorization']='Bearer '+token
        opener=urllib.request.build_opener(SafeRedirect())
        count=0
        with os.fdopen(fd,'wb') as f,opener.open(urllib.request.Request(url,headers=headers),timeout=90) as response:
            emit('private_model_download_connected',http_status=response.status)
            while part:=response.read(1024*1024):
                count+=len(part)
                if count>200_000_000:raise RuntimeError('Model artifact too large')
                f.write(part)
        emit('private_model_download_finished',bytes=count,elapsed_seconds=round(time.monotonic()-started,3))
        if digest(partial)!=expected:raise RuntimeError('Downloaded model hash mismatch')
        os.replace(partial,target)
        emit('private_model_verified',bytes=count,elapsed_seconds=round(time.monotonic()-started,3))
    finally:
        if partial.exists():partial.unlink()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        # Do not print private URLs, query signatures or authentication headers.
        emit('private_model_preparation_failed',error_type=type(exc).__name__,http_status=exc.code if isinstance(exc,urllib.error.HTTPError) else None)
        raise SystemExit('Private model preparation failed. Check server-only model settings and expected SHA-256.')
