"""Fail if weights, LFS pointers, or old client-side inference enter the public tree."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN={'.pt','.pth','.onnx','.bin','.ckpt','.safetensors','.zip','.whl'}


def main():
    paths=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ['.git','.venv','__pycache__'])]
    bad=[]
    for p in paths:
        rel=p.relative_to(ROOT)
        if p.suffix.lower() in FORBIDDEN or 'private' in rel.parts:bad.append(str(rel));continue
        with p.open('rb') as f:head=f.read(150)
        if head.startswith(b'version https://git-lfs.github.com/spec/'):bad.append(str(rel)+' (LFS pointer)')
    assert not bad,'Private artifacts in public checkout: '+repr(bad)
    frontend='\n'.join(p.read_text(encoding='utf-8') for p in (ROOT/'frontend').glob('*.js'))
    assert all(x not in frontend for x in ['InferenceSession','onnxruntime','loadPyodide','mof_fusion.onnx','MOF_MODEL_BEARER_TOKEN'])
    # If staged, inspect index filenames too. Never rely only on .gitignore.
    result=subprocess.run(['git','diff','--cached','--name-only','-z'],cwd=ROOT,capture_output=True)
    if result.returncode==0:
        for name in result.stdout.decode('utf-8').split('\0'):
            if name:assert Path(name).suffix.lower() not in FORBIDDEN,name
    print('PASS: source-only public tree; no weights, LFS pointers, model runner or private secrets in frontend.')


if __name__=='__main__':main()
