"""Check syntax, notebook cleanliness, frozen asset hashes and disjoint splits."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'app'/'assets'/'asset_manifest.json'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-asset-manifest',action='store_true')
    parser.add_argument('--private-checkpoint',type=Path)
    args=parser.parse_args()
    if args.write_asset_manifest:
        assets = [p for p in (ROOT/'app'/'assets').iterdir() if p.is_file() and p!=MANIFEST]
        payload={'files':{p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(assets)}}
        MANIFEST.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        path=ROOT/name
        if not path.is_file() or path.stat().st_size!=expected['bytes'] or sha(path)!=expected['sha256']:
            raise RuntimeError(f'Public preprocessing asset mismatch: {name}')
    if args.private_checkpoint:
        expected=next(iter(manifest['private_artifacts'].values()))
        assert args.private_checkpoint.stat().st_size==expected['bytes']
        assert sha(args.private_checkpoint)==expected['sha256']
    sources=list(ROOT.rglob('*.py'))
    for path in sources:
        if '.venv' not in path.parts:
            ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    notebooks=list(ROOT.rglob('*.ipynb'))
    for path in notebooks:
        nb=json.loads(path.read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                assert not cell.get('outputs'),path
                assert cell.get('execution_count') is None,path
    split=json.loads((ROOT/'data'/'splits'/'fixed_split_70_10_20.json').read_text(encoding='utf-8'))
    ids={name:set(split[f'{name}_mof_ids']) for name in ['train','val','test']}
    assert [len(ids[n]) for n in ids]==[35000,5000,10000]
    assert not ids['train']&ids['val'] and not ids['train']&ids['test'] and not ids['val']&ids['test']
    with (ROOT/'data'/'splits'/'external_10k_ids.csv').open(encoding='utf-8-sig',newline='') as handle:
        external_rows=list(csv.DictReader(handle))
    external_ids=[row['mof_id'] for row in external_rows]
    assert len(external_ids)==10000 and len(set(external_ids))==10000
    assert not set(external_ids)&set.union(*ids.values())
    sampling=json.loads((ROOT/'data'/'external_sampling_manifest.json').read_text(encoding='utf-8'))
    ordered_hash=hashlib.sha256(''.join(mof_id+'\n' for mof_id in external_ids).encode('utf-8')).hexdigest()
    assert ordered_hash==sampling['primary_id_order_sha256']
    broken=[]
    for path in ROOT.rglob('*.md'):
        for target in re.findall(r'\]\(([^\s)]+)\)',path.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#','mailto:')):
                continue
            if not (path.parent/target.split('#')[0]).exists():
                broken.append((str(path.relative_to(ROOT)),target))
    assert not broken,broken
    print(f'PASS: {len(manifest["files"])} public frozen assets; {len(sources)} Python sources; {len(notebooks)} clean notebooks.')
    print('Private checkpoint: '+('SHA-256 verified.' if args.private_checkpoint else 'not included and not checked.'))
    print('PASS: 35,000/5,000/10,000 disjoint IDs and local Markdown links.')
    print('PASS: additional 10,000 IDs excluded from all original 50K; sample-order hash matches.')


if __name__=='__main__':
    main()
