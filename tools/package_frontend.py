"""Export only the tiny API client for Netlify. Never copies model files."""
from pathlib import Path
import argparse
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--api-origin',default='')
args=parser.parse_args()
if args.api_origin and not args.api_origin.startswith('https://'):raise SystemExit('Production API origin must use HTTPS')
if args.output.exists():raise SystemExit('Output already exists; choose a new folder')
args.output.mkdir(parents=True)
for name in ['index.html','app.js','styles.css','config.js']:shutil.copy2(ROOT/'frontend'/name,args.output/name)
if args.api_origin:
    import json
    (args.output/'config.js').write_text('window.MOF_API_BASE = '+json.dumps(args.api_origin.rstrip('/'))+';\n')
archive=args.output.with_suffix('.zip')
if archive.exists():raise SystemExit('Archive already exists')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(args.output.iterdir()):z.write(p,p.name)
print('Source-only frontend:',args.output,'ZIP bytes:',archive.stat().st_size)
