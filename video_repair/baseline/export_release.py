"""Export code+SAM source zip, excluding big weights and experiment outputs."""
from pathlib import Path
import json,hashlib,zipfile
root=Path(__file__).parent.resolve()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
files=[p for p in root.rglob('*') if p.is_file() and not any(s in ('__pycache__','validation','.git','models') for s in p.relative_to(root).parts) and p.suffix not in ('.pyc','.zip') and p.name!='RELEASE_MANIFEST.json']
manifest={str(p.relative_to(root)):sha(p) for p in sorted(files)};(root/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest,indent=2));files.append(root/'RELEASE_MANIFEST.json')
archive=root.parent/'mito_video_repair_v1_code.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,Path(root.name)/p.relative_to(root))
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),note='Models excluded; complete ready-to-run folder includes assets/models')))
