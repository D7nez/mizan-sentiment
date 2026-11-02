from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]/'artifacts/legacy'
manifest=json.loads((root/'manifest.json').read_text())
for name,digest in manifest['sha256'].items():
    path=root/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise SystemExit('Missing or altered artifact: '+name)
print('All model artifact checksums match.')
