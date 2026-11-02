import json,hashlib
from pathlib import Path
FILES=['model.keras','tokenizer.json','labels.json','stopwords.json']
def manifest(folder,version,status):
    folder=Path(folder)
    payload={'model_version':version,'evaluation_status':status,'sha256':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in FILES}}
    (folder/'manifest.json').write_text(json.dumps(payload,indent=2))
    return payload
