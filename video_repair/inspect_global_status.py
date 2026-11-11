import json,urllib.request
from pathlib import Path
ROOT=Path(__file__).parent
req=urllib.request.Request('http://127.0.0.1:18085',headers={'Authorization':'Bearer '+(ROOT/'io_access.token').read_text().strip()})
print(json.dumps(json.load(urllib.request.urlopen(req,timeout=20))))
