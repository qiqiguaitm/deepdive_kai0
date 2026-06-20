import json, os, sys
from volcengine.ApiInfo import ApiInfo
from volcengine.Credentials import Credentials
from volcengine.ServiceInfo import ServiceInfo
from volcengine.base.Service import Service
jid = sys.argv[1] if len(sys.argv)>1 else "t-20260619191017-2ngpc"
si = ServiceInfo('open.volcengineapi.com', {'Accept':'application/json'},
                 Credentials(os.environ['VOLC_AK'], os.environ['VOLC_SK'], 'ml_platform', 'cn-beijing'), 10, 10)
svc = Service(si, {'GetJob': ApiInfo('POST','/',{'Action':'GetJob','Version':'2024-07-01'},{},{})})
d = json.loads(svc.json('GetJob', {}, json.dumps({"Id":jid}).encode())).get('Result', {})
st = (d.get('Status') or {}).get('State') or d.get('State')
print(f"{jid} {st} created={d.get('CreateTime')}")
