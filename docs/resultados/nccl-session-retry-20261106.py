"""Retry within the original session's absolute 30-minute budget."""
from datetime import datetime, timezone
from pathlib import Path
import json, subprocess, sys

deadline=datetime(2026,11,6,15,38,25,tzinfo=timezone.utc)
started=datetime.now(timezone.utc)
budget=min(1800,int((deadline-started).total_seconds())-10)
if budget<45:
    sys.exit('No remaining session budget; no GPU invocation launched')
command=['timeout','--signal=INT','--kill-after=10s',str(budget)+'s',
         'python3','-B','-m','infraestructura.nccl','run','--cluster',
         'experimentos/configs/cluster.json','--config',
         'experimentos/configs/nccl-diagnostic.json','--execute']
print('External session budget seconds:',budget,flush=True)
result=subprocess.run(command)
finished=datetime.now(timezone.utc)
path=Path('docs/resultados')/('nccl-session-budget-'+started.strftime('%Y%m%dT%H%M%SZ')+'.json')
with path.open('x') as f:
    json.dump({'deadline_utc':deadline.isoformat(),'started_at':started.isoformat(),
               'ended_at':finished.isoformat(),'budget_seconds':budget,'command':command,
               'returncode':result.returncode},f,indent=2)
print(path,flush=True)
sys.exit(result.returncode)
