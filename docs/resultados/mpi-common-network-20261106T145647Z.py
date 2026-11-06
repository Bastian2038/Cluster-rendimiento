"""Read-only network inspection and bounded CPU launcher validation."""
import concurrent.futures, json, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
sys.path.insert(0, '/home/bastiangl/ipre-rendimiento')
from infraestructura.cluster import load_config, ssh_command
from infraestructura.nccl import make_plan, _execute

root=Path('/home/bastiangl/ipre-rendimiento')
cluster=load_config(root/'experimentos/configs/cluster.json')
probe='''import json,subprocess
checks=[]
for command in [['hostname'],['ip','-j','addr','show','dev','enp1s0f0np0']]+[['ip','-j','route','get','10.10.0.'+str(i)] for i in range(10,14)]:
 p=subprocess.run(command,capture_output=True,text=True,timeout=5)
 checks.append({'command':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
print(json.dumps(checks))
'''
def inspect(n):
 p=subprocess.run(ssh_command(n,['python3','-c',probe]),capture_output=True,text=True,timeout=30)
 return {'node_id':n['node_id'],'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 observations=list(pool.map(inspect,cluster['nodes']))
valid=True
for idx,n in enumerate(observations):
 checks=json.loads(n['stdout']) if n['returncode']==0 else []
 valid=valid and len(checks)==6 and all(c['returncode']==0 for c in checks)
 if len(checks)==6:
  address=json.loads(checks[1]['stdout'])
  valid=valid and any(i.get('local')=='10.10.0.'+str(10+idx) and i.get('prefixlen')==24 for d in address for i in d.get('addr_info',[]))
  for c in checks[2:]:
   route=json.loads(c['stdout'])[0]
   valid=valid and route.get('dev') in ('lo','enp1s0f0np0') and route.get('prefsrc')=='10.10.0.'+str(10+idx)
print('Common subnet interfaces and routes verified:',valid,flush=True)
ompi=subprocess.run(['ompi_info','--param','oob','tcp','--level','9'],capture_output=True,text=True,timeout=10)
result=None; command=None
if valid:
 p=make_plan(root/'experimentos/configs/cluster.json',root/'experimentos/configs/nccl-diagnostic.json')
 command=p['jobs'][0]['command'][:-7]+['/bin/hostname']
 command[1:1]=['--mca','oob_tcp_if_include','10.10.0.0/24','--mca','plm_base_verbose','5','--mca','oob_base_verbose','5']
 result=_execute(command,45)
 print('MPI hostname rc',result['returncode'],'duration',result['duration_seconds'],flush=True)
 print(result['stdout'],flush=True)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=root/'docs/resultados'/f'mpi-common-network-{stamp}.json'
with out.open('x') as f:json.dump({'scope':'passive routes plus CPU hostname; OOB interface selection only for this MPI invocation','observations':observations,'routes_valid':valid,'ompi_oob_help':{'returncode':ompi.returncode,'stdout':ompi.stdout,'stderr':ompi.stderr},'command':command,'result':result},f,indent=2)
print(out,flush=True)
