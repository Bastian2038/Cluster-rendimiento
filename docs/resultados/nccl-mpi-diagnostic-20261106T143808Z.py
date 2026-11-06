import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
sys.path.insert(0, '/home/bastiangl/ipre-rendimiento')
from infraestructura.cluster import load_config, ssh_command
from infraestructura.nccl import _execute

root=Path('/home/bastiangl/ipre-rendimiento')
cluster=load_config(root/'experimentos/configs/cluster.json')
query="""import json,subprocess
p=subprocess.run(['ps','-eo','pid,ppid,lstart,args'],capture_output=True,text=True,timeout=5)
rows=[line for line in p.stdout.splitlines() if any(x in line for x in ('mpirun','orted','/bin/all_reduce_perf','ipre/nccl/')) and 'python3 -c' not in line and 'ssh -o' not in line]
print(json.dumps({'hostname':subprocess.check_output(['hostname'],text=True).strip(),'processes':rows}))
"""
def inspect(n):
 r=subprocess.run(ssh_command(n,['python3','-c',query]),capture_output=True,text=True,timeout=15)
 return {'node_id':n['node_id'],'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: before=list(pool.map(inspect,cluster['nodes']))
checks=[]
base=['env','-u','DISPLAY','-u','XAUTHORITY','mpirun','--tag-output','--mca','plm_rsh_args',
      '-o BatchMode=yes -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=8']
for label,hosts,n in [('single','spark-cfea:1',1),('four','spark-cfea:1,192.168.100.11:1,146.155.13.134:1,146.155.13.135:1',4)]:
 command=base+['--mca','plm_base_verbose','5','--mca','oob_base_verbose','5','--host',hosts,'--map-by','ppr:1:node','-np',str(n),'/bin/hostname']
 result=_execute(command,25)
 checks.append({'label':label,'command':command,**result})
 print(label,'rc',result['returncode'],'timeout',result['timed_out'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: after=list(pool.map(inspect,cluster['nodes']))
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
path=root/'docs/resultados'/f'nccl-mpi-diagnostic-{stamp}.json'
with path.open('x') as f:json.dump({'scope':'CPU-only MPI hostname after failed NCCL smoke; no network settings changed','processes_before':before,'checks':checks,'processes_after':after},f,indent=2)
print(path,flush=True)
