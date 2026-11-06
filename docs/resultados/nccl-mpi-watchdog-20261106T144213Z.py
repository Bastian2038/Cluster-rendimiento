import concurrent.futures,json,re,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,'/home/bastiangl/ipre-rendimiento')
from infraestructura.cluster import load_config,ssh_command
from infraestructura.nccl import make_plan,_execute
root=Path('/home/bastiangl/ipre-rendimiento')
plan=make_plan(root/'experimentos/configs/cluster.json',root/'experimentos/configs/nccl-diagnostic.json')
# Retain the exact MPI launcher under test; replace the GPU rank command with hostname.
command=plan['jobs'][0]['command'][:-7]+['/bin/hostname']
command[1:1]=['--mca','plm_base_verbose','5','--mca','oob_base_verbose','5']
result=_execute(command,45)
ids=sorted(set(re.findall(r'ess_base_jobid\s+"?(\d+)',result['stderr'])))
time.sleep(3)
query='''import json,subprocess,sys
ids=json.loads(sys.argv[1]); p=subprocess.run(['ps','-eo','pid,ppid,lstart,args'],capture_output=True,text=True,timeout=5)
rows=[l for l in p.stdout.splitlines() if any(j in l for j in ids) and ('orted' in l or 'mpirun' in l or '/usr/bin/ssh' in l) and 'python3 -c' not in l]
print(json.dumps({'hostname':subprocess.check_output(['hostname'],text=True).strip(),'remaining_own_jobs':rows}))
'''
def inspect(n):
 r=subprocess.run(ssh_command(n,['python3','-c',query,json.dumps(ids+['1692467200','2617049088'])]),capture_output=True,text=True,timeout=15)
 return {'node_id':n['node_id'],'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 observations=list(pool.map(inspect,load_config(root/'experimentos/configs/cluster.json')['nodes']))
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=root/'docs/resultados'/f'nccl-mpi-watchdog-{stamp}.json'
with out.open('x') as f:json.dump({'scope':'CPU hostname; bounded launcher validation only, not NCCL','command':command,'job_ids':ids,'result':result,'postcheck':observations},f,indent=2)
print(json.dumps({'rc':result['returncode'],'timed_out':result['timed_out'],'duration':result['duration_seconds'],'job_ids':ids,'postcheck':observations},indent=2))
print(out)
