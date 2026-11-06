"""Repeat the same tiny CPU collective, changing only MPI TCP interface selection."""
import concurrent.futures,json,re,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,'/home/bastiangl/ipre-rendimiento')
from infraestructura.cluster import load_config,ssh_command
from infraestructura.nccl import _execute
root=Path('/home/bastiangl/ipre-rendimiento')
previous=root/'docs/resultados/mpi-cpu-collective-20261106T145807Z.json'
base=json.loads(previous.read_text()); binary=base['binary']
command=base['command'].copy()
command[1:1]=['--mca','btl_tcp_if_include','10.10.0.0/24']
result=_execute(command,45)
rows=[json.loads(s) for s in re.findall(r'IPRE_MPI_RESULT=(\{[^\n]+\})',result['stdout'])]
ids=sorted(set(re.findall(r'ess_base_jobid\s+"?(\d+)',result['stderr'])))
cluster=load_config(root/'experimentos/configs/cluster.json')
success=result['returncode']==0 and not result['timed_out'] and len(rows)==4 and sorted(r['rank'] for r in rows)==[0,1,2,3] and {r['hostname'] for r in rows}=={n['expected_hostname'] for n in cluster['nodes']} and all(r['size']==4 and r['sum']==10 for r in rows)
time.sleep(1)
query='''import json,subprocess,sys
ids=json.loads(sys.argv[1]); binary=sys.argv[2]
p=subprocess.run(['ps','-eo','pid,ppid,args'],capture_output=True,text=True,timeout=5)
rows=[l for l in p.stdout.splitlines() if (any(j in l for j in ids) or binary in l) and 'python3 -c' not in l and 'ssh -o' not in l]
print(json.dumps({'remaining_own_jobs':rows}))
'''
def inspect(n):
 p=subprocess.run(ssh_command(n,['python3','-c',query,json.dumps(ids),binary]),capture_output=True,text=True,timeout=12)
 return {'node_id':n['node_id'],'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:postcheck=list(pool.map(inspect,cluster['nodes']))
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=root/'docs/resultados'/f'mpi-data-network-{stamp}.json'
with out.open('x') as f:json.dump({'scope':'CPU MPI collective; same binary as previous attempt; add only btl_tcp_if_include to align MPI TCP payload network with OOB','previous_evidence':str(previous),'binary':binary,'binary_sha256':base['binary_sha256'],'command':command,'result':result,'observed_results':rows,'success':success,'postcheck':postcheck},f,indent=2)
print(json.dumps({'success':success,'rc':result['returncode'],'timed_out':result['timed_out'],'duration':result['duration_seconds'],'rows':rows,'postcheck':postcheck},indent=2),flush=True)
print(out,flush=True)
