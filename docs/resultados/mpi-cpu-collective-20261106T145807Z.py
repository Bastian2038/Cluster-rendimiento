"""Small CPU MPI collective using the verified OOB subnet; no GPU workload."""
import concurrent.futures,hashlib,json,re,shlex,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,'/home/bastiangl/ipre-rendimiento')
from infraestructura.cluster import load_config,ssh_command
from infraestructura.nccl import make_plan,_execute
root=Path('/home/bastiangl/ipre-rendimiento')
cluster=load_config(root/'experimentos/configs/cluster.json')
source=Path('/tmp/ipre_mpi_cpu_check.c')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
binary=Path('/tmp')/('ipre-mpi-cpu-'+stamp)
compile_command=['mpicc','-O2','-Wall','-Wextra','-o',str(binary),str(source)]
subprocess.run(compile_command,check=True,capture_output=True,text=True,timeout=15)
data=binary.read_bytes(); digest=hashlib.sha256(data).hexdigest()
copies=[]
for node in cluster['nodes']:
 if node['local']:continue
 script='''import hashlib,os,pathlib,sys
p=pathlib.Path(sys.argv[1]); data=sys.stdin.buffer.read()
assert hashlib.sha256(data).hexdigest()==sys.argv[2]
with p.open('xb') as f:f.write(data)
p.chmod(0o700)
print(hashlib.sha256(p.read_bytes()).hexdigest())
'''
 command=ssh_command(node,['python3','-c',script,str(binary),digest])
 p=subprocess.run(command,input=data,capture_output=True,timeout=12)
 copies.append({'node_id':node['node_id'],'command':command,'returncode':p.returncode,'stdout':p.stdout.decode(),'stderr':p.stderr.decode()})
 assert p.returncode==0 and p.stdout.decode().strip()==digest,copies[-1]
plan=make_plan(root/'experimentos/configs/cluster.json',root/'experimentos/configs/nccl-diagnostic.json')
command=plan['jobs'][0]['command'][:-7]+[str(binary)]
command[1:1]=['--mca','oob_tcp_if_include','10.10.0.0/24','--mca','plm_base_verbose','5']
result=_execute(command,45)
rows=[json.loads(s) for s in re.findall(r'IPRE_MPI_RESULT=(\{[^\n]+\})',result['stdout'])]
ids=sorted(set(re.findall(r'ess_base_jobid\s+"?(\d+)',result['stderr'])))
success=result['returncode']==0 and not result['timed_out'] and len(rows)==4 and sorted(r['rank'] for r in rows)==[0,1,2,3] and {r['hostname'] for r in rows}=={n['expected_hostname'] for n in cluster['nodes']} and all(r['size']==4 and r['sum']==10 for r in rows)
time.sleep(1)
query='''import json,subprocess,sys
ids=json.loads(sys.argv[1]); binary=sys.argv[2]
p=subprocess.run(['ps','-eo','pid,ppid,args'],capture_output=True,text=True,timeout=5)
rows=[l for l in p.stdout.splitlines() if (any(j in l for j in ids) or binary in l) and 'python3 -c' not in l and 'ssh -o' not in l]
print(json.dumps({'remaining_own_jobs':rows}))
'''
def inspect(n):
 p=subprocess.run(ssh_command(n,['python3','-c',query,json.dumps(ids),str(binary)]),capture_output=True,text=True,timeout=12)
 return {'node_id':n['node_id'],'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:postcheck=list(pool.map(inspect,cluster['nodes']))
out=root/'docs/resultados'/f'mpi-cpu-collective-{stamp}.json'
with out.open('x') as f:json.dump({'scope':'CPU MPI_Init + MPI_Allreduce integers + MPI_Finalize; no NCCL/GPU; only OOB subnet specified','source':source.read_text(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'compile_command':compile_command,'binary':str(binary),'binary_sha256':digest,'copies':copies,'command':command,'result':result,'observed_results':rows,'success':success,'postcheck':postcheck},f,indent=2)
print(json.dumps({'success':success,'rc':result['returncode'],'timed_out':result['timed_out'],'duration':result['duration_seconds'],'rows':rows,'postcheck':postcheck},indent=2),flush=True)
print(out,flush=True)
