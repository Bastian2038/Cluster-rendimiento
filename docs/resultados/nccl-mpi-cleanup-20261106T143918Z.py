import concurrent.futures,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,'/home/bastiangl/ipre-rendimiento')
from infraestructura.cluster import load_config,ssh_command
root=Path('/home/bastiangl/ipre-rendimiento')
targets={'pegasus':[(556673,'1692467200'),(556674,'1692467200')], 'octopus':[],
         'griffin':[(309976,'1692467200'),(310346,'2617049088')],
         'draggon':[(1964982,'1692467200'),(1965377,'2617049088')]}
script='''import json,os,pathlib,select,signal,subprocess,sys
results=[]
for pid,job in json.loads(sys.argv[1]):
 p=pathlib.Path('/proc')/str(pid)
 try:
  fd=os.pidfd_open(pid)
 except ProcessLookupError:
  results.append({'pid':pid,'status':'already_gone'}); continue
 try:
  args=(p/'cmdline').read_bytes().decode().split('\\0')
  owner=p.stat().st_uid
  if owner!=os.getuid() or pathlib.Path(args[0]).name not in ('orted','ssh') or 'ess_base_jobid' not in ' '.join(args) or job not in ' '.join(args):
   results.append({'pid':pid,'status':'identity_mismatch_not_signalled'}); continue
  signal.pidfd_send_signal(fd,signal.SIGTERM)
  poll=select.poll(); poll.register(fd,select.POLLIN)
  if not poll.poll(3000):
   signal.pidfd_send_signal(fd,signal.SIGKILL); poll.poll(1000)
  results.append({'pid':pid,'job_id':job,'status':'own_process_terminated'})
 except (FileNotFoundError,ProcessLookupError):
  results.append({'pid':pid,'status':'already_gone'})
 finally: os.close(fd)
p=subprocess.run(['ps','-eo','pid,ppid,lstart,args'],capture_output=True,text=True,timeout=5)
remaining=[l for l in p.stdout.splitlines() if any(j in l for j in ('1692467200','2617049088')) and 'python3 -c' not in l and ('orted' in l or '/usr/bin/ssh' in l)]
print(json.dumps({'results':results,'remaining_own_jobs':remaining}))
'''
def clean(n):
 r=subprocess.run(ssh_command(n,['python3','-c',script,json.dumps(targets[n['node_id']])]),capture_output=True,text=True,timeout=20)
 return {'node_id':n['node_id'],'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 results=list(pool.map(clean,load_config(root/'experimentos/configs/cluster.json')['nodes']))
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=root/'docs/resultados'/f'nccl-mpi-cleanup-{stamp}.json'
with out.open('x') as f:json.dump({'scope':'Only exact PIDs and job ids from NCCL smoke and subsequent CPU diagnostic; pidfd signals after owner/command verification','nodes':results},f,indent=2)
print(json.dumps(results,indent=2));print(out)
