"""Passive postcheck after the bounded NCCL session; no GPU workload."""
import concurrent.futures,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,'/home/bastiangl/ipre-rendimiento')
from infraestructura.cluster import load_config,ssh_command
root=Path('/home/bastiangl/ipre-rendimiento')
query='''import json,os,subprocess
p=subprocess.run(['ps','-u',str(os.getuid()),'-o','pid,ppid,lstart,args'],capture_output=True,text=True,timeout=5)
ids=[str(31797*65536),str(26058*65536)]
rows=[l for l in p.stdout.splitlines() if (any(j in l for j in ids) or '/.local/share/ipre/nccl/' in l) and 'python3 -c' not in l and 'ssh -o' not in l]
g=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
print(json.dumps({'hostname':subprocess.check_output(['hostname'],text=True).strip(),'remaining_matching_processes':rows,'compute_apps':{'returncode':g.returncode,'stdout':g.stdout,'stderr':g.stderr}}))
'''
def inspect(node):
 command=ssh_command(node,['python3','-c',query])
 p=subprocess.run(command,capture_output=True,text=True,timeout=15)
 return {'node_id':node['node_id'],'command':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 results=list(pool.map(inspect,load_config(root/'experimentos/configs/cluster.json')['nodes']))
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
path=root/'docs/resultados'/f'nccl-postcheck-{stamp}.json'
with path.open('x') as f:json.dump({'scope':'passive postcheck; session ended; no new NCCL launch','timestamp_utc':stamp,'nodes':results},f,indent=2)
for item in results:print(item['node_id'],item['returncode'],item['stdout'],item['stderr'])
print(path)
