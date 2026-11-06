"""Passive, read-only node checks after the f0 NCCL session closes."""
import concurrent.futures
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/bastiangl/ipre-rendimiento')
sys.path.insert(0, str(ROOT))
from infraestructura import manifest
from infraestructura.cluster import load_config, ssh_command

QUERY = r'''
import json, os, subprocess, time
from pathlib import Path
names = {'mpirun', 'mpiexec', 'orterun', 'orted', 'all_reduce_perf',
         'all_gather_perf', 'reduce_scatter_perf', 'sendrecv_perf'}
candidates = []
scan_errors = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        if proc.stat().st_uid != os.getuid():
            continue
        args = [x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        if not args:
            continue
        name = Path(args[0]).name
        is_wrapper = name in {'timeout', 'sh', 'bash'} and any(
            a.startswith('/home/bastiangl/.local/share/ipre/nccl/') for a in args[1:])
        if name not in names and not is_wrapper:
            continue
        fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        candidates.append({'pid': int(proc.name), 'ppid': int(fields[1]),
                           'start_ticks': fields[19], 'argv': args})
    except FileNotFoundError:
        pass
    except (PermissionError, OSError, IndexError) as exc:
        scan_errors.append({'pid': int(proc.name), 'error': type(exc).__name__})
gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                      '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8)
print(json.dumps({'hostname': subprocess.check_output(['hostname'], text=True).strip(),
                  'uid': os.getuid(), 'timestamp_epoch': time.time(),
                  'mpi_or_nccl_process_candidates': candidates, 'process_scan_errors': scan_errors,
                  'compute_apps': {'returncode': gpu.returncode, 'stdout': gpu.stdout,
                                   'stderr': gpu.stderr}}))
'''

def inspect(node):
    command = ssh_command(node, ['python3', '-c', QUERY])
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return {'node_id': node['node_id'], 'command': command,
                'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except subprocess.TimeoutExpired as exc:
        return {'node_id': node['node_id'], 'command': command,
                'returncode': None, 'error': str(exc)}

run_path = Path(sys.argv[1]).resolve()
run = manifest.load_run(run_path)
if run['status'] not in {'complete', 'failed'}:
    raise SystemExit('Run is still active; no postcheck executed')
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(inspect, load_config(ROOT/'experimentos/configs/cluster.json')['nodes']))
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out = ROOT/'docs/resultados'/f'nccl-postcheck-{stamp}.json'
document = {'run_id': run['run_id'], 'timestamp_utc': stamp,
            'scope': 'passive process and compute-app checks; no GPU workload or process termination',
            'limitations': 'Candidates belong to the user, not necessarily this run. No complete auxiliary SSH-client census. GPU apps are reported separately without ownership attribution.',
            'query_source': QUERY, 'nodes': results}
with out.open('x') as stream:
    json.dump(document, stream, indent=2)
for item in results:
    print(item['node_id'], item['returncode'], item.get('stdout', ''), item.get('stderr', ''), item.get('error', ''))
print(out)
