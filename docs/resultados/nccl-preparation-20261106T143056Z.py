"""Prepare the explicitly authorized isolated NCCL runtime; no GPU workloads."""
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone

ROOT = Path('/home/bastiangl/ipre-rendimiento')
sys.path.insert(0, str(ROOT))
from infraestructura.cluster import load_config, ssh_command

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

sources = {f'bin/{name}': Path('/home/bastiangl/nccl-tests/build') / name
           for name in ('all_reduce_perf', 'all_gather_perf', 'reduce_scatter_perf', 'sendrecv_perf')}
sources['lib/libnccl.so.2.30.7'] = Path('/home/bastiangl/nccl/build/lib/libnccl.so.2.30.7')
for project in ('nccl', 'nccl-tests'):
    sources[f'licenses/{project}.txt'] = Path('/home/bastiangl') / project / 'LICENSE.txt'
    if not sources[f'licenses/{project}.txt'].exists():
        sources[f'licenses/{project}.txt'] = Path('/home/bastiangl') / project / 'LICENSE'
files = {name: {'sha256': digest(path), 'bytes': path.stat().st_size} for name, path in sources.items()}
identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
destination = Path('/home/bastiangl/.local/share/ipre/nccl') / identity
metadata = {'bundle_id': identity, 'destination': str(destination), 'files': files,
            'source_checkouts_observed': {project: subprocess.check_output(
                ['git', '-C', f'/home/bastiangl/{project}', 'rev-parse', 'HEAD'], text=True).strip()
                for project in ('nccl', 'nccl-tests')},
            'provenance_note': 'Source checkouts observed; existing executable build provenance not independently proven.'}
archive = Path('/tmp') / f'ipre-nccl-{identity}.tar'
with tarfile.open(archive, 'x') as tar:
    for name, path in sources.items():
        tar.add(path, arcname=name, recursive=False)
    data = json.dumps(metadata, indent=2).encode()
    item = tarfile.TarInfo('bundle.json'); item.size = len(data); item.mode = 0o644
    tar.addfile(item, io.BytesIO(data))
    for name in ('libnccl.so', 'libnccl.so.2'):
        item = tarfile.TarInfo('lib/' + name); item.type = tarfile.SYMTYPE
        item.linkname = 'libnccl.so.2.30.7'; tar.addfile(item)
verify = '''import ctypes,hashlib,json,os,pathlib,subprocess,sys
p=pathlib.Path(sys.argv[1]); m=json.loads((p/'bundle.json').read_text())
for name, info in m['files'].items():
 with (p/name).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==info['sha256'], name
env=dict(os.environ,LD_LIBRARY_PATH=str(p/'lib'))
checks=[]
for name in ('all_reduce_perf','all_gather_perf','reduce_scatter_perf','sendrecv_perf'):
 r=subprocess.run(['ldd',str(p/'bin'/name)],env=env,capture_output=True,text=True,timeout=8)
 checks.append({'binary':name,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 assert r.returncode==0 and 'not found' not in r.stdout, checks[-1]
lib=ctypes.CDLL(str(p/'lib/libnccl.so.2')); v=ctypes.c_int(); assert lib.ncclGetVersion(ctypes.byref(v))==0
print(json.dumps({'hostname':subprocess.check_output(['hostname'],text=True).strip(),'bundle_id':m['bundle_id'],'nccl_version':v.value,'link_checks':checks}))
'''
cluster = load_config(ROOT/'experimentos/configs/cluster.json')
records=[]
deadline=time.monotonic()+110
for node in cluster['nodes']:
    # mkdir of the final directory is exclusive: never overwrite existing files.
    shell = f'mkdir -p {shlex.quote(str(destination.parent))} && mkdir {shlex.quote(str(destination))} && tar --extract --keep-old-files --file=- --directory={shlex.quote(str(destination))}'
    remaining = max(1, min(45, deadline-time.monotonic()))
    with archive.open('rb') as data:
        r=subprocess.run(ssh_command(node,['sh','-c',shell]), stdin=data, capture_output=True, timeout=remaining)
    record={'node_id':node['node_id'],'copy_returncode':r.returncode,'copy_stderr':r.stderr.decode()}
    records.append(record)
    if r.returncode:
        break
    r=subprocess.run(ssh_command(node,['python3','-c',verify,str(destination)]),capture_output=True,text=True,
                     timeout=max(1,min(40,deadline-time.monotonic())))
    record.update(verify_returncode=r.returncode, verify_stderr=r.stderr)
    if r.returncode==0:
        record['verification']=json.loads(r.stdout)
    else:
        record['verify_stdout']=r.stdout
        break
    print(node['node_id'], 'copied and linker verified', flush=True)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
output=ROOT/'docs/resultados'/f'nccl-preparation-{stamp}.json'
with output.open('x') as f:
    json.dump({'timestamp_utc':stamp,'bundle':metadata,'script_sha256':digest(__file__),'nodes':records},f,indent=2)
print('EVIDENCE',output, flush=True)
assert len(records)==4 and all(r.get('verify_returncode')==0 for r in records), 'preparation incomplete; see evidence'
config_path=ROOT/'experimentos/configs/nccl-diagnostic.json'
config=json.loads(config_path.read_text())
config['binary_paths']={n['node_id']:str(destination/'bin') for n in cluster['nodes']}
config['library_paths']={n['node_id']:str(destination/'lib') for n in cluster['nodes']}
config_path.write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
print('CONFIG_UPDATED',str(destination),flush=True)
