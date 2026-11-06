"""Comandos y sondas acotadas de inferencia; sin efectos al importar."""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

from infraestructura import cluster


# Solo consulta estado. La sonda opcional de paquetes crea un contenedor CPU
# efímero sin red, sin GPU, sin bind mounts y con filesystem de solo lectura.
PROBE_SOURCE = r'''
import hashlib, json, os, platform, pwd, shutil, socket, subprocess, sys
from pathlib import Path
request = json.loads(sys.argv[1])
def command(argv, timeout=8):
    try:
        p = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
        return {"returncode":p.returncode, "stdout":p.stdout, "stderr":p.stderr}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"returncode":None, "stdout":"", "stderr":str(exc)}
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
mem = {}
for line in Path('/proc/meminfo').read_text().splitlines():
    key, value = line.split(':', 1)
    if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
        mem[key] = int(value.split()[0]) * 1024
cache = Path(request['hf_hub'])
models = {}
for model in request['models']:
    snapshot = cache / ('models--' + model['id'].replace('/', '--')) / 'snapshots' / model['revision']
    record = {"snapshot":str(snapshot), "present":snapshot.is_dir(), "complete":False}
    required = ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors.index.json']
    record['missing'] = [name for name in required if not (snapshot/name).is_file()]
    record['metadata_sha256'] = {name:digest(snapshot/name) for name in required if (snapshot/name).is_file()}
    index = snapshot/'model.safetensors.index.json'
    if index.is_file():
        try:
            shards = sorted(set(json.loads(index.read_text())['weight_map'].values()))
            if any(Path(s).name != s for s in shards):
                raise ValueError('unsafe shard path')
            record['shards'] = [{'name':s, 'present':(snapshot/s).is_file(),
                                 'size_bytes':(snapshot/s).stat().st_size if (snapshot/s).is_file() else None,
                                 'blob_name':(snapshot/s).resolve().name if (snapshot/s).is_file() else None}
                                for s in shards]
            record['complete'] = bool(shards) and not record['missing'] and all(s['present'] and s['size_bytes'] > 0 for s in record['shards'])
        except (ValueError, KeyError, OSError) as exc:
            record['error'] = str(exc)
    record['weights_content_rehashed'] = False
    models[model['key']] = record
image = command(['docker','inspect','--type','image',request['image_id'],
                 '--format','{{json .Id}} {{json .Architecture}}'])
packages = None
if request.get('container_probe') and image['returncode'] == 0:
    code = """import importlib.metadata as m,json,platform,io
from pathlib import Path
names=['ray','vllm','torch','transformers','tokenizers','huggingface-hub','apache-tvm-ffi','pandas']
d={}
for name in names:
 try:d[name]=m.version(name)
 except m.PackageNotFoundError:d[name]=None
import flashinfer.comm
from flashinfer.jit import env
p=env.FLASHINFER_WORKSPACE_DIR
assert p.is_relative_to(Path('/tmp/ipre-cache'))
(p/'ipre-write-check').write_text('CPU import passed')
import pandas
row={'prompt':'CPU fixture','output_tokens':128}
assert pandas.read_json(io.StringIO(json.dumps(row)+chr(10)),lines=True).to_dict('records')==[row]
print(json.dumps({'python':platform.python_version(),'packages':d,'cpu_imports':{'flashinfer.comm':'passed','cache_write':'passed','pandas_jsonl':'passed','cuda_arch_override':'12.1; not hardware validation'}}))
"""
    packages = command(['docker','run','--rm','--pull=never','--network','none','--read-only',
                        '--tmpfs','/tmp:rw,nosuid,size=256m','--user',str(os.getuid())+':'+str(os.getgid()),
                        '--env','USER='+pwd.getpwuid(os.getuid()).pw_name,
                        '--env','LOGNAME='+pwd.getpwuid(os.getuid()).pw_name,
                        '--env','NVIDIA_VISIBLE_DEVICES=void','--env','XDG_CACHE_HOME=/tmp/ipre-cache',
                        '--env','FLASHINFER_WORKSPACE_BASE=/tmp/ipre-cache','--env','FLASHINFER_CUDA_ARCH_LIST=12.1',
                        '--entrypoint','python3',request['image_id'],'-c',code], 45)
ports = set()
for name in ('/proc/net/tcp','/proc/net/tcp6'):
    for line in Path(name).read_text().splitlines()[1:]:
        parts = line.split()
        if parts[3] == '0A': ports.add(int(parts[1].split(':')[1],16))
ray_pids = []
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit(): continue
    try:
        if (entry/'comm').read_text().strip() in ('raylet','gcs_server'):
            ray_pids.append(int(entry.name))
    except OSError: pass
print(json.dumps({
 'node_id':request['node_id'], 'hostname':socket.gethostname(), 'architecture':platform.machine(),
 'uid':os.getuid(), 'gid':os.getgid(), 'user':pwd.getpwuid(os.getuid()).pw_name, 'groups':os.getgroups(),
 'memory_bytes':mem, 'disk_free_bytes':shutil.disk_usage(cache if cache.exists() else Path.cwd()).free,
 'image':image, 'container_probe':packages, 'models':models,
 'gpu':command(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader,nounits']),
 'gpu_apps':command(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits']),
 'docker_running':command(['docker','ps','--format','{{.ID}} {{.Names}} {{.Image}}']),
 'network':command(['ip','-j','address','show','dev','enp1s0f0np0']),
 'rdma_devices':sorted(p.name for p in Path('/dev/infiniband').glob('*')),
 'occupied_ports':sorted(ports.intersection(request['ports'])), 'ray_pids':ray_pids
}, sort_keys=True))
'''


def execute(argv, *, timeout=30, stdin=None):
    """Bound subprocess and its local descendants; argv is never shell-expanded."""
    started = time.monotonic()
    process = subprocess.Popen(argv, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); process.communicate()
        raise
    return {"command":argv, "returncode":process.returncode, "stdout":stdout,
            "stderr":stderr, "timed_out":timed_out,
            "duration_seconds":time.monotonic()-started}


def remote(node, argv, *, timeout=30, stdin=None):
    # The remote watchdog survives loss of the local SSH process.
    bounded = ["timeout", "--signal=TERM", "--kill-after=2s", f"{max(1, int(timeout)-3)}s", *argv]
    return execute(cluster.ssh_command(node, bounded), timeout=timeout, stdin=stdin)


def snapshot_subdir(model):
    return f"models--{model['id'].replace('/', '--')}/snapshots/{model['revision']}"


def model_path(model):
    return "/models/" + snapshot_subdir(model)


def used_ports(config):
    runtime = config['runtime']
    return [runtime[k] for k in ('api_port','ray_port','ray_node_manager_port','ray_object_manager_port')] + list(
        range(runtime['ray_worker_port_min'], runtime['ray_worker_port_max']+1))


def probe_command(node, config, *, container_probe=False):
    request = {"node_id":node['node_id'], "image_id":config['image']['id'],
               "hf_hub":config['paths']['hf_hub'], "models":config['models'],
               "ports":used_ports(config), "container_probe":container_probe}
    return ["python3", "-c", "import sys; exec(sys.stdin.read())", json.dumps(request, separators=(',', ':'))]


def preflight(cluster_config, config, *, container_probe=False):
    started_at=datetime.now(timezone.utc).isoformat()
    def one(node):
        try:
            result = remote(node, probe_command(node, config, container_probe=container_probe),
                            stdin=PROBE_SOURCE, timeout=90)
            if result['returncode'] != 0:
                return {"node_id":node['node_id'], "probe_error":result}
            return json.loads(result['stdout'].splitlines()[-1])
        except (OSError, ValueError, IndexError) as exc:
            return {"node_id":node['node_id'], "probe_error":str(exc)}
    nodes = cluster.selected_nodes(cluster_config, 'four')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(one, nodes))
    errors, uuids, envs = [], [], []
    for node, record in zip(nodes, records):
        name = node['node_id']
        if 'probe_error' in record:
            errors.append(f"{name}: consulta de host falló")
            continue
        if record['hostname'] != node['expected_hostname']:
            errors.append(f"{name}: hostname distinto al inventario")
        if record['architecture'] != 'aarch64' or record['uid'] == 0:
            errors.append(f"{name}: se requiere ARM64 y usuario no root")
        if record['image']['returncode'] != 0:
            errors.append(f"{name}: imagen fijada ausente o Docker inaccesible")
        else:
            try:
                image_id, architecture = shlex.split(record['image']['stdout'].strip())
                if (image_id, architecture) != (config['image']['id'], config['image']['architecture']):
                    errors.append(f"{name}: imagen o arquitectura diferente")
            except ValueError:
                errors.append(f"{name}: respuesta de inspección de imagen inválida")
        for key, cache in record['models'].items():
            if not cache['complete']:
                errors.append(f"{name}: snapshot completo no disponible: {key}")
        gpu = record['gpu']
        lines = gpu['stdout'].strip().splitlines()
        if gpu['returncode'] != 0 or len(lines) != 1 or not lines[0].startswith('GPU-'):
            errors.append(f"{name}: se requiere exactamente una GPU observada")
        else:
            uuids.append(lines[0].split(',')[0].strip())
        if record['gpu_apps']['returncode'] != 0 or record['gpu_apps']['stdout'].strip():
            errors.append(f"{name}: GPU ocupada o estado de aplicaciones desconocido")
        if record['occupied_ports'] or record['ray_pids']:
            errors.append(f"{name}: puerto reservado o proceso Ray ya ocupado")
        if record['memory_bytes']['MemAvailable'] < config['runtime']['minimum_available_memory_bytes']:
            errors.append(f"{name}: memoria disponible inferior al mínimo")
        if not record['rdma_devices']:
            errors.append(f"{name}: /dev/infiniband ausente")
        try:
            addresses = [a['local'] for iface in json.loads(record['network']['stdout'])
                         for a in iface.get('addr_info', [])]
            if config['network']['node_ips'][name] not in addresses:
                errors.append(f"{name}: IP f0 esperada no observada")
        except (ValueError, TypeError, KeyError):
            errors.append(f"{name}: no se pudo observar f0")
        package_probe = record.get('container_probe')
        if container_probe and package_probe:
            try:
                if package_probe['returncode']!=0:
                    raise ValueError('sonda de paquetes/importación falló')
                env = json.loads(package_probe['stdout'].splitlines()[-1])
                envs.append(env)
                versions = env['packages']
                if not versions.get('ray'):
                    errors.append(f"{name}: Ray ausente en la imagen")
                elif versions['ray']!=config['image']['ray_version']:
                    errors.append(f"{name}: versión Ray fuera del protocolo")
                if versions.get('apache-tvm-ffi')!=config['image']['tvm_ffi_version']:
                    errors.append(f"{name}: versión apache-tvm-ffi fuera del protocolo")
                if versions.get('pandas')!=config['image']['pandas_version']:
                    errors.append(f"{name}: pandas para benchmark ausente o fuera del protocolo")
                if not str(versions.get('vllm', '')).startswith(config['image']['vllm_version_prefix']):
                    errors.append(f"{name}: versión vLLM fuera del protocolo")
            except (ValueError, IndexError, KeyError):
                errors.append(f"{name}: falló sonda CPU de paquetes")
        elif container_probe:
            errors.append(f"{name}: paquetes sin verificar")
    if len(uuids) != len(set(uuids)):
        errors.append('GPU UUID repetido entre hosts')
    if envs and any(env != envs[0] for env in envs[1:]):
        errors.append('versiones de Python/paquetes diferentes entre nodos')
    for model in config['models']:
        snapshots = [r.get('models', {}).get(model['key']) for r in records]
        snapshots = [s for s in snapshots if s and s['complete']]
        fingerprints = [json.dumps({'metadata':s['metadata_sha256'], 'shards':s.get('shards')},sort_keys=True)
                        for s in snapshots]
        if fingerprints and any(f != fingerprints[0] for f in fingerprints[1:]):
            errors.append(f"{model['key']}: identidades de snapshot difieren entre nodos")
    if not container_probe:
        errors.append('pendiente sonda CPU de paquetes: --container-probe')
    return {"status":"ready_for_smoke" if not errors else 'blocked',
            "started_at":started_at,"ended_at":datetime.now(timezone.utc).isoformat(),
            "container_probe":container_probe, "errors":errors, "nodes":records,
            "limitations":["No inicia Ray/vLLM ni asigna GPU; no acredita inferencia.",
                            "Verifica metadatos y existencia de shards, no rehashea sus pesos."]}


def checked(result, description):
    if result['returncode'] != 0 or result.get('timed_out'):
        raise RuntimeError(f"{description}: exit={result['returncode']}, timeout={result.get('timed_out', False)}; {result['stderr'][-1500:]}")
    return result


def docker_base(node, config, node_probe, workspace, name, *, gpu=True):
    argv = ['docker','run','--pull=never','--name',name,'--label','io.ipre.inference='+name,
            '--network','host','--shm-size=8g','--user',f"{node_probe['uid']}:{node_probe['gid']}"]
    if gpu:
        argv += ['--gpus','all','--device','/dev/infiniband','--ulimit','memlock=-1:-1']
    for group in node_probe['groups']:
        if group != node_probe['gid']: argv += ['--group-add',str(group)]
    argv += ['--volume',config['paths']['hf_hub']+':/models:ro',
             '--volume',workspace+':/work', '--workdir','/work/code']
    environment = {**config['network']['environment'],
        'USER':node_probe['user'], 'LOGNAME':node_probe['user'],
        'PYTHONPATH':'/work/code', 'PYTHONUNBUFFERED':'1',
        'HF_HOME':'/work/cache/huggingface', 'HF_HUB_CACHE':'/models', 'HF_HUB_OFFLINE':'1',
        'TRANSFORMERS_OFFLINE':'1', 'HF_HUB_DISABLE_TELEMETRY':'1',
        'XDG_CACHE_HOME':'/work/cache', 'TORCH_HOME':'/work/cache/torch',
        'TRITON_CACHE_DIR':'/work/cache/triton', 'TORCHINDUCTOR_CACHE_DIR':'/work/cache/torchinductor',
        'FLASHINFER_WORKSPACE_BASE':'/work/cache',
        'VLLM_CACHE_ROOT':'/work/cache/vllm', 'VLLM_NO_USAGE_STATS':'1',
        'RAY_USAGE_STATS_ENABLED':'0', 'RAY_DEDUP_LOGS':'0',
        'RAY_TMPDIR':'/work/ray', 'VLLM_USE_RAY_V2_EXECUTOR_BACKEND':'1',
        'IPRE_NODE_ID':node['node_id'], 'IPRE_EVIDENCE_DIR':'/work/evidence',
        'VLLM_HOST_IP':config['network']['node_ips'][node['node_id']],
        'RAY_ADDRESS':f"{config['network']['node_ips']['pegasus']}:{config['runtime']['ray_port']}"}
    if not gpu: environment['NVIDIA_VISIBLE_DEVICES']='void'
    for key, value in environment.items(): argv += ['--env',f'{key}={value}']
    return argv


def ray_command(node, config):
    runtime = config['runtime']
    args = ['ray','start','--node-ip-address',config['network']['node_ips'][node['node_id']],
            '--num-gpus','1','--object-store-memory',str(runtime['ray_object_store_bytes']),
            '--node-manager-port',str(runtime['ray_node_manager_port']),
            '--object-manager-port',str(runtime['ray_object_manager_port']),
            '--min-worker-port',str(runtime['ray_worker_port_min']),
            '--max-worker-port',str(runtime['ray_worker_port_max']), '--disable-usage-stats']
    if node['node_id'] == 'pegasus':
        args += ['--head','--port',str(runtime['ray_port']), '--include-dashboard=false',
                 '--temp-dir','/work/ray']
    else:
        args += ['--address',f"{config['network']['node_ips']['pegasus']}:{runtime['ray_port']}"]
    return args


def server_command(config, model, world_size):
    r = config['runtime']
    args = ['vllm','serve',model_path(model),'--served-model-name',model['key'],
        '--tokenizer',model_path(model),'--revision',model['revision'],
        '--tokenizer-revision',model['tokenizer_revision'], '--dtype',r['dtype'],
        '--tensor-parallel-size',str(world_size),'--pipeline-parallel-size','1',
        '--distributed-executor-backend','ray', '--worker-cls','infraestructura.inference_worker.ObservedWorker',
        '--host','127.0.0.1','--port',str(r['api_port']),
        '--max-model-len',str(r['max_model_len']), '--max-num-seqs',str(r['max_num_seqs']),
        '--max-num-batched-tokens',str(r['max_num_batched_tokens']),
        '--gpu-memory-utilization',str(r['gpu_memory_utilization']),
        '--kv-cache-dtype','auto','--no-enable-prefix-caching', '--enable-chunked-prefill',
        '--generation-config','vllm','--seed',str(config['workload']['seed'])]
    if model['language_model_only']:
        args += ['--language-model-only','--default-chat-template-kwargs','{"enable_thinking":false}']
    return args


def bench_command(config, model, input_tokens, concurrency, phase, output_directory):
    count = config['measurement']['warmup_requests' if phase == 'warmup' else 'measured_requests']
    return ['vllm','bench','serve','--backend','openai','--base-url',f"http://127.0.0.1:{config['runtime']['api_port']}",
        '--endpoint','/v1/completions','--model',model['key'], '--tokenizer',model_path(model),
        '--dataset-name','custom','--dataset-path',f"/work/workloads/{model['key']}/{input_tokens}/{phase}.jsonl",
        '--skip-chat-template','--disable-shuffle','--no-oversample','--custom-output-len','-1',
        '--num-prompts',str(count),'--num-warmups','0','--max-concurrency',str(concurrency),
        '--request-rate','inf','--temperature','0','--ignore-eos','--seed',str(config['workload']['seed']),
        '--save-result','--save-detailed','--result-dir',output_directory,
        '--result-filename',phase+'.json','--percentile-metrics','ttft,tpot,itl,e2el',
        '--metric-percentiles','50,95','--disable-tqdm']
