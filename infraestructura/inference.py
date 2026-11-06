"""Campaña de inferencia 4/2: plan, preflight, run y report.

El plan no ejecuta comandos. El runner no instala, descarga, modifica red ni
detiene procesos ajenos. Los prerrequisitos ausentes bloquean antes de usar GPU.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
from datetime import datetime
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from infraestructura import cluster, manifest
from infraestructura import inference_runtime as runtime
from infraestructura.inference_metrics import summarize_bench_result, paired_speedup
from infraestructura.inference_observation import validate_worker_records
from infraestructura.rutas import REPO_ROOT


DEFAULT_CONFIG = REPO_ROOT / 'experimentos/configs/inferencia-4-2.json'
SAFE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*$')
SHA40 = re.compile(r'^[0-9a-f]{40}$')
SHA256 = re.compile(r'^sha256:[0-9a-f]{64}$')


class InferenceError(ValueError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def validate_config(config, cluster_config):
    if config.get('schema_version') != '1.0' or config.get('protocol_id') != 'inference-4-2-v1':
        raise InferenceError('protocolo de inferencia desconocido')
    models = config.get('models', [])
    if len(models) != 2 or [m.get('key') for m in models] != ['qwen35-9b','llama31-8b']:
        raise InferenceError('se requieren Qwen y Llama en el orden acordado')
    ids = ['Qwen/Qwen3.5-9B','meta-llama/Llama-3.1-8B-Instruct']
    for model, expected in zip(models, ids):
        if model.get('id') != expected or model.get('tokenizer_id') != expected:
            raise InferenceError('modelo/tokenizer fuera de la campaña')
        for key in ('revision','tokenizer_revision'):
            if not SHA40.fullmatch(model.get(key, '')):
                raise InferenceError(f'{key} debe ser SHA inmutable')
        if model['revision'] != model['tokenizer_revision'] or model.get('enable_thinking') is not False:
            raise InferenceError('tokenizer pareado y thinking desactivado requeridos')
    if not models[0].get('language_model_only') or models[1].get('language_model_only'):
        raise InferenceError('language-model-only requerido exclusivamente para Qwen')
    image = config['image']
    if not SHA256.fullmatch(image['id']) or image['architecture'] != 'arm64':
        raise InferenceError('imagen ARM64 por ID SHA256 requerida')
    for path in config['paths'].values():
        p = Path(path)
        if not p.is_absolute() or '..' in p.parts or '\x00' in path:
            raise InferenceError('rutas absolutas sin traversal requeridas')
    r, w, m = config['runtime'], config['workload'], config['measurement']
    expected_runtime = {'dtype':'bfloat16','quantization':None,'max_model_len':4096,
        'max_num_seqs':8,'max_num_batched_tokens':4096,'gpu_memory_utilization':0.70,
        'ray_object_store_bytes':1073741824,'pipeline_parallel_size':1,
        'prefix_caching':False,'speculative_decoding':False}
    if any(r.get(k) != v for k,v in expected_runtime.items()):
        raise InferenceError('runtime fuera del protocolo BF16 aprobado')
    expected_workload = {'seed':20261106,'input_tokens':[512,2048], 'output_tokens':128,
        'concurrencies':[1,4,8],'temperature':0,'ignore_eos':True}
    if w != expected_workload:
        raise InferenceError('carga fuera del protocolo aprobado')
    expected_measurement = {'blocks':3,'campaign_order':['four','pair'], 'warmup_requests':16,
        'measured_requests':32,'telemetry_interval_seconds':1,'session_limit_seconds':28800,
        'startup_limit_seconds':900,'cell_limit_seconds':900,'cleanup_reserve_seconds':300}
    if m != expected_measurement:
        raise InferenceError('medición fuera de la campaña de ocho horas aprobada')
    ports = runtime.used_ports(config)
    if any(type(p) is not int or not 1024 <= p <= 65535 for p in ports) or len(ports) != len(set(ports)):
        raise InferenceError('puertos reservados inválidos o duplicados')
    if r['ray_worker_port_max']-r['ray_worker_port_min'] != 99:
        raise InferenceError('se requiere intervalo de 100 puertos para workers Ray')
    expected_environment = {'NCCL_DEBUG':'INFO','NCCL_DEBUG_SUBSYS':'INIT,BOOTSTRAP,NET,GRAPH',
        'NCCL_SOCKET_IFNAME':'=enp1s0f0np0',
        'NCCL_IB_HCA':'=rocep1s0f0:1,roceP2p1s0f0:1','GLOO_SOCKET_IFNAME':'enp1s0f0np0'}
    if config['network']['environment'] != expected_environment:
        raise InferenceError('selectores f0 fuera del protocolo')
    ips = config['network']['node_ips']
    if set(ips) != set(cluster_config['groups']['four']):
        raise InferenceError('IP requerida por cada nodo')
    for node in cluster_config['nodes']:
        if ips[node['node_id']] not in node['data_addresses'] or not ips[node['node_id']].startswith('10.10.0.'):
            raise InferenceError('IP Ray debe pertenecer al f0 inventariado')


def load_config(path=DEFAULT_CONFIG):
    config = read_json(path)
    cluster_path = Path(config['cluster_config'])
    if not cluster_path.is_absolute(): cluster_path = REPO_ROOT / cluster_path
    cluster_config = cluster.load_config(cluster_path)
    validate_config(config, cluster_config)
    return config, cluster_config, cluster_path


def jobs(config):
    for block in range(1, config['measurement']['blocks']+1):
        for model in config['models']:
            for group in config['measurement']['campaign_order']:
                yield {'name':f"b{block}-{model['key']}-{group}", 'block':block,
                       'model':model, 'group':group,
                       'cells':[{'input_tokens':length,'concurrency':concurrency}
                                for length in config['workload']['input_tokens']
                                for concurrency in config['workload']['concurrencies']]}


def make_plan(config_path=DEFAULT_CONFIG):
    config, cluster_config, _ = load_config(config_path)
    preview = []
    for job in jobs(config):
        nodes = cluster.selected_nodes(cluster_config, job['group'])
        preview.append({**job, 'nodes':[n['node_id'] for n in nodes],
            'ray_commands':{n['node_id']:runtime.ray_command(n, config) for n in nodes},
            'server_command':runtime.server_command(config,job['model'],len(nodes)),
            'cells':[{**cell, 'commands':{phase:runtime.bench_command(config,job['model'],
                     cell['input_tokens'],cell['concurrency'],phase,
                     f"/work/metrics/{cell['input_tokens']}-c{cell['concurrency']}")
                     for phase in ('warmup','measured')}} for cell in job['cells']]})
    return {'status':'planned_only','protocol':config,'jobs':preview,
            'resources_max':{'nodes':4,'gpus':4,'replicas':1},
            'job_count':12,'cell_count':72,'measured_requests':2304,'warmup_requests':1152,
            'maximum_occupation_seconds':28800,
            'preflight_commands':{n['node_id']:cluster.ssh_command(n,runtime.probe_command(n,config,container_probe=True))
                                  for n in cluster_config['nodes']},
            'docker_policy':{'pull':'never','user':'UID:GID observados no root','network':'host',
                'shm_size':'8g','models_mount':'read-only','work_mount':'nuevo por corrida',
                'cleanup':'solo IDs creados por esta campaña con etiqueta verificada'},
            'notice':'plan no abre conexiones, inicia contenedores ni escribe resultados'}


STAGE = r'''
import base64,json,sys
from pathlib import Path
request=json.load(sys.stdin); root=Path(request['root'])
root.mkdir(parents=True,exist_ok=False)
for name in ('cache','evidence','metrics','logs','workloads','ray'):(root/name).mkdir()
for relative, encoded in request['files'].items():
 p=Path(relative)
 if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe stage path')
 dest=root/p;dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('xb') as f:f.write(base64.b64decode(encoded,validate=True))
print(json.dumps({'workspace':str(root),'files':len(request['files'])}))
'''

COLLECT = r'''
import base64,json,sys
from pathlib import Path
root=Path(sys.argv[1]); out={}
for subdir in ('evidence','metrics','logs'):
 for p in (root/subdir).rglob('*'):
  if p.is_file() and not p.is_symlink():
   if p.stat().st_size>33554432:raise ValueError('artifact exceeds32MiB')
   out[str(p.relative_to(root))]=base64.b64encode(p.read_bytes()).decode()
for p in (root/'ray').rglob('*'):
 if p.is_file() and not p.is_symlink() and 'logs' in p.relative_to(root/'ray').parts and p.suffix in ('.log','.out','.err'):
  if p.stat().st_size>33554432:raise ValueError('Ray log exceeds32MiB')
  out[str(p.relative_to(root))]=base64.b64encode(p.read_bytes()).decode()
print(json.dumps(out))
'''


def source_files():
    return {f'code/infraestructura/{path.name}':path.read_bytes()
            for path in sorted((REPO_ROOT/'infraestructura').glob('*.py'))}


def stage(node, workspace, files, *, timeout=60):
    request = {'root':workspace, 'files':{key:base64.b64encode(value).decode() for key,value in files.items()}}
    return runtime.checked(runtime.remote(node, ['python3','-c',STAGE],
        stdin=json.dumps(request),timeout=timeout),'preparar workspace propio')


def container_command(node, config, probe, workspace, name, inner, *, gpu=True, seconds=900, detached=False):
    argv = runtime.docker_base(node,config,probe,workspace,name,gpu=gpu)
    argv += ['--hostname',node['expected_hostname']]
    if detached: argv += ['--detach']
    # timeout inside the container bounds its entire process namespace lifetime,
    # even if the controller or SSH is lost. Stopping PID1 ends only this job.
    return argv + ['--entrypoint','timeout',config['image']['id'],
                   '--signal=TERM','--kill-after=10s',f'{max(1,int(seconds))}s',*inner]


def _http(port, path, data=None, timeout=5):
    payload = None if data is None else json.dumps(data).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=payload,
                                    headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=timeout) as response:
        return json.load(response)


def wait_ready(port, model_key, deadline, check_running=None):
    next_check=0
    while time.monotonic() < deadline:
        if check_running is not None and time.monotonic()>=next_check:
            check_running()
            next_check=time.monotonic()+5
        try:
            models = _http(port,'/v1/models',timeout=min(3,max(.1,deadline-time.monotonic())))
            if [m['id'] for m in models.get('data',[])] == [model_key]: return
        except (OSError, ValueError, KeyError): pass
        time.sleep(min(1,max(0,deadline-time.monotonic())))
    raise TimeoutError('vLLM no estuvo listo dentro del plazo de arranque')


def remaining(deadline, cap):
    left=deadline-time.monotonic()
    if left <= 1: raise TimeoutError('plazo operativo agotado')
    return min(cap,left)


def ensure_running(node, name, deadline):
    result=runtime.checked(runtime.remote(node,['docker','inspect','--type','container',name,
        '--format','{{.State.Running}} {{.State.ExitCode}}'],timeout=remaining(deadline,10)),
        'verificar proceso propio de arranque')
    fields=result['stdout'].strip().split()
    if len(fields)!=2 or fields[0]!='true':
        raise InferenceError(f'contenedor propio {name} terminó antes de estar listo: {result["stdout"].strip()}')


def wait_ray_head(config, deadline, check_running=None):
    address=(config['network']['node_ips']['pegasus'],config['runtime']['ray_port'])
    next_check=0
    while time.monotonic()<deadline:
        if check_running is not None and time.monotonic()>=next_check:
            check_running()
            next_check=time.monotonic()+5
        try:
            with socket.create_connection(address,timeout=remaining(deadline,2)): return
        except OSError: time.sleep(min(.5,max(0,deadline-time.monotonic())))
    raise TimeoutError('GCS de Ray no estuvo accesible en el plazo de arranque')


def cleanup_container(node, name, directory):
    """Inspect the exact name and label before touching a campaign process."""
    check = runtime.remote(node,['docker','inspect','--type','container',name,'--format',
        '{{.Id}} {{index .Config.Labels "io.ipre.inference"}}'],timeout=15)
    if check['returncode'] != 0:
        if 'No such container' in check['stderr'] or 'No such object' in check['stderr']:
            return {'name':name,'status':'confirmed_absent','inspection':check}
        raise InferenceError('no se pudo verificar estado de contenedor propio para limpieza')
    parts = check['stdout'].strip().split()
    if len(parts) != 2 or parts[1] != name or not re.fullmatch('[0-9a-f]{64}',parts[0]):
        raise InferenceError('rechazo limpieza de contenedor sin identidad propia')
    container_id = parts[0]
    logs = runtime.remote(node,['docker','logs',container_id],timeout=30)
    write_new(directory / f'{name}-container-log.json', logs)
    stopped = runtime.remote(node,['docker','stop','--time','10',container_id],timeout=20)
    removed = runtime.remote(node,['docker','rm',container_id],timeout=15)
    runtime.checked(stopped,'detener contenedor propio')
    runtime.checked(removed,'retirar contenedor propio')
    absence=runtime.remote(node,['docker','inspect','--type','container',container_id,'--format','{{.Id}}'],timeout=15)
    if absence['returncode']==0 or not ('No such object' in absence['stderr'] or 'No such container' in absence['stderr']):
        raise InferenceError('ausencia de contenedor propio no confirmada después de cierre')
    return {'name':name,'id':container_id,'stop':stopped,'remove':removed}


def collect(node, workspace, directory):
    result = runtime.checked(runtime.remote(node,['python3','-c',COLLECT,workspace],timeout=45),
                             'recoger evidencia')
    payload = json.loads(result['stdout'])
    for relative, encoded in payload.items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise InferenceError('ruta de evidencia externa inválida')
        target = directory / node['node_id'] / path
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream: stream.write(base64.b64decode(encoded,validate=True))


def prepare_workloads(config, cluster_config, preflight_result, campaign_id):
    """Prepare local tokenized data before the GPU occupation clock starts."""
    head = cluster.selected_nodes(cluster_config,'single')[0]
    probe = next(n for n in preflight_result['nodes'] if n['node_id']=='pegasus')
    workspace = str(Path(config['paths']['workspace'])/campaign_id/'prepare')
    root=Path(config['paths']['workspace'])/campaign_id
    intents=[]; launched=[]; results=[]; cleanup=[]; errors=[]; failed=False
    for model in config['models']:
        for length in config['workload']['input_tokens']:
            name=f'ipre-prep-{campaign_id[-8:]}-{model["key"]}-{length}'
            inner=['python3','-m','infraestructura.inference_workload','--model-path',runtime.model_path(model),
                '--model-id',model['id'],'--revision',model['revision'],'--input-tokens',str(length),
                '--output',f"/work/workloads/{model['key']}/{length}", '--seed',str(config['workload']['seed'])]
            command=container_command(head,config,probe,workspace,name,inner,gpu=False,seconds=100)
            command.insert(2,'--rm')
            intents.append({'name':name,'node_id':head['node_id'],'command':command,'timeout_seconds':110})
    write_new(root/'preparation-intent.json',{'campaign_id':campaign_id,'gpu':False,
        'watchdog_seconds':100,'stage_workspace':workspace,'commands':intents})
    try:
        stage(head,workspace,source_files())
        for item in intents:
            launched.append((head,item['name']))
            result=runtime.remote(head,item['command'],timeout=item['timeout_seconds'])
            results.append({'name':item['name'],'result':result})
            runtime.checked(result,'generar datos con tokenizer real')
    except (Exception,KeyboardInterrupt) as exc:
        failed=True
        errors.append(f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for node,name in launched:
            try: cleanup.append(cleanup_container(node,name,root/'logs'))
            except Exception as exc: errors.append(f'cleanup {name}: {exc}')
        write_new(root/'preparation-closure.json',{'status':'failed' if errors else 'complete',
            'results':results,'cleanup':cleanup,'errors':errors})
        if errors and not failed:
            raise InferenceError('limpieza de preparación no confirmada: '+'; '.join(errors))
    return Path(workspace)/'workloads'


def _job_shell(config, job, node, lifetime):
    ray = shlex.join(runtime.ray_command(node,config))
    telemetry = shlex.join(['python3','-m','infraestructura.inference_observation','telemetry',
                           '--seconds',str(lifetime),'--interval','1'])
    # Quoted argv assembled by us, not arbitrary shell from a config file.
    steps = ['set -eu',f'{telemetry} > /work/logs/telemetry.jsonl 2> /work/logs/telemetry.stderr &',
             f'{ray} > /work/logs/ray-start.log 2>&1']
    if node['node_id'] == 'pegasus':
        world = 4 if job['group']=='four' else 2
        expected_ips=[config['network']['node_ips'][key] for key in
                      (['pegasus','octopus','griffin','draggon'] if world==4 else ['pegasus','octopus'])]
        barrier=("import json,ray,time; ray.init(address="+repr(f"{config['network']['node_ips']['pegasus']}:{config['runtime']['ray_port']}")+
                 "); end=time.monotonic()+"+str(config['measurement']['startup_limit_seconds'])+
                 "; expected=set("+repr(expected_ips)+");\nwhile time.monotonic()<end:\n"
                 " nodes=[n for n in ray.nodes() if n['Alive']];\n"
                 " if {n['NodeManagerAddress'] for n in nodes}==expected and all(n['Resources'].get('GPU')==1 for n in nodes):break\n"
                 " time.sleep(.5)\nelse:raise TimeoutError('membresia Ray incompleta')\n"
                 "print(json.dumps(nodes));ray.shutdown()")
        steps.append(shlex.join(['python3','-c',barrier])+' > /work/logs/ray-membership.json 2> /work/logs/ray-membership.stderr')
        steps.append('exec '+shlex.join(runtime.server_command(config,job['model'],world))+
                     ' > /work/logs/server.log 2>&1')
    else:
        # Ray subprocesses stay in this container's PID namespace.
        steps.append(f'sleep {int(lifetime)}')
    return ['/bin/sh','-c','\n'.join(steps)]


def _rank_records(nodes, workspace, deadline):
    records=[]
    code = "import json,sys;from pathlib import Path;print(json.dumps([json.loads(p.read_text()) for p in Path(sys.argv[1]).glob('load_model-rank*.json')]))"
    for node in nodes:
        result = runtime.checked(runtime.remote(node,['python3','-c',code,workspace+'/evidence'],timeout=remaining(deadline,15)),'leer ranks')
        records.extend(json.loads(result['stdout']))
    return records


def _telemetry_valid(directory, nodes, cells):
    errors=[]
    for node in nodes:
        path=directory/'raw'/'nodes'/node['node_id']/'logs'/'telemetry.jsonl'
        if not path.is_file():
            errors.append(f"{node['node_id']}: telemetría ausente"); continue
        try: samples=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        except ValueError:
            errors.append(f"{node['node_id']}: telemetría inválida"); continue
        if len(samples)<2 or any(s.get('hostname')!=node['expected_hostname'] for s in samples):
            errors.append(f"{node['node_id']}: cobertura/identidad de telemetría insuficiente")
        if not all(s.get('gpus') and s.get('memory',{}).get('MemAvailable_kib') is not None for s in samples):
            errors.append(f"{node['node_id']}: métricas GPU/memoria ausentes")
        intervals=[(b['monotonic_ns']-a['monotonic_ns'])/1e9 for a,b in zip(samples,samples[1:])]
        if intervals and max(intervals)>10:
            errors.append(f"{node['node_id']}: hueco de telemetría mayor a10s")
        try:
            stamps=[datetime.fromisoformat(s['timestamp_utc']).timestamp() for s in samples]
            for cell in cells:
                start=datetime.fromisoformat(cell['measurement_started_at']).timestamp()
                end=datetime.fromisoformat(cell['measurement_ended_at']).timestamp()
                # Allow one sample period plus bounded sampler/query jitter.
                if not stamps or min(stamps)>start+3 or max(stamps)<end-3:
                    errors.append(f"{node['node_id']}: telemetría no cubre {cell['cell']}")
                in_window=[s for s,t in zip(samples,stamps) if start-3<=t<=end+3]
                if not in_window:
                    errors.append(f"{node['node_id']}: sin muestras durante {cell['cell']}")
                if any(s['memory'].get('MemAvailable_kib',0)<8388608 for s in in_window):
                    errors.append(f"{node['node_id']}: presión de memoria en {cell['cell']}")
        except (ValueError,KeyError,TypeError):
            errors.append(f"{node['node_id']}: marcas temporales de telemetría inválidas")
    return errors


WORKER_LOGS = r'''
import json,sys
from pathlib import Path
request=json.load(sys.stdin);root=Path(request['root']);offsets=request.get('offsets');out={}
for p in sorted((root/'logs').glob('worker-rank*.log')):
 name=p.name
 if offsets is None:out[name]=p.stat().st_size
 else:
  if name not in offsets or p.stat().st_size<offsets[name]:raise ValueError('worker log changed')
  with p.open('rb') as f:
   f.seek(offsets[name]);out[name]=f.read(8388609).decode(errors='replace')
  if len(out[name].encode())>8388608:raise ValueError('worker delta too large')
if len(out)!=1:raise ValueError('expected one observed rank log per node')
print(json.dumps(out))
'''


def worker_log_window(nodes, workspace, deadline, offsets=None):
    data={}
    for node in nodes:
        request={'root':workspace,'offsets':None if offsets is None else offsets[node['node_id']]}
        result=runtime.checked(runtime.remote(node,['python3','-c',WORKER_LOGS],stdin=json.dumps(request),
                                              timeout=remaining(deadline,15)),'leer ventana de logs de rank')
        data[node['node_id']]=json.loads(result['stdout'])
    return data


def run_job(config, cluster_config, config_path, cluster_path, job, campaign_id,
            preflight_result, workloads, deadline, results_root):
    model=job['model']; nodes=cluster.selected_nodes(cluster_config,job['group'])
    request={'protocol':config,'job':job,'campaign_id':campaign_id}
    inputs={'cluster':cluster_path,'protocol':config_path}
    inputs.update({name:REPO_ROOT/name for name in [
        'infraestructura/inference.py','infraestructura/inference_runtime.py',
        'infraestructura/inference_worker.py','infraestructura/inference_workload.py',
        'infraestructura/inference_metrics.py','infraestructura/inference_observation.py']})
    for path in sorted(workloads.rglob('*')):
        if path.is_file(): inputs['workload/'+str(path.relative_to(workloads))]=path
    directory=manifest.start_run(kind='inference-serving',model_id=model['id'],model_revision=model['revision'],
        tokenizer_id=model['tokenizer_id'],tokenizer_revision=model['tokenizer_revision'],
        requested_config=request,requested_config_path=config_path,inputs=inputs,
        repo_path=REPO_ROOT,results_root=results_root)
    print(json.dumps({'event':'job_start','job':job['name'],'run_dir':str(directory)}),flush=True)
    workspace=str(Path(config['paths']['workspace'])/campaign_id/job['name'])
    files={}
    created=[]; active_clients=[]; errors=[]; cells=[]; cleanup=[]; launch=[]
    probe_by_id={n['node_id']:n for n in preflight_result['nodes']}
    started=time.monotonic(); startup_deadline=min(deadline,started+config['measurement']['startup_limit_seconds'])
    try:
        for name,path in inputs.items():
            target=directory/'raw'/'inputs'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream: stream.write(Path(path).read_bytes())
        files=source_files()
        files.update({'workloads/'+str(p.relative_to(workloads)):p.read_bytes()
                      for p in workloads.rglob('*') if p.is_file()})
        write_new(directory/'raw'/'preflight.json',preflight_result)
        write_new(directory/'raw'/'requested-job.json',request)
        # Launch head Ray first; vLLM waits for resources while workers join.
        for node in nodes:
            if time.monotonic()>=startup_deadline: raise TimeoutError('plazo de arranque agotado')
            stage(node,workspace,files,timeout=remaining(startup_deadline,60))
            name=f"ipre-{campaign_id[-8:]}-{job['name']}-{node['node_id']}"
            lifetime=max(1,int(deadline-time.monotonic()))
            argv=container_command(node,config,probe_by_id[node['node_id']],workspace,name,
                _job_shell(config,job,node,lifetime),seconds=lifetime,detached=True)
            launch.append({'node_id':node['node_id'],'command':argv})
            created.append((node,name))
            result=runtime.remote(node,argv,timeout=min(45,max(5,startup_deadline-time.monotonic())))
            # Record potential own container even on a lost SSH reply. A cleanup
            # checks its label and ID, so collisions never stop someone else's.
            runtime.checked(result,'arrancar contenedor Ray/vLLM')
            if node['node_id']=='pegasus':
                head_name=name
                check_running=lambda:ensure_running(nodes[0],head_name,startup_deadline)
                wait_ray_head(config,startup_deadline,check_running)
        wait_ready(config['runtime']['api_port'],model['key'],startup_deadline,check_running)
        ranks=validate_worker_records(_rank_records(nodes,workspace,startup_deadline),
             {n['node_id']:n for n in nodes},len(nodes),runtime.model_path(model))
        for rank in ranks:
            observed=probe_by_id[rank['node_id']]
            if rank['gpu_uuid']!=observed['gpu']['stdout'].split(',')[0].strip():
                raise InferenceError('UUID de rank difiere del preflight')
            versions=json.loads(observed['container_probe']['stdout'].splitlines()[-1])['packages']
            if any(rank['versions'].get(k)!=versions[k] for k in ('vllm','torch','ray')):
                raise InferenceError('versiones efectivas de worker difieren del preflight')
        participants=[]
        for rank in ranks:
            relative=f"raw/participants/rank-{rank['rank']}.json"
            write_new(directory/relative,rank)
            participants.append({k:rank[k] for k in ('node_id','hostname','gpu_uuid','rank')}|
                                {'evidence_path':relative})
        manifest.register_participants(directory,participants)
        write_new(directory/'raw'/'startup.json',{'seconds':time.monotonic()-started,'ranks':ranks})
        # Server-side tokenization must match the persisted client dataset;
        # the benchmark's input_lens alone are not independent proof.
        for length in config['workload']['input_tokens']:
            metadata=read_json(workloads/model['key']/str(length)/'metadata.json')
            prompts=[]
            for phase in ('warmup','measured'):
                prompts.extend(json.loads(line) for line in (workloads/model['key']/str(length)/(phase+'.jsonl')).read_text().splitlines())
            for row,prompt in zip(metadata['rows'],prompts):
                response=_http(config['runtime']['api_port'],'/tokenize',
                    {'model':model['key'],'prompt':prompt['prompt'],'add_special_tokens':True},
                    timeout=remaining(startup_deadline,5))
                if response.get('tokens')!=row['token_ids']:
                    raise InferenceError('tokenización efectiva del servidor difiere del dataset')
        write_new(directory/'raw'/'tokenization-verified.json',{'verified':True,'profiles':[512,2048]})
        for cell in job['cells']:
            if time.monotonic()>=deadline: raise TimeoutError('presupuesto global agotado')
            cell_deadline=min(deadline,time.monotonic()+config['measurement']['cell_limit_seconds'])
            cell_name=f"{cell['input_tokens']}-c{cell['concurrency']}"
            record={**cell,'cell':cell_name,'status':'running','started_at':manifest.utc_now()}
            # Logs before and after the measured phase distinguish lazy compile
            # from setup. The warmup is never silently extended.
            for phase in ('warmup','measured'):
                left=cell_deadline-time.monotonic()
                if left<=3: raise TimeoutError('plazo de celda agotado')
                name=f"ipre-{campaign_id[-8:]}-{job['name']}-{cell_name}-{phase}"
                cmd=container_command(nodes[0],config,probe_by_id['pegasus'],workspace,name,
                    runtime.bench_command(config,model,cell['input_tokens'],cell['concurrency'],phase,
                                          '/work/metrics/'+cell_name),gpu=False,seconds=left-2)
                cmd.insert(2,'--rm')
                log_path=Path(workspace)/'logs'/'server.log'
                log_offset=log_path.stat().st_size if log_path.exists() else 0
                offsets=worker_log_window(nodes,workspace,cell_deadline) if phase=='measured' else None
                phase_start=manifest.utc_now()
                active_clients.append((nodes[0],name))
                result=runtime.remote(nodes[0],cmd,timeout=remaining(cell_deadline,left))
                write_new(directory/'logs'/f'{cell_name}-{phase}.json',result)
                if result['returncode']!=0 or result.get('timed_out'):
                    # --rm removes normal CPU clients; a lost Docker CLI needs
                    # explicit, label-checked cleanup of its bounded container.
                    cleanup.append(cleanup_container(nodes[0],name,directory/'logs'))
                active_clients.remove((nodes[0],name))
                runtime.checked(result,f'{cell_name} {phase}')
                raw=read_json(Path(workspace)/'metrics'/cell_name/(phase+'.json'))
                summary=summarize_bench_result(raw,expected_prompt_tokens=cell['input_tokens'],
                     expected_count=config['measurement']['warmup_requests' if phase=='warmup' else 'measured_requests'])
                if phase=='measured':
                    with log_path.open('rb') as stream:
                        stream.seek(log_offset); new_log=stream.read().decode(errors='replace')
                    if re.search(r'(?i)(compil(?:ing|ation)|captur(?:ing|e).*cuda.*graph)',new_log):
                        raise InferenceError('compilación observada en la ventana medida')
                    deltas=worker_log_window(nodes,workspace,cell_deadline,offsets)
                    write_new(directory/'logs'/f'{cell_name}-worker-deltas.json',deltas)
                    if any(re.search(r'(?i)(compil(?:ing|ation)|captur(?:ing|e).*cuda.*graph)',text)
                           for logs in deltas.values() for text in logs.values()):
                        raise InferenceError('compilación de rank observada en la ventana medida')
                    record.update(summary)
                    record['measurement_started_at']=phase_start
                    record['measurement_ended_at']=manifest.utc_now()
            record.update(status='valid',ended_at=manifest.utc_now())
            cells.append(record)
            write_new(directory/'metrics'/(cell_name+'.json'),record)
            print(json.dumps({'event':'cell_complete','job':job['name'],'cell':cell_name,
                              't32_seconds':record['t32_seconds']}),flush=True)
    except (Exception,KeyboardInterrupt) as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    finally:
        write_new(directory/'raw'/'launch.json',launch)
        for node,name in active_clients:
            try: cleanup.append(cleanup_container(node,name,directory/'logs'))
            except Exception as exc: errors.append(f"cleanup client {name}: {exc}")
        def close_node(item):
            node,name=item
            node_errors=[]; result=None
            try: result=cleanup_container(node,name,directory/'logs')
            except Exception as exc: node_errors.append(f"cleanup {node['node_id']}: {exc}")
            try: collect(node,workspace,directory/'raw'/'nodes')
            except Exception as exc: node_errors.append(f"collect {node['node_id']}: {exc}")
            return result,node_errors
        # Each host has <=140s of bounded cleanup and collection; parallel hosts
        # fit inside the reserved300s instead of multiplying by four.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for result,node_errors in pool.map(close_node,reversed(created)):
                if result: cleanup.append(result)
                errors.extend(node_errors)
        errors.extend(_telemetry_valid(directory,nodes,cells))
        if len(cells)!=len(job['cells']): errors.append('matriz de seis celdas incompleta')
        write_new(directory/'raw'/'cleanup.json',cleanup)
        write_new(directory/'metrics'/'assessment.json',{'valid':not errors,'errors':errors,'cells':cells})
        manifest.finalize_run(directory,status='failed' if errors else 'complete',errors=errors)
    return {'job':job['name'],'group':job['group'],'block':job['block'],'model_key':model['key'],
            'run_dir':str(directory),'status':'failed' if errors else 'complete',
            'manifest_sha256':manifest.sha256_file(directory/'manifest.json'),'errors':errors}


def run_campaign(config_path=DEFAULT_CONFIG, *, results_root=REPO_ROOT/'resultados/raw', session_limit_seconds=None):
    config,cluster_config,cluster_path=load_config(config_path)
    maximum=config['measurement']['session_limit_seconds']
    effective_limit=maximum if session_limit_seconds is None else session_limit_seconds
    if (type(effective_limit) is not int or
            not config['measurement']['cleanup_reserve_seconds']<effective_limit<=maximum):
        raise InferenceError('el límite efectivo solo puede reducir las ocho horas y debe reservar el cierre')
    check=runtime.preflight(cluster_config,config,container_probe=True)
    preflight_path=REPO_ROOT/'resultados/tmp'/f'inference-preflight-{manifest.generate_run_id()}.json'
    write_new(preflight_path,check)
    if check['errors']:
        raise InferenceError(f"preflight bloqueado; {preflight_path}: "+'; '.join(check['errors']))
    git=manifest.capture_git(REPO_ROOT)
    if not git['commit'] or git['dirty']:
        raise InferenceError('campaña requiere commit identificable y árbol limpio; no se usará allow-dirty')
    campaign_id=manifest.generate_run_id()
    workloads=prepare_workloads(config,cluster_config,check,campaign_id)
    directory=Path(results_root)/'inference-campaign'/campaign_id
    directory.mkdir(parents=True,exist_ok=False)
    write_new(directory/'plan.json',make_plan(config_path))
    write_new(directory/'preflight.json',check)
    started=time.monotonic()
    deadline=started+effective_limit-config['measurement']['cleanup_reserve_seconds']
    records=[]; errors=[]
    try:
        for job in jobs(config):
            if time.monotonic()>=deadline:
                errors.append('presupuesto global agotado'); break
            record=run_job(config,cluster_config,Path(config_path).resolve(),cluster_path,job,campaign_id,
                           check,workloads,deadline,results_root)
            records.append(record)
            if record['status']!='complete':
                errors.append('sesión detenida tras fallo; sin reintentos automáticos'); break
    except (Exception,KeyboardInterrupt) as exc: errors.append(f'{type(exc).__name__}: {exc}')
    status='complete' if len(records)==12 and not errors else 'partial'
    campaign={'campaign_id':campaign_id,'status':status,'git':git,'protocol':config,
        'duration_seconds':time.monotonic()-started,'effective_session_limit_seconds':effective_limit,
        'runs':records,'errors':errors,
        'pending_jobs':[j['name'] for j in jobs(config) if j['name'] not in {r['job'] for r in records}]}
    write_new(directory/'campaign.json',campaign)
    write_new(directory/('COMPLETE' if status=='complete' else 'PARTIAL'),
              {'campaign_sha256':manifest.sha256_file(directory/'campaign.json'),
               'plan_sha256':manifest.sha256_file(directory/'plan.json'),
               'preflight_sha256':manifest.sha256_file(directory/'preflight.json')})
    print(json.dumps({'event':'campaign_closed','directory':str(directory),'status':status}),flush=True)
    return directory


def build_report(directory):
    directory=Path(directory).resolve()
    campaign=read_json(directory/'campaign.json')
    markers=[directory/name for name in ('COMPLETE','PARTIAL') if (directory/name).is_file()]
    if len(markers)!=1: raise InferenceError('campaña abierta o cierre ambiguo')
    marker=read_json(markers[0])
    for key,name in [('campaign_sha256','campaign.json'),('plan_sha256','plan.json'),('preflight_sha256','preflight.json')]:
        if marker[key]!=manifest.sha256_file(directory/name): raise InferenceError('integridad de campaña inválida')
    pairs={}; rows=[]; seen_jobs=set(); seen_slots=set(); seen_paths=set()
    for ref in campaign['runs']:
        job_name=ref['job']; slot=(ref['model_key'],ref['block'],ref['group'])
        run_path=Path(ref['run_dir']).resolve()
        if job_name in seen_jobs or slot in seen_slots or run_path in seen_paths:
            raise InferenceError('referencia de job, grupo o corrida duplicada en campaña')
        seen_jobs.add(job_name); seen_slots.add(slot); seen_paths.add(run_path)
        path=Path(ref['run_dir']); run=manifest.load_run(path); manifest.verify_artifacts(path,run)
        if ref['manifest_sha256']!=manifest.sha256_file(path/'manifest.json'):
            raise InferenceError('manifiesto cambió desde cierre de campaña')
        request=run['requested_config']; job=request['job']; model=run['model']
        if request.get('campaign_id')!=campaign['campaign_id']:
            raise InferenceError('corrida pertenece a otra campaña')
        if (ref['job']!=job['name'] or ref['group']!=job['group'] or ref['block']!=job['block']
                or ref['model_key']!=job['model']['key'] or model['id']!=job['model']['id']
                or model['revision']!=job['model']['revision']):
            raise InferenceError('índice de campaña contradice la identidad de la corrida')
        assessment=read_json(path/'metrics'/'assessment.json')
        if run['status']!='complete' or not assessment['valid']: continue
        world=4 if job['group']=='four' else 2
        if len(run['participants'])!=world or {p['rank'] for p in run['participants']}!=set(range(world)):
            raise InferenceError('roster observado no corresponde a la topología comparada')
        for cell in assessment['cells']:
            if cell['status']!='valid': continue
            # Identity includes source commit, protocol, model/tokenizer revision,
            # exact datasets and the trial. Node count alone differs.
            identity={'commit':run['git']['commit'],'protocol':request['protocol'],'model':model,
                'block':job['block'],'input_tokens':cell['input_tokens'],'concurrency':cell['concurrency'],
                'workload_hashes':{i['name']:i['sha256'] for i in run['inputs'] if i['name'].startswith('workload/')},
                'effective_environment':json.loads(read_json(path/'raw'/'preflight.json')['nodes'][0]['container_probe']['stdout'].splitlines()[-1]),
                'snapshot_identity':read_json(path/'raw'/'preflight.json')['nodes'][0]['models'][job['model']['key']]}
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            summary={**cell,'comparison_key':key,'model_key':ref['model_key'],'block':ref['block'],
                     'group':ref['group'],'run_dir':str(path)}
            group_pairs=pairs.setdefault(key,{})
            if ref['group'] in group_pairs:
                raise InferenceError('celda de grupo duplicada en comparación pareada')
            group_pairs[ref['group']]=summary; rows.append(summary)
    comparisons=[]
    for group in pairs.values():
        if set(group)=={'four','pair'}:
            four,two=group['four'],group['pair']
            comparisons.append({'model':four['model_key'],'block':four['block'],
                'input_tokens':four['input_tokens'],'concurrency':four['concurrency'],
                **paired_speedup(two,four)})
    return {'campaign_id':campaign['campaign_id'],'status':campaign['status'],
            'pending_jobs':campaign['pending_jobs'],'runs':campaign['runs'],'cells':rows,
            'paired_comparisons':comparisons,'limitations':campaign['protocol']['limitations']+
                ['p50/p95 descriptivos con32solicitudes por celda; no inferir p99 poblacional.',
                 'Cliente en Pegasus compartido con servicio; saturación del cliente no descartada solo por benchmark JSON.']}


def markdown_report(report):
    lines=[f"# Inferencia {report['campaign_id']}",'',f"Estado: **{report['status']}**.",'',
           '## Rendimiento y latencias','',
           'Cada fila resume bloques válidos de un mismo modelo y carga. Se muestra la mediana '
           'entre bloques; los percentiles son medianas de percentiles por bloque, no percentiles '
           'de solicitudes agrupadas. El JSON conserva cada celda y sus conteos.','',
           '| Modelo | Nodos | Entrada | Concurrencia | Bloques | T32 s (mín–máx) | Salida tokens/s (mín–máx) |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    grouped={}
    for cell in report['cells']:
        key=(cell['model_key'],cell['group'],cell['input_tokens'],cell['concurrency'])
        grouped.setdefault(key,[]).append(cell)
    def spread(values):
        return f'{statistics.median(values):.3f} ({min(values):.3f}–{max(values):.3f})'
    for (model,group,length,concurrency),cells in sorted(grouped.items()):
        lines.append(f"| {model} | {4 if group=='four' else 2} | {length} | {concurrency} | {len(cells)} | "
                     f"{spread([c['t32_seconds'] for c in cells])} | "
                     f"{spread([c['output_tokens_per_second'] for c in cells])} |")
    lines += ['','| Modelo | Nodos | Entrada | Concurrencia | TTFT p50/p95 ms | TPOT p50/p95 ms | E2E p50/p95 ms |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for (model,group,length,concurrency),cells in sorted(grouped.items()):
        values=[' / '.join(f"{statistics.median(c[metric][p] for c in cells):.3f}" for p in ('p50_ms','p95_ms'))
                for metric in ('ttft','tpot','e2el')]
        lines.append(f"| {model} | {4 if group=='four' else 2} | {length} | {concurrency} | "+' | '.join(values)+' |')
    if not grouped: lines += ['','No hay celdas válidas para resumir.']
    lines += ['','## Comparaciones pareadas por bloque','',
           '| Modelo | Bloque | Entrada | Concurrencia | S4/2 | E4/2 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in report['paired_comparisons']:
        lines.append(f"| {row['model']} | {row['block']} | {row['input_tokens']} | {row['concurrency']} | {row['s4_2']:.3f} | {row['e4_2']:.3f} |")
    if not report['paired_comparisons']: lines += ['','No hay pares completos y válidos para calcular speedup.']
    lines += ['','## Limitaciones','']+[f'- {s}' for s in report['limitations']]
    lines += ['','## Corridas y evidencia','']
    for run in report['runs']:
        lines.append(f"- {run['job']}: {run['status']}; `{run['run_dir']}`; manifiesto SHA256 `{run['manifest_sha256']}`.")
    if report['pending_jobs']: lines += ['','Pendientes: '+', '.join(report['pending_jobs'])+'.']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('plan','preflight','run'):
        p=sub.add_parser(name); p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
        if name!='run': p.add_argument('--output',type=Path)
        if name=='preflight': p.add_argument('--container-probe',action='store_true',
                  help='sonda CPU efímera de paquetes; sin GPU/red del contenedor')
        if name=='run':
            p.add_argument('--execute',action='store_true',help='ejecutar plan operativo previamente revisado')
            p.add_argument('--session-limit-seconds',type=int,
                help='reducir el presupuesto de ocupación; nunca amplía el máximo aprobado de ocho horas')
    p=sub.add_parser('report');p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot',action='store_true',help='añadir SVG científico con Matplotlib ya instalado')
    args=parser.parse_args(argv)
    try:
        if args.command=='plan': data=make_plan(args.config)
        elif args.command=='preflight':
            config,cluster_config,_=load_config(args.config)
            data=runtime.preflight(cluster_config,config,container_probe=args.container_probe)
        elif args.command=='run':
            if not args.execute: raise InferenceError('use plan y después run --execute para iniciar la campaña')
            def interrupt(_signum,_frame): raise KeyboardInterrupt('signal de parada')
            signal.signal(signal.SIGTERM,interrupt)
            closed=run_campaign(args.config,session_limit_seconds=args.session_limit_seconds)
            return 0 if read_json(closed/'campaign.json')['status']=='complete' else 2
        else:
            data=build_report(args.directory)
            if args.output.resolve().is_relative_to((REPO_ROOT/'resultados/raw').resolve()):
                raise InferenceError('informes derivados deben ir fuera de raw')
            if args.plot:
                from infraestructura.inference_plot import plot_report
                data['plot']=plot_report(data,args.output.with_suffix('.svg'))
            write_new(args.output.with_suffix('.json'),data)
            with args.output.with_suffix('.md').open('x',encoding='utf-8') as stream: stream.write(markdown_report(data))
            return 0
        if args.output: write_new(args.output,data)
        else: print(json.dumps(data,ensure_ascii=False,indent=2))
        return 2 if args.command=='preflight' and data['errors'] else 0
    except (OSError,ValueError,RuntimeError) as exc:
        print(f'inference: {exc}',file=sys.stderr);return 2


if __name__=='__main__':
    raise SystemExit(main())
