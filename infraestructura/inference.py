"""Campañas de inferencia 4/2, 4/2/1 y diagnóstico: plan, preflight, run y report.

El plan no ejecuta comandos. El runner no instala, descarga, modifica red ni
detiene procesos ajenos. Los prerrequisitos ausentes bloquean antes de usar GPU.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
from datetime import datetime
import errno
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
import uuid
from pathlib import Path

from infraestructura import cluster, manifest
from infraestructura import inference_runtime as runtime
from infraestructura.inference_evidence import (RUNNER_SOURCES, archived_job_evidence,
    verify_client_cell, runner_sources, cell_id, DIAGNOSIS_PROTOCOL, diagnostic_cell_issues,
    SERVING_CAPACITY_PROTOCOL)
from infraestructura.inference_metrics import paired_speedup, summarize_bench_result, validate_client_identity
from infraestructura.inference_observation import validate_worker_records
from infraestructura.rutas import REPO_ROOT


DEFAULT_CONFIG = REPO_ROOT / 'experimentos/configs/inferencia-4-2.json'
SAFE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*$')
SHA40 = re.compile(r'^[0-9a-f]{40}$')
SHA256 = re.compile(r'^sha256:[0-9a-f]{64}$')
TOPOLOGIES = cluster.TOPOLOGIES
PEGASUS_REFERENCE_PROTOCOL = 'inference-4-2-1-v1'
PROTOCOL_ORDERS = {'inference-4-2-v1':['four','pair'],
                   PEGASUS_REFERENCE_PROTOCOL:['four','pair','single'],
                   DIAGNOSIS_PROTOCOL:['four','pair','single']}


class InferenceError(ValueError):
    pass


class CleanupError(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__('; '.join(receipt['errors']))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, data):
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.ipre-json-{uuid.uuid4().hex}.tmp')
    # open('x') applies the same 0666 & umask permissions as the historical
    # destination. Publish only a fully flushed JSON, without replacing a name.
    stream = temporary.open('x', encoding='utf-8')
    try:
        with stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_protocol(config):
    if config.get('schema_version') != '1.0' or config.get('protocol_id') not in PROTOCOL_ORDERS:
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
        raise InferenceError('runtime fuera del protocolo BF16')
    expected_workload = {'seed':20261106,'input_tokens':[512,2048], 'output_tokens':128,
        'concurrencies':[1,4,8],'temperature':0,'ignore_eos':True}
    if w != expected_workload:
        raise InferenceError('carga fuera del protocolo')
    expected_measurement = {'blocks':3,'campaign_order':PROTOCOL_ORDERS[config['protocol_id']], 'warmup_requests':16,
        'measured_requests':32,'telemetry_interval_seconds':1,'session_limit_seconds':28800,
        'startup_limit_seconds':900,'cell_limit_seconds':900,'cleanup_reserve_seconds':300}
    if config['protocol_id'] == DIAGNOSIS_PROTOCOL:
        expected_measurement['session_limit_seconds'] = 14400
        diagnostic = config.get('diagnosis', {})
        if (set(diagnostic) != {'capture_limit_seconds', 'trace_collection_limit_seconds',
                'historical_campaign', 'historical_campaign_sha256', 'historical_jsonl_sha256'}
                or diagnostic.get('capture_limit_seconds') != 30
                or diagnostic.get('trace_collection_limit_seconds') != 60):
            raise InferenceError('presupuestos de diagnóstico fuera del protocolo')
        from infraestructura import inference_diag_protocol as diagnosis
        diagnosis.validate_config(config)
    if m != expected_measurement:
        raise InferenceError('medición fuera del presupuesto del protocolo')
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


def validate_config(config, cluster_config):
    _validate_protocol(config)
    cluster.validate_config(cluster_config)
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
    if config['protocol_id'] == DIAGNOSIS_PROTOCOL:
        from infraestructura import inference_diag_protocol as diagnosis
        yield from diagnosis.jobs(config)
        return
    for block in range(1, config['measurement']['blocks']+1):
        for model in config['models']:
            for group in config['measurement']['campaign_order']:
                yield {'name':f"b{block}-{model['key']}-{group}", 'block':block,
                       'model':model, 'group':group,
                       'cells':[{'input_tokens':length,'concurrency':concurrency}
                                for length in config['workload']['input_tokens']
                                for concurrency in config['workload']['concurrencies']]}


def _plan_counts(measurement, planned_jobs):
    cells = sum(len(job['cells']) for job in planned_jobs)
    return {'job_count': len(planned_jobs), 'cell_count': cells,
            'measured_requests': cells * measurement['measured_requests'],
            'warmup_requests': cells * measurement['warmup_requests']}


def _cell_commands(config, model, cell):
    length, concurrency = cell['input_tokens'], cell['concurrency']
    directory = '/work/metrics/' + cell_id(cell)
    commands = {phase: runtime.bench_command(config, model, length, concurrency, phase, directory)
                for phase in ('warmup', 'measured')}
    if config['protocol_id'] == DIAGNOSIS_PROTOCOL:
        for command in commands.values():
            command += ['--ready-check-timeout-sec', '0']
    if 'mode' in cell:
        from infraestructura import inference_diag_client
        commands['measured'] = inference_diag_client.command(commands['measured'],
            profile_port=config['runtime']['api_port'] if cell['mode'] == 'on' else None)
    return commands


def _server_command(config, job, world):
    command = runtime.server_command(config, job['model'], world)
    if job.get('series') == 'auxiliary':
        from infraestructura import inference_diag_traces
        command += inference_diag_traces.server_args()
    return command


def make_plan(config_path=DEFAULT_CONFIG):
    config, cluster_config, _ = load_config(config_path)
    preview = []
    for job in jobs(config):
        nodes = cluster.selected_nodes(cluster_config, job['group'])
        preview.append({**job, 'nodes':[n['node_id'] for n in nodes],
            'ray_commands':{n['node_id']:runtime.ray_command(n, config) for n in nodes},
            'server_command':_server_command(config,job,len(nodes)),
            'cells':[{**cell, 'commands':_cell_commands(config,job['model'],cell)} for cell in job['cells']]})
    plan = {'status':'planned_only','protocol':config,'jobs':preview,
            'resources_max':{'nodes':4,'gpus':4,'replicas':1},
            **_plan_counts(config['measurement'], preview),
            'maximum_occupation_seconds':config['measurement']['session_limit_seconds'],
            'preflight_commands':{n['node_id']:cluster.ssh_command(n,runtime.probe_command(n,config,container_probe=True))
                                  for n in cluster_config['nodes']},
            'docker_policy':{'pull':'never','user':'UID:GID observados no root','network':'host',
                'shm_size':'8g','models_mount':'read-only','work_mount':'nuevo por corrida',
                'cleanup':'solo IDs creados por esta campaña con etiqueta verificada'},
            'notice':'plan no abre conexiones, inicia contenedores ni escribe resultados'}
    if config['protocol_id'] == DIAGNOSIS_PROTOCOL:
        plan['diagnostic_capture'] = {'scrape_limit_seconds': 30, 'trace_window_limit_seconds': 60,
            'trace_max_iterations': 32, 'trace_collection_limit_seconds': 60,
            'cleanup_reserve_seconds': 300, 'traces_collector': 'inference_diag_traces.collect_traces'}
    return plan


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
   if p.name.endswith(('.pt.trace.json','.pt.trace.json.gz','.trace.json','.trace.json.gz')):continue
   if any(part in ('traces','profiler') for part in p.relative_to(root).parts):continue
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
    started = time.monotonic()
    receipt = {'name': name, 'id': container_id, 'started_at': manifest.utc_now(),
               'status': 'failed', 'absence_confirmed': False, 'errors': [], 'timing_seconds': {}}
    logs = None
    def attempt(label, function):
        began = time.monotonic()
        try:
            return function()
        except (Exception, KeyboardInterrupt) as exc:
            receipt['errors'].append(f'{label}: {type(exc).__name__}: {exc}')
        finally:
            receipt['timing_seconds'][label] = time.monotonic() - began
    logs = attempt('logs', lambda: runtime.checked(
        runtime.remote(node,['docker','logs',container_id],timeout=30), 'leer logs propios'))
    # Each attempt is independent: failed capture, stop or removal must never
    # prevent the next operation on this already verified, exact container ID.
    receipt['stop'] = attempt('stop', lambda: runtime.checked(
        runtime.remote(node,['docker','stop','--time','10',container_id],timeout=20), 'detener contenedor propio'))
    receipt['remove'] = attempt('remove', lambda: runtime.checked(
        runtime.remote(node,['docker','rm',container_id],timeout=15), 'retirar contenedor propio'))
    def verify_absence():
        absence = runtime.remote(node,['docker','inspect','--type','container',container_id,'--format','{{.Id}}'],timeout=15)
        receipt['absence'] = absence
        if absence['returncode']==0 or not ('No such object' in absence['stderr'] or 'No such container' in absence['stderr']):
            raise InferenceError('ausencia de contenedor propio no confirmada después de cierre')
        receipt['absence_confirmed'] = True
    attempt('absence', verify_absence)
    # Persist only after attempting stop/rm/absence; ENOSPC cannot skip them.
    if logs is not None:
        attempt('write_logs', lambda: write_new(directory / f'{name}-container-log.json', logs))
    receipt.update(ended_at=manifest.utc_now(), duration_seconds=time.monotonic()-started)
    if receipt['errors']:
        raise CleanupError(receipt)
    receipt['status'] = 'confirmed_absent'
    return receipt


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
        contents = base64.b64decode(encoded,validate=True)
        if target.exists():
            if target.read_bytes() != contents:
                raise InferenceError('evidencia de cliente cambió antes de la colección')
        else:
            with target.open('xb') as stream: stream.write(contents)


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
        roster=TOPOLOGIES[job['group']]
        world=len(roster)
        expected_ips=[config['network']['node_ips'][key] for key in roster]
        barrier=("import json,ray,time; ray.init(address="+repr(f"{config['network']['node_ips']['pegasus']}:{config['runtime']['ray_port']}")+
                 "); end=time.monotonic()+"+str(config['measurement']['startup_limit_seconds'])+
                 "; expected=set("+repr(expected_ips)+");\nwhile time.monotonic()<end:\n"
                 " nodes=[n for n in ray.nodes() if n['Alive']];\n"
                 " ids={n.get('NodeID') for n in nodes};\n"
                 " if (len(nodes)==len(expected)==len(ids) and all(ids) and "
                 "{n['NodeManagerAddress'] for n in nodes}==expected and "
                 "all(n['Resources'].get('GPU')==1 for n in nodes)):break\n"
                 " time.sleep(.5)\nelse:raise TimeoutError('membresia Ray incompleta')\n"
                 "print(json.dumps(nodes));ray.shutdown()")
        steps.append(shlex.join(['python3','-c',barrier])+' > /work/logs/ray-membership.json 2> /work/logs/ray-membership.stderr')
        steps.append('exec '+shlex.join(_server_command(config,job,world))+
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


def _experiment_inputs(workloads, protocol_id=PEGASUS_REFERENCE_PROTOCOL):
    inputs = {name: REPO_ROOT / name for name in runner_sources(protocol_id)}
    for path in sorted(workloads.rglob('*')):
        if path.is_file():
            inputs['workload/' + str(path.relative_to(workloads))] = path
    return inputs


def _diagnostic_call(function, *args, **kwargs):
    """A failed instrument must not discard an otherwise valid client batch."""
    try:
        result = function(*args, **kwargs)
        if not isinstance(result, dict):
            raise InferenceError('diagnóstico no devolvió un objeto JSON')
        return result
    except Exception as exc:
        return {'valid': False, 'error': f'{type(exc).__name__}: {exc}'}


EXECUTION_LOGS = r"""
import json,sys
from pathlib import Path
request=json.load(sys.stdin);root=Path(request['root'])/'logs';old=request['offsets'];out={}
for name in request['names']:
 p=root/name;size=p.stat().st_size
 if old is None:out[name]=size
 else:
  start=old[name]
  if not 0<=start<=size or size-start>8388608:raise ValueError('log window changed or too large')
  with p.open('rb') as f:f.seek(start);text=f.read(size-start).decode(errors='replace')
  out[name]={'bounds':[start,size],'text':text}
print(json.dumps(out))
"""


def execution_logs(nodes, workspace, deadline, ranks, offsets=None):
    data = {}
    by_node = {r['node_id']: r for r in ranks}
    for node in nodes:
        key = node['node_id']
        names = [f"worker-rank{by_node[key]['rank']:02d}.log"]
        if key == 'pegasus': names.append('server.log')
        request = {'root': workspace, 'names': names,
                   'offsets': None if offsets is None else offsets[key]}
        result = runtime.checked(runtime.remote(node, ['python3', '-c', EXECUTION_LOGS],
            stdin=json.dumps(request), timeout=remaining(deadline, 15)), 'ventana real de logs')
        data[key] = json.loads(result['stdout'])
    return data


def _archive_client_file(directory, workspace, name, filename):
    if any((directory / marker).exists() for marker in ('COMPLETE', 'FAILED')):
        raise InferenceError('no se modifica ni recupera una corrida sellada')
    target = directory / 'raw/nodes/pegasus/metrics' / name / filename
    source = Path(workspace) / 'metrics' / name / filename
    if source.is_file():
        contents = source.read_bytes()
        if target.exists():
            if target.read_bytes() != contents:
                raise InferenceError('artefacto nativo de cliente cambió')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream: stream.write(contents)
    return read_json(target)


def _persist_native_client(config, model, cell, directory, workspace, phase, start, end, result):
    """Preserve the native JSON and timestamps before optional instrument writes."""
    name = cell_id(cell)
    raw = _archive_client_file(directory, workspace, name, phase + '.json')
    if phase == 'measured':
        window = directory / 'metrics' / (name + '-client-window.json')
        if not window.exists():
            write_new(window, {'measurement_started_at': start, 'measurement_ended_at': end,
                'client_returncode': result.get('returncode'), 'client_timed_out': result.get('timed_out', False)})
    validate_client_identity(raw, model=model, concurrency=cell['concurrency'])
    summary = summarize_bench_result(raw, expected_prompt_tokens=cell['input_tokens'],
        expected_output_tokens=config['workload']['output_tokens'],
        expected_count=config['measurement'][phase + '_requests'])
    if phase != 'measured': return raw, None
    window = read_json(window)
    record = dict(cell, **summary, cell=name, status='valid', ended_at=manifest.utc_now(), **window)
    path = directory / 'metrics' / (name + '.json')
    if not path.exists(): write_new(path, record)
    return raw, record


def _cell_quality_issues(path, cell, ranks, *, verify_receipt=False):
    from infraestructura import inference_diag_quality as quality
    try:
        issues = list(quality.cell_issues(path, cell, ranks))
    except Exception as exc:
        issues = [f'execution quality: {type(exc).__name__}: {exc}']
    if cell.get('client_returncode') != 0 or cell.get('client_timed_out') is not False:
        issues.append('client execution: missing or unsuccessful measured CLI exit')
    if verify_receipt:
        try:
            receipt = read_json(Path(path)/'raw/diagnostic'/cell['cell']/'quality.json')
            if receipt != {'valid': not issues, 'issues': issues}:
                raise InferenceError('quality receipt differs from recomputation')
        except (OSError, ValueError, KeyError) as exc:
            issues.append(f'execution quality receipt: {exc}')
    return issues


def _run_diagnostic_cell(config, job, cell, nodes, probes, workspace, campaign_id,
                         directory, deadline, active_clients, cleanup, ranks, seen):
    from infraestructura import inference_diag_phases as phases
    from infraestructura import inference_diag_client as client
    from infraestructura import inference_diag_traces as traces
    from infraestructura import inference_diag_quality as quality
    model = job['model']; name = cell_id(cell)
    dest = directory / 'raw/diagnostic' / name
    port = config['runtime']['api_port']; errors = []; record = None
    def budget(key):
        return min(deadline, time.monotonic() + config['diagnosis'][key])
    def save(path, value, kind):
        try: write_new(path, value)
        except Exception as exc: errors.append(f'{kind}: {type(exc).__name__}: {exc}')
    def capture(label):
        snapshot = _diagnostic_call(phases.capture, port, budget('capture_limit_seconds'), model['key'])
        save(dest / (label + '.json'), snapshot, 'phases')
        return snapshot
    commands = _cell_commands(config, model, cell)
    for phase in ('warmup', 'measured'):
        left = remaining(deadline, config['measurement']['cell_limit_seconds'])
        container = f"ipre-{campaign_id[-8:]}-{job['name']}-{name}-{phase}"
        command = container_command(nodes[0], config, probes['pegasus'], workspace, container,
                                    commands[phase], gpu=False, seconds=left-2)
        command.insert(2, '--rm')
        offsets = _diagnostic_call(execution_logs, nodes, workspace, deadline, ranks) if phase == 'measured' else None
        if phase == 'measured': before = capture('before')
        active_clients.append((nodes[0], container))
        start = manifest.utc_now(); result = None; client_error = None
        try:
            result = runtime.remote(nodes[0], command, timeout=remaining(deadline, left))
        except (Exception, KeyboardInterrupt) as exc:
            client_error = exc
        end = manifest.utc_now()
        # Native results survive failed profile receipts and even CLI failures.
        fallback = {'returncode': None, 'timed_out': True} if result is None else result
        raw, record = _persist_native_client(config, model, cell, directory, workspace,
                                             phase, start, end, fallback)
        save(directory / 'logs' / f'{name}-{phase}.json', fallback, 'client execution')
        if client_error is not None or result['returncode'] != 0 or result.get('timed_out'):
            # Leave the bounded client registered for label-checked cleanup;
            # the enclosing finally closes this server before any more clients.
            if client_error is not None: raise client_error
            runtime.checked(result, f'{name} {phase}')
        active_clients.remove((nodes[0], container))
        if phase != 'measured': continue
        window = _diagnostic_call(execution_logs, nodes, workspace, deadline, ranks, offsets)
        logs = {node: {filename: value['bounds'] for filename, value in files.items()}
                for node, files in window.items()} if window.get('valid') is not False else {}
        execution = {'cell': name, 'measurement_started_at': record['measurement_started_at'],
                     'measurement_ended_at': record['measurement_ended_at'], 'logs': logs}
        save(dest / 'execution-window.json', execution, 'execution quality')
        text = '\n'.join(value['text'] for files in window.values() for value in files.values()) if logs else ''
        text += '\n' + result.get('stdout', '') + '\n' + result.get('stderr', '')
        control = None; events = None
        if 'mode' in cell:
            events = _diagnostic_call(_archive_client_file, directory, workspace, name, 'client-events.json')
            receipt = _diagnostic_call(client.verify_events, events, raw)
            save(dest / 'client-validation.json', receipt, 'client events')
            if receipt.get('valid') is not True: errors.append('client events: invalid admission receipt')
        if cell.get('mode') == 'on':
            control = _diagnostic_call(_archive_client_file, directory, workspace, name, 'profile-control.json')
            save(dest / 'profile-control.json', control, 'profile control')
            coverage = _diagnostic_call(client.profile_coverage, control, events, raw)
            save(dest / 'profile-coverage.json', coverage, 'profile coverage')
            if coverage.get('valid') is not True: errors.append('profile coverage: no verified measured overlap')
            # START not acknowledged or STOP not confirmed poisons this server.
            from infraestructura.inference_evidence import _verify_profile_control
            verified = _diagnostic_call(lambda: (_verify_profile_control(control), {'valid': True})[1])
            if verified.get('valid') is not True: errors.append('profile control: invalid START/STOP receipt')
            if (not isinstance(control.get('start'), dict) or not control['start'].get('ack')
                    or control['start'].get('http_status') not in (200, 204) or control['start'].get('error')
                    or not isinstance(control.get('stop'), dict) or not control['stop'].get('ack')
                    or control['stop'].get('http_status') not in (200, 204)
                    or control['stop'].get('error') or control.get('requires_server_shutdown') is not False
                    or quality.profiler_stop_failed(text) or not logs):
                record['server_must_close'] = 'profiler START/STOP unconfirmed or explicit stop failure'
                errors.append('profile control: close this server before any further capture or client')
                return record, errors
        after = capture('after')
        delta = _diagnostic_call(phases.phase_deltas, before, after, model['key'], expected_count=32, measured=raw)
        save(dest / 'phases.json', delta, 'phases')
        if delta.get('valid') is not True: errors.append('phases: invalid native measured window or counters')
        if cell.get('mode') == 'on':
            receipt = _diagnostic_call(traces.collect_traces, nodes, workspace, dest / 'traces',
                                       ranks, seen, budget('trace_collection_limit_seconds'))
            save(dest / 'traces.json', receipt, 'traces')
            if receipt.get('valid') is not True: errors.append('traces: invalid rank capture')
    return record, errors


def run_job(config, cluster_config, config_path, cluster_path, job, campaign_id,
            preflight_result, workloads, deadline, results_root, *, run_id=None, occupation_deadline=None):
    model=job['model']; nodes=cluster.selected_nodes(cluster_config,job['group'])
    request={'protocol':config,'job':job,'campaign_id':campaign_id}
    diagnostic = config['protocol_id'] == DIAGNOSIS_PROTOCOL
    capacity = config['protocol_id'] == SERVING_CAPACITY_PROTOCOL
    inputs={'cluster':cluster_path,'protocol':config_path,
            **_experiment_inputs(workloads, config['protocol_id'])}
    if diagnostic:
        from infraestructura import inference_diag_protocol as diagnosis
        inputs['diagnosis/historical-campaign.json'] = diagnosis.historical_path(config, REPO_ROOT)
    if capacity:
        from infraestructura import serving_capacity_protocol as capacity_protocol
        from infraestructura import serving_capacity_runner as capacity_runner
        inputs['reference/campaign.json'] = capacity_protocol.reference_index_path(config)
        inputs['reference/preflight.json'] = capacity_protocol.historical_preflight_path(config)
    directory=manifest.start_run(kind='inference-serving',model_id=model['id'],model_revision=model['revision'],
        tokenizer_id=model['tokenizer_id'],tokenizer_revision=model['tokenizer_revision'],
        requested_config=request,requested_config_path=config_path,inputs=inputs,
        repo_path=REPO_ROOT,results_root=results_root, **({'run_id': run_id} if run_id is not None else {}))
    print(json.dumps({'event':'job_start','job':job['name'],'run_dir':str(directory)}),flush=True)
    workspace=str(Path(config['paths']['workspace'])/campaign_id/job['name'])
    files={}
    created=[]; active_clients=[]; errors=[]; cells=[]; cleanup=[]; launch=[]; instrument_errors=[]; quality_errors=[]; ranks=[]
    probe_by_id={n['node_id']:n for n in preflight_result['nodes']}
    started=time.monotonic(); occupation_started_at=manifest.utc_now()
    occupation=None
    startup_deadline=min(deadline,started+config['measurement']['startup_limit_seconds'])
    try:
        for name,path in inputs.items():
            target=directory/'raw'/'inputs'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            contents = Path(path).read_bytes()
            with target.open('xb') as stream:
                stream.write(contents)
            if name in runner_sources(config['protocol_id']):
                files['code/' + name] = contents
            elif name.startswith('workload/'):
                files['workloads/' + name.removeprefix('workload/')] = contents
        write_new(directory/'raw'/'preflight.json',preflight_result)
        write_new(directory/'raw'/'requested-job.json',request)
        if capacity:
            capacity_protocol.validate_reference_index(config, directory/'raw/inputs/reference/campaign.json')
            capacity_protocol.verify_hardware(config, directory/'raw/inputs/reference/preflight.json', preflight_result)
            if read_json(directory/'raw/inputs/protocol') != config:
                raise InferenceError('configuración efectiva difiere del protocolo archivado')
        if diagnostic:
            diagnosis.verify_workloads(config, {name: manifest.sha256_file(path) for name, path in inputs.items()})
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
        archived_profiles = directory / 'raw/inputs/workload' / model['key']
        for length in config['workload']['input_tokens']:
            profile = archived_profiles / str(length)
            metadata=read_json(profile/'metadata.json')
            prompts=[]
            for phase in ('warmup','measured'):
                prompts.extend(json.loads(line) for line in (profile/(phase+'.jsonl')).read_text().splitlines())
            for row,prompt in zip(metadata['rows'],prompts):
                response=_http(config['runtime']['api_port'],'/tokenize',
                    {'model':model['key'],'prompt':prompt['prompt'],'add_special_tokens':True},
                    timeout=remaining(startup_deadline,5))
                if response.get('tokens')!=row['token_ids']:
                    raise InferenceError('tokenización efectiva del servidor difiere del dataset')
        write_new(directory/'raw'/'tokenization-verified.json',{'verified':True,'profiles':[512,2048]})
        seen = {node['node_id']: {} for node in nodes}
        for cell in job['cells']:
            if time.monotonic()>=deadline: raise TimeoutError('presupuesto global agotado')
            cell_deadline=min(deadline,time.monotonic()+config['measurement']['cell_limit_seconds'])
            if capacity:
                cells.append(capacity_runner.run_point(config, job, cell, nodes, probe_by_id,
                    workspace, campaign_id, directory, cell_deadline, active_clients, ranks))
                continue
            if diagnostic:
                record, issues = _run_diagnostic_cell(config, job, cell, nodes, probe_by_id, workspace,
                    campaign_id, directory, cell_deadline, active_clients, cleanup, ranks, seen)
                cells.append(record)
                cell_errors = [f"{record['cell']}: {issue}" for issue in issues]
                instrument_errors.extend(error for error, issue in zip(cell_errors, issues)
                    if issue.startswith(('phases:', 'traces:', 'client events:', 'profile control:', 'profile coverage:')))
                errors.extend(cell_errors)
                if record.get('server_must_close'):
                    errors.append(f"{record['cell']}: {record['server_must_close']}; parada operacional global, sin reintento")
                    break
                continue
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
                validate_client_identity(raw, model=model, concurrency=cell['concurrency'])
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
        if isinstance(exc, CleanupError): cleanup.append(exc.receipt)
        errors.append(f'{type(exc).__name__}: {exc}')
    finally:
        close_deadline = min(deadline + config['measurement']['cleanup_reserve_seconds'],
                             time.monotonic() + config['measurement']['cleanup_reserve_seconds'])
        try: write_new(directory/'raw'/'launch.json',launch)
        except (Exception, KeyboardInterrupt) as exc:
            errors.append(f'launch receipt: {type(exc).__name__}: {exc}')
        for node,name in active_clients:
            try: cleanup.append(cleanup_container(node,name,directory/'logs'))
            except Exception as exc:
                if isinstance(exc, CleanupError): cleanup.append(exc.receipt)
                errors.append(f"cleanup client {name}: {exc}")
        def close_node(item):
            node,name=item
            node_errors=[]; result=None
            try: result=cleanup_container(node,name,directory/'logs')
            except Exception as exc:
                if isinstance(exc, CleanupError): result = exc.receipt
                node_errors.append(f"cleanup {node['node_id']}: {exc}")
            if not capacity:
                try: collect(node,workspace,directory/'raw'/'nodes')
                except Exception as exc: node_errors.append(f"collect {node['node_id']}: {exc}")
            return result,node_errors
        # Each host has <=140s of bounded cleanup and collection; parallel hosts
        # fit inside the reserved300s instead of multiplying by four.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for result,node_errors in pool.map(close_node,reversed(created)):
                if result: cleanup.append(result)
                errors.extend(node_errors)
        if capacity:
            # Stop the occupation clock immediately after resource retirement,
            # before collection, point assessment and manifest hashing.
            retired=time.monotonic(); retired_at=manifest.utc_now()
            expected={name for _,name in created+active_clients}
            confirmed={r.get('name') for r in cleanup if r.get('status')=='confirmed_absent'
                       or r.get('absence_confirmed') is True}
            removal_confirmed=expected<=confirmed
            stop_limit=occupation_deadline if occupation_deadline is not None else deadline+config['measurement']['cleanup_reserve_seconds']
            occupation={'started_at':occupation_started_at,'started_monotonic_s':started,
                'ended_at':retired_at if removal_confirmed else None,
                'ended_monotonic_s':retired if removal_confirmed else None,
                'observed_until_at':retired_at,'observed_until_monotonic_s':retired,
                'duration_seconds':retired-started,'removal_confirmed':removal_confirmed,
                'limit_exceeded':retired>stop_limit}
            if not removal_confirmed: errors.append('retirada de recursos sin confirmar; ocupación abierta')
            if occupation['limit_exceeded']: errors.append('ocupación excedió el límite efectivo')
            try: write_new(directory/'raw/occupation.json',occupation)
            except (Exception,KeyboardInterrupt) as exc: errors.append(f'occupation receipt: {type(exc).__name__}: {exc}')
            def collect_capacity(item):
                node,_=item
                try: collect(node,workspace,directory/'raw'/'nodes'); return None
                except Exception as exc: return f"collect {node['node_id']}: {exc}"
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                errors.extend(issue for issue in pool.map(collect_capacity,reversed(created)) if issue)
        if diagnostic:
            # A later instrument/transport failure can occur after the verified
            # client summary was persisted. Recover it before sealing assessment.
            by_name = {record['cell']: record for record in cells}
            for coordinate in job['cells']:
                name = cell_id(coordinate)
                if coordinate.get('mode') == 'on':
                    native = directory/'raw/nodes/pegasus/metrics'/name
                    local = Path(workspace)/'metrics'/name
                    if (native/'profile-control.json').is_file() or (local/'profile-control.json').is_file():
                        from infraestructura import inference_diag_client as client
                        try:
                            control = _archive_client_file(directory, workspace, name, 'profile-control.json')
                            target = directory/'raw/diagnostic'/name/'profile-control.json'
                            if not target.exists(): write_new(target, control)
                            target = target.with_name('profile-coverage.json')
                            if not target.exists():
                                events = _diagnostic_call(_archive_client_file, directory, workspace, name, 'client-events.json')
                                measured = _diagnostic_call(_archive_client_file, directory, workspace, name, 'measured.json')
                                write_new(target, _diagnostic_call(client.profile_coverage, control, events, measured))
                            target = target.with_name('traces.json')
                            if not target.exists():
                                from infraestructura import inference_diag_traces as traces
                                receipt = _diagnostic_call(traces.collect_traces, nodes, workspace,
                                    target.parent/'traces', ranks, seen,
                                    min(close_deadline, time.monotonic()+config['diagnosis']['trace_collection_limit_seconds']))
                                write_new(target, receipt)
                        except Exception as exc: errors.append(f'{name}: profile receipt recovery: {exc}')
                saved = directory/'metrics'/(name+'.json')
                if name not in by_name and saved.is_file():
                    record = read_json(saved)
                    if record.get('status') == 'valid': by_name[name] = record
                elif name not in by_name:
                    window = directory/'metrics'/(name+'-client-window.json')
                    if window.is_file():
                        try:
                            receipt = read_json(window)
                            _, record = _persist_native_client(config, model, coordinate, directory, workspace,
                                'measured', receipt['measurement_started_at'], receipt['measurement_ended_at'],
                                {'returncode': receipt.get('client_returncode'), 'timed_out': receipt.get('client_timed_out')})
                            by_name[name] = record
                        except Exception as exc: errors.append(f'{name}: native client recovery: {exc}')
            cells = [by_name[cell_id(c)] for c in job['cells'] if cell_id(c) in by_name]
            for record in cells:
                issues = _cell_quality_issues(directory, record, ranks)
                quality_errors.extend(f"{record['cell']}: {issue}" for issue in issues)
                try:
                    write_new(directory/'raw/diagnostic'/record['cell']/'quality.json',
                              {'valid': not issues, 'issues': issues})
                except Exception as exc:
                    quality_errors.append(f"{record['cell']}: execution quality receipt: {exc}")
            errors.extend(quality_errors)
            errors.extend(diagnosis.cell_matrix_issues(job, cells, complete=True))
        elif capacity:
            try:
                capacity_runner.require_tokenization(directory, config)
                cells, issues = capacity_runner.assess_job(directory, config, job, ranks)
                errors.extend(issues)
            except Exception as exc:
                # Never leave partially checked client cells marked valid.
                cells = []
                errors.append(f'capacity quality: {type(exc).__name__}: {exc}')
        else:
            errors.extend(_telemetry_valid(directory,nodes,cells))
        if len(cells)!=len(job['cells']): errors.append('matriz de celdas incompleta')
        write_new(directory/'raw'/'cleanup.json',cleanup)
        write_new(directory/'metrics'/'assessment.json',{'valid':not errors,'errors':errors,'cells':cells})
        try:
            manifest.finalize_run(directory,status='failed' if errors else 'complete',errors=errors)
        except Exception:
            # finalize_run can seal FAILED and then raise on changed inputs.
            # Never mutate a sealed archive or lose the real attempt's index.
            if not (directory / 'FAILED').is_file():
                raise
            closed = manifest.load_run(directory)
            errors = [e.get('message', str(e)) if isinstance(e, dict) else str(e)
                      for e in closed['errors']] or ['cierre FAILED']
    return {'job':job['name'],'group':job['group'],'block':job['block'],'model_key':model['key'],
            'run_dir':str(directory),'status':'failed' if errors else 'complete',
            'manifest_sha256':manifest.sha256_file(directory/'manifest.json'),'errors':errors,
            **({'occupation':occupation} if capacity else {}),
            **({'series': job['series'], 'client_cells_complete': len(cells) == len(job['cells']),
                'cell_counts': diagnosis.cell_counts(job['series'], cells),
                'failure_scope': 'diagnostic' if errors and all(e in instrument_errors for e in errors)
                                else ('cell_quality' if errors and all(e in instrument_errors + quality_errors for e in errors)
                                      else ('operational' if errors else None))}
               if diagnostic else {})}


def run_campaign(config_path=DEFAULT_CONFIG, *, results_root=REPO_ROOT/'resultados/raw', session_limit_seconds=None):
    config,cluster_config,cluster_path=load_config(config_path)
    if config['protocol_id'] == DIAGNOSIS_PROTOCOL:
        from infraestructura import inference_diag_protocol as diagnosis
        diagnosis.validate_historical(config, REPO_ROOT)
    maximum=config['measurement']['session_limit_seconds']
    effective_limit=maximum if session_limit_seconds is None else session_limit_seconds
    if (type(effective_limit) is not int or
            not config['measurement']['cleanup_reserve_seconds']<effective_limit<=maximum):
        raise InferenceError('el límite efectivo solo puede reducir el presupuesto configurado y debe reservar el cierre')
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
    input_hashes = {}
    diagnostic = config['protocol_id'] == DIAGNOSIS_PROTOCOL
    if config['protocol_id'] in (PEGASUS_REFERENCE_PROTOCOL, DIAGNOSIS_PROTOCOL):
        input_hashes = {name: manifest.sha256_file(path)
                        for name, path in _experiment_inputs(workloads, config['protocol_id']).items()}
    if diagnostic:
        from infraestructura import inference_diag_protocol as diagnosis
        diagnosis.verify_workloads(config, input_hashes)
        for name, path in {'cluster': cluster_path, 'protocol': Path(config_path),
                'diagnosis/historical-campaign.json': diagnosis.historical_path(config, REPO_ROOT)}.items():
            input_hashes[name] = manifest.sha256_file(path)
    directory=Path(results_root)/'inference-campaign'/campaign_id
    directory.mkdir(parents=True,exist_ok=False)
    write_new(directory/'plan.json',make_plan(config_path))
    write_new(directory/'preflight.json',check)
    # CPU preparation and review artifacts precede the occupation clock in
    # every protocol. Reserve the final 300 seconds for bounded closure.
    started=time.monotonic()
    deadline=started+effective_limit-config['measurement']['cleanup_reserve_seconds']
    records=[]; errors=[]
    planned_jobs = list(jobs(config))
    try:
        for job in planned_jobs:
            if time.monotonic()>=deadline:
                errors.append('presupuesto global agotado'); break
            record=run_job(config,cluster_config,Path(config_path).resolve(),cluster_path,job,campaign_id,
                           check,workloads,deadline,results_root)
            records.append(record)
            if record['status']!='complete':
                if diagnostic and record.get('client_cells_complete') and record.get('failure_scope') in ('diagnostic', 'cell_quality'):
                    errors.append(f"{job['name']}: evidencia diagnóstica o cierre inválido; métricas conservadas")
                    continue
                errors.append('sesión detenida tras fallo; sin reintentos automáticos'); break
    except (Exception,KeyboardInterrupt) as exc: errors.append(f'{type(exc).__name__}: {exc}')
    if diagnostic and not errors and len(records) == len(planned_jobs):
        totals = {key: sum(ref.get('cell_counts', {}).get(key, 0) for ref in records)
                  for key in ('primary', 'off', 'on')}
        if totals != {'primary': 54, 'off': 12, 'on': 12}:
            errors.append('COMPLETE requiere totales exactos 54 primary, 12 off y 12 on')
    status='complete' if len(records)==len(planned_jobs) and not errors else 'partial'
    campaign={'campaign_id':campaign_id,'status':status,'git':git,'protocol':config,
        'duration_seconds':time.monotonic()-started,'effective_session_limit_seconds':effective_limit,
        'runs':records,'errors':errors,
        'pending_jobs':[j['name'] for j in planned_jobs[len(records):]]}
    if config['protocol_id'] in (PEGASUS_REFERENCE_PROTOCOL, DIAGNOSIS_PROTOCOL):
        campaign['input_hashes'] = input_hashes
    write_new(directory/'campaign.json',campaign)
    write_new(directory/('COMPLETE' if status=='complete' else 'PARTIAL'),
              {'campaign_sha256':manifest.sha256_file(directory/'campaign.json'),
               'plan_sha256':manifest.sha256_file(directory/'plan.json'),
               'preflight_sha256':manifest.sha256_file(directory/'preflight.json')})
    print(json.dumps({'event':'campaign_closed','directory':str(directory),'status':status}),flush=True)
    return directory


def _probe_identity(probe, model_key):
    return (probe['hostname'], probe['architecture'], probe['gpu']['stdout'].strip(),
            probe['image']['stdout'].strip(), probe['container_probe']['stdout'].strip(),
            probe['models'][model_key])


def _observed_topology(path, run, job, preflight, protocol, campaign_probes):
    """Validate evidence, then return identities for the participating nodes only.

    Preflight covers all four hosts even for TP1. Extra preflight hosts must not
    enter a single/pair trial's identity. Rank order may differ between TP4 and
    TP2; it is verified within each trial rather than inferred from node IDs.
    """
    expected=set(TOPOLOGIES[job['group']])
    participants=run['participants']
    if (len(participants)!=len(expected) or {p['node_id'] for p in participants}!=expected
            or any(type(p['rank']) is not int for p in participants)
            or {p['rank'] for p in participants}!=set(range(len(expected)))):
        raise InferenceError('roster observado no corresponde a la topología comparada')
    probes={p['node_id']:p for p in preflight['nodes']}
    if len(probes)!=len(preflight['nodes']) or not expected<=probes.keys():
        raise InferenceError('preflight sin identidad única por participante')
    inventory={node_id:{'expected_hostname':probes[node_id]['hostname']} for node_id in expected}
    # Prefer the immutable inventory copied by the runner, never today's hosts.
    inventory_path=path/'raw'/'inputs'/'cluster'
    if inventory_path.is_file():
        recorded=next((i for i in run['inputs'] if i['name']=='cluster'),None)
        if recorded is None or manifest.sha256_file(inventory_path)!=recorded['sha256']:
            raise InferenceError('hash del inventario copiado no coincide')
        archived=read_json(inventory_path)
        inventory={n['node_id']:n for n in cluster.selected_nodes(archived,job['group'])}
    evidence=[read_json(path/p['evidence_path']) for p in participants]
    ranks=validate_worker_records(evidence,inventory,len(expected),runtime.model_path(job['model']))
    identities = {}
    comparable = inventory_path.is_file() or campaign_probes is None
    model_key = job['model']['key']
    for participant,rank in zip(sorted(participants,key=lambda p:p['rank']),ranks):
        if any(participant[key]!=rank[key] for key in ('node_id','hostname','gpu_uuid','rank')):
            raise InferenceError('identidad efectiva contradice el manifiesto')
        probe=probes[participant['node_id']]
        if campaign_probes is not None:
            reference = campaign_probes.get(participant['node_id'])
            if reference is None or _probe_identity(reference, model_key) != _probe_identity(probe, model_key):
                comparable = False
        gpu=probe['gpu']; package_probe=probe['container_probe']
        if (probe['hostname']!=participant['hostname'] or gpu['returncode']!=0
                or package_probe['returncode']!=0):
            raise InferenceError('identidad de host/GPU difiere del preflight')
        gpu_lines=gpu['stdout'].strip().splitlines()
        if len(gpu_lines)!=1 or gpu_lines[0].split(',')[0].strip()!=participant['gpu_uuid']:
            raise InferenceError('UUID de participante difiere del preflight')
        env=json.loads(package_probe['stdout'].splitlines()[-1])
        if any(rank['versions'].get(k)!=env['packages'].get(k) for k in ('vllm','torch','ray')):
            raise InferenceError('versiones efectivas de worker difieren del preflight')
        snapshot=probe['models'][job['model']['key']]
        if not snapshot.get('complete') or not snapshot.get('metadata_sha256') or not snapshot.get('shards'):
            raise InferenceError('snapshot efectivo sin identidad completa')
        image=protocol['image']; versions=env['packages']
        try: image_identity=shlex.split(probe['image']['stdout'].strip())
        except ValueError: image_identity=[]
        if (probe['architecture']!='aarch64' or probe['image']['returncode']!=0
                or image_identity!=[image['id'],image['architecture']]
                or versions.get('ray')!=image['ray_version']
                or versions.get('apache-tvm-ffi')!=image['tvm_ffi_version']
                or versions.get('pandas')!=image['pandas_version']
                or not str(versions.get('vllm','')).startswith(image['vllm_version_prefix'])):
            comparable=False
        identities[participant['node_id']]={
            'hostname':participant['hostname'],'gpu_uuid':participant['gpu_uuid'],
            'gpu':gpu_lines[0], 'architecture':probe['architecture'],
            'image':probe['image']['stdout'].strip(), 'environment':env,
            'worker_versions':rank['versions'],
            'snapshot':{'metadata_sha256':snapshot['metadata_sha256'],'shards':snapshot['shards']}}
    head = identities['pegasus']
    common_fields = ('image', 'architecture', 'environment', 'worker_versions', 'snapshot')
    if any(node[field] != head[field] for node in identities.values() for field in common_fields):
        comparable = False
    return identities, probes['pegasus'], comparable


def _pair_compatible(baseline, scaled, identities):
    if baseline['comparison_key']!=scaled['comparison_key']:
        return False
    small=identities[baseline['run_dir']]; large=identities[scaled['run_dir']]
    return small.keys() < large.keys() and all(small[node] == large[node] for node in small)


def _validate_planned_matrix(plan, protocol, expected_jobs):
    measurement = protocol['measurement']
    saved_jobs = plan.get('jobs')
    if not isinstance(saved_jobs, list) or len(saved_jobs) != len(expected_jobs):
        raise InferenceError('cantidad de corridas del plan incompatible con el protocolo')
    for field, expected in _plan_counts(measurement, expected_jobs).items():
        if type(plan.get(field)) is not int or plan[field] != expected:
            raise InferenceError(f'total del plan incompatible con el protocolo: {field}')
    if protocol['protocol_id'] == DIAGNOSIS_PROTOCOL:
        if (plan.get('maximum_occupation_seconds') != 14400
                or plan.get('resources_max') != {'nodes': 4, 'gpus': 4, 'replicas': 1}
                or plan.get('diagnostic_capture') != {
                    'scrape_limit_seconds': 30, 'trace_window_limit_seconds': 60,
                    'trace_max_iterations': 32, 'trace_collection_limit_seconds': 60,
                    'cleanup_reserve_seconds': 300, 'traces_collector': 'inference_diag_traces.collect_traces'}):
            raise InferenceError('presupuestos o recursos del plan diagnóstico incompatibles')
    for saved, expected in zip(saved_jobs, expected_jobs):
        if (not isinstance(saved, dict)
                or any(saved.get(k) != expected[k] for k in ('name', 'block', 'model', 'group'))
                or saved.get('nodes') != TOPOLOGIES[expected['group']]):
            raise InferenceError('identidad u orden de corridas incorrecto en el plan')
        if protocol['protocol_id'] == DIAGNOSIS_PROTOCOL:
            if (saved.get('series') != expected['series']
                    or saved.get('server_command') != _server_command(protocol, expected, len(saved['nodes']))):
                raise InferenceError('serie o flags del servidor diagnóstico incompatibles')
            if saved.get('ray_commands') != {node: runtime.ray_command({'node_id': node}, protocol)
                                             for node in saved['nodes']}:
                raise InferenceError('comandos Ray del diagnóstico incompatibles')
        cells = saved.get('cells')
        if not isinstance(cells, list) or len(cells) != len(expected['cells']):
            raise InferenceError(f"matriz de celdas incompleta en el plan de {expected['name']}")
        for cell, coordinate in zip(cells, expected['cells']):
            if not isinstance(cell, dict) or any(
                    type(cell.get(k)) is not type(value) or cell[k] != value for k, value in coordinate.items()):
                raise InferenceError(f"coordenadas de celda incorrectas en el plan de {expected['name']}")
            if protocol['protocol_id'] == DIAGNOSIS_PROTOCOL and set(cell) != {*coordinate, 'commands'}:
                raise InferenceError('campos de celda no previstos en el plan diagnóstico')
            if cell.get('commands') != _cell_commands(protocol, expected['model'], coordinate):
                raise InferenceError(f"comandos de celda incompatibles con el plan de {expected['name']}")


def build_report(directory):
    directory=Path(directory).resolve()
    campaign=read_json(directory/'campaign.json')
    markers=[directory/name for name in ('COMPLETE','PARTIAL') if (directory/name).is_file()]
    if len(markers)!=1: raise InferenceError('campaña abierta o cierre ambiguo')
    marker=read_json(markers[0])
    for key,name in [('campaign_sha256','campaign.json'),('plan_sha256','plan.json'),('preflight_sha256','preflight.json')]:
        if marker[key]!=manifest.sha256_file(directory/name): raise InferenceError('integridad de campaña inválida')
    protocol=campaign['protocol']
    _validate_protocol(protocol)
    protocol_id=protocol['protocol_id']
    diagnostic = protocol_id == DIAGNOSIS_PROTOCOL
    reference_protocol = protocol_id in (PEGASUS_REFERENCE_PROTOCOL, DIAGNOSIS_PROTOCOL)
    expected_jobs={job['name']:job for job in jobs(protocol)}
    if (campaign['status'] not in ('complete','partial')
            or markers[0].name!=('COMPLETE' if campaign['status']=='complete' else 'PARTIAL')):
        raise InferenceError('estado y marcador de campaña contradictorios')
    plan=read_json(directory/'plan.json')
    if plan.get('protocol', None if reference_protocol else protocol)!=protocol:
        raise InferenceError('plan y protocolo de campaña contradictorios')
    if reference_protocol:
        _validate_planned_matrix(plan, protocol, list(expected_jobs.values()))
    if campaign['status']=='complete' and (
            campaign['pending_jobs'] or campaign.get('errors')
            or [r['job'] for r in campaign['runs']]!=list(expected_jobs)
            or any(r['status']!='complete' for r in campaign['runs'])
            or [j['name'] for j in plan.get('jobs',[])]!=list(expected_jobs)):
        raise InferenceError('COMPLETE requiere todos los jobs y el plan íntegros')
    campaign_probes = None
    if reference_protocol:
        probe_list = read_json(directory/'preflight.json')['nodes']
        campaign_probes = {probe['node_id']: probe for probe in probe_list}
        if len(campaign_probes) != len(probe_list) or set(campaign_probes) != set(TOPOLOGIES['four']):
            raise InferenceError('preflight de campaña sin identidad única de los cuatro nodos')
    campaign_hashes = campaign.get('input_hashes', {})
    if diagnostic:
        actual = [ref['job'] for ref in campaign['runs']]
        expected = list(expected_jobs)
        if actual != expected[:len(actual)] or campaign['pending_jobs'] != expected[len(actual):]:
            raise InferenceError('índice de intentos o pendientes contradictorio con el plan')
    pairs={}; rows=[]; auxiliary_rows=[]; raw_rows=[]; excluded=[]; identities={}; seen_jobs=set(); seen_slots=set(); seen_paths=set()
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
        expected_job=expected_jobs.get(job_name)
        if request.get('campaign_id')!=campaign['campaign_id']:
            raise InferenceError('corrida pertenece a otra campaña')
        if (ref['job']!=job['name'] or ref['group']!=job['group'] or ref['block']!=job['block']
                or ref['model_key']!=job['model']['key'] or model['id']!=job['model']['id']
                or model['revision']!=job['model']['revision']
                or model['tokenizer']!={'id':job['model']['tokenizer_id'],
                                        'revision':job['model']['tokenizer_revision']}
                or expected_job is None or ref['status']!=run['status']
                or any(job[k]!=expected_job[k] for k in ('name','group','block','cells'))
                or job['model']['key']!=expected_job['model']['key']):
            raise InferenceError('índice de campaña contradice la identidad de la corrida')
        if diagnostic and (job.get('series') != expected_job['series'] or ref.get('series') != job['series']):
            raise InferenceError('serie del índice contradice el plan')
        if run['status'] != 'complete' and not diagnostic:
            continue
        if diagnostic and not (path/'metrics/assessment.json').is_file():
            if campaign['status'] == 'complete':
                raise InferenceError('COMPLETE sin evaluación')
            continue
        assessment=read_json(path/'metrics'/'assessment.json')
        if diagnostic:
            from infraestructura import inference_diag_protocol as diagnosis
            matrix_issues = diagnosis.cell_matrix_issues(job, assessment.get('cells'),
                complete=campaign['status'] == 'complete' or run['status'] == 'complete' or assessment.get('valid') is True)
            if matrix_issues:
                raise InferenceError('; '.join(matrix_issues))
        if not assessment['valid']:
            if campaign['status']=='complete': raise InferenceError('COMPLETE contiene corrida inválida')
            if not diagnostic: continue
        if diagnostic and not assessment.get('cells'):
            continue
        observed,head,comparable=_observed_topology(
            path,run,job,read_json(path/'raw'/'preflight.json'),protocol,campaign_probes)
        identities[str(path)]=observed
        run_issues = []
        if not comparable:
            run_issues.append('El inventario o el entorno de los participantes difiere de la referencia.')
        if request['protocol'] != protocol or job['model'] != expected_job['model']:
            run_issues.append('La configuración o revisión del modelo difiere del plan.')
        commit = run['git'].get('commit')
        if run['git'].get('dirty') is not False or not SHA40.fullmatch(commit or ''):
            run_issues.append('La corrida carece de un commit limpio e identificable.')
        if reference_protocol:
            campaign_git = campaign.get('git', {})
            if campaign_git.get('dirty') is not False or campaign_git.get('commit') != commit:
                run_issues.append('El commit de la corrida difiere del commit limpio de la campaña.')
            code_hashes, profile_hashes, profile_issues = archived_job_evidence(path, run, job)
            if any(campaign_hashes.get(name) != digest for name, digest in code_hashes.items()):
                run_issues.append('El código archivado difiere del fijado antes de la campaña.')
            if diagnostic:
                for name in ('cluster', 'protocol', 'diagnosis/historical-campaign.json'):
                    archived = manifest._internal_file(path, 'raw/inputs/' + name)
                    if campaign_hashes.get(name) != manifest.sha256_file(archived):
                        run_issues.append(f'{name}: entrada archivada difiere del índice de campaña.')
        else:
            legacy_workloads = {i['name']:i['sha256'] for i in run['inputs'] if i['name'].startswith('workload/')}
        seen_cells=set()
        for cell in assessment['cells']:
            if cell['status']!='valid': continue
            slot=(cell['input_tokens'],cell['concurrency'])
            if diagnostic: slot += (cell.get('mode'),)
            if slot in seen_cells:
                raise InferenceError('celda de grupo duplicada en comparación pareada')
            seen_cells.add(slot)
            coordinate = {'input_tokens': slot[0], 'concurrency': slot[1]}
            if diagnostic and 'mode' in cell: coordinate['mode'] = cell['mode']
            if coordinate not in job['cells'] or (diagnostic and cell.get('cell') != cell_id(coordinate)):
                raise InferenceError('celda fuera de la carga solicitada')
            # Identity includes source commit, protocol, model/tokenizer revision,
            # exact datasets and the trial. Node count alone differs.
            issues = list(run_issues)
            length = cell['input_tokens']
            if reference_protocol:
                workload_hashes = profile_hashes[length]
                issues.extend(profile_issues[length])
                if any(campaign_hashes.get(name) != digest for name, digest in workload_hashes.items()):
                    issues.append('La carga archivada difiere de la preparada para este modelo y longitud.')
                issues.extend(verify_client_cell(path, job['model'], cell, protocol))
            else:
                workload_hashes = legacy_workloads
            identity={'commit':commit,'protocol':request['protocol'],'model':model,
                'block':job['block'],'input_tokens':cell['input_tokens'],'concurrency':cell['concurrency'],
                'workload_hashes':workload_hashes,
                'effective_environment':json.loads(head['container_probe']['stdout'].splitlines()[-1]),
                'snapshot_identity':head['models'][job['model']['key']]}
            # Keep legacy comparison keys byte-for-byte reproducible. The new
            # protocol also binds code inputs and campaign identity explicitly.
            if reference_protocol:
                identity['campaign_id']=request['campaign_id']
                identity['code_hashes']=code_hashes
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            summary={**cell,'comparison_key':key,'model_key':ref['model_key'],'block':ref['block'],
                     'group':ref['group'],'run_dir':str(path)}
            if diagnostic:
                ranks = [read_json(path / p['evidence_path']) for p in run['participants']]
                diagnostic_issues = diagnostic_cell_issues(path, job['model'], cell, ranks)
                quality_issues = _cell_quality_issues(path, cell, ranks, verify_receipt=True)
                issues.extend(quality_issues)
                summary.update(series=job['series'], diagnostic_valid=not diagnostic_issues,
                               diagnostic_issues=diagnostic_issues, execution_valid=not quality_issues,
                               execution_issues=quality_issues)
                if cell.get('mode') == 'on':
                    summary['profile_control_valid'] = not any(
                        issue.startswith('profile control:') for issue in diagnostic_issues)
                    summary['traces_valid'] = not any(
                        issue.startswith('traces:') for issue in diagnostic_issues)
                    summary['profile_coverage_valid'] = not any(
                        issue.startswith('profile coverage:') for issue in diagnostic_issues)
                    summary['trace_coverage'] = 'limited; START.sent to STOP.ack, at most 60s/32 iterations; may cover only a prefix of T32'
                if campaign['status'] == 'complete' and diagnostic_issues:
                    raise InferenceError('COMPLETE contiene evidencia diagnóstica incompleta')
                raw_rows.append(summary)
            expected_counts={'requested':protocol['measurement']['measured_requests'],
                'completed':protocol['measurement']['measured_requests'],'failed':0,
                'input_tokens':protocol['measurement']['measured_requests']*cell['input_tokens'],
                'output_tokens':protocol['measurement']['measured_requests']*protocol['workload']['output_tokens']}
            counts=cell.get('raw_counts',{})
            if not isinstance(counts,dict) or any(
                    type(counts.get(k)) is not int or counts[k]!=v for k,v in expected_counts.items()):
                issues.append('Los conteos difieren del lote medido fijado para la celda.')
            if not workload_hashes:
                issues.append('Faltan hashes de la carga.')
            if reference_protocol:
                summary['node_count']=len(observed)
                if issues:
                    if diagnostic and campaign['status'] == 'complete':
                        raise InferenceError('COMPLETE contiene una celda sin evidencia íntegra')
                    excluded.append({**summary,'status':'excluded','reasons':issues})
                    continue
            if diagnostic and job['series'] == 'auxiliary':
                auxiliary_rows.append(summary)
                continue
            rows.append(summary)
            if not issues:
                pairs.setdefault((ref['model_key'],ref['block'],*slot[:2]),{})[ref['group']] = summary
        if campaign['status']=='complete' and len(seen_cells)!=len(job['cells']):
            raise InferenceError('COMPLETE contiene una matriz de celdas incompleta')
    comparisons=[]
    for group in pairs.values():
        sample=next(iter(group.values()))
        comparison={'model':sample['model_key'],'block':sample['block'],
                    'input_tokens':sample['input_tokens'],'concurrency':sample['concurrency']}
        for scaled_group,baseline_group in (('four','pair'),('pair','single'),('four','single')):
            if not {scaled_group,baseline_group}<=group.keys(): continue
            scaled,baseline=group[scaled_group],group[baseline_group]
            if not _pair_compatible(baseline,scaled,identities): continue
            try:
                comparison.update(paired_speedup(baseline,scaled,
                    baseline_nodes=len(identities[baseline['run_dir']]),
                    scaled_nodes=len(identities[scaled['run_dir']])))
            except ValueError:
                # A rejected pair must not discard another valid pair in this cell.
                continue
        if 's4_2' in comparison or 's2_1' in comparison or 's4_1' in comparison:
            comparisons.append(comparison)
    report={'campaign_id':campaign['campaign_id'],'status':campaign['status'],
            'pending_jobs':campaign['pending_jobs'],'runs':campaign['runs'],'cells':rows,
            'paired_comparisons':comparisons,'limitations':campaign['protocol']['limitations']+
                ['p50/p95 descriptivos con32solicitudes por celda; no inferir p99 poblacional.',
                 'Cliente en Pegasus compartido con servicio; saturación del cliente no descartada solo por benchmark JSON.']}
    if reference_protocol:
        report.update(protocol_id=protocol_id, excluded_cells=excluded)
    if diagnostic:
        if campaign['status'] == 'complete' and diagnosis.cell_counts('primary', rows) != {
                'primary': 54, 'off': 0, 'on': 0}:
            raise InferenceError('COMPLETE requiere exactamente 54 celdas primarias válidas')
        if campaign['status'] == 'complete' and diagnosis.cell_counts('auxiliary', auxiliary_rows) != {
                'primary': 0, 'off': 12, 'on': 12}:
            raise InferenceError('COMPLETE requiere exactamente 12 OFF y 12 ON válidas')
        report['auxiliary_cells'] = auxiliary_rows
        report['raw_cells'] = raw_rows
        report['limitations'].append('La traza ON cubre como máximo 60 s entre START.sent y STOP.ack y 32 iteraciones; puede ser solo un prefijo de T32.')
    return report


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
        lines.append(f"| {model} | {len(TOPOLOGIES[group])} | {length} | {concurrency} | {len(cells)} | "
                     f"{spread([c['t32_seconds'] for c in cells])} | "
                     f"{spread([c['output_tokens_per_second'] for c in cells])} |")
    lines += ['','| Modelo | Nodos | Entrada | Concurrencia | TTFT p50/p95 ms | TPOT p50/p95 ms | E2E p50/p95 ms |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for (model,group,length,concurrency),cells in sorted(grouped.items()):
        values=[' / '.join(f"{statistics.median(c[metric][p] for c in cells):.3f}" for p in ('p50_ms','p95_ms'))
                for metric in ('ttft','tpot','e2el')]
        lines.append(f"| {model} | {len(TOPOLOGIES[group])} | {length} | {concurrency} | "+' | '.join(values)+' |')
    if not grouped: lines += ['','No hay celdas válidas para resumir.']
    if report.get('excluded_cells'):
        lines += ['', f"Celdas excluidas de medianas y comparaciones: {len(report['excluded_cells'])}. "
                  'El JSON conserva sus observaciones y motivos en `excluded_cells`.']
    metric_columns=['s4_2','e4_2']
    if (report.get('protocol_id') in (PEGASUS_REFERENCE_PROTOCOL, DIAGNOSIS_PROTOCOL)
            or any(run['group']=='single' for run in report['runs'])):
        metric_columns+=['s2_1','e2_1','s4_1','e4_1']
    titles={'s4_2':'S4/2','e4_2':'E4/2','s2_1':'S2/1','e2_1':'E2/1','s4_1':'S4/1','e4_1':'E4/1'}
    lines += ['','## Comparaciones pareadas por bloque','',
           '| Modelo | Bloque | Entrada | Concurrencia | '+' | '.join(titles[k] for k in metric_columns)+' |',
           '| --- | ---: | ---: | ---: | '+' | '.join('---:' for _ in metric_columns)+' |']
    for row in report['paired_comparisons']:
        values=[f'{row[k]:.3f}' if k in row else '—' for k in metric_columns]
        lines.append(f"| {row['model']} | {row['block']} | {row['input_tokens']} | {row['concurrency']} | "+' | '.join(values)+' |')
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
            p.add_argument('--execute',action='store_true',help='ejecutar campaña; requiere autorización específica de comandos, duración, recursos y salidas')
            p.add_argument('--session-limit-seconds',type=int,
                help='reducir el presupuesto de ocupación; nunca amplía el máximo del protocolo configurado')
    p=sub.add_parser('report');p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot',action='store_true',help='añadir SVG científico con Matplotlib ya instalado')
    args=parser.parse_args(argv)
    try:
        output = getattr(args, 'output', None)
        if output and output.resolve().is_relative_to((REPO_ROOT/'resultados/raw').resolve()):
            raise InferenceError('el plan, el preflight y los informes derivados deben escribirse fuera de raw')
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
