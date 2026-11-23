"""Sondas y ejecución acotada del SFT en hosts inventariados."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shlex
import time

from infraestructura import cluster, inference, inference_runtime as remote_runtime, manifest
from infraestructura.rutas import REPO_ROOT
from infraestructura.serving_capacity_protocol import _hardware_roster


GIB = 1024 ** 3
OWNER_LABEL = "io.ipre.sft.owner"
CAMPAIGN_LABEL = "io.ipre.sft.campaign"
EVIDENCE_FILES = ("identity.json", "group.json", "steps.jsonl", "receipt.json", "telemetry.jsonl", "failure.json")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")

WORKSPACE_PROBE = r'''
import os
from pathlib import Path

def workspace_resources(reference):
    root = Path(reference)
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError('se requiere un workspace absoluto sin enlaces')
    ancestor = next(path for path in (root, *root.parents) if path.exists())
    fs = os.statvfs(ancestor)
    return {
        'cpu_affinity_count': len(os.sched_getaffinity(0)), 'load_average': os.getloadavg(),
        'workspace': reference, 'workspace_mount_probe': str(ancestor),
        'workspace_writable': os.access(ancestor, os.W_OK | os.X_OK),
        'checkpoint_disk_free_bytes': fs.f_bavail * fs.f_frsize,
        'checkpoint_disk_free_inodes': fs.f_favail,
    }
'''

HOST_PROBE = remote_runtime.PROBE_SOURCE + WORKSPACE_PROBE + r'''
host_resources = workspace_resources(request['workspace'])
# Check the real processes reported by the driver, including graphics clients.
gpu_summary = command(['nvidia-smi'])
# Include bound TCP sockets and IPv6. A listener-only scan misses concurrent bind.
for family in ('tcp', 'tcp6'):
    for line in Path('/proc/net/' + family).read_text().splitlines()[1:]:
        fields = line.split()
        if fields[3] != '06':  # TIME_WAIT survives a normal previous execution.
            ports.add(int(fields[1].split(':')[1], 16))
model = request['models'][0]
snapshot = Path(models[model['key']]['snapshot'])
weights = {}
for shard in models[model['key']].get('shards', []):
    if not shard['present']:
        continue
    p = snapshot / shard['name']
    if not p.resolve().is_relative_to(snapshot.parent.parent.resolve()):
        raise ValueError('peso fuera del cache fijado')
    with p.open('rb') as stream:
        weights[shard['name']] = hashlib.file_digest(stream, 'sha256').hexdigest()
print(json.dumps({'resources': host_resources, 'weight_sha256': weights,
                  'gpu_summary': gpu_summary, 'ports_in_use': sorted(ports.intersection(request['ports']))}))
'''

CPU_PROBE = """
import importlib.metadata
import json, os, platform
from pathlib import Path
import torch, torch.distributed
import transformers, tokenizers
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

def read(path):
    p = Path('/sys/fs/cgroup') / path
    return p.read_text().strip() if p.is_file() else None

print(json.dumps({
    'python': platform.python_version(), 'torch_version': importlib.metadata.version('torch'),
    'torch_runtime_version': str(torch.__version__),
    'transformers_version': transformers.__version__, 'tokenizers_version': tokenizers.__version__,
    'torch_cuda_version': torch.version.cuda, 'nccl_available': torch.distributed.is_nccl_available(),
    'nccl_version': torch.cuda.nccl.version(),
    'cpu_affinity_count': len(os.sched_getaffinity(0)), 'cpu_max': read('cpu.max'),
    'memory_max': read('memory.max'), 'memory_current': read('memory.current'),
    'gpus_hidden': not torch.cuda.is_available()
}))
"""

COLLECT = r'''
import base64, json, sys
from pathlib import Path
root = Path(sys.argv[1])
if root.exists() and root.resolve() != root:
    raise ValueError('directorio de evidencia con enlaces')
out = {}
for name in sys.argv[2:]:
    p = root / name
    if p.is_symlink():
        raise ValueError('evidencia con enlaces: ' + name)
    if p.is_file():
        with p.open('rb') as stream:
            data = stream.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024:
            raise ValueError('evidencia excede 16 MiB: ' + name)
        out[name] = base64.b64encode(data).decode('ascii')
print(json.dumps(out))
'''

STAGE = r'''
import base64, hashlib, json, os, sys
from pathlib import Path
request = json.load(sys.stdin)
root = Path(request['root'])
if not root.is_absolute() or root.resolve() != root:
    raise ValueError('workspace fuera del área operativa o con enlaces')
os.umask(0o077)
root.mkdir(parents=True, exist_ok=False)
for name in ('cache', 'out', 'checkpoint'):
    (root / name).mkdir()
hashes = {}
for relative, encoded in request['files'].items():
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('ruta de entrada inválida')
    dest = root / p
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('xb') as stream:
        stream.write(base64.b64decode(encoded, validate=True))
    dest.chmod(0o400)
    with dest.open('rb') as stream:
        hashes[relative] = hashlib.file_digest(stream, 'sha256').hexdigest()
print(json.dumps({'workspace': str(root), 'uid': os.getuid(), 'files': hashes}))
'''

READY_PROBE = WORKSPACE_PROBE + r'''
import json, subprocess, sys
from pathlib import Path

def call(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=5, check=True)
    return result.stdout

summary = call(['nvidia-smi'])
gpu = call(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader,nounits'])
apps = call(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'])
network = call(['ip', '-j', 'address', 'show', 'dev', 'enp1s0f0np0'])
port = int(sys.argv[1])
ports = set()
for name in ('tcp', 'tcp6'):
    for row in (Path('/proc/net') / name).read_text().splitlines()[1:]:
        fields = row.split()
        if fields[3] != '06' and int(fields[1].split(':')[1], 16) == port:
            ports.add(port)
available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                 if line.startswith('MemAvailable:'))
print(json.dumps({'gpu': gpu, 'apps': apps, 'summary': summary, 'network': network,
                  'ports_in_use': sorted(ports),
                  'memory_bytes': {'MemAvailable': available}, 'resources': workspace_resources(sys.argv[2])}))
'''

STATE_FORMAT = "{{json .State}}"
LABEL_FORMAT = '{{index .Config.Labels "io.ipre.sft.owner"}} {{index .Config.Labels "io.ipre.sft.campaign"}}'


def _describe(exc):
    return f"{type(exc).__name__}: {exc}"


def parallel(nodes, action):
    """Cada llamada conserva su resultado aunque falle otro host."""
    with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        tasks = [(node["node_id"], pool.submit(action, node)) for node in nodes]
        values, errors = {}, {}
        for name, task in tasks:
            try:
                values[name] = task.result()
            except Exception as exc:
                errors[name] = _describe(exc)
    return values, errors


def remaining(deadline, cap):
    # inference_runtime.execute puede dedicar hasta 2 s adicionales a TERM/KILL.
    seconds = min(cap, deadline - time.monotonic() - 3)
    if seconds < 5:
        raise TimeoutError("plazo operativo agotado")
    return seconds


def invoke(node, argv, deadline, *, cap=25, stdin=None):
    return remote_runtime.remote(node, argv, timeout=remaining(deadline, cap), stdin=stdin)


def checked(node, argv, deadline, *, cap=25, stdin=None):
    return remote_runtime.checked(invoke(node, argv, deadline, cap=cap, stdin=stdin), argv[0])


def _labels(campaign_id, owner):
    return ["--label", f"{OWNER_LABEL}={owner}", "--label", f"{CAMPAIGN_LABEL}={campaign_id}"]


def _user_options(probe):
    args = ["--user", f"{probe['uid']}:{probe['gid']}"]
    for group in sorted(set(probe["groups"]) - {probe["gid"]}):
        args += ["--group-add", str(group)]
    return args


def _owned_ids(node, campaign_id, owner, deadline):
    result = checked(node, ["docker", "ps", "-aq", "--no-trunc",
        "--filter", f"label={OWNER_LABEL}={owner}", "--filter", f"label={CAMPAIGN_LABEL}={campaign_id}"], deadline)
    ids = result["stdout"].split()
    if any(not _CONTAINER_ID.fullmatch(cid) for cid in ids):
        raise ValueError("respuesta inesperada al enumerar contenedores propios")
    return ids


def _inspect_owner(node, cid, campaign_id, owner, deadline):
    result = checked(node, ["docker", "inspect", "--type", "container", "--format", LABEL_FORMAT, cid], deadline)
    if result["stdout"].strip() != f"{owner} {campaign_id}":
        raise ValueError(f"etiquetas inesperadas en {cid}")


def cleanup(nodes, campaign_id, owner, deadline):
    def one(node):
        removed = []
        for cid in _owned_ids(node, campaign_id, owner, deadline):
            _inspect_owner(node, cid, campaign_id, owner, deadline)
            checked(node, ["docker", "rm", "--force", cid], deadline)
            removed.append(cid)
        remaining_ids = _owned_ids(node, campaign_id, owner, deadline)
        if remaining_ids:
            raise RuntimeError(f"persisten contenedores propios: {remaining_ids}")
        return {"removed_ids": removed, "remaining_ids": remaining_ids, "confirmed": True}

    values, errors = parallel(nodes, one)
    for name, error in errors.items():
        values[name] = {"confirmed": False, "error": error}
    return {"nodes": values, "removal_confirmed": not errors, "ended_at": manifest.utc_now()}


def _host_probe(node, config, deadline):
    request = {"node_id": node["node_id"], "image_id": config["image"]["id"],
        "hf_hub": config["paths"]["hf_hub"], "workspace": config["paths"]["workspace"],
        "models": [config["model"]], "ports": [config["network"]["master_port"]]}
    result = checked(node, ["python3", "-c", "import sys; exec(sys.stdin.read())",
        json.dumps(request)], deadline, cap=600, stdin=HOST_PROBE)
    host, resources = map(json.loads, result["stdout"].splitlines())
    if resources["weight_sha256"]:
        host["models"][config["model"]["key"]]["weights_content_rehashed"] = True
    return {**host, **resources}


def _cpu_probe(node, config, probe, campaign_id, owner, deadline):
    name = f"ipre-sft-{owner}-preflight"
    argv = ["docker", "create", "--pull=never", "--name", name,
        *_labels(campaign_id, owner), "--network=none", "--read-only",
        "--tmpfs", "/tmp:rw,nosuid,size=256m", *_user_options(probe),
        "--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "CUDA_VISIBLE_DEVICES=",
        "--env", f"USER={probe['user']}", "--env", f"LOGNAME={probe['user']}",
        "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "HOME=/tmp", "--entrypoint", "timeout",
        config["image"]["id"], "--kill-after=2s", "55s", "python3", "-B", "-c", CPU_PROBE]
    cid = create(node, argv, campaign_id, owner, deadline)
    start(node, cid, deadline)
    stopped = checked(node, ["docker", "wait", cid], deadline, cap=64)
    exit_code = stopped["stdout"].strip()[:64]
    try:
        result = logs(node, cid, deadline)
    except Exception as exc:
        raise RuntimeError(f"sonda CPU exit={exit_code!r}; logs no disponibles: {_describe(exc)[-2048:]}") from exc
    diagnostics = (f"exit={exit_code!r}; stdout={result['stdout'][-2048:]!r}; "
                   f"stderr={result['stderr'][-2048:]!r}")
    if exit_code != "0":
        raise RuntimeError(f"el contenedor de preflight CPU terminó con {diagnostics}")
    try:
        payload = json.loads(result["stdout"].splitlines()[-1])
        if not isinstance(payload, dict):
            raise ValueError("la sonda CPU debe emitir un objeto JSON")
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"sonda CPU sin objeto JSON interpretable; {diagnostics}") from exc
    return payload


def _f0(record):
    output = record["network"]
    interfaces = json.loads(output["stdout"]) if output["returncode"] == 0 else []
    return {address["local"] for interface in interfaces if interface["ifname"] == "enp1s0f0np0"
            and "LOWER_UP" in interface["flags"] for address in interface["addr_info"]}


def _table_cells(line):
    text = line.strip()
    if not text.startswith("|") or not text.endswith("|"):
        return []
    return [" ".join(cell.split()) for cell in text[1:-1].split("|")]


def gpu_idle_diagnosis(record):
    """Exige consultas válidas, GPU al 0 % y solo dos ejecutables G conocidos.

    Se lee la tabla completa, incluidos encabezados y cierre, para no aceptar
    tablas truncadas ni confundir ausencia de cómputo con ausencia de clientes.
    Las cantidades de memoria no soportadas conservan su texto observado.
    """
    apps, summary = record["gpu_apps"], record["gpu_summary"]
    for query in (apps, summary):
        if query.get("returncode") != 0 or query.get("timed_out") or not isinstance(query.get("stdout"), str):
            raise ValueError("consulta GPU fallida o sin verificar")
    if apps["stdout"].strip():
        raise ValueError("aplicaciones de cómputo GPU observadas")
    lines = [line.strip() for line in summary["stdout"].splitlines() if line.strip()]
    markers = [i for i, line in enumerate(lines) if _table_cells(line) == ["Processes:"]]
    if len(markers) != 1 or not any(line.startswith("| NVIDIA-SMI ") for line in lines):
        raise ValueError("tabla de procesos GPU ilegible")
    prefix, processes = lines[:markers[0]], lines[markers[0] + 1:]
    headers = [cells for line in prefix if len(cells := _table_cells(line)) == 3
               and any("GPU-Util" in cell for cell in cells)]
    devices = [(i, cells) for i, line in enumerate(prefix)
               if len(cells := _table_cells(line)) == 3 and re.match(r"^\d+\s+", cells[0])]
    if len(headers) != 1 or len(devices) != 1:
        raise ValueError("tabla GPU ausente, ambigua o ilegible")
    index, device = devices[0]
    if device[0].split()[0] != "0" or index + 1 >= len(prefix):
        raise ValueError("se requiere exactamente la GPU local 0")
    column = next(i for i, cell in enumerate(headers[0]) if "GPU-Util" in cell)
    utilization = _table_cells(prefix[index + 1])
    match = re.fullmatch(r"(\d{1,3})%\s+(?:Default|Exclusive_Process)", utilization[column]) if len(utilization) == 3 else None
    if match is None or int(match[1]) != 0:
        raise ValueError("GPU sin utilización 0 % demostrada")
    if (len(processes) < 5
            or _table_cells(processes[0]) != ["GPU GI CI PID Type Process name GPU Memory"]
            or _table_cells(processes[1]) != ["ID ID Usage"]
            or not re.fullmatch(r"\|=+\|", processes[2])
            or not re.fullmatch(r"\+-+\+", processes[-1])):
        raise ValueError("tabla de procesos GPU incompleta o ilegible")
    rows = processes[3:-1]
    if len(rows) == 1 and _table_cells(rows[0]) == ["No running processes found"]:
        graphics = []
    else:
        graphics = []
        for line in rows:
            cells = _table_cells(line)
            row = re.fullmatch(
                r"(\d+)\s+(N/A|\d+)\s+(N/A|\d+)\s+([1-9]\d*)\s+(\S+)\s+(\S+)\s+(\d+(?:\.\d+)?MiB|N/A|Not Supported)",
                cells[0]) if len(cells) == 1 else None
            if (row is None or row[1] != "0" or row[5] != "G"
                    or row[6] not in {"/usr/lib/xorg/Xorg", "/usr/bin/gnome-shell"}):
                raise ValueError("proceso, ruta o tipo GPU no autorizado o ilegible")
            graphics.append({"gpu_index": 0, "gi": None if row[2] == "N/A" else int(row[2]),
                             "ci": None if row[3] == "N/A" else int(row[3]), "pid": int(row[4]),
                             "type": row[5], "executable": row[6], "gpu_memory": row[7]})
        if not graphics or len({row["pid"] for row in graphics}) != len(graphics):
            raise ValueError("tabla de procesos GPU vacía sin constancia o con PIDs repetidos")
    return {"status": "idle", "policy": "desktop-g-zero-util-v1", "gpu_index": 0,
            "gpu_utilization_percent": 0, "compute_apps_empty": True, "graphics_processes": graphics}


def _snapshot(record, model):
    value = record["models"][model["key"]]
    if value["complete"] is not True or not value.get("shards"):
        raise ValueError("snapshot local incompleto")
    return {field: value[field] for field in ("metadata_sha256", "shards")}


def _check_host(node, config, historic, observed):
    name = node["node_id"]
    if observed["hostname"] != node["expected_hostname"] or not 0 < observed["uid"]:
        raise ValueError("identidad de host o usuario inesperada")
    image = observed["image"]
    old_image = historic["image"]
    expected_image = [config["image"]["id"], config["image"]["architecture"]]
    if any(item["returncode"] != 0 or shlex.split(item["stdout"]) != expected_image
           for item in (image, old_image)):
        raise ValueError("imagen inmutable ausente o arquitectura distinta")
    if observed["docker_running"]["returncode"] != 0:
        raise ValueError("estado de Docker desconocido")
    if config["network"]["node_ips"][name] not in _f0(observed):
        raise ValueError("IP f0 no observada en enlace activo")
    if not {"rdma_cm", "uverbs0"} <= set(observed["rdma_devices"]):
        raise ValueError("dispositivos RDMA requeridos ausentes")
    observed["gpu_idle_assessment"] = gpu_idle_diagnosis(observed)
    if observed.get("ports_in_use") != []:
        raise ValueError("puerto de rendezvous ocupado o sin verificar")
    reference = _snapshot(historic, config["model"])
    if _snapshot(observed, config["model"]) != reference:
        raise ValueError("metadatos o shards difieren del preflight de serving")
    expected_digests = {shard["name"]: shard["blob_name"] for shard in reference["shards"]}
    if observed["weight_sha256"] != expected_digests:
        raise ValueError("SHA-256 de contenido de pesos difiere del preflight de serving")


def _check_cpu(config, historic, observed):
    baseline = json.loads(historic["container_probe"]["stdout"].splitlines()[-1])
    actual = observed["sft_container_probe"]
    for package in ("torch", "transformers", "tokenizers"):
        version = config["image"][package + "_version"]
        if actual[package + "_version"] != version or baseline["packages"][package] != version:
            raise ValueError(f"versión de {package} distinta del protocolo o de serving")
    if actual["torch_runtime_version"] != config["image"]["torch_runtime_version"]:
        raise ValueError("versión de ejecución de torch distinta del protocolo")
    if (actual["python"] != baseline["python"] or actual["gpus_hidden"] is not True
            or not actual["torch_cuda_version"] or not actual["nccl_available"] or not actual["nccl_version"]):
        raise ValueError("sonda CPU de Python, Torch CUDA/NCCL o aislamiento falló")


def resource_limits(config, nodes, jobs):
    # SGD sin estado persistente por parámetro. Se retienen todos los pilotos.
    # Cada shard conserva FP32 más 1 GiB para redondeo y archivos auxiliares.
    # El checkpoint ocupa un archivo por tensor, más estado, índice y directorios.
    # Reservamos 8 inodos extra por shard y un piso adicional para el workspace.
    # El piso de memoria cubre maestro + gradiente FP32 y 8 GiB de holgura.
    # La presión de activaciones/all-gather se decide experimentalmente.
    params = config["model"]["parameter_count"]
    limits = {node["node_id"]: {"minimum_unified_bytes": 0, "checkpoint_bytes": 0, "checkpoint_inodes": 0}
              for node in nodes}
    for job in jobs:
        world = len(job["nodes"])
        for name in job["nodes"]:
            limit = limits[name]
            limit["minimum_unified_bytes"] = max(limit["minimum_unified_bytes"], 8 * params // world + 8 * GIB)
            if job["block"] == 0:
                limit["checkpoint_bytes"] += 4 * params // world + GIB
                limit["checkpoint_inodes"] += config["model"]["tensor_count"] + 8
    return limits


def _check_resources(host, limit):
    resources, cpu = host["resources"], host["sft_container_probe"]
    available = host["memory_bytes"]["MemAvailable"]
    memory_max = cpu["memory_max"]
    if memory_max not in (None, "max"):
        available = min(available, int(memory_max) - int(cpu["memory_current"]))
    cores = min(resources["cpu_affinity_count"], cpu["cpu_affinity_count"])
    if cpu["cpu_max"]:
        quota, period = cpu["cpu_max"].split()
        if quota != "max":
            cores = min(cores, int(quota) / int(period))
    if cores <= 0 or available < limit["minimum_unified_bytes"]:
        raise ValueError("CPU disponible o memoria unificada disponible inferior al piso por nodo")
    minimum_inodes = limit["checkpoint_inodes"] + 256
    minimum_disk = limit["checkpoint_bytes"] + GIB
    if (not resources["workspace_writable"] or resources["checkpoint_disk_free_inodes"] < minimum_inodes
            or resources["checkpoint_disk_free_bytes"] < minimum_disk):
        raise ValueError("workspace sin permisos, inodos o espacio para checkpoints retenidos")
    return {"effective_unified_available_bytes": available, "effective_cpu_count": cores,
            "minimum_disk_inodes": minimum_inodes, "minimum_disk_bytes": minimum_disk, **limit}


def preflight(config, cluster_config, jobs, campaign_id, owner, deadline, cleanup_deadline):
    started_at = manifest.utc_now()
    nodes = cluster.selected_nodes(cluster_config, "four")
    first_limits = resource_limits(config, nodes, jobs[:1])
    planned_limits = resource_limits(config, nodes, jobs)
    observed, errors, roster = {}, [], {}
    removal = {"removal_confirmed": True, "nodes": {}}
    containers_possible = False
    try:
        path = REPO_ROOT / config["reference_preflight"]
        if manifest.sha256_file(path) != config["reference_preflight_sha256"]:
            raise ValueError("huella del preflight de serving incompatible")
        reference = inference.read_json(path)
        if reference["status"] != "ready_for_smoke" or reference["errors"] != []:
            raise ValueError("referencia de serving incompleta")
        original, expected_roster = _hardware_roster(reference)
        observed, failures = parallel(nodes, lambda node: _host_probe(node, config, deadline))
        errors.extend(f"{name}: {reason}" for name, reason in failures.items())
        if not errors:
            _, roster = _hardware_roster({"nodes": list(observed.values())})
            for node in nodes:
                name = node["node_id"]
                try:
                    if roster[name] != expected_roster[name]:
                        raise ValueError("hostname, arquitectura, GPU UUID, modelo o driver cambió")
                    observed[name]["gpu_uuid"] = roster[name]["gpu_uuid"]
                    _check_host(node, config, original[name], observed[name])
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append(f"{name}: {_describe(exc)}")
        if not errors:
            containers_possible = True
            versions, failures = parallel(nodes, lambda node: _cpu_probe(
                node, config, observed[node["node_id"]], campaign_id, owner, deadline))
            errors.extend(f"{name}: {reason}" for name, reason in failures.items())
            for name, cpu in versions.items():
                record = observed[name]
                record["sft_container_probe"] = cpu
                try:
                    _check_cpu(config, original[name], record)
                    record["resource_assessment"] = _check_resources(record, first_limits[name])
                    record["planned_peak_and_retained_checkpoints"] = planned_limits[name]
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append(f"{name}: {_describe(exc)}")
    except (Exception, KeyboardInterrupt) as exc:
        errors.append(_describe(exc))
    finally:
        if containers_possible:
            removal = cleanup(nodes, campaign_id, owner, cleanup_deadline)
            if not removal["removal_confirmed"]:
                errors.append("retirada de sondas CPU sin verificar")
    if deadline <= time.monotonic():
        errors.append("preflight agotó el plazo operativo")
    return {"status": "blocked" if errors else "ready", "errors": errors,
        "started_at": started_at, "ended_at": manifest.utc_now(),
        "reference_preflight_sha256": config["reference_preflight_sha256"],
        "nodes": list(observed.values()), "roster": roster, "removal": removal}


def stage(node, workspace, files, hashes, probe, deadline):
    request = {"root": str(workspace), "files": {
        key: base64.b64encode(value).decode("ascii") for key, value in files.items()}}
    result = checked(node, ["python3", "-c", STAGE], deadline, cap=60, stdin=json.dumps(request))
    receipt = json.loads(result["stdout"])
    if receipt != {"workspace": str(workspace), "files": hashes, "uid": probe["uid"]}:
        raise ValueError("copia remota o usuario difiere de las entradas fijadas")
    return receipt


def check_ready(nodes, config, probes, deadline, *, job, phase, workspace):
    limits = resource_limits(config, nodes, [job])
    def check(node):
        name = node["node_id"]
        result = checked(node, ["python3", "-c", READY_PROBE, str(config["network"]["master_port"]), str(workspace)],
                         deadline, cap=30)
        current = json.loads(result["stdout"])
        baseline = probes[name]
        if current["gpu"] != baseline["gpu"]["stdout"]:
            raise ValueError("identidad GPU cambió desde el preflight")
        gpu_evidence = {"gpu_apps": {"returncode": 0, "stdout": current["apps"]},
                        "gpu_summary": {"returncode": 0, "stdout": current["summary"]}}
        gpu_assessment = gpu_idle_diagnosis(gpu_evidence)
        if current.get("ports_in_use") != []:
            raise ValueError("puerto de rendezvous ocupado o sin verificar")
        if config["network"]["node_ips"][name] not in _f0({
            "network": {"returncode": 0, "stdout": current["network"]}
        }):
            raise ValueError("f0 cambió desde el preflight")
        limit = dict(limits[name])
        if phase != "pilot":
            limit.update(checkpoint_bytes=0, checkpoint_inodes=0)
        assessment = _check_resources({**baseline, **current}, limit)
        return {"node_id": name, "phase": phase, "observed_at": manifest.utc_now(),
                "memory_bytes": current["memory_bytes"], "resources": current["resources"],
                "assessment": assessment, "gpu_idle_assessment": gpu_assessment,
                **gpu_evidence, "gpu": {"returncode": 0, "stdout": current["gpu"]},
                "network": {"returncode": 0, "stdout": current["network"]},
                "ports_in_use": current["ports_in_use"]}
    observations, failures = parallel(nodes, check)
    if failures:
        raise RuntimeError(str(failures))
    return observations


def phase_command(config, probe, workspace, campaign_id, owner, name, worker, seconds, *, phase):
    argv = ["docker", "create", "--pull=never", "--name", name,
        *_labels(campaign_id, owner), "--uts=host", "--network=host", "--gpus=all",
        "--device", "/dev/infiniband", "--ulimit", "memlock=-1:-1", "--shm-size=8g",
        *_user_options(probe), "--mount", f"type=bind,src={config['paths']['hf_hub']},dst=/models,readonly",
        "--mount", f"type=bind,src={workspace},dst=/work",
        "--mount", f"type=bind,src={workspace / 'code'},dst=/work/code,readonly",
        "--mount", f"type=bind,src={workspace / 'config.json'},dst=/work/config.json,readonly",
        "--workdir", "/work/code", "--stop-timeout=5", "--init"]
    if phase == "resume":
        argv += ["--mount", f"type=bind,src={workspace / 'checkpoint'},dst=/work/checkpoint,readonly"]
    env = {**config["network"]["environment"], "USER": probe["user"], "LOGNAME": probe["user"],
        "HOME": "/work/cache", "PYTHONPATH": "/work/code", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1", "HF_HOME": "/work/cache/huggingface", "HF_HUB_CACHE": "/models",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "XDG_CACHE_HOME": "/work/cache", "TORCH_HOME": "/work/cache/torch"}
    if "determinism" in config["training"]:
        env["CUBLAS_WORKSPACE_CONFIG"] = config["training"]["determinism"]["cublas_workspace_config"]
    for key, value in env.items():
        argv += ["--env", f"{key}={value}"]
    return [*argv, "--entrypoint", "timeout", config["image"]["id"], "--signal=TERM",
            "--kill-after=5s", f"{seconds}s", *worker]


def create(node, command, campaign_id, owner, deadline):
    cid = checked(node, command, deadline)["stdout"].strip()
    if not _CONTAINER_ID.fullmatch(cid):
        raise ValueError("Docker no devolvió un identificador completo")
    _inspect_owner(node, cid, campaign_id, owner, deadline)
    return cid


def start(node, cid, deadline):
    return checked(node, ["docker", "start", cid], deadline)


def state(node, cid, deadline):
    result = checked(node, ["docker", "inspect", "--type", "container", "--format",
        STATE_FORMAT, cid], deadline, cap=15)
    return json.loads(result["stdout"])


def logs(node, cid, deadline):
    return checked(node, ["docker", "logs", "--tail", "5000", cid], deadline, cap=15)


def collect(node, workspace, phase, deadline):
    result = checked(node, ["python3", "-c", COLLECT, str(workspace / "out" / phase),
        *EVIDENCE_FILES], deadline, cap=30)
    payload = json.loads(result["stdout"])
    if not set(payload) <= set(EVIDENCE_FILES):
        raise ValueError("archivos de evidencia inesperados")
    return {name: base64.b64decode(value, validate=True) for name, value in payload.items()}
