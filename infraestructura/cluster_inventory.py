"""Inventario pasivo distribuido. `plan` no escribe ni abre conexiones."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
from pathlib import Path

from infraestructura import cluster, manifest
from infraestructura.rutas import REPO_ROOT


# Se transmite por stdin a `python3 -c ...`; no requiere instalar este repositorio.
COLLECTOR = r'''
import json, os, platform, socket, subprocess, sys, time
from pathlib import Path
DEADLINE=time.monotonic()+39

def read(path):
    try:
        return {"status":"ok", "value":Path(path).read_text(errors="replace").strip(), "source":path}
    except FileNotFoundError:
        return {"status":"missing", "value":None, "source":path}
    except PermissionError:
        return {"status":"denied", "value":None, "source":path}
    except OSError as exc:
        return {"status":"error", "value":None, "source":path, "error":str(exc)}

def _command(argv, timeout=5):
    remaining=DEADLINE-time.monotonic()
    if remaining <= 0:
        return {"status":"timeout", "command":argv, "error":"global collector deadline"}
    try:
        p=subprocess.run(argv, text=True, capture_output=True, timeout=min(timeout,remaining), check=False)
        return {"status":"ok" if p.returncode == 0 else "error", "command":argv,
                "returncode":p.returncode, "stdout":p.stdout, "stderr":p.stderr}
    except FileNotFoundError:
        return {"status":"missing", "command":argv}
    except PermissionError:
        return {"status":"denied", "command":argv}
    except subprocess.TimeoutExpired:
        return {"status":"timeout", "command":argv, "timeout_seconds":timeout}
    except OSError as exc:
        return {"status":"error", "command":argv, "error":str(exc)}

def command(argv, timeout=5):
    result=_command(argv, timeout)
    print(json.dumps({"event":"probe", "result":result}, ensure_ascii=False), flush=True)
    return result

def sysfs(base, names):
    return {name:read(str(base/name)) for name in names}

request=json.loads(sys.argv[1])
interfaces={}
for base in sorted(Path('/sys/class/net').glob('*')):
    if base.is_dir():
        interfaces[base.name]=sysfs(base, ['address','mtu','speed','duplex','operstate',
            'device/vendor','device/device','device/firmware_node/path',
            'statistics/rx_bytes','statistics/tx_bytes','statistics/rx_errors',
            'statistics/tx_errors','statistics/rx_dropped','statistics/tx_dropped'])
disks={}
for base in sorted(Path('/sys/block').glob('*')):
    if base.is_dir() and not base.name.startswith(('loop','ram')):
        disks[base.name]=sysfs(base,['size','queue/rotational','device/model'])
output={"schema_version":"1.0", "node_id":request['node_id'],
    "hostname":socket.gethostname(), "rank":None,
    "observed_at":__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
    "probes":{
      "dmi":sysfs(Path('/sys/class/dmi/id'),['sys_vendor','product_name','product_version']),
      "cpu":read('/proc/cpuinfo'), "cpu_details":command(['lscpu','--json']),
      "memory":read('/proc/meminfo'),
      "os_release":read('/etc/os-release'), "kernel":{"status":"ok","value":platform.release(),"source":"platform.release"},
      "disks":{"status":"ok","value":disks,"source":"/sys/block"},
      "interfaces":{"status":"ok","value":interfaces,"source":"/sys/class/net"},
      "addresses":command(['ip','-j','address','show']),
      "routes":{peer:command(['ip','-j','route','get',peer]) for peer in request['peers']},
      "gpu":command(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader,nounits']),
      "gpu_driver_cuda":command(['nvidia-smi']),
      "cuda_compiler":command(['nvcc','--version']),
      "nccl_host":command(['ldconfig','-p']),
      "nccl_packages":command(['dpkg-query','-W','-f=${Package} ${Version}\\n','libnccl*','nccl*']),
      "pci":command(['lspci','-nn']),
      "storage_available":command(['df','-B1','-P']),
      "nic_firmware":{name:command(['ethtool','-i',name]) for name in interfaces},
      "nic_link":{name:command(['ethtool',name]) for name in interfaces},
      "rdma":command(['rdma','-j','link','show']),
      "rdma_mapping":command(['ibdev2netdev','-v']),
      "lldp":command(['lldpctl','-f','json']),
    }}
print(json.dumps(output,ensure_ascii=False,allow_nan=False))
'''


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _required_errors(record: dict, node: dict, nodes: list[dict]) -> list[str]:
    errors = []
    if record.get("hostname") != node["expected_hostname"]:
        errors.append(f"{node['node_id']}: hostname observado inesperado")
    probes = record.get("probes") or {}
    for key in ("cpu", "memory", "os_release", "gpu", "interfaces", "addresses"):
        if probes.get(key, {}).get("status") != "ok":
            errors.append(f"{node['node_id']}: {key} sin evidencia completa")
    expected_peers = {address for other in nodes if other["node_id"] != node["node_id"]
                      for address in other["data_addresses"]}
    if not isinstance(probes.get("routes"), dict) or set(probes["routes"]) != expected_peers:
        errors.append(f"{node['node_id']}: rutas hacia peers incompletas")
    for address, route in (probes.get("routes") or {}).items():
        if route.get("status") != "ok":
            errors.append(f"{node['node_id']}: ruta hacia {address} sin evidencia completa")
    gpu = probes.get("gpu", {})
    rows = [line.strip() for line in gpu.get("stdout", "").splitlines() if line.strip()]
    uuids = [row.split(",")[1].strip() for row in rows if len(row.split(",")) >= 2]
    if not uuids or any(not uuid.startswith("GPU-") for uuid in uuids):
        errors.append(f"{node['node_id']}: UUID GPU ausente o inválido")
    record["gpu_uuids"] = uuids
    record["gpu_uuid"] = uuids[0] if uuids else None
    return errors


def _node_request(node: dict, nodes: list[dict]) -> str:
    peers = sorted({address for other in nodes if other["node_id"] != node["node_id"]
                    for address in other["data_addresses"]})
    return json.dumps({"node_id": node["node_id"], "peers": peers}, separators=(",", ":"))


def _collector_command(node: dict, nodes: list[dict]) -> list[str]:
    python = ["python3", "-c", "import sys; exec(sys.stdin.read())", _node_request(node, nodes)]
    return cluster.ssh_command(node, ["timeout", "-k", "2s", "42s", *python])


def plan(config_path: str | Path) -> dict:
    config = cluster.load_config(config_path)
    nodes = cluster.selected_nodes(config, "four")
    return {"cluster_id": config["cluster_id"], "status": "planned_only",
            "collector_sha256": hashlib.sha256(COLLECTOR.encode()).hexdigest(),
            "collector_source": str(Path(__file__).resolve()),
            "collector_stdin_bytes": len(COLLECTOR.encode()),
            "nodes": [{"node_id": n["node_id"], "expected_hostname": n["expected_hostname"],
                       "local": n["local"], "access_status": n["access_status"],
                       "command": _collector_command(n, nodes)}
                      for n in nodes],
            "notice": "plan no verifica acceso, hardware, enlace ni reposo"}


def run(config_path: str | Path, *, allow_dirty: bool = False,
        results_root: str | Path = "resultados/raw") -> Path:
    config_path = Path(config_path).resolve()
    config = cluster.load_config(config_path)
    # Inventariar los cuatro verifica disponibilidad; no ejecuta carga distribuida.
    nodes = cluster.selected_nodes(config, "four")
    sources = [Path(__file__).resolve(), Path(cluster.__file__).resolve(),
               Path(manifest.__file__).resolve(), REPO_ROOT / "infraestructura/rutas.py",
               REPO_ROOT / "infraestructura/schemas/run-manifest.schema.json",
               REPO_ROOT / "infraestructura/schemas/run-context.schema.json"]
    directory = manifest.start_run(kind="cluster-inventory", model_id=None,
        model_revision=None, tokenizer_id=None, tokenizer_revision=None,
        requested_config=config, requested_config_path=config_path,
        inputs={str(p.relative_to(REPO_ROOT)): p for p in sources}, repo_path=REPO_ROOT,
        allow_dirty=allow_dirty, results_root=results_root)
    print(f"RUN_DIRECTORY={directory}", flush=True)
    errors = []
    participants = []
    try:
        _write_json(directory / "raw" / "launch.json", {
            "command": shlex.join([sys.executable, "-m", "infraestructura.cluster_inventory", *sys.argv[1:]]),
            "argv": sys.argv, "node_count_requested": len(nodes),
            "nodes_requested": [{"node_id": n["node_id"], "hostname": n["expected_hostname"],
                                  "role": "inventory", "rank": None} for n in nodes],
            "strategy": "independent passive probes", "parallelism": {"dp": None, "tp": None, "pp": None},
            "warmup_seconds": 0, "measurement_window": "single snapshot; no benchmark",
            "model": None, "dataset": None})
        source_dir = directory / "raw" / "sources"
        source_dir.mkdir()
        for source in [*sources, config_path]:
            (source_dir / source.name).write_bytes(source.read_bytes())
        for node in nodes:
            command = _collector_command(node, nodes)
            log = directory / "logs" / f"{node['node_id']}.json"
            try:
                process = subprocess.run(command, input=COLLECTOR, text=True,
                                         capture_output=True, timeout=45, check=False)
                _write_json(log, {"command": command, "returncode": process.returncode,
                                  "stdout": process.stdout, "stderr": process.stderr})
                if process.returncode != 0:
                    errors.append(f"{node['node_id']}: collector exit {process.returncode}")
                    continue
                lines = process.stdout.splitlines()
                if not lines:
                    raise ValueError("colector sin respuesta JSON final")
                record = json.loads(lines[-1])
                if not isinstance(record, dict):
                    raise ValueError("respuesta del colector no es objeto JSON")
                if record.get("node_id") != node["node_id"]:
                    errors.append(f"{node['node_id']}: node_id observado inesperado")
                    continue
                errors.extend(_required_errors(record, node, nodes))
                path = directory / "raw" / "nodes" / f"{node['node_id']}.json"
                _write_json(path, record)
                if record.get("hostname") == node["expected_hostname"] and record["gpu_uuids"]:
                    participants.append({"node_id": node["node_id"], "hostname": record["hostname"],
                                         "gpu_uuid": record["gpu_uuids"][0], "rank": None,
                                         "evidence_path": str(path.relative_to(directory))})
            except (OSError, subprocess.TimeoutExpired, ValueError, TypeError) as exc:
                errors.append(f"{node['node_id']}: {type(exc).__name__}: {exc}")
                if not log.exists():
                    def partial(value):
                        return value.decode(errors="replace") if isinstance(value, bytes) else value
                    _write_json(log, {"command": command, "status": "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "error",
                                      "error": str(exc),
                                      "stdout_partial": partial(exc.stdout)
                                          if isinstance(exc, subprocess.TimeoutExpired) else None,
                                      "stderr_partial": partial(exc.stderr)
                                          if isinstance(exc, subprocess.TimeoutExpired) else None})
        if participants:
            manifest.register_participants(directory, participants)
        if len(participants) != len(nodes):
            errors.append(f"participantes verificados {len(participants)}/{len(nodes)}")
        manifest.finalize_run(directory, status="failed" if errors else "complete", errors=errors)
    except BaseException as exc:
        if not (directory / "COMPLETE").exists() and not (directory / "FAILED").exists():
            manifest.finalize_run(directory, status="failed", errors=[f"{type(exc).__name__}: {exc}", *errors])
        raise
    return directory


def _lldp_neighbors(probe: dict) -> list[dict]:
    if probe.get("status") != "ok":
        return []
    try:
        document = json.loads(probe.get("stdout", ""))
    except (ValueError, TypeError):
        return []
    if not isinstance(document, dict) or not isinstance(document.get("lldp"), dict):
        return []
    interfaces = document.get("lldp", {}).get("interface", {})
    if isinstance(interfaces, dict):
        entries = interfaces.items()
    elif isinstance(interfaces, list):
        entries = ((local, entry) for index, item in enumerate(interfaces)
                   if isinstance(item, dict)
                   for local, entry in (item.items() if len(item) == 1 else [(str(index), item)]))
    else:
        return []
    def unwrap(value):
        if isinstance(value, dict) and "id" not in value and "name" not in value and len(value) == 1:
            name, inner = next(iter(value.items()))
            return inner, name
        return value, None
    def ident(value):
        if isinstance(value, dict):
            return str(value.get("value") or value.get("name") or "desconocido")
        return str(value or "desconocido")
    neighbors = []
    for local, entry in entries:
        if not isinstance(entry, dict):
            continue
        chassis, chassis_key = unwrap(entry.get("chassis", {}))
        port, _ = unwrap(entry.get("port", {}))
        if not isinstance(chassis, dict) or not isinstance(port, dict):
            continue
        chassis_id = ident(chassis.get("id"))
        if chassis_id == "desconocido":
            continue
        neighbors.append({"interface": str(local), "chassis_id": chassis_id,
                          "chassis_name": str(chassis.get("name") or chassis_key or "desconocido"),
                          "chassis_descr": str(chassis.get("descr") or "desconocido"),
                          "mgmt_ips": chassis.get("mgmt-ip") if isinstance(chassis.get("mgmt-ip"), list)
                                      else ([chassis["mgmt-ip"]] if chassis.get("mgmt-ip") else []),
                          "remote_port": ident(port.get("id")),
                          "port_descr": str(port.get("descr") or "desconocido")})
    return neighbors


def _nccl_packages(probe: dict) -> str:
    rows = (probe.get("stdout") or "").replace("\\n", "\n").splitlines()
    packages = [line.strip() for line in rows if re.match(r"^(?:libnccl|nccl)[a-z0-9+.-]*\s", line)]
    if probe.get("status") != "ok":
        return ("consulta parcial (dpkg no completó): " + "; ".join(packages)
                if packages else "consulta dpkg fallida; paquetes NCCL desconocidos")
    return "; ".join(packages) or "ningún paquete NCCL registrado por dpkg"


def render(run_directory: str | Path, output: str | Path) -> Path:
    run, output = Path(run_directory).resolve(), Path(output).resolve()
    raw_root = next((parent for parent in run.parents if parent.name == "raw"), run)
    if output.is_relative_to(raw_root) or any(
        (parent / "manifest.json").exists() or (parent / "COMPLETE").exists()
        or (parent / "FAILED").exists() for parent in output.parents
    ):
        raise ValueError("informe debe quedar fuera de raw")
    data = manifest.load_run(run)
    if data["kind"] != "cluster-inventory" or data["status"] not in {"complete", "failed"}:
        raise ValueError("se requiere inventario cluster finalizado")
    manifest.verify_artifacts(run, data)
    indexed = {item["path"]: item for item in data["artifacts"]}
    nodes = data["requested_config"]["nodes"]
    lines = ["# Inventario distribuido", "", f"- Corrida: `{data['run_id']}`; estado: **{data['status']}**.",
             f"- Commit: `{data['git'].get('commit')}`; dirty: `{data['git'].get('dirty')}`.",
             f"- Generador del informe: `infraestructura/cluster_inventory.py`; SHA-256: `{manifest.sha256_file(Path(__file__))}`.",
             "- El commit y la configuración declarada corresponden a la captura; el generador puede ser posterior.",
             "- Las direcciones de configuración son declaradas. El inventario no verifica reposo ni readiness NCCL.",
             "", "| Nodo | Host esperado | Host observado | GPU UUID observado | Estado |", "| --- | --- | --- | --- | --- |"]
    observed = {}
    for node in nodes:
        relative = f"raw/nodes/{node['node_id']}.json"
        if relative in indexed:
            rec = json.loads((run / relative).read_text())
            observed[node["node_id"]] = rec
            uuid = ", ".join(rec.get("gpu_uuids", [])) or "desconocido"
            status = "observado" if rec.get("hostname") == node["expected_hostname"] else "mismatch"
            host = rec.get("hostname", "desconocido")
        else:
            uuid, status, host = "desconocido", "sin evidencia", "desconocido"
        lines.append(f"| {node['node_id']} | {node['expected_hostname']} | {host} | {uuid} | {status} |")
    lines += ["", "## Hardware y sistema observados", "",
              "| Nodo | Fabricante DMI | Modelo DMI | CPU | Memoria MemTotal | GPU/driver | SO | Discos |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    def value(probe):
        return str(probe.get("value", "desconocido")) if probe.get("status") == "ok" else "desconocido"
    for node in nodes:
        rec = observed.get(node["node_id"], {})
        probes = rec.get("probes", {})
        dmi = probes.get("dmi", {})
        memory = probes.get("memory", {})
        memline = next((line.strip() for line in memory.get("value", "").splitlines()
                        if line.startswith("MemTotal:")), "desconocido") if memory.get("status") == "ok" else "desconocido"
        gpu = probes.get("gpu", {})
        gpuline = gpu.get("stdout", "").splitlines()[0] if gpu.get("status") == "ok" and gpu.get("stdout") else "desconocido"
        interfaces = probes.get("interfaces", {}).get("value", {})
        cpu = probes.get("cpu", {})
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpu.get("value", "").splitlines()
                          if line.lower().startswith(("model name", "hardware")) and ":" in line),
                         "desconocido") if cpu.get("status") == "ok" else "desconocido"
        if cpu_model == "desconocido" and probes.get("cpu_details", {}).get("status") == "ok":
            try:
                details = json.loads(probes["cpu_details"]["stdout"]).get("lscpu", [])
                def models(entries):
                    found = []
                    for item in entries:
                        if item.get("field", "").rstrip(":") == "Model name" and item.get("data"):
                            found.append(str(item["data"]))
                        found.extend(models(item.get("children", [])))
                    return found
                cpu_model = " + ".join(dict.fromkeys(models(details))) or "desconocido"
            except (ValueError, KeyError, TypeError):
                pass
        if cpu.get("status") == "ok":
            cpu_count = sum(line.lower().startswith("processor") and ":" in line
                            for line in cpu.get("value", "").splitlines())
            if cpu_count:
                cpu_model += f" ({cpu_count} CPU lógicas)"
        os_release = probes.get("os_release", {})
        os_name = next((line.split("=", 1)[1].strip('"') for line in os_release.get("value", "").splitlines()
                        if line.startswith("PRETTY_NAME=")), "desconocido") if os_release.get("status") == "ok" else "desconocido"
        disks = probes.get("disks", {}).get("value", {})
        disk_entries = []
        if isinstance(disks, dict):
            for disk_name, fields in sorted(disks.items()):
                disk_model = value(fields.get("device/model", {})) if isinstance(fields, dict) else "desconocido"
                sectors = value(fields.get("size", {})) if isinstance(fields, dict) else "desconocido"
                size = f"{int(sectors) * 512} bytes" if sectors.isdigit() else "capacidad desconocida"
                disk_entries.append(f"{disk_name}: {disk_model} ({size})")
        disk_names = "; ".join(disk_entries) or "desconocido"
        lines.append(f"| {node['node_id']} | {value(dmi.get('sys_vendor', {}))} | "
                     f"{value(dmi.get('product_name', {}))} | {cpu_model} | {memline} | {gpuline} | {os_name} | {disk_names} |")
    lines += ["", "## Acceso y red", "",
              "| Nodo | Acceso declarado | Resultado colector observado | Interfaces observadas (velocidad/MTU/estado) | Firmware NIC | RDMA/mapeo | Almacenamiento disponible |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for node in nodes:
        probes = observed.get(node["node_id"], {}).get("probes", {})
        log_relative = f"logs/{node['node_id']}.json"
        log = json.loads((run / log_relative).read_text()) if log_relative in indexed else {}
        outcome = f"exit {log['returncode']}" if "returncode" in log else log.get("status", "desconocido")
        interfaces = probes.get("interfaces", {}).get("value", {})
        entries = []
        if isinstance(interfaces, dict):
            for name, fields in sorted(interfaces.items()):
                speed = value(fields.get("speed", {})) if isinstance(fields, dict) else "desconocido"
                mtu = value(fields.get("mtu", {})) if isinstance(fields, dict) else "desconocido"
                state = value(fields.get("operstate", {})) if isinstance(fields, dict) else "desconocido"
                speed_text = f"{speed} Mb/s" if speed.isdigit() else "velocidad desconocida"
                entries.append(f"{name} ({speed_text}; MTU {mtu}; estado {state})")
        rdma = probes.get("rdma_mapping", {})
        rdma_text = rdma.get("stdout", "").strip().replace("\n", "; ") if rdma.get("status") == "ok" else "desconocido"
        firmware = []
        for name, probe in sorted(probes.get("nic_firmware", {}).items()):
            if probe.get("status") == "ok":
                version = next((line.split(":", 1)[1].strip() for line in probe.get("stdout", "").splitlines()
                                if line.startswith("firmware-version:")), "desconocida")
                firmware.append(f"{name}: {version}")
        storage = probes.get("storage_available", {})
        storage_text = "desconocido"
        if storage.get("status") == "ok":
            for line in storage.get("stdout", "").splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 6 and parts[-1] == "/":
                    storage_text = f"{parts[3]} bytes libres en /"
                    break
        access = ("local" if node["local"] else
                  f"ssh -p {node['ssh']['port']} {node['ssh']['user']}@{node['ssh']['host']}")
        lines.append(f"| {node['node_id']} | `{access}`; {node['access_status']} | {outcome} | "
                     f"{'; '.join(entries) or 'desconocido'} | {'; '.join(firmware) or 'desconocido'} | "
                     f"{rdma_text or 'ningún puerto listado'} | {storage_text} |")
    lines += ["", "## Versiones de software observadas", "",
              "| Nodo | Kernel | CUDA compiler | NCCL packages | PCI inventory |",
              "| --- | --- | --- | --- | --- |"]
    for node in nodes:
        probes = observed.get(node["node_id"], {}).get("probes", {})
        cuda = probes.get("cuda_compiler", {})
        cuda_release = next((line.strip() for line in cuda.get("stdout", "").splitlines()
                             if "release" in line), "desconocido") if cuda.get("status") == "ok" else "desconocido"
        nccl = probes.get("nccl_packages", {})
        nccl_version = _nccl_packages(nccl)
        pci = "lspci registrado en raw" if probes.get("pci", {}).get("status") == "ok" else "desconocido"
        lines.append(f"| {node['node_id']} | {value(probes.get('kernel', {}))} | "
                     f"{cuda_release} | {nccl_version} | {pci} |")
    lines += ["", "La consulta dpkg no determina si libnccl está instalada fuera de paquetes dpkg."]
    lines += ["", "## Sondas sin resultado", ""]
    missing = []
    def failures(prefix, item):
        if not isinstance(item, dict):
            return
        if "status" in item and item["status"] != "ok":
            missing.append(f"- `{prefix}`: {item['status']}.")
            return
        for key, child in item.items():
            if isinstance(child, dict):
                failures(f"{prefix}.{key}", child)
    for node in nodes:
        probes = observed.get(node["node_id"], {}).get("probes", {})
        for name, probe in probes.items():
            failures(f"{node['node_id']}.{name}", probe)
        if not probes:
            missing.append(f"- `{node['node_id']}`: sin evidencia de sondas.")
    lines.extend(missing or ["- Ninguna sonda principal fallida registrada."])
    neighbors = []
    for node in nodes:
        probe = observed.get(node["node_id"], {}).get("probes", {}).get("lldp", {})
        neighbors.extend({"node_id": node["node_id"], **entry} for entry in _lldp_neighbors(probe))
    lines += ["", "## Vecinos LLDP observados", "",
              "| Nodo | Interfaz local | Chassis ID anunciado | Nombre anunciado | Descripción chassis anunciada | Direcciones gestión anunciadas | Puerto remoto anunciado | Descripción puerto anunciada |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for entry in neighbors:
        lines.append(f"| {entry['node_id']} | {entry['interface']} | {entry['chassis_id']} | "
                     f"{entry['chassis_name']} | {entry['chassis_descr']} | "
                     f"{', '.join(map(str, entry['mgmt_ips'])) or 'desconocido'} | "
                     f"{entry['remote_port']} | {entry['port_descr']} |")
    if not neighbors:
        lines.append("| desconocido | desconocido | desconocido | desconocido | desconocido | desconocido | desconocido | desconocido |")
    lines.append("Los nombres LLDP son anuncios del vecino; no prueban fabricante ni modelo del switch.")
    lines += ["", "## Conexiones", "",
              "Una ruta del kernel indica dónde se enviaría tráfico; no comprueba que el destino responda ni confirma cableado.",
              "Ninguna conexión física se clasifica como confirmada con estas sondas.", "",
              "| Par | Clasificación | Evidencia |", "| --- | --- | --- |"]
    edges = []
    chassis_nodes = {}
    for entry in neighbors:
        chassis_nodes.setdefault(entry["chassis_id"], f"neighbor{len(chassis_nodes)}")
    # Una ruta de kernel hacia la IP de un peer no confirma cableado ni switch.
    for a in nodes:
        for b in nodes:
            if a["node_id"] >= b["node_id"]:
                continue
            ra = observed.get(a["node_id"], {}).get("probes", {}).get("routes", {})
            rb = observed.get(b["node_id"], {}).get("probes", {}).get("routes", {})
            routed = any(ra.get(ip, {}).get("status") == "ok" for ip in b["data_addresses"]) and any(
                rb.get(ip, {}).get("status") == "ok" for ip in a["data_addresses"])
            label = "rutas observadas; conectividad pendiente" if routed else "desconocido"
            classification = "trayecto lógico inferido; enlace físico desconocido" if routed else "desconocida"
            lines.append(f"| {a['node_id']} ↔ {b['node_id']} | {classification} | {label} |")
            edges.append(f"  {a['node_id']} -. \"{label}\" .- {b['node_id']}")
    lines += ["", "```mermaid", "flowchart LR"]
    for node in nodes:
        lines.append(f"  {node['node_id']}[\"{node['node_id']}\"]")
    for chassis, ident in chassis_nodes.items():
        lines.append(f"  {ident}[\"Vecino LLDP {ident}\"]")
    for entry in neighbors:
        lines.append(f"  {entry['node_id']} -- \"LLDP {entry['interface']}\" --> {chassis_nodes[entry['chassis_id']]}")
    lines.extend(edges)
    lines += ["```", "", "No se confirma switch ni cableado mediante rutas del kernel.", "", "## Errores", ""]
    lines.extend(f"- {error.get('message', str(error))}" for error in data.get("errors", []))
    if not data.get("errors"):
        lines.append("- Ninguno registrado.")
    lines += ["", "## Evidencia", ""]
    for node in nodes:
        for relative in (f"raw/nodes/{node['node_id']}.json", f"logs/{node['node_id']}.json"):
            if relative in indexed:
                lines.append(f"- [{node['node_id']} {relative.split('/')[0]}]({os.path.relpath(run / relative, output.parent)}).")
    lines += ["", f"Manifiesto SHA-256: `{manifest.sha256_file(run / 'manifest.json')}`", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--config", required=True)
    p = sub.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--results-root", default="resultados/raw")
    p = sub.add_parser("report")
    p.add_argument("run_directory")
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "plan":
            print(json.dumps(plan(args.config), ensure_ascii=False, indent=2))
        elif args.action == "run":
            directory = run(args.config, allow_dirty=args.allow_dirty, results_root=args.results_root)
            print(directory)
            return 2 if (directory / "FAILED").exists() else 0
        else:
            print(render(args.run_directory, args.output))
        return 0
    except (ValueError, OSError, manifest.ManifestError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
