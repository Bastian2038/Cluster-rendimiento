"""Diagnóstico NCCL con plan sin efectos y ejecución autorizada explícitamente.

El plan no consulta hosts ni crea archivos. ``run --execute`` sí inicia procesos;
la autorización operacional externa sigue siendo necesaria.
"""

from __future__ import annotations

import argparse
import getpass
import itertools
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

from infraestructura import manifest
from infraestructura.rutas import REPO_ROOT


class NcclError(RuntimeError):
    pass


OPERATIONS = {"all_reduce": "all_reduce_perf", "all_gather": "all_gather_perf",
              "reduce_scatter": "reduce_scatter_perf", "sendrecv": "sendrecv_perf"}
IDENTITY = "IPRE_RANK_IDENTITY="
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
RANK_LINE = re.compile(r"^\[(?:\d+,)?(\d+)\]<(?:stdout|stderr)>:(.*)$")


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise NcclError(f"se requiere objeto JSON: {path}")
    return value


def _validate_protocol(config, cluster):
    if config.get("schema_version") != "1.0" or config.get("protocol_id") != "nccl-diagnostic-v1":
        raise NcclError("protocolo NCCL no soportado")
    if config.get("session_limit_seconds") != 1800 or config.get("invocation_limit_seconds") != 45:
        raise NcclError("los límites de sesión e invocación son fijos: 1800/45 s")
    if config.get("cleanup_grace_seconds") != 5 or config.get("gpu_per_node") != 1:
        raise NcclError("se requiere 1 GPU/rank y gracia de 5 s")
    if (config.get("dtype"), config.get("warmup_iterations"), config.get("measured_iterations"),
        config.get("check"), config.get("repetitions")) != ("float", 5, 20, 1, 3):
        raise NcclError("parámetros de protocolo inválidos")
    if config.get("environment") != {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,BOOTSTRAP,NET,GRAPH"}:
        raise NcclError("entorno NCCL no permitido")
    if config.get("mpi_launcher") != "mpirun":
        raise NcclError("solo mpirun soportado")
    expected_cells = {
        "smoke": {"operation": "all_reduce", "group": "four", "minimum_bytes": 8,
                  "maximum_bytes": 1048576, "factor": 2},
        "main": {"groups": ["four", "pair"], "operations": ["all_reduce", "all_gather", "reduce_scatter"],
                 "minimum_bytes_by_operation": {"all_reduce": 8, "all_gather": 64, "reduce_scatter": 64},
                 "minimum_reason": "nccl-tests alinea a 16 bytes por rank; con cuatro ranks, tamaños menores a 64 bytes se truncan a cero",
                 "maximum_bytes": 268435456, "factor": 2},
        "diagnostic": {"operation": "sendrecv", "pairs": "all-six", "minimum_bytes": 8,
                       "maximum_bytes": 1048576, "factor": 2},
        "control": {"group": "single", "operation": "all_reduce", "minimum_bytes": 8,
                    "maximum_bytes": 1048576, "factor": 2},
    }
    if any(config.get(name) != cell for name, cell in expected_cells.items()):
        raise NcclError("matriz NCCL no coincide con el protocolo aprobado")
    nodes = cluster.get("nodes", [])
    paths = config.get("binary_paths", {})
    if set(paths) != {node["node_id"] for node in nodes}:
        raise NcclError("binary_paths debe cubrir exactamente los nodos")
    libraries = config.get("library_paths", {})
    if libraries and set(libraries) != set(paths):
        raise NcclError("library_paths debe cubrir exactamente los nodos")
    for path in [*paths.values(), *libraries.values()]:
        if not isinstance(path, str) or not path.startswith("/") or any(c in path for c in "\n\r\0"):
            raise NcclError("ruta de binario absoluta inválida")
    for node in nodes:
        for key in ("node_id", "expected_hostname"):
            if not SAFE_NAME.fullmatch(node[key]):
                raise NcclError(f"{key} inválido")
    groups = cluster.get("groups", {})
    if len(groups.get("four", [])) != 4 or len(groups.get("pair", [])) != 2 or len(groups.get("single", [])) != 1:
        raise NcclError("se requieren grupos four/pair/single de 4/2/1 nodos")
    if groups["single"] != [groups["pair"][0]]:
        raise NcclError("control single debe ser Pegasus")
    return nodes


def _jobs(cluster):
    four, pair, single = (cluster["groups"][key] for key in ("four", "pair", "single"))
    yield {"name": "smoke-four", "group": four, "operation": "all_reduce", "minimum": 8,
           "maximum": 1048576, "repetition": 1, "phase": "smoke"}
    for group_name, group in (("four", four), ("pair", pair)):
        for operation in ("all_reduce", "all_gather", "reduce_scatter"):
            for repetition in range(1, 4):
                yield {"name": f"main-{group_name}-{operation}-{repetition}", "group": group,
                       "operation": operation, "minimum": 8 if operation == "all_reduce" else 64,
                       "maximum": 268435456,
                       "repetition": repetition, "phase": "main"}
    other = [group for group in itertools.combinations(four, 2) if list(group) != pair]
    for index, group in enumerate(other, 1):
        yield {"name": f"pair-{index}-allreduce", "group": list(group), "operation": "all_reduce",
               "minimum": 8, "maximum": 1048576, "repetition": 1, "phase": "diagnostic"}
    for index, group in enumerate(itertools.combinations(four, 2), 1):
        yield {"name": f"pair-{index}-sendrecv", "group": list(group), "operation": "sendrecv",
               "minimum": 8, "maximum": 1048576, "repetition": 1, "phase": "diagnostic"}
    yield {"name": "control-single", "group": single, "operation": "all_reduce",
           "minimum": 8, "maximum": 1048576, "repetition": 1, "phase": "control"}


def _rank_script(job, nodes, config):
    # The only shell text is constructed from validated names and quoted paths.
    cases = " ".join(f"{shlex.quote(n['expected_hostname'])}) binary={shlex.quote(config['binary_paths'][n['node_id']] + '/' + OPERATIONS[job['operation']])}; library={shlex.quote(config.get('library_paths', {}).get(n['node_id'], ''))};;"
                     for n in nodes)
    args = ["-b", str(job["minimum"]), "-e", str(job["maximum"]), "-f", "2", "-g", "1",
            "-d", "float", "-w", "5", "-n", "20", "-c", "1"]
    return ("set -u; host=$(hostname); case \"$host\" in " + cases +
            " *) exit 86;; esac; if [ -n \"$library\" ]; then export LD_LIBRARY_PATH=\"$library\"; fi; rank=${OMPI_COMM_WORLD_RANK:?}; "
            "if [ -n \"${NCCL_TESTS_SPLIT:-}\" ] || [ -n \"${NCCL_TESTS_SPLIT_MASK:-}\" ] || "
            "[ -n \"${NCCL_TESTS_DEVICE:-}\" ] || [ -n \"${NCCL_DEBUG_FILE:-}\" ] || "
            "{ [ -n \"${CUDA_VISIBLE_DEVICES:-}\" ] && [ \"$CUDA_VISIBLE_DEVICES\" != 0 ]; }; then "
            "printf 'IPRE_REJECTED_ENV=split/device/debug_file/cuda_visible_devices\\n' >&2; exit 87; fi; "
            "gpu=$(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | head -n 1); "
            "printf 'IPRE_RANK_IDENTITY={\"rank\":%s,\"hostname\":\"%s\",\"gpu_uuid\":\"%s\"}\\n' \"$rank\" \"$host\" \"$gpu\"; "
            "telemetry() { printf 'IPRE_TELEMETRY=%s|' \"$1\"; "
            "nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits | head -n 1; "
            "for dev in /sys/class/net/*; do [ -d \"$dev\" ] || continue; "
            "printf 'IPRE_NIC=%s|%s|%s|%s\\n' \"$1\" \"${dev##*/}\" "
            "\"$(cat \"$dev/statistics/rx_bytes\")\" \"$(cat \"$dev/statistics/tx_bytes\")\"; done; }; "
            "for port in /sys/class/infiniband/*/ports/*; do [ -f \"$port/link_layer\" ] || continue; "
            "printf 'IPRE_RDMA_LINK=%s|%s\\n' \"$port\" \"$(cat \"$port/link_layer\")\"; done; "
            "env | grep -E '^(NCCL_DEBUG|NCCL_DEBUG_SUBSYS|NCCL_DEBUG_FILE|NCCL_CONF_FILE|"
            "NCCL_SOCKET_IFNAME|NCCL_IB_HCA|NCCL_IB_DISABLE|NCCL_IB_GID_INDEX|NCCL_NET|"
            "NCCL_NET_PLUGIN|NCCL_NET_GDR_LEVEL|NCCL_P2P_DISABLE|NCCL_SHM_DISABLE|"
            "NCCL_CROSS_NIC|NCCL_ALGO|NCCL_PROTO|NCCL_TESTS_SPLIT|NCCL_TESTS_SPLIT_MASK|"
            "NCCL_TESTS_DEVICE|CUDA_VISIBLE_DEVICES|UCX_TLS|UCX_NET_DEVICES|LD_LIBRARY_PATH)=' | sed 's/^/IPRE_ENV=/'; "
            "for file in /etc/nccl.conf \"${HOME:-/dev/null}/.nccl.conf\" \"${NCCL_CONF_FILE:-/dev/null}\"; do "
            "if [ -f \"$file\" ]; then printf 'IPRE_CONF_HASH=%s|' \"$file\"; sha256sum \"$file\"; "
            "else printf 'IPRE_CONF_MISSING=%s\\n' \"$file\"; fi; done; "
            "telemetry before; timeout -k 3s 29s \"$binary\" " + shlex.join(args) +
            "; rc=$?; telemetry after; exit \"$rc\"")


def _command(job, cluster, config):
    by_id = {node["node_id"]: node for node in cluster["nodes"]}
    nodes = [by_id[node_id] for node_id in job["group"]]
    hosts = ",".join((node["expected_hostname"] if node["local"] else node["ssh"]["host"]) + ":1"
                     for node in nodes)
    return ["mpirun", "--tag-output", "--host", hosts, "--map-by", "ppr:1:node",
            "--mca", "plm_rsh_agent", "ssh", "--mca", "plm_rsh_args",
            "-o BatchMode=yes -o StrictHostKeyChecking=yes -o UpdateHostKeys=no "
            "-o ControlMaster=no -o ControlPath=none -o ConnectTimeout=8",
            "-np", str(len(nodes)), "-x", "NCCL_DEBUG=INFO",
            "-x", "NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET,GRAPH",
            "timeout", "-k", "5s", "34s", "/bin/sh", "-c", _rank_script(job, nodes, config)]


def make_plan(cluster_path, config_path):
    from infraestructura import cluster as cluster_module
    cluster = cluster_module.load_config(cluster_path)
    cluster_module.validate_config(cluster)
    config = _json(config_path)
    _validate_protocol(config, cluster)
    jobs = list(_jobs(cluster))
    pending = ["autorización operacional explícita", "hosts y SSH accesibles",
               "mpirun y binarios NCCL con soporte MPI y linker verificados por nodo",
               "cuatro GPU UUID distintos y ranks observables"]
    pending.extend(f"{node['node_id']}: access_status={node['access_status']}"
                   for node in cluster["nodes"] if node["access_status"] not in ("local", "verified"))
    return {"protocol": config["protocol_id"], "cluster_id": cluster["cluster_id"],
            "session_limit_seconds": 1800, "invocation_limit_seconds": 45,
            "resources_max": {"nodes": 4, "gpus": 4, "ranks": 4},
            "pending_prerequisites": pending,
            "jobs": [dict(job, command=_command(job, cluster, config)) for job in jobs]}


def parse_nccl_table(output):
    rows = []
    for line in output.splitlines():
        line = RANK_LINE.sub(lambda m: m.group(2), line).lstrip("# ")
        fields = line.split()
        if len(fields) < 13 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        size = int(fields[0])
        values = fields[5:13]
        def number(value):
            if value in ("N/A", "-"):
                return None
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError("métrica no finita")
            return parsed
        try:
            out_time, out_alg, out_bus, out_wrong, in_time, in_alg, in_bus = (
                number(values[0]), number(values[1]), number(values[2]), number(values[3]),
                number(values[4]), number(values[5]), number(values[6]))
            in_wrong = number(values[7])
        except (ValueError, IndexError):
            continue
        rows.append({"bytes": size, "count": int(fields[1]), "dtype": fields[2],
                     "redop": fields[3], "root": fields[4],
                     "in_place": {"time_us": in_time, "algbw_GBps": in_alg,
                        "busbw_GBps": in_bus, "wrong": in_wrong, "wrong_token": values[7]},
                     "out_of_place": {"time_us": out_time, "algbw_GBps": out_alg,
                        "busbw_GBps": out_bus, "wrong": out_wrong, "wrong_token": values[3]}})
    return rows


def classify_transport(text):
    data = [line for line in text.splitlines() if "NET/" in line and "Bootstrap" not in line]
    used = [line for line in data if "Using network" in line or
            ("Using" in line and "No device" not in line)]
    # INIT anuncia el transporte seleccionado; NET también enumera candidatos.
    selected = [line for line in text.splitlines() if "Using network " in line]
    modes = {"NET/" + match.group(1) for line in selected
             if (match := re.search(r"\bUsing network (IB|Socket)\b", line))}
    if not selected:
        modes = {name for name in ("NET/IB", "NET/Socket") if any(name in line for line in used)}
    if len(modes) == 1:
        mode = modes.pop()
    elif len(modes) > 1:
        mode = "mixed"
    else:
        mode = "indeterminate"
    hcas = sorted(set(re.findall(r"\b(?:mlx\w+|hca\d+|roce\w+)\b", "\n".join(used))))
    per_rank = {}
    for line in text.splitlines():
        tagged = RANK_LINE.match(line)
        if tagged:
            per_rank.setdefault(tagged.group(1), []).append(tagged.group(2))
    return {"data_transport": mode, "hca_mentions": hcas,
            "selection_evidence": selected or used,
            "by_rank": {rank: classify_transport("\n".join(lines))
                        for rank, lines in per_rank.items()},
            "bootstrap_socket_is_data_transport": False}


def _rank_rdma_verified(stdout, stderr):
    """Correlate used NCCL virtual NICs with observed Ethernet RDMA ports."""
    log = stdout + "\n" + stderr
    if classify_transport(log)["data_transport"] != "NET/IB":
        return False
    channels = re.findall(r"\bvia NET/([^/\s]+)/([0-9]+)", log)
    if not channels or any(mode != "IB" for mode, _ in channels):
        return False
    physical = {}
    for hca, port, layer in re.findall(r"\[\d+\]([A-Za-z0-9_]+):(\d+)(?:[:\d]*)/(Ethernet|InfiniBand)", log):
        physical[(hca, port)] = layer
    virtual = {number: name for number, name in
               re.findall(r"NET/IB : Made virtual device \[(\d+)\] name=([A-Za-z0-9_+]+)", log)}
    observed = {(hca, port): layer for hca, port, layer in re.findall(
        r"IPRE_RDMA_LINK=/sys/class/infiniband/([A-Za-z0-9_]+)/ports/(\d+)\|(Ethernet|InfiniBand)", log)}
    for _, number in channels:
        names = virtual.get(number, "").split("+")
        if not names or not all(any(physical.get((name, port)) == "Ethernet" and
                                    observed.get((name, port)) == "Ethernet"
                                    for _, port in observed if _ == name)
                                for name in names):
            return False
    return True


def _expected_sizes(job):
    value = job["minimum"]
    sizes = []
    while value <= job["maximum"]:
        sizes.append(value)
        value *= 2
    return sizes


def _validate_measurement(job, result, text, preflight=None):
    if result["returncode"] != 0 or result["timed_out"] or "rank_error" in result:
        raise NcclError(f"job falló o faltan ranks: {job['name']}")
    rows = result["rows"]
    if sorted(row["bytes"] for row in rows) != _expected_sizes(job):
        raise NcclError(f"tabla NCCL incompleta: {job['name']}")
    tagged_lines = [RANK_LINE.sub(lambda m: m.group(2), line) for line in text.splitlines()]
    header = "\n".join(tagged_lines)
    if not re.search(rf"# Collective test starting: .*{re.escape(OPERATIONS[job['operation']])}", header):
        raise NcclError(f"operación NCCL no verificable: {job['name']}")
    expected_header = (rf"# nThread 1 nGpus 1 minBytes {job['minimum']} maxBytes {job['maximum']} "
                       r"step: 2\(factor\) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 ")
    if not re.search(expected_header, header):
        raise NcclError(f"parámetros efectivos NCCL inesperados: {job['name']}")
    for row in rows:
        ranks = len(job["group"])
        count = row["bytes"] // 4
        if job["operation"] in ("all_gather", "reduce_scatter"):
            count = ((count // ranks) // 4) * 4
        redop = "none" if job["operation"] == "all_gather" else "sum"
        if (row["count"], row["dtype"], row["redop"], row["root"]) != (count, "float", redop, "-1"):
            raise NcclError(f"tabla NCCL contradice protocolo: {job['name']}")
        for side in (row["in_place"], row["out_of_place"]):
            if any(side[field] is None for field in ("time_us", "algbw_GBps", "busbw_GBps")):
                raise NcclError(f"métricas NCCL ausentes: {job['name']}")
            if side["time_us"] <= 0 or side["algbw_GBps"] < 0 or side["busbw_GBps"] < 0:
                raise NcclError(f"métricas NCCL fuera de rango: {job['name']}")
            if side["wrong"] is None and not (job["operation"] == "sendrecv" and
                                               side is row["in_place"] and side.get("wrong_token") == "N/A"):
                raise NcclError(f"check ausente: {job['name']}")
            if side["wrong"] not in (0, None):
                raise NcclError(f"wrong != 0: {job['name']}")
    match = re.search(r"\bNCCL version\s+([^\s]+)", text)
    result["nccl_runtime_version"] = match.group(1) if match else None
    if result["nccl_runtime_version"] is None:
        raise NcclError(f"versión NCCL efectiva ausente: {job['name']}")
    if len(job["group"]) > 1 and result["transport"]["data_transport"] in ("indeterminate", "mixed"):
        raise NcclError(f"transporte NCCL de datos indeterminado: {job['name']}")
    if len(job["group"]) > 1 and result.get("ranks"):
        by_rank = result["transport"].get("by_rank", {})
        if any(str(rank["rank"]) not in by_rank or
               by_rank[str(rank["rank"])]["data_transport"] != result["transport"]["data_transport"]
               for rank in result["ranks"]):
            raise NcclError(f"transporte NCCL sin cobertura por rank: {job['name']}")
    devices = re.findall(r"#\s+Rank\s+(\d+)\s+Group\s+(\d+)\s+Pid\s+\d+\s+on\s+(\S+)\s+device\s+(\d+)", text)
    if any(group != "0" or device != "0" for _, group, _, device in devices):
        raise NcclError(f"grupo NCCL o dispositivo inesperado: {job['name']}")
    if sorted((int(rank), host) for rank, _, host, _ in devices) != sorted(
            (rank["rank"], rank["hostname"]) for rank in result["ranks"]):
        raise NcclError(f"ranks NCCL efectivos incompletos: {job['name']}")
    versions = {}
    for line in text.splitlines():
        tagged = RANK_LINE.match(line)
        if tagged:
            version = re.search(r"\bNCCL version\s+([^\s]+)", tagged.group(2))
            if version:
                versions[int(tagged.group(1))] = version.group(1)
    result["nccl_versions_by_rank"] = versions
    if preflight:
        expected = {item["node_id"]: item["nccl_library_versions"][0] for item in preflight}
        by_rank = {rank["rank"]: rank["node_id"] for rank in result["ranks"]}
        for rank, banner in versions.items():
            parsed = re.match(r"^(\d+)\.(\d+)\.(\d+)", banner)
            if not parsed or int(parsed[1]) * 10000 + int(parsed[2]) * 100 + int(parsed[3]) != expected[by_rank[rank]]:
                raise NcclError(f"banner NCCL difiere de libnccl efectiva: {job['name']}")


def _observed_ranks(output, job, cluster, preflight=None):
    expected = {node["expected_hostname"]: node["node_id"] for node in cluster["nodes"]
                if node["node_id"] in job["group"]}
    found = []
    for line in output.splitlines():
        tagged = RANK_LINE.match(line)
        if tagged and tagged.group(2).startswith(IDENTITY):
            value = json.loads(tagged.group(2)[len(IDENTITY):])
            if int(tagged.group(1)) != value["rank"]:
                raise NcclError("rank MPI y marcador no coinciden")
            if value["hostname"] not in expected or not re.fullmatch(r"GPU-[A-Fa-f0-9-]+", value["gpu_uuid"]):
                raise NcclError("host o GPU UUID observado inválido")
            found.append({"node_id": expected[value["hostname"]], **value})
    ranks = [item["rank"] for item in found]
    if len(found) != len(job["group"]) or sorted(ranks) != list(range(len(found))):
        raise NcclError("evidencia de ranks incompleta o duplicada")
    if len({item["hostname"] for item in found}) != len(found) or len({item["gpu_uuid"] for item in found}) != len(found):
        raise NcclError("hostname o GPU UUID duplicados")
    if preflight:
        baseline = {item["node_id"]: item["gpu_uuid"] for item in preflight}
        if any(baseline.get(item["node_id"]) != item["gpu_uuid"] for item in found):
            raise NcclError("GPU UUID de rank difiere del preflight")
    return sorted(found, key=lambda item: item["rank"])


def _execute(command, seconds):
    start = time.monotonic()
    started_at = manifest.utc_now()
    deadline = start + seconds
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=max(0, seconds - 5))
        return {"returncode": process.returncode, "stdout": stdout, "stderr": stderr,
                "timed_out": False, "duration_seconds": time.monotonic() - start,
                "started_at": started_at, "ended_at": manifest.utc_now()}
    except BaseException as interrupted:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        stdout = interrupted.stdout if isinstance(interrupted, subprocess.TimeoutExpired) else ""
        stderr = interrupted.stderr if isinstance(interrupted, subprocess.TimeoutExpired) else ""
        try:
            stdout, stderr = process.communicate(timeout=max(0, deadline - time.monotonic() - 0.2))
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=max(0, deadline - time.monotonic()))
            except BaseException:
                for pipe in (process.stdout, process.stderr):
                    if pipe is not None:
                        pipe.close()
                try:
                    process.wait(timeout=0)
                except subprocess.TimeoutExpired:
                    pass
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        result = {"returncode": process.returncode, "stdout": stdout or "", "stderr": stderr or "",
                  "timed_out": True, "duration_seconds": time.monotonic() - start,
                  "started_at": started_at, "ended_at": manifest.utc_now()}
        if not isinstance(interrupted, subprocess.TimeoutExpired):
            result["interrupted"] = type(interrupted).__name__
            interrupted.execution_result = result
            raise
        return result


def _preflight(cluster, config, deadline, evidence_dir=None):
    from infraestructura import cluster as cluster_module
    observations = []
    for node in cluster["nodes"]:
        commands = [["hostname"], ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"]]
        commands.extend((
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
            ["cat", "/proc/meminfo"],
        ))
        commands.extend(["test", "-x", config["binary_paths"][node["node_id"]] + "/" + name]
                        for name in OPERATIONS.values())
        commands.append(["sh", "-c", "command -v mpirun"])
        commands.append(["mpirun", "--version"])
        commands.append(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"])
        commands.extend(["sha256sum", config["binary_paths"][node["node_id"]] + "/" + name]
                        for name in OPERATIONS.values())
        library = config.get("library_paths", {}).get(node["node_id"])
        prefix = ["env", f"LD_LIBRARY_PATH={library}"] if library else []
        commands.extend([*prefix, "ldd", config["binary_paths"][node["node_id"]] + "/" + name]
                        for name in OPERATIONS.values())
        commands.extend(["readelf", "-Ws", config["binary_paths"][node["node_id"]] + "/" + name]
                        for name in OPERATIONS.values())
        commands.append(["nvcc", "--version"])
        evidence = {"node_id": node["node_id"], "checks": [], "binary_hashes": {}}
        for index, command in enumerate(commands):
            if time.monotonic() >= deadline:
                raise NcclError("presupuesto de sesión agotado en preflight")
            argv = cluster_module.ssh_command(node, ["timeout", "-k", "3s", "11s", *command])
            try:
                result = _execute(argv, min(15, max(1, deadline - time.monotonic())))
            except OSError as exc:
                result = {"returncode": None, "stdout": "", "stderr": str(exc),
                          "timed_out": False, "duration_seconds": 0.0}
            evidence["checks"].append({"command": argv, **result})
            checked_command = command[2:] if command[:1] == ["env"] else command
            if evidence_dir is not None:
                _write_new(evidence_dir / f"{node['node_id']}-{index:02d}.json",
                           {"node_id": node["node_id"], "expected_hostname": node["expected_hostname"],
                            "check": evidence["checks"][-1]})
            if (result["returncode"] != 0 or result["timed_out"] or "not found" in result["stdout"]) \
                    and command[0] != "nvcc":
                raise NcclError(f"preflight falló en {node['node_id']}: {shlex.join(command)}")
            if checked_command[0] == "ldd" and "libmpi" not in result["stdout"]:
                raise NcclError(f"binario sin enlace MPI verificable: {checked_command[1]}")
            if command[0] == "readelf" and not re.search(r"\bMPI_Init(?:_thread)?\b", result["stdout"]):
                raise NcclError(f"binario sin inicialización MPI verificable: {command[-1]}")
            if command[0] == "sha256sum":
                digest = result["stdout"].split()[0] if result["stdout"].split() else ""
                if not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
                    raise NcclError(f"hash de ejecutable inválido: {command[1]}")
                evidence["binary_hashes"][Path(command[1]).name] = digest.lower()
            if command[1:2] == ["--query-compute-apps=pid,gpu_uuid,used_gpu_memory"] and re.search(
                    r"^\s*\d+\s*,", result["stdout"], flags=re.MULTILINE):
                raise NcclError(f"GPU ocupada por compute app en {node['node_id']}; no se detiene")
            if command == ["cat", "/proc/meminfo"] and not all(
                    re.search(rf"^{key}:\s+\d+\s+kB", result["stdout"], flags=re.MULTILINE)
                    for key in ("MemAvailable", "SwapFree")):
                raise NcclError(f"memoria host no observable en {node['node_id']}")
            if checked_command[0] == "ldd":
                libraries = re.findall(r"\blibnccl\S*\s+=>\s+(/\S+)", result["stdout"])
                if len(libraries) != 1:
                    raise NcclError(f"libnccl efectiva no identificable: {checked_command[1]}")
                library_command = ["sha256sum", libraries[0]]
                library_argv = cluster_module.ssh_command(
                    node, ["timeout", "-k", "3s", "11s", *library_command])
                library_result = _execute(library_argv, min(15, max(1, deadline - time.monotonic())))
                if evidence_dir is not None:
                    _write_new(evidence_dir / f"{node['node_id']}-{index:02d}-libnccl.json",
                               {"node_id": node["node_id"], "check": {"command": library_argv, **library_result}})
                if library_result["returncode"] != 0 or library_result["timed_out"]:
                    raise NcclError(f"hash libnccl efectivo falló: {libraries[0]}")
                digest = library_result["stdout"].split()[0] if library_result["stdout"].split() else ""
                if not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
                    raise NcclError("SHA-256 libnccl inválido")
                evidence.setdefault("nccl_library_hashes", []).append(digest.lower())
                getter = [*prefix, "python3", "-c", "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); v=ctypes.c_int(); "
                          "r=lib.ncclGetVersion(ctypes.byref(v)); assert r==0; print(v.value)", libraries[0]]
                getter_argv = cluster_module.ssh_command(node, ["timeout", "-k", "3s", "11s", *getter])
                getter_result = _execute(getter_argv, min(15, max(1, deadline - time.monotonic())))
                if evidence_dir is not None:
                    _write_new(evidence_dir / f"{node['node_id']}-{index:02d}-nccl-version.json",
                               {"node_id": node["node_id"], "check": {"command": getter_argv, **getter_result}})
                if getter_result["returncode"] != 0 or not getter_result["stdout"].strip().isdigit():
                    raise NcclError(f"ncclGetVersion falló: {node['node_id']}")
                evidence.setdefault("nccl_library_versions", []).append(int(getter_result["stdout"].strip()))
            if command[:2] == ["mpirun", "--version"] and "Open MPI" not in result["stdout"]:
                raise NcclError("launcher MPI no es Open MPI verificable")
            if command[:2] == ["mpirun", "--version"]:
                evidence["mpi_version"] = result["stdout"].strip()
        actual = evidence["checks"][0]["stdout"].strip()
        gpu_uuids = evidence["checks"][1]["stdout"].splitlines()
        if actual != node["expected_hostname"] or len(gpu_uuids) != 1 or not gpu_uuids[0].startswith("GPU-"):
            raise NcclError(f"identidad/GPUs inesperadas en {node['node_id']}")
        evidence.update(hostname=actual, gpu_uuid=gpu_uuids[0])
        observations.append(evidence)
    if len({item["gpu_uuid"] for item in observations}) != len(observations):
        raise NcclError("GPU UUID duplicados entre nodos")
    if len({tuple(item["nccl_library_hashes"]) for item in observations}) != 1:
        raise NcclError("drift de hash libnccl entre nodos")
    if len({tuple(item["nccl_library_versions"]) for item in observations}) != 1:
        raise NcclError("drift de versión libnccl entre nodos")
    if len({tuple(sorted(item["binary_hashes"].items())) for item in observations}) != 1:
        raise NcclError("drift de ejecutables nccl-tests entre nodos")
    if len({item["mpi_version"] for item in observations}) != 1:
        raise NcclError("drift de versión MPI entre nodos")
    return observations


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _rank_artifacts(job_dir, full_output, ranks):
    for rank in ranks:
        rank_id = rank["rank"]
        lines = {"stdout": [], "stderr": []}
        for line in full_output.splitlines():
            match = RANK_LINE.match(line)
            if match and int(match.group(1)) == rank_id:
                channel = "stderr" if "<stderr>" in line else "stdout"
                lines[channel].append(match.group(2))
        for channel, content in lines.items():
            (job_dir / f"rank-{rank_id}-{channel}.log").write_text("\n".join(content) + "\n", encoding="utf-8")
        telemetry = {"node_id": rank["node_id"], "rank": rank_id,
                     "gpu": [line for line in sum(lines.values(), []) if line.startswith("IPRE_TELEMETRY=")],
                     "nic": [line for line in sum(lines.values(), []) if line.startswith("IPRE_NIC=")],
                     "effective_env": [line.removeprefix("IPRE_ENV=") for line in sum(lines.values(), [])
                                       if line.startswith("IPRE_ENV=")],
                     "nccl_config_hashes": [line.removeprefix("IPRE_CONF_HASH=") for line in sum(lines.values(), [])
                                            if line.startswith("IPRE_CONF_HASH=")],
                     "nccl_config_missing": [line.removeprefix("IPRE_CONF_MISSING=") for line in sum(lines.values(), [])
                                             if line.startswith("IPRE_CONF_MISSING=")]}
        parsed_gpu = {}
        missing_gpu_metrics = []
        for line in telemetry["gpu"]:
            phase, _, values = line.removeprefix("IPRE_TELEMETRY=").partition("|")
            fields = [part.strip() for part in values.split(",")]
            if phase in ("before", "after") and len(fields) == 4:
                metrics = {"gpu_uuid": fields[0]}
                for key, value in zip(("utilization_pct", "memory_used_MiB", "power_W"), fields[1:]):
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        metrics[key] = None
                        missing_gpu_metrics.append(f"{phase}.{key}")
                parsed_gpu[phase] = metrics
        parsed_nic = {}
        for line in telemetry["nic"]:
            fields = line.removeprefix("IPRE_NIC=").split("|")
            if len(fields) == 4 and fields[0] in ("before", "after"):
                try:
                    parsed_nic.setdefault(fields[1], {})[fields[0]] = [int(fields[2]), int(fields[3])]
                except ValueError:
                    pass
        telemetry["parsed_gpu"] = parsed_gpu
        telemetry["missing_gpu_metrics"] = missing_gpu_metrics
        telemetry["nic_deltas_bytes"] = {
            name: {"rx": values["after"][0] - values["before"][0],
                   "tx": values["after"][1] - values["before"][1]}
            for name, values in parsed_nic.items() if set(values) == {"before", "after"}
            and all(after >= before for before, after in zip(values["before"], values["after"]))}
        _write_new(job_dir / f"rank-{rank_id}-telemetry.json", telemetry)
        if set(parsed_gpu) != {"before", "after"} or parsed_gpu["before"]["gpu_uuid"] != rank["gpu_uuid"] \
                or parsed_gpu["after"]["gpu_uuid"] != rank["gpu_uuid"] or not telemetry["nic_deltas_bytes"] \
                or "NCCL_DEBUG=INFO" not in telemetry["effective_env"] \
                or "NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET,GRAPH" not in telemetry["effective_env"]:
            raise NcclError(f"telemetría incompleta para rank {rank_id}")


def run(cluster_path, config_path, *, allow_dirty=False, results_root=None):
    from infraestructura import cluster as cluster_module
    cluster = cluster_module.load_config(cluster_path)
    cluster_module.validate_config(cluster)
    config = _json(config_path)
    _validate_protocol(config, cluster)
    plan = make_plan(cluster_path, config_path)
    if any(node["ssh"]["user"] != getpass.getuser() or node["ssh"]["port"] != 22
           for node in cluster["nodes"]):
        raise NcclError("launcher MPI requiere usuario local y SSH puerto 22 en todos los nodos")
    deadline = time.monotonic() + 1800
    requested = {"cluster_id": cluster["cluster_id"], "protocol": config,
                 "jobs": plan["jobs"], "strategy": "one-rank-per-GPU", "parallelism": {"gpus_per_rank": 1}}
    source_inputs = {"cluster": cluster_path, "runner": Path(__file__),
                     "cluster_helper": Path(cluster_module.__file__),
                     "manifest_module": Path(manifest.__file__),
                     "rutas_module": REPO_ROOT / "infraestructura/rutas.py",
                     "schema_manifest": REPO_ROOT / "infraestructura/schemas/run-manifest.schema.json",
                     "schema_context": REPO_ROOT / "infraestructura/schemas/run-context.schema.json"}
    run_dir = manifest.start_run(kind="nccl-test", model_id=None, model_revision=None,
                                 tokenizer_id=None, tokenizer_revision=None,
                                 requested_config=requested, requested_config_path=config_path,
                                 inputs=source_inputs, allow_dirty=allow_dirty, repo_path=REPO_ROOT,
                                 **({"results_root": results_root} if results_root else {}))
    errors = []
    participants = []
    try:
        source_dir = run_dir / "raw" / "sources"
        source_dir.mkdir(exist_ok=False)
        for name, source in (("cluster.json", cluster_path), ("protocol.json", config_path),
                             ("nccl.py", Path(__file__)), ("cluster.py", Path(cluster_module.__file__)),
                             ("manifest.py", Path(manifest.__file__)),
                             ("rutas.py", REPO_ROOT / "infraestructura/rutas.py"),
                             ("run-manifest.schema.json", REPO_ROOT / "infraestructura/schemas/run-manifest.schema.json"),
                             ("run-context.schema.json", REPO_ROOT / "infraestructura/schemas/run-context.schema.json")):
            with (source_dir / name).open("xb") as stream:
                stream.write(Path(source).read_bytes())
        preflight_dir = run_dir / "raw" / "preflight"
        preflight_dir.mkdir(exist_ok=False)
        preflight = _preflight(cluster, config, deadline, preflight_dir)
        _write_new(run_dir / "raw" / "preflight.json", preflight)
        for job in plan["jobs"]:
            if deadline - time.monotonic() < 45:
                raise NcclError("presupuesto insuficiente para otra invocación de 45 s")
            job_dir = run_dir / "raw" / job["name"]
            job_dir.mkdir(exist_ok=False)
            try:
                result = _execute(job["command"], 45)
            except BaseException as exc:
                if not hasattr(exc, "execution_result"):
                    raise
                result = exc.execution_result
            (job_dir / "stdout.log").write_text(result.pop("stdout"), encoding="utf-8")
            (job_dir / "stderr.log").write_text(result.pop("stderr"), encoding="utf-8")
            full_output = (job_dir / "stdout.log").read_text() + "\n" + (job_dir / "stderr.log").read_text()
            result.update(job=job, transport=classify_transport(full_output), rows=parse_nccl_table(full_output))
            try:
                ranks = _observed_ranks(full_output, job, cluster, preflight)
                for rank in ranks:
                    evidence_path = job_dir / f"rank-{rank['rank']}-identity.json"
                    _write_new(evidence_path, rank)
                    rank["evidence_path"] = str(evidence_path.relative_to(run_dir))
                _rank_artifacts(job_dir, full_output, ranks)
                result["ranks"] = ranks
            except (NcclError, ValueError, KeyError, json.JSONDecodeError) as exc:
                result["rank_error"] = str(exc)
            validation_error = None
            try:
                _validate_measurement(job, result, full_output, preflight)
            except NcclError as exc:
                validation_error = str(exc)
                result["validation_error"] = validation_error
            _write_new(job_dir / "result.json", result)
            if validation_error:
                raise NcclError(validation_error)
            if job["phase"] == "smoke":
                participants = ranks
                manifest.register_participants(run_dir, participants)
        if len(participants) != 4:
            raise NcclError("roster de cuatro ranks no registrado")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    try:
        manifest.finalize_run(run_dir, status="failed" if errors else "complete", errors=errors)
    except BaseException as exc:
        raise NcclError(f"cierre de corrida falló: {exc}; run={run_dir}") from exc
    if errors:
        raise NcclError(f"corrida FAILED: {run_dir}: {'; '.join(errors)}")
    return run_dir


def report(run_path, output):
    run_path = Path(run_path)
    output = Path(output).resolve()
    if output.is_relative_to(run_path.resolve().parent.parent) or output.is_relative_to((Path("resultados/raw")).resolve()):
        raise NcclError("el reporte debe escribirse fuera de raw")
    run = manifest.load_run(run_path)
    manifest.verify_artifacts(run_path, run=run)
    if run["kind"] != "nccl-test" or run["status"] not in ("complete", "failed"):
        raise NcclError("se requiere corrida NCCL cerrada")
    planned = run["requested_config"]["jobs"]
    jobs = []
    grouped = {}
    planned_order = {job["name"]: index for index, job in enumerate(planned)}
    paths = sorted((run_path / "raw").glob("*/result.json"),
                   key=lambda path: planned_order.get(path.parent.name, len(planned)))
    for path in paths:
        item = _json(path)
        invalid = item.get("validation_error") or item.get("rank_error") or (
            f"returncode={item['returncode']}" if item["returncode"] != 0 else None)
        if item.get("timed_out"):
            invalid = "timeout"
        telemetry = []
        for rank in item.get("ranks", []):
            evidence = path.parent / f"rank-{rank['rank']}-telemetry.json"
            if evidence.is_file():
                info = _json(evidence)
                telemetry.append({"rank": rank["rank"], "node_id": rank["node_id"],
                                  "sha256": manifest.sha256_file(evidence),
                                  "nic_deltas_bytes": info.get("nic_deltas_bytes", {})})
        rdma_by_rank = []
        for rank in item.get("ranks", []):
            evidence_file = path.parent / f"rank-{rank['rank']}-stderr.log"
            stdout_file = path.parent / f"rank-{rank['rank']}-stdout.log"
            if not evidence_file.is_file() or not stdout_file.is_file():
                rdma_by_rank.append(False)
                continue
            evidence_text = evidence_file.read_text(encoding="utf-8")
            stdout_text = stdout_file.read_text(encoding="utf-8")
            rdma_by_rank.append(_rank_rdma_verified(stdout_text, evidence_text))
        rdma_status = ("verified" if len(item["job"]["group"]) == len(rdma_by_rank)
                       and all(rdma_by_rank) and len(rdma_by_rank) > 1 else
                       "not_verified" if len(item["job"]["group"]) > 1 else "not_applicable")
        jobs.append({"name": item["job"]["name"], "result_sha256": manifest.sha256_file(path),
                     "transport": item["transport"], "valid": invalid is None,
                     "failure": invalid, "rdma_status": rdma_status,
                     "sendrecv_inplace_check": "not_supported" if item["job"].get("operation") == "sendrecv" else "checked",
                     "rows": len(item.get("rows", [])),
                     "returncode": item["returncode"], "telemetry": telemetry})
        job = item["job"]
        if invalid is not None:
            continue
        for row in item.get("rows", []):
            for placement in ("in_place", "out_of_place"):
                for metric in ("time_us", "algbw_GBps", "busbw_GBps"):
                    value = row[placement][metric]
                    if value is not None:
                        key = (job["phase"], tuple(job["group"]), job["operation"], row["bytes"], placement, metric)
                        grouped.setdefault(key, []).append(value)
    dispersion = [{"phase": phase, "group": list(group), "operation": operation,
                   "bytes": size, "placement": placement, "metric": metric,
                   "count": len(values), "median": statistics.median(values),
                   "min": min(values), "max": max(values)}
                  for (phase, group, operation, size, placement, metric), values in sorted(grouped.items())]
    summary = {"run_id": run["run_id"], "status": run["status"], "manifest_sha256":
               manifest.sha256_file(run_path / "manifest.json"), "jobs": jobs,
               "pending_jobs": [job["name"] for job in planned if job["name"] not in {item["name"] for item in jobs}],
               "dispersion": dispersion,
               "errors": run["errors"], "metric_note": "NCCL busbw GB/s; no es velocidad física del enlace"}
    summary["functional_success"] = (run["status"] == "complete" and not summary["pending_jobs"]
                                     and len(jobs) == len(planned) and all(job["valid"] for job in jobs))
    summary["rdma_success"] = (summary["functional_success"] and
                               all(job["rdma_status"] == "verified" for job in jobs
                                   if len(planned[planned_order[job["name"]]]["group"]) > 1))
    summary["overall_success"] = summary["functional_success"] and summary["rdma_success"]
    if output.suffix not in {".json", ".md"}:
        raise NcclError("--output debe terminar en .json o .md")
    if any((parent / "manifest.json").exists() or (parent / "COMPLETE").exists()
           or (parent / "FAILED").exists() for parent in output.parents):
        raise NcclError("el reporte debe escribirse fuera de raw")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".json":
        _write_new(output, summary)
    elif output.suffix == ".md":
        sample_sizes = {1048576, 268435456}
        lines = [f"# NCCL {run['run_id']}", "", f"Estado: **{run['status']}**.",
                 f"Corrección funcional: **{'sí' if summary['functional_success'] else 'no'}**. RoCE/RDMA: **{'sí' if summary['rdma_success'] else 'no'}**. Éxito completo: **{'sí' if summary['overall_success'] else 'no'}**.", "",
                 f"SHA-256 manifest: `{summary['manifest_sha256']}`.", "",
                 "## Jobs", "", "| Job | Válido | Transporte | RDMA | Falla |", "|---|---|---|---|---|"]
        for item in jobs:
            lines.append(f"| {item['name']} | {'sí' if item['valid'] else 'no'} | "
                         f"{item['transport']['data_transport']} | {item['rdma_status']} | {item['failure'] or ''} |")
        lines += ["", "## Mediciones seleccionadas", "",
                  "| Grupo | Operación | Bytes | Métrica | Mediana | Mín | Máx | n |",
                  "|---|---|---:|---|---:|---:|---:|---:|"]
        for item in dispersion:
            if item["bytes"] in sample_sizes and item["placement"] == "out_of_place":
                lines.append(f"| {','.join(item['group'])} | {item['operation']} | {item['bytes']} | "
                             f"{item['metric']} | {item['median']:.4g} | {item['min']:.4g} | "
                             f"{item['max']:.4g} | {item['count']} |")
        lines += ["", f"Jobs pendientes: {', '.join(summary['pending_jobs']) or 'ninguno'}.",
                  "", "NCCL busbw se expresa en GB/s; no mide directamente la velocidad física del enlace.", ""]
        if summary["errors"]:
            lines += ["## Errores de la sesión", ""]
            lines.extend(f"- {error.get('message', str(error))}" for error in summary["errors"])
        with output.open("x", encoding="utf-8") as stream:
            stream.write("\n".join(lines))
    else:
        raise NcclError("--output debe terminar en .json o .md")
    return Path(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("plan", "run"):
        command = sub.add_parser(name)
        command.add_argument("--cluster", required=True)
        command.add_argument("--config", required=True)
        if name == "run":
            command.add_argument("--execute", action="store_true", required=True)
            command.add_argument("--allow-dirty", action="store_true")
            command.add_argument("--results-root")
    command = sub.add_parser("report")
    command.add_argument("run")
    command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "plan":
            print(json.dumps(make_plan(args.cluster, args.config), indent=2))
        elif args.action == "run":
            print(run(args.cluster, args.config, allow_dirty=args.allow_dirty,
                      results_root=args.results_root))
        else:
            print(report(args.run, args.output))
        return 0
    except (NcclError, ValueError, OSError, manifest.ManifestError) as exc:
        print(f"nccl: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
