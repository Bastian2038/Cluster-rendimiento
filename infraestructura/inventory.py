"""Inventario Linux y referencia de reposo: solo biblioteca estándar, sin cargas GPU."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shlex
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

from infraestructura import manifest
from infraestructura.rutas import REPO_ROOT


GPU_FIELDS = ("uuid", "utilization.gpu", "memory.total", "memory.used",
              "temperature.gpu", "power.draw", "clocks.sm", "pstate")
GPU_KEYS = ("uuid", "utilization_pct", "memory_total_mib", "memory_used_mib",
            "temperature_c", "power_w", "sm_clock_mhz", "pstate")
COMMANDS = {
    "os_release": ["cat", "/etc/os-release"],
    "kernel": ["uname", "-a"],
    "cpu": ["lscpu", "--json"],
    "memory": ["cat", "/proc/meminfo"],
    "gpu": ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader,nounits"],
    "nvidia_detail": ["nvidia-smi", "-q"],
    "gpu_topology": ["nvidia-smi", "topo", "-m"],
    "cuda_compiler": ["nvcc", "--version"],
    "nccl_packages": ["dpkg-query", "-W", "-f=${Package}\t${Version}\n", "libnccl*"],
    "network": ["ip", "-j", "address", "show"],
    "routes": ["ip", "-j", "route", "show"],
    "storage": ["lsblk", "-J", "-o", "NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS"],
    "disk_space": ["df", "-h"],
    "containers": ["docker", "ps", "-a", "--no-trunc", "--format",
                   "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}"],
    "docker_version": ["docker", "--version"],
    "services": ["systemctl", "--no-pager", "--type=service", "--state=running"],
    "processes": ["ps", "-eo", "pid,ppid,comm,rss", "--sort=-rss"],
}
# Explicit identity fields only: never capture Docker command strings or environments.


def probe(command, timeout=10):
    """Keep command evidence, including failed/unsupported probes; never shell=True."""
    result = {"command": command, "observed_at": manifest.utc_now(),
              "status": "ok", "returncode": None, "stdout": "", "stderr": ""}
    start = time.monotonic()
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        result.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        if proc.returncode:
            result["status"] = "error"
    except FileNotFoundError as exc:
        result.update(status="unavailable", stderr=str(exc))
    except subprocess.TimeoutExpired:
        result.update(status="timeout", stderr=f"timeout={timeout}s")
    except OSError as exc:
        result.update(status="error", stderr=str(exc))
    result["duration_seconds"] = time.monotonic() - start
    return result


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def parse_meminfo(text):
    return {parts[0].rstrip(":"): int(parts[1])
            for line in text.splitlines() if len(parts := line.split()) >= 2}


def cpu_counters(text):
    line = next(line for line in text.splitlines() if line.startswith("cpu "))
    values = [int(value) for value in line.split()[1:9]]
    if len(values) != 8:
        raise ValueError("se requieren 8 contadores CPU")
    # guest/guest_nice already included in user/nice; don't double count them.
    return values


def cpu_delta(previous, current):
    deltas = [new - old for old, new in zip(previous, current)]
    total = sum(deltas)
    if total <= 0 or any(value < 0 for value in deltas):
        return None, None
    return (100 * (total - deltas[3] - deltas[4]) / total,
            100 * deltas[4] / total)


def parse_gpu(result):
    if result["status"] != "ok":
        return []
    rows = []
    for values in csv.reader(io.StringIO(result["stdout"])):
        if len(values) != len(GPU_KEYS):
            raise ValueError("salida GPU con número inesperado de campos")
        row, availability = {}, {}
        for key, value in zip(GPU_KEYS, values):
            value = value.strip()
            if not value or value.startswith("[") or value in {"N/A", "Not Supported"}:
                row[key] = None
                availability[key] = "unsupported_or_unavailable"
            elif key in {"uuid", "pstate"}:
                row[key] = value
                availability[key] = "ok"
            else:
                number = float(value)
                row[key] = number if math.isfinite(number) else None
                availability[key] = "ok" if row[key] is not None else "invalid"
        row["availability"] = availability
        rows.append(row)
    return rows


def gpu_processes(result):
    if result["status"] != "ok":
        return None
    rows = list(csv.reader(io.StringIO(result["stdout"])))
    processes = []
    for row in rows:
        if len(row) != 3 or not row[1].strip().isdigit():
            return None
        processes.append({"uuid": row[0].strip(), "pid": int(row[1]),
                          "name": row[2].strip()})
    return processes


def stats(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {"count": len(values), "min": values[0], "median": statistics.median(values),
            "p95": values[math.ceil(.95 * len(values)) - 1], "max": values[-1]}


def summarize(samples, config, inventory):
    thresholds = config["idle_thresholds"]
    expected = round(config["duration_seconds"] / config["interval_seconds"])
    missing, violations = [], []
    if len(samples) != expected:
        missing.append("sample_count")
    cpu = stats([s.get("cpu_busy_pct") for s in samples])
    memory = stats([s.get("memory", {}).get("MemAvailable") for s in samples])
    gpu = stats([row.get("utilization_pct") for s in samples for row in s.get("gpus", [])])
    for key in ("cpu", "memory", "gpu"):
        if {"cpu": cpu, "memory": memory, "gpu": gpu}[key]["count"] != expected:
            missing.append(key)
    if any(s.get("gpu_processes") is None for s in samples):
        missing.append("gpu_process_visibility")
    if any(any(key not in s.get("memory", {}) for key in
               ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")) for s in samples):
        missing.append("memory_fields")
    if any(s.get("gpu_processes") for s in samples):
        violations.append("gpu_compute_process_present")
    if any(len(s.get("gpus", [])) != 1 for s in samples):
        missing.append("expected_single_gpu")
    for name in ("cpu", "memory", "gpu", "processes", "services"):
        if inventory["probes"][name]["status"] != "ok":
            missing.append(f"inventory_{name}")
    if cpu["p95"] is not None and cpu["p95"] > thresholds["cpu_busy_p95_pct"]:
        violations.append("cpu_busy_p95")
    if gpu["p95"] is not None and gpu["p95"] > thresholds["gpu_utilization_p95_pct"]:
        violations.append("gpu_utilization_p95")
    edge = max(1, min(len(samples), round(30 / config["interval_seconds"])))
    first = [s.get("memory", {}).get("MemAvailable") for s in samples[:edge]]
    last = [s.get("memory", {}).get("MemAvailable") for s in samples[-edge:]]
    drift = None
    if first and last and all(v is not None for v in first + last):
        drift = abs(statistics.median(last) - statistics.median(first)) / 1024
        if drift > thresholds["memory_available_drift_mib"]:
            violations.append("memory_available_drift")
    lateness = stats([s["lateness_seconds"] for s in samples])
    if lateness["max"] is not None and lateness["max"] > thresholds["max_sample_lateness_seconds"]:
        missing.append("sampling_schedule_out_of_tolerance")
    optional = {}
    for key in ("memory_total_mib", "memory_used_mib", "temperature_c", "power_w", "sm_clock_mhz"):
        optional[key] = stats([row.get(key) for s in samples for row in s.get("gpus", [])])
    return {
        "idle_status": "not_idle" if violations else "unverified" if missing else "validated",
        "missing_or_invalid": sorted(set(missing)), "violations": violations,
        "sample_count": len(samples), "expected_sample_count": expected,
        "cpu_busy_pct": cpu, "memory_available_kib": memory,
        "gpu_utilization_pct": gpu, "optional_gpu_metrics": optional,
        "memory_available_drift_mib": drift, "lateness_seconds": lateness,
        "sampling_cost_seconds": stats([s["collection_seconds"] for s in samples]),
        "interval_seconds": stats([s["interval_seconds"] for s in samples]),
        "thresholds": thresholds,
        "limitations": ["Umbrales provisionales, no garantía de reposo térmico ni ausencia de procesos breves entre muestras.",
                        "El recolector y servicios de fondo contribuyen al consumo observado.",
                        "Potencia GPU no equivale a potencia total de pared del equipo.",
                        "Sin inferencia, SFT ni prueba activa de red; no es un benchmark de rendimiento."]}


def validate_config(config):
    node = config["node"]
    if node["node_count"] != 1:
        raise ValueError("inventario de reposo admite exactamente un nodo por captura")
    if not isinstance(node.get("alias"), str) or not node["alias"].strip():
        raise ValueError("alias de nodo obligatorio")
    if not isinstance(node.get("expected_hostname"), str) or not node["expected_hostname"].strip():
        raise ValueError("expected_hostname obligatorio")
    for key in ("duration_seconds", "interval_seconds"):
        if isinstance(config[key], bool) or not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} debe ser positivo y finito")
    if config["duration_seconds"] < 60:
        raise ValueError("reposo requiere al menos 60 segundos; usar mocks para pruebas cortas")
    if config["interval_seconds"] < 1:
        raise ValueError("intervalo mínimo de 1 segundo")
    count = config["duration_seconds"] / config["interval_seconds"]
    if not count.is_integer():
        raise ValueError("duración debe ser múltiplo del intervalo")
    for key in ("cpu_busy_p95_pct", "gpu_utilization_p95_pct", "memory_available_drift_mib", "max_sample_lateness_seconds"):
        value = config["idle_thresholds"][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"umbral inválido: {key}")
    if config["warmup_seconds"] != 0 or config["trial"] != 1 or config["repetitions"] != 1:
        raise ValueError("este protocolo implementa una ventana sin warmup, trial 1")


def capture(config_path, results_root, allow_dirty=False):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    validate_config(config)
    if socket.gethostname() != config["node"]["expected_hostname"]:
        raise ValueError(f"host inesperado: {socket.gethostname()}; no se capturó")
    repo = REPO_ROOT
    source_paths = [Path(__file__).resolve(), repo / "infraestructura/manifest.py",
                    repo / "infraestructura/rutas.py",
                    repo / "infraestructura/schemas/run-manifest.schema.json",
                    repo / "infraestructura/schemas/run-context.schema.json"]
    run = manifest.start_run(
        kind="system-baseline", model_id=None, model_revision=None,
        tokenizer_id=None, tokenizer_revision=None, requested_config=config,
        requested_config_path=config_path, repo_path=repo, allow_dirty=allow_dirty,
        results_root=results_root,
        inputs={str(path.relative_to(repo)): path for path in source_paths})
    print(f"RUN_DIRECTORY={run}", flush=True)
    try:
        # Preserve exact collector sources even when the worktree is dirty.
        sources = run / "raw" / "sources"
        sources.mkdir()
        for path in source_paths + [config_path]:
            with (sources / path.name).open("xb") as stream:
                stream.write(path.read_bytes())
        write_json(run / "raw" / "launch.json", {
            "argv": sys.argv, "command": shlex.join([sys.executable, "-m", "infraestructura.inventory", *sys.argv[1:]]),
            "cwd": str(Path.cwd()), "host": socket.gethostname(), "node": config["node"],
            "model": None, "dataset": None, "parallelism": None,
            "notes": config["notes"]})
        inventory = {"observed_at": manifest.utc_now(), "probes": {}}
        for name, command in COMMANDS.items():
            inventory["probes"][name] = probe(command)
        write_json(run / "raw" / "inventory.json", inventory)
        samples = []
        previous_cpu = cpu_counters(Path("/proc/stat").read_text())
        initial_vm = parse_meminfo(Path("/proc/vmstat").read_text())
        started_at = manifest.utc_now()
        started = previous_time = time.monotonic()
        count = round(config["duration_seconds"] / config["interval_seconds"])
        print(f"Capturando {count} muestras; duración objetivo {config['duration_seconds']}s", flush=True)
        with (run / "metrics" / "idle.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(count):
                target = started + (index + 1) * config["interval_seconds"]
                time.sleep(max(0, target - time.monotonic()))
                tick = time.monotonic()
                sample = {"index": index, "timestamp": manifest.utc_now(),
                          "elapsed_seconds": tick - started, "interval_seconds": tick - previous_time,
                          "lateness_seconds": max(0, tick - target), "errors": []}
                previous_time = tick
                current_cpu = cpu_counters(Path("/proc/stat").read_text())
                sample["cpu_busy_pct"], sample["cpu_iowait_pct"] = cpu_delta(previous_cpu, current_cpu)
                previous_cpu = current_cpu
                sample["memory"] = parse_meminfo(Path("/proc/meminfo").read_text())
                sample["gpu_query"] = probe(["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"], timeout=2)
                try:
                    sample["gpus"] = parse_gpu(sample["gpu_query"])
                except ValueError as exc:
                    sample["gpus"] = []
                    sample["errors"].append(str(exc))
                sample["process_query"] = probe(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"], timeout=2)
                sample["gpu_processes"] = gpu_processes(sample["process_query"])
                sample["collection_seconds"] = time.monotonic() - tick
                stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                samples.append(sample)
                if (index + 1) % 30 == 0:
                    print(f"Muestras {index + 1}/{count}", flush=True)
        ended = time.monotonic()
        ended_at = manifest.utc_now()
        final_vm = parse_meminfo(Path("/proc/vmstat").read_text())
        write_json(run / "raw" / "after.json", {name: probe(COMMANDS[name]) for name in ("processes", "services", "containers", "memory")})
        summary = summarize(samples, config, inventory)
        summary.update(window_started_at=started_at, window_ended_at=ended_at,
                       elapsed_seconds=ended - started,
                       swap_io_pages={key: final_vm[key] - initial_vm[key] for key in ("pswpin", "pswpout")})
        write_json(run / "raw" / "assessment.json", summary)
        acquisition = "failed" if summary["missing_or_invalid"] else "complete"
        manifest.finalize_run(run, status=acquisition,
                              errors=[f"required evidence: {item}" for item in summary["missing_or_invalid"]])
    except BaseException as exc:
        if not any((run / name).exists() for name in ("COMPLETE", "FAILED")):
            manifest.finalize_run(run, status="failed", errors=[f"{type(exc).__name__}: {exc}"])
        raise
    print(f"CAPTURE={acquisition} IDLE={summary['idle_status']}", flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experimentos/configs/pegasus-inventory.json")
    parser.add_argument("--results-root", default="resultados/raw")
    parser.add_argument("--allow-dirty", action="store_true", help="captura exploratoria con fuentes preservadas")
    args = parser.parse_args()
    try:
        run = capture(args.config, args.results_root, args.allow_dirty)
        if (run / "FAILED").exists():
            parser.exit(2, "Captura con evidencia obligatoria incompleta; revisar manifest y assessment.\n")
    except (ValueError, KeyError, OSError, manifest.ManifestError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
