"""Small, read-only host sampler for inference evidence.

No torch or vLLM import occurs here, so ``probe`` can run before GPU startup.
Missing device fields remain ``None`` rather than becoming numerical zero.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone


GPU_FIELDS = (
    "uuid", "utilization.gpu", "memory.used", "memory.total", "power.draw",
    "temperature.gpu", "clocks.sm", "clocks.mem",
)
F0_INTERFACES = ("enp1s0f0np0", "enP2p1s0f0np0")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number(value: str):
    value = value.strip()
    if not value or value.lower() in {"n/a", "not supported", "[not supported]"}:
        return None
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except (OSError, UnicodeError):
        return None


def _query(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _gpu_data() -> list[dict]:
    output = _query(["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"])
    if output is None:
        return []
    rows = []
    for line in output.splitlines():
        values = [v.strip() for v in line.split(",")]
        if len(values) == len(GPU_FIELDS):
            rows.append({key.replace(".", "_"): value if key == "uuid" else _number(value)
                         for key, value in zip(GPU_FIELDS, values)})
    return rows


def _gpu_apps() -> list[dict]:
    output = _query(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                     "--format=csv,noheader,nounits"])
    if output is None:
        return []
    apps = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            apps.append({"gpu_uuid": parts[0], "pid": _number(parts[1]),
                         "used_gpu_memory_mib": _number(parts[2])})
    return apps


def _meminfo() -> dict:
    data = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        key, sep, value = line.partition(":")
        if sep and key in {"MemTotal", "MemAvailable"}:
            data[key + "_kib"] = _number(value.split()[0])
    return {key: data.get(key) for key in ("MemTotal_kib", "MemAvailable_kib")}


def _pressure() -> dict:
    data = {}
    for line in (_read("/proc/pressure/memory") or "").splitlines():
        parts = line.split()
        if not parts or parts[0] not in {"some", "full"}:
            continue
        metrics = {}
        for part in parts[1:]:
            key, sep, value = part.partition("=")
            if sep:
                metrics[key] = _number(value)
        data[parts[0]] = metrics
    return data


def _cpu() -> dict:
    for line in (_read("/proc/stat") or "").splitlines():
        parts = line.split()
        if parts and parts[0] == "cpu":
            names = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
            return {name: _number(value) for name, value in zip(names, parts[1:])}
    return {}


def _nic() -> dict:
    # Separate PCIe ports; these counters must not be summed as independent links.
    return {
        interface: {
            direction + "_bytes": _number(
                _read(f"/sys/class/net/{interface}/statistics/{direction}_bytes") or ""
            )
            for direction in ("rx", "tx")
        }
        for interface in F0_INTERFACES
    }


def collect_sample() -> dict:
    """Return one timestamped host sample; counters and units stay explicit."""
    return {"timestamp_utc": _utc(), "monotonic_ns": time.monotonic_ns(),
            "hostname": socket.gethostname(), "memory": _meminfo(),
            "memory_pressure": _pressure(), "cpu_ticks": _cpu(),
            "nic_f0": _nic(), "gpus": _gpu_data(), "gpu_apps": _gpu_apps()}


def probe() -> dict:
    """Read package metadata without importing torch, Ray, or vLLM."""
    versions = {}
    for name in ("ray", "vllm", "torch", "nvidia-cuda-runtime-cu12",
                 "nvidia-cuda-runtime-cu13", "cuda-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    driver = _query(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return {"timestamp_utc": _utc(), "hostname": socket.gethostname(),
            "versions": versions,
            "cuda_runtime_package_version": (versions["nvidia-cuda-runtime-cu13"]
                                             or versions["nvidia-cuda-runtime-cu12"]),
            "gpu_driver_versions": sorted(set(driver.splitlines())) if driver else [],
            "gpus": _gpu_data()}


def validate_worker_records(records: list[dict], expected_nodes_by_id: dict,
                            expected_tp: int, model_path: str) -> list[dict]:
    """Validate observed, loaded ranks against the requested topology.

    ``expected_nodes_by_id`` maps node IDs to hostnames or inventory dicts with
    ``expected_hostname`` (or ``hostname``).
    The caller must supply records read from individual worker evidence files.
    """
    if not isinstance(expected_tp, int) or isinstance(expected_tp, bool) or expected_tp < 1:
        raise ValueError("expected_tp inválido")
    loaded = [r for r in records if r.get("stage") == "load_model"]
    if len(loaded) != expected_tp:
        raise ValueError("cantidad de ranks observados distinta de TP")
    ranks = set()
    devices = set()
    nodes = set()
    for record in loaded:
        if not {"node_id", "hostname", "gpu_uuid", "rank"}.issubset(record):
            raise ValueError("identidad de worker incompleta")
        node_id, rank = record["node_id"], record["rank"]
        if node_id not in expected_nodes_by_id or not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError("nodo o rank inesperado")
        host = expected_nodes_by_id[node_id]
        if isinstance(host, dict):
            host = host.get("expected_hostname", host.get("hostname"))
        if not isinstance(host, str) or not host or record["hostname"] != host:
            raise ValueError("hostname inesperado")
        if not isinstance(record["gpu_uuid"], str) or not record["gpu_uuid"].startswith("GPU-"):
            raise ValueError("GPU UUID ausente")
        if record["gpu_uuid"] in devices:
            raise ValueError("GPU UUID duplicado")
        devices.add(record["gpu_uuid"])
        nodes.add(node_id)
        if record.get("model_path") != model_path or record.get("world_size") != expected_tp:
            raise ValueError("modelo o world size inesperado")
        if type(record.get("world_size")) is not int:
            raise ValueError("world size inválido")
        if record.get("effective_tp") != expected_tp or record.get("effective_pp") != 1:
            raise ValueError("paralelismo efectivo inesperado")
        if type(record.get("effective_tp")) is not int or type(record.get("effective_pp")) is not int:
            raise ValueError("paralelismo efectivo inválido")
        if record.get("dtype") != "torch.bfloat16" or record.get("tokenizer_path") != model_path:
            raise ValueError("dtype o tokenizer inesperado")
        versions = record.get("versions")
        if not isinstance(versions, dict) or not all(
            isinstance(versions.get(name), str) and versions[name]
            for name in ("vllm", "ray", "torch")
        ):
            raise ValueError("versiones de paquetes ausentes")
        if rank in ranks:
            raise ValueError("rank duplicado")
        ranks.add(rank)
    if ranks != set(range(expected_tp)):
        raise ValueError("ranks no contiguos")
    if nodes != set(expected_nodes_by_id):
        raise ValueError("nodos observados incompletos")
    return sorted(loaded, key=lambda record: record["rank"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    telemetry = sub.add_parser("telemetry")
    telemetry.add_argument("--seconds", type=float, required=True)
    telemetry.add_argument("--interval", type=float, default=1.0)
    sub.add_parser("probe")
    args = parser.parse_args(argv)
    if args.command == "probe":
        print(json.dumps(probe(), sort_keys=True), flush=True)
        return 0
    if args.seconds <= 0 or args.interval <= 0:
        parser.error("seconds e interval deben ser positivos")
    end = time.monotonic() + args.seconds
    while True:
        print(json.dumps(collect_sample(), sort_keys=True), flush=True)
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(args.interval, remaining))
    return 0


if __name__ == "__main__":
    sys.exit(main())
