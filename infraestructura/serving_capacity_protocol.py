"""Frozen workload, open-loop schedule and reservation plan for serving capacity."""

from pathlib import Path
import re

from infraestructura import cluster, inference, inference_runtime as runtime, manifest
from infraestructura.rutas import REPO_ROOT


PROTOCOL = "serving-slo-4-2-1-v1"
DEFAULT_CONFIG = REPO_ROOT / "experimentos/configs/serving-capacidad-slo.json"
REFERENCE_CONFIG = REPO_ROOT / "experimentos/configs/inferencia-4-2-1.json"
HISTORICAL_ID = "20261119T175208.562948Z-e2644646"
HISTORICAL_INDEX = f"resultados/raw/inference-campaign/{HISTORICAL_ID}/campaign.json"
HISTORICAL_SHA256 = "7643c834a5105e43554df196c6a452a6aff1f5d251d47d91b3151e061e1a29f7"
HISTORICAL_PREFLIGHT_SHA256 = "a86ca0d58a0de592a39a17fc18afab6d505bc332462159f25fb3bcc33150fb1d"
RATES = (0.25, 0.5, 0.75, 1, 1.25, 1.5, 2)
GROUPS = ("four", "pair", "single")
PHASES = ("warmup", "measured")


def validate_protocol(config):
    """Check the approved design, including the historical input pins."""
    if config.get("protocol_id") != PROTOCOL or config.get("schema_version") != "1.0":
        raise ValueError("protocolo de capacidad desconocido")
    measurement = config["measurement"]
    counts = {"blocks": 3, "warmup_requests": 16, "measured_requests": 128,
              "telemetry_interval_seconds": 1, "startup_limit_seconds": 900,
              "cell_limit_seconds": 900, "cleanup_reserve_seconds": 300, "session_limit_seconds": 43200}
    if any(type(measurement.get(k)) is not int or measurement[k] != v for k, v in counts.items()):
        raise ValueError("conteos o presupuesto de medición incompatibles con el protocolo")
    if measurement.get("campaign_order") != list(GROUPS):
        raise ValueError("la comparación requiere bloques en orden cuatro, dos y un nodo")
    capacity = config["capacity"]
    limits = {"ttft_s": 3, "e2e_s": 12, "compliance": 0.95, "request_timeout_s": 30,
              "dispatch_lag_limit_s": 0.05}
    if any(type(capacity.get(k)) not in (int, float) or capacity[k] != v for k, v in limits.items()):
        raise ValueError("límites del SLO, timeout o tolerancia del generador inválidos")
    rates = capacity.get("rates_rps")
    if (capacity.get("arrival_process") != "regular" or rates != list(RATES)
            or any(type(rate) not in (int, float) for rate in rates)):
        raise ValueError("calendario de tasas fuera del protocolo")
    if [model["key"] for model in config["models"]] != ["qwen35-9b", "llama31-8b"]:
        raise ValueError("modelos u orden de la campaña inválidos")
    if config["workload"] != {"seed": 20261106, "input_tokens": [512, 2048],
                              "output_tokens": 128, "temperature": 0, "ignore_eos": True}:
        raise ValueError("carga mixta fuera del protocolo")
    for value in config["paths"].values():
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts or "\x00" in value:
            raise ValueError("directorio del runtime inválido")
    ref = config["reference"]
    expected = {f"workload/{model['key']}/{length}/{phase}.jsonl"
                for model in config["models"] for length in (512, 2048) for phase in PHASES}
    pins = ref["historical_jsonl_sha256"]
    if (ref.get("historical_campaign") != HISTORICAL_INDEX
            or ref.get("historical_campaign_sha256") != HISTORICAL_SHA256
            or ref.get("historical_preflight_sha256") != HISTORICAL_PREFLIGHT_SHA256
            or set(pins) != expected or not all(isinstance(v, str) and re.fullmatch("[0-9a-f]{64}", v)
                                                for v in pins.values())):
        raise ValueError("índice histórico u ocho hashes JSONL inválidos")


def _match_reference(config, historical):
    for field in ("models", "image", "network", "runtime", "cluster_config", "paths"):
        if config[field] != historical[field]:
            raise ValueError(f"{field} difiere de la referencia 4/2/1")
    historical_workload = {k: v for k, v in historical["workload"].items() if k != "concurrencies"}
    if config["workload"] != historical_workload:
        raise ValueError("tokenización o generación difieren de la referencia 4/2/1")


def load_config(path=DEFAULT_CONFIG):
    config = inference.read_json(path)
    validate_protocol(config)
    _match_reference(config, inference.read_json(REFERENCE_CONFIG))
    cluster_path = REPO_ROOT / config["cluster_config"]
    inventory = cluster.load_config(cluster_path)
    ips = config["network"]["node_ips"]
    if any(ips[node["node_id"]] not in node["data_addresses"] for node in inventory["nodes"]):
        raise ValueError("la dirección de datos del servicio difiere del inventario")
    return config, inventory, cluster_path


def reference_index_path(config, root=REPO_ROOT):
    return Path(root) / config["reference"]["historical_campaign"]


def historical_preflight_path(config, root=REPO_ROOT):
    return reference_index_path(config, root).with_name("preflight.json")


def _hardware_roster(preflight):
    expected = set(inference.TOPOLOGIES["four"])
    nodes = {node["node_id"]: node for node in preflight["nodes"]}
    if len(preflight["nodes"]) != len(expected) or set(nodes) != expected:
        raise ValueError("la referencia de hardware requiere los cuatro nodos sin duplicados")
    roster = {}
    for name, node in nodes.items():
        gpu = node["gpu"]
        lines = gpu["stdout"].strip().splitlines()
        if gpu["returncode"] != 0 or len(lines) != 1:
            raise ValueError(f"{name}: identidad de GPU ausente o ambigua")
        fields = [field.strip() for field in lines[0].split(",")]
        if len(fields) != 3 or not all(fields) or not fields[0].startswith("GPU-"):
            raise ValueError(f"{name}: se requieren UUID, modelo de GPU y driver")
        roster[name] = {"hostname": node["hostname"], "architecture": node["architecture"],
                        "gpu_uuid": fields[0], "gpu_model": fields[1], "driver": fields[2]}
    if any(len({row[field] for row in roster.values()}) != len(expected)
           for field in ("hostname", "gpu_uuid")):
        raise ValueError("host o GPU duplicado en el preflight")
    return nodes, roster


def verify_hardware(config, historical_path, observed=None):
    """Bind a live or archived observation to the original four-node preflight."""
    if (config["reference"]["historical_preflight_sha256"] != HISTORICAL_PREFLIGHT_SHA256
            or manifest.sha256_file(historical_path) != HISTORICAL_PREFLIGHT_SHA256):
        raise ValueError("SHA-256 del preflight histórico incompatible")
    reference = inference.read_json(historical_path)
    if reference.get("status") != "ready_for_smoke" or reference.get("errors") != []:
        raise ValueError("la referencia histórica de los nodos está incompleta")
    original, expected = _hardware_roster(reference)
    if observed is not None:
        current, actual = _hardware_roster(observed)
        for name, identity in expected.items():
            for field, value in identity.items():
                if actual[name][field] != value:
                    raise ValueError(f"{name}: {field} difiere de la campaña histórica")
            for model in config["models"]:
                key = model["key"]
                if inference._probe_identity(current[name], key) != inference._probe_identity(original[name], key):
                    raise ValueError(f"{name}/{key}: el entorno o los pesos difieren del preflight histórico")
    return expected


def validate_reference_index(config, path):
    """Validate the pinned index, including when archived inside a new run."""
    if manifest.sha256_file(path) != HISTORICAL_SHA256:
        raise ValueError("el SHA-256 del índice histórico no coincide")
    historical = inference.read_json(path)
    if (historical["campaign_id"] != HISTORICAL_ID or historical["status"] != "complete"
            or historical["pending_jobs"] or historical["errors"]):
        raise ValueError("la campaña histórica no tiene un cierre completo")
    _match_reference(config, historical["protocol"])
    hashes = historical["input_hashes"]
    jsonl = {k: v for k, v in hashes.items() if k.startswith("workload/") and k.endswith(".jsonl")}
    if jsonl != config["reference"]["historical_jsonl_sha256"]:
        raise ValueError("las cargas solicitadas difieren del índice histórico")
    return historical


def frozen_workloads(config, root=REPO_ROOT):
    """Locate the archived, tokenized prompts without running a tokenizer or model."""
    index = reference_index_path(config, root)
    historical = validate_reference_index(config, index)
    marker = inference.read_json(index.parent / "COMPLETE")
    if marker["preflight_sha256"] != HISTORICAL_PREFLIGHT_SHA256:
        raise ValueError("el cierre difiere de la referencia histórica de los cuatro nodos")
    for field, filename in (("campaign_sha256", "campaign.json"), ("plan_sha256", "plan.json"),
                            ("preflight_sha256", "preflight.json")):
        if marker[field] != manifest.sha256_file(index.parent / filename):
            raise ValueError(f"cierre histórico inconsistente para {filename}")
    first = historical["runs"][0]
    # The historical index records a host-specific absolute path. Run IDs are portable.
    source = Path(root) / "resultados/raw/inference-serving" / Path(first["run_dir"]).name
    if manifest.sha256_file(source / "manifest.json") != first["manifest_sha256"]:
        raise ValueError("el manifiesto de las cargas históricas cambió")
    if manifest.load_run(source)["status"] != "complete":
        raise ValueError("la corrida original de los prompts está incompleta")
    directory = source / "raw/inputs/workload"
    expected = {name: digest for name, digest in historical["input_hashes"].items()
                if name.startswith("workload/")}
    actual_names = {"workload/" + file.relative_to(directory).as_posix()
                    for file in directory.rglob("*") if file.is_file()}
    if actual_names != set(expected):
        raise ValueError("archivos del workload histórico ausentes o adicionales")
    for name, digest in expected.items():
        candidate = manifest._internal_file(source, "raw/inputs/" + name)
        if manifest.sha256_file(candidate) != digest:
            raise ValueError(f"hash de la carga histórica inválido para {name}")
    return directory


def jobs(config):
    for block in range(1, config["measurement"]["blocks"] + 1):
        for model in config["models"]:
            for group in config["measurement"]["campaign_order"]:
                yield {"name": f"b{block}-{model['key']}-{group}", "block": block,
                       "model": model, "group": group,
                       "cells": [{"rate_rps": rate} for rate in config["capacity"]["rates_rps"]]}


def point_name(rate):
    return "rate-" + format(rate, "g").replace(".", "p")


def client_command(config, model, rate, phase):
    count = config["measurement"][phase + "_requests"]
    return ["python3", "-m", "infraestructura.serving_capacity_client", "--model", model["key"],
            "--port", str(config["runtime"]["api_port"]), "--workloads", f"/work/workloads/{model['key']}",
            "--rate", str(rate), "--phase", phase, "--count", str(count),
            "--request-timeout", str(config["capacity"]["request_timeout_s"]),
            "--output", f"/work/metrics/{point_name(rate)}/{phase}.json"]


def planned_jobs(config, inventory):
    preview = []
    for job in jobs(config):
        nodes = cluster.selected_nodes(inventory, job["group"])
        cells = [{"rate_rps": c["rate_rps"], "commands": {
            phase: client_command(config, job["model"], c["rate_rps"], phase) for phase in PHASES}}
            for c in job["cells"]]
        preview.append({**job, "nodes": [n["node_id"] for n in nodes], "cells": cells,
                        "server_command": runtime.server_command(config, job["model"], len(nodes)),
                        "ray_commands": {n["node_id"]: runtime.ray_command(n, config) for n in nodes}})
    return preview


def make_plan(path=DEFAULT_CONFIG):
    config, inventory, _ = load_config(path)
    preview = planned_jobs(config, inventory)
    m = config["measurement"]
    rates = config["capacity"]["rates_rps"]
    job_count = len(preview)
    points = job_count * len(rates)
    per_job = sum(1 / rate for rate in rates)
    measured = round(job_count * m["measured_requests"] * per_job, 3)
    warm = round(job_count * m["warmup_requests"] * per_job, 3)
    return {
        "status": "planned_only", "protocol": config, "jobs": preview,
        "counts": {"jobs": job_count, "blocks": m["blocks"], "points": points,
                   "warmup_requests": m["warmup_requests"] * points,
                   "measured_requests": m["measured_requests"] * points},
        "resources_max": {"nodes": 4, "gpus": 4, "replicas": 1},
        "budget": {"measured_emission_seconds": measured, "warmup_emission_seconds": warm,
                   "emission_seconds_per_block": round((measured + warm) / m["blocks"], 3),
                   "proposed_reservation_seconds": m["session_limit_seconds"],
                   "proposed_seconds_per_block": m["session_limit_seconds"] // m["blocks"],
                   "cleanup_reserve_seconds": m["cleanup_reserve_seconds"],
                   "basis": "Ventana de N/r segundos por fase. El presupuesto añade arranque, "
                            "drenaje entre fases, recolección y cierre."},
        "preflight_commands": {
            n["node_id"]: cluster.ssh_command(n, runtime.probe_command(n, config, container_probe=True))
            for n in inventory["nodes"]},
        "outputs": {"index": "resultados/raw/serving-capacity-campaign/<campaign_id>/",
                    "runs": "resultados/raw/inference-serving/<run_id>/",
                    "report": "JSON y Markdown en la ruta nueva indicada por --output"},
    }
