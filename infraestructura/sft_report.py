"""Reconstrucción local de evidencia y comparaciones de SFT completo."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
from statistics import median

from . import cluster as cluster_inventory, manifest, sft_protocol
from .rutas import REPO_ROOT
from .serving_capacity_protocol import _hardware_roster
from .sft_runtime import GIB, resource_limits


_GROUPS = ("four", "pair", "single")
_PAIRS = (("four", "pair", 2), ("pair", "single", 2), ("four", "single", 4))
_PAIR_NAMES = {"four": "4", "pair": "2", "single": "1"}
_CAMPAIGN_FILES = ("plan", "config", "cluster", "preflight", "campaign")
_IDENTITY = ("campaign_id", "run_id", "node_id", "hostname", "gpu_uuid", "rank", "world_size", "phase")
_REQUIRED_INPUTS = {"requested_config", "cluster.json", "reference/preflight.json", "corpus.jsonl"} | {
    f"code/infraestructura/{name}.py" for name in sft_protocol.SOURCE_MODULES
}
_GPU_FIELDS = ("temperature_gpu", "power_draw", "utilization_gpu", "memory_used", "memory_total",
               "clocks_sm", "clocks_mem")
_MEMORY_FIELDS = ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "device_free_bytes",
                  "device_total_bytes")
_OOM_STAGES = {"model_materialize", "snapshot_load", "restore", "checkpoint_write", "train_step"}
_TORCH_VERSIONS = ("torch_version", "torch_runtime_version")
_EVIDENCE_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, manifest.ManifestError)


# Lectura y validación de evidencia


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_object(value):
    _require(isinstance(value, dict), "se esperaba un objeto JSON")
    return value


def _read(path):
    return _json_object(json.loads(path.read_text(encoding="utf-8")))


def _file(root, relative):
    path = root / relative
    _require(path.is_relative_to(root) and path == path.resolve() and path.is_file(),
             f"archivo ausente o ruta no canónica: {relative}")
    return path


def _object(root, relative):
    return _read(_file(root, relative))


def _rows(root, relative):
    with _file(root, relative).open(encoding="utf-8") as source:
        return [_json_object(json.loads(line)) for line in source]


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _expect(record, expected, location):
    for field, value in expected.items():
        _require(field in record and _same(record[field], value), f"{location}: {field} incompatible")


def _number(value, label, *, minimum=0, positive=False):
    _require(type(value) in (int, float) and math.isfinite(value)
             and (value > minimum if positive else value >= minimum), f"{label}: valor inválido")
    return value


def _count(value, label):
    _require(type(value) is int and value >= 0, f"{label}: conteo inválido")
    return value


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _timestamp(value):
    timestamp = datetime.fromisoformat(value)
    _require(timestamp.utcoffset() is not None, "marca temporal sin zona horaria")
    return timestamp


def _spread(values):
    return {"median": median(values) if values else None,
            "range": [min(values), max(values)] if values else None}


# Entradas y control de los trabajos


def _input_hashes(root, run):
    records = list(run["inputs"])
    if run["requested_config_file"] is not None:
        records.append(run["requested_config_file"])
    indexed = {}
    for record in records:
        name = record["name"]
        _require(isinstance(name, str) and name not in indexed, "entrada inválida o duplicada")
        archived = _file(root, f"raw/inputs/{name}")
        digest = manifest.sha256_file(archived)
        _require(digest == record["sha256"] and archived.stat().st_size == record["size_bytes"],
                 f"entrada archivada incompatible: {name}")
        indexed[name] = digest
    if run["status"] != "running":
        verification = {row["name"]: row for row in run["input_verification"]}
        _require(len(verification) == len(run["input_verification"]) == len(indexed),
                 "verificación de entradas incompleta o duplicada")
        for name, digest in indexed.items():
            _expect(verification[name], {"matches": True, "expected_sha256": digest,
                                         "observed_sha256": digest}, name)
    return indexed


def _validate_manifest(run_directory, run, job, config):
    model = config["model"]
    _expect(run, {"kind": "full-sft", "schema_version": "2.2", "run_id": run_directory.name,
                  "model": {"id": model["id"], "revision": model["revision"],
                            "tokenizer": {"id": model["tokenizer_id"],
                                          "revision": model["tokenizer_revision"]}}}, "manifiesto")
    request = _json_object(run["requested_config"])
    _expect(request, {"job": job, "protocol": config}, "solicitud")
    manifest._validate_component(request.get("campaign_id"), "campaign_id")
    _require(run["policy"]["strict"] is True and run["git"]["dirty"] is False
             and isinstance(run["git"]["commit"], str), "política Git o manifiesto incompatible")
    _require(run["status"] != "complete" or not run["errors"], "corrida completa con errores")
    return request["campaign_id"]


def _preflight_environment(preflight, config):
    _expect(preflight, {"status": "ready", "errors": []}, "preflight")
    nodes, roster = _hardware_roster(preflight)
    _expect(preflight, {"roster": roster}, "preflight")
    _removal(preflight["removal"], cluster_inventory.TOPOLOGIES["four"])
    versions = (*_TORCH_VERSIONS, "transformers_version", "tokenizers_version")
    environment = {}
    for node, row in nodes.items():
        image = row["image"]
        _require(image["returncode"] == 0 and shlex.split(image["stdout"]) ==
                 [config["image"]["id"], config["image"]["architecture"]], f"{node}: imagen observada distinta")
        probe = row["sft_container_probe"]
        _expect(probe, {**{key: config["image"][key] for key in versions}, "gpus_hidden": True,
                        "nccl_available": True}, f"{node}: sonda de entorno")
        _require(all(probe[key] for key in ("python", "torch_cuda_version", "nccl_version")),
                 f"{node}: entorno de CUDA, NCCL o Python incompleto")
        environment[node] = {**roster[node], "image_id": config["image"]["id"],
                             **{key: probe[key] for key in (*versions, "python", "torch_cuda_version", "nccl_version")}}
    return environment


def _validate_workload(root, hashes, config, preflight):
    _require(_REQUIRED_INPUTS <= hashes.keys(), "faltan fuentes esenciales en la corrida")
    workload = config["workload"]
    corpus = hashlib.sha256(sft_protocol.corpus_bytes()).hexdigest()
    _require(corpus == workload["corpus_sha256"] == hashes.get("corpus.jsonl"),
             "corpus archivado incompatible con la carga")
    _require(hashes.get("reference/preflight.json") == config["reference_preflight_sha256"],
             "preflight histórico ausente o incompatible")
    _require(hashes.get(f"code/{config['cluster_config']}") == hashes["cluster.json"],
             "copia de inventario del worker distinta")
    _require(_same(_object(root, "raw/inputs/requested_config"), config), "config archivada incompatible")
    historical = _object(root, "raw/inputs/reference/preflight.json")
    _expect(historical, {"status": "ready_for_smoke", "errors": []}, "preflight histórico")
    history_nodes, hardware = _hardware_roster(historical)
    if preflight is not None:
        _expect(preflight, {"reference_preflight_sha256": config["reference_preflight_sha256"],
                            "roster": hardware}, "preflight efectivo")
        _preflight_environment(preflight, config)
    _require(sft_protocol.tokens_sha256(sft_protocol.frozen_tokens()) == workload["tokens_sha256"],
             "tokens de referencia incompatibles con la carga")
    identities = {node: {key: item[key] for key in ("hostname", "gpu_uuid")} for node, item in hardware.items()}
    model_records = {node: row["models"][config["model"]["key"]] for node, row in history_nodes.items()}
    return identities, model_records


def _workspace(root, job, campaign_id, config, hashes):
    path = Path(config["paths"]["workspace"]) / campaign_id / root.name
    receipt = _object(root, "raw/workspace.json")
    _expect(receipt, {"path": str(path), "retained": True, "errors": {}}, "workspace")
    staged = _json_object(receipt["staged"])
    _require(set(staged) == set(job["nodes"]), "copias remotas incompletas")
    expected = {"config.json": hashes["requested_config"], "corpus.jsonl": hashes["corpus.jsonl"]}
    expected.update({name: digest for name, digest in hashes.items() if name.startswith("code/")})
    for node, copy in staged.items():
        _expect(copy, {"workspace": str(path), "files": expected}, f"copia remota en {node}")
        _require(_count(copy["uid"], "UID") > 0, f"usuario remoto inválido en {node}")
    return _number(receipt["setup_before_launch_seconds"], "preparación antes del lanzamiento")


def _removal(removal, nodes):
    _expect(removal, {"removal_confirmed": True}, "retirada")
    observations = _json_object(removal["nodes"])
    _require(set(observations) == set(nodes), "retirada sin cobertura de cuatro nodos")
    for node, row in observations.items():
        _expect(row, {"confirmed": True, "remaining_ids": []}, f"retirada en {node}")
        removed = row["removed_ids"]
        _require(isinstance(removed, list) and all(_digest(cid) for cid in removed)
                 and len(set(removed)) == len(removed), f"IDs de contenedor inválidos en {node}")
    _timestamp(removal["ended_at"])


def _cleanup(root, job, campaign_id, phases):
    cleanup = _object(root, "raw/cleanup.json")
    _expect(cleanup, {"campaign_id": campaign_id, "run_id": root.name,
                      "removal_confirmed": True}, "retirada")
    rows = cleanup["nodes"]
    _require(isinstance(rows, list) and len(rows) == len(job["nodes"]), "retirada sin recibo de todos los nodos")
    _require({row["node_id"] for row in rows} == set(job["nodes"])
             and all(row["status"] == "confirmed_absent" and not row.get("errors") for row in rows),
             "retirada de recursos sin confirmar")
    nodes = cluster_inventory.TOPOLOGIES["four"]
    _expect(cleanup, {"coverage": nodes}, "cobertura de retirada")
    _require(set(cleanup["phases"]) == {*phases, "final"}, "fases de retirada incompletas")
    for phase in (*phases, "final"):
        removal = cleanup["phases"][phase]
        _removal(removal, nodes)
        if phase != "final":
            terminal = _object(root, f"logs/{phase}-terminal.json")
            _expect(terminal, {"removal": removal, "errors": []}, f"cierre de {phase}")
    return _number(cleanup["removal_seconds"], "tiempo de retirada")


def _phase_control(root, job, config, campaign_id, phase, oom_nodes):
    commands = _object(root, f"logs/{phase}-commands.json")
    containers = _object(root, f"logs/{phase}-containers.json")
    terminal = _object(root, f"logs/{phase}-terminal.json")
    resources = _object(root, f"logs/{phase}-resources.json")
    nodes = set(job["nodes"])
    _require(set(commands) == set(containers) == set(terminal["states"]) == set(resources) == nodes,
             f"{phase}: recursos, comandos, contenedores o estados incompletos")
    _require(all(_digest(cid) for cid in containers.values()) and len(set(containers.values())) == len(nodes),
             "IDs de contenedor incompletos o duplicados")
    failed = set()
    running = set()
    workspace = str(Path(config["paths"]["workspace"]) / campaign_id / root.name)
    limits = resource_limits(config, [{"node_id": node} for node in job["nodes"]], [job])
    for rank, node in enumerate(job["nodes"]):
        _phase_resources(resources[node], node, phase, workspace, limits[node])
        log = _object(root, f"logs/{phase}/{node}.json")
        _expect(log, {"returncode": 0, "timed_out": False}, f"logs de {node}/{phase}")
        _require(all(isinstance(log[key], str) for key in ("stdout", "stderr")), f"logs inválidos de {node}/{phase}")
        argv = log["command"]
        _require(isinstance(argv, list) and argv and all(isinstance(part, str) for part in argv),
                 f"comando de captura inválido de {node}/{phase}")
        argv = shlex.split(argv[-1]) if argv[0] == "ssh" else argv
        _require(argv[-5:] == ["docker", "logs", "--tail", "5000", containers[node]],
                 f"logs de otro contenedor en {node}/{phase}")
        _number(log["duration_seconds"], f"duración de captura de logs de {node}/{phase}")
        command = commands[node]
        worker = sft_protocol.worker_argv(
            config, job, rank, phase, config_path="/work/config.json",
            model_dir=Path("/models") / sft_protocol.model_subdir(config), out=f"/work/out/{phase}",
            checkpoint_dir="/work/checkpoint", campaign_id=campaign_id, run_id=root.name,
        )
        _require(isinstance(command, list) and command[:2] == ["docker", "create"]
                 and command[-len(worker):] == worker and "--pull=never" in command
                 and config["image"]["id"] in command, f"{phase}: lanzamiento distinto en {node}")
        for name, value in config["network"]["environment"].items():
            _require(f"{name}={value}" in command, f"{phase}: entorno de red distinto en {node}")
        state = terminal["states"][node]
        _require(state["OOMKilled"] is False and state["Error"] == "",
                 f"{phase}: terminación del contenedor incompatible en {node}")
        if state["Status"] == "running":
            running.add(node)
        elif state["Status"] != "exited" or state["ExitCode"] != 0:
            failed.add(node)
        _require(containers[node] in terminal["removal"]["nodes"][node]["removed_ids"],
                 f"{phase}: falta recibo del contenedor de {node}")
    if oom_nodes:
        states = terminal["states"]
        _require(bool(failed) and failed <= oom_nodes
                 and all(states[node]["Status"] == "exited" and states[node]["ExitCode"] == 1 for node in failed),
                 f"{phase}: fallo sin evidencia CUDA OOM o código de salida incompatible")
    else:
        _require(not failed and not running, f"{phase}: ranks sin salida correcta")
    elapsed = _number(terminal["launch_to_terminal_seconds"], f"tiempo de lanzamiento y fase {phase}")
    return containers, elapsed


def _phase_resources(receipt, node, phase, workspace, limit):
    _expect(receipt, {"node_id": node, "phase": phase}, "sonda de recursos")
    _timestamp(receipt["observed_at"])
    host, assessment = receipt["resources"], receipt["assessment"]
    _expect(host, {"workspace": workspace, "workspace_writable": True}, f"workspace en {node}")
    expected = dict(limit)
    if phase != "pilot":
        expected.update(checkpoint_bytes=0, checkpoint_inodes=0)
    expected.update(minimum_disk_bytes=expected["checkpoint_bytes"] + GIB,
                    minimum_disk_inodes=expected["checkpoint_inodes"] + 256)
    _expect(assessment, expected, f"pisos de recursos en {node}")
    available = _count(assessment["effective_unified_available_bytes"], "memoria unificada")
    _require(expected["minimum_unified_bytes"] <= available <= _count(receipt["memory_bytes"]["MemAvailable"], "RAM")
             and 0 < _number(assessment["effective_cpu_count"], "CPU efectiva") <= _count(host["cpu_affinity_count"], "CPU")
             and _count(host["checkpoint_disk_free_bytes"], "disco libre") >= expected["minimum_disk_bytes"]
             and _count(host["checkpoint_disk_free_inodes"], "inodos libres") >= expected["minimum_disk_inodes"],
             f"capacidad insuficiente o incoherente en {node}/{phase}")


def _oom_evidence(root, job, campaign_id, phases, hardware, issues):
    evidence = []
    for rank, node in enumerate(job["nodes"]):
        for phase in phases:
            base = f"raw/nodes/{node}/{phase}"
            if not (root / base / "failure.json").exists():
                continue
            try:
                failure = _object(root, f"{base}/failure.json")
                _expect(failure, {"node_id": node, "rank": rank, "phase": phase, "run_id": root.name,
                                  "campaign_id": campaign_id, "world_size": len(job["nodes"]),
                                  "hostname": hardware[node]["hostname"]}, "recibo de fallo")
                _require("gpu_uuid" in failure and failure["gpu_uuid"] in (None, hardware[node]["gpu_uuid"]),
                         "GPU contradictoria en recibo de fallo")
                message = failure["message"]
                _require(failure.get("exception_type") in ("torch.OutOfMemoryError", "torch.cuda.OutOfMemoryError")
                         and isinstance(message, str) and re.search(r"\bcuda\b", message, re.I)
                         and re.search(r"\bout of memory\b", message, re.I),
                         "fallo de worker sin evidencia inequívoca del asignador CUDA")
                _timestamp(failure["timestamp_utc"])
                stage = failure["stage"]
                _require(isinstance(stage, str) and bool(stage), "etapa del fallo ausente")
                evidence.append({"node_id": node, "phase": phase, "rank": rank,
                                 "path": f"{base}/failure.json", "message": message, "stage": stage,
                                 "timestamp_utc": failure["timestamp_utc"]})
                _expect(failure, {"type": "cuda_oom", "status": "cuda_oom", "oom_origin": "cuda"}, "recibo CUDA OOM")
                _require(stage in _OOM_STAGES, "CUDA OOM fuera de una etapa atribuible al entrenamiento")
                identity = _object(root, f"{base}/identity.json")
                _validate_identity(identity, node, phase, rank, job, root.name, campaign_id)
                _expect(identity, hardware[node], "preflight")
                _expect(failure, {key: identity[key] for key in _IDENTITY}, "recibo de fallo")
            except _EVIDENCE_ERRORS as exc:
                issues.append(f"{node}/{phase}: {exc}")
    return evidence


# Evidencia de los ranks


def _validate_identity(identity, node, phase, rank, job, run_id, campaign_id):
    _expect(identity, {"node_id": node, "phase": phase, "rank": rank, "run_id": run_id,
                       "world_size": len(job["nodes"]), "campaign_id": campaign_id}, "identidad rank")
    _require(isinstance(identity.get("hostname"), str) and bool(identity["hostname"])
             and isinstance(identity.get("gpu_uuid"), str) and identity["gpu_uuid"].startswith("GPU-")
             and len(identity["gpu_uuid"]) > 4, "host o UUID de GPU ausente")


def _groups_observed(root, run, job, campaign_id, phases, cluster, hardware):
    hostnames = {row["node_id"]: row["expected_hostname"] for row in cluster["nodes"]}
    for phase in phases:
        roster = []
        for rank, node in enumerate(job["nodes"]):
            base = f"raw/nodes/{node}/{phase}"
            identity = _object(root, f"{base}/identity.json")
            _validate_identity(identity, node, phase, rank, job, root.name, campaign_id)
            _expect(identity, hardware[node], "preflight")
            _expect(identity, {"hostname": hostnames[node]}, "inventario")
            roster.append({key: identity[key] for key in _IDENTITY})
        for identity in roster:
            node = identity["node_id"]
            group = _object(root, f"raw/nodes/{node}/{phase}/group.json")
            _expect(group, {**identity, "backend": "nccl", "participants": roster}, f"grupo en {node}/{phase}")
    fields = ("node_id", "hostname", "gpu_uuid", "rank")
    participants = run["participants"]
    observed = [{key: identity[key] for key in fields} for identity in roster]
    _require(len(participants) == len(observed)
             and all(len({item[key] for item in observed}) == len(observed) for key in fields),
             "manifiesto del grupo incompleto o identidades duplicadas")
    by_rank = {item["rank"]: item for item in participants}
    for identity in observed:
        base = f"raw/nodes/{identity['node_id']}/{phases[0]}"
        _expect(by_rank[identity["rank"]], {**identity, "evidence_path": f"{base}/identity.json"},
                "participante del manifiesto")
    return observed


def _check_receipt_rosters(phases):
    for phase, ranks in phases.items():
        roster = [{key: rank["identity"][key] for key in _IDENTITY} for rank in ranks]
        for rank in ranks:
            _expect(rank["receipt"], {"participants": roster, "model_source": ranks[0]["receipt"]["model_source"]},
                    f"recibo de {phase}")


def _receipt_evidence(root, base, receipt, identity, config, config_sha256, historical):
    expected = {key: identity[key] for key in _IDENTITY}
    expected.update({key: config["workload"][key] for key in ("corpus_sha256", "tokens_sha256")})
    expected.update({key: config["image"][key] for key in (*_TORCH_VERSIONS, "transformers_version")})
    expected.update(schema_version="1.0", status="ok", config_sha256=config_sha256,
                    model_revision=config["model"]["revision"], tokenizer_revision=config["model"]["tokenizer_revision"],
                    tensor_count=config["model"]["tensor_count"], parameter_count=config["model"]["parameter_count"],
                    tensor_shapes_sha256=config["model"]["tensor_shapes_sha256"],
                    gradient_accumulation_steps=config["training"]["global_batch"] // identity["world_size"],
                    reduce_dtype="float32", attention="sdpa", activation_checkpointing="non-reentrant")
    keys = ("global_batch", "microbatch", "compute_dtype", "master_dtype", "gradient_dtype", "use_cache",
            "optimizer", "learning_rate")
    expected.update({key: config["training"][key] for key in keys})
    _expect(receipt, expected, "recibo")
    duration = _number(receipt["phase_seconds"], "duración de fase", positive=True)
    startup = _number(receipt["startup_seconds"], "arranque", positive=True)
    _expect(receipt, {"startup_scope": "worker_execute_to_first_step"}, "arranque")
    _require(startup < duration, "arranque fuera de la fase")
    _require(_count(receipt["peak_allocated_bytes"], "memoria CUDA máxima") > 0,
             "memoria CUDA máxima ausente")
    if identity["phase"] != "measure":
        _expect(receipt, {"window_seconds": None}, "recibo")
    if identity["phase"] != "pilot":
        _expect(receipt["proof"], {"audit_performed": False, "tensor_count": expected["tensor_count"],
                                   "parameter_count": expected["parameter_count"]}, "auditoría")
    if identity["phase"] == "measure":
        _expect(receipt, {"checkpoint": None}, "recibo")
    records = _json_object(receipt["artifacts"])
    _require(set(records) == {"identity.json", "group.json", "steps.jsonl", "telemetry.jsonl"}, "índice de fase incompleto")
    for name, record in records.items():
        path = _file(root, f"{base}/{name}")
        _expect(record, {"sha256": manifest.sha256_file(path), "bytes": path.stat().st_size}, name)
    _model_source(receipt["model_source"], historical)


def _model_source(source, historical):
    metadata = dict(historical["metadata_sha256"])
    _expect(source, {"index_sha256": metadata.pop("model.safetensors.index.json"),
                     "shards": [{"path": row["name"], "sha256": row["blob_name"], "bytes": row["size_bytes"]}
                                for row in historical["shards"]]}, "snapshot")
    _expect(source["metadata"], metadata, "tokenizer y configuración de modelo")
    _require(all(_digest(value) for value in source["metadata"].values()), "metadato de snapshot inválido")


def _cuda_memory(memory):
    values = {field: _count(memory[field], field) for field in _MEMORY_FIELDS}
    allocated = values["allocated_bytes"]
    _require(allocated <= values["reserved_bytes"] and allocated <= values["peak_allocated_bytes"]
             and 0 <= values["device_free_bytes"] <= values["device_total_bytes"]
             and values["device_total_bytes"] > 0, "contadores de memoria CUDA incompatibles")
    return values


def _phase_steps(rows, steps, identity, config):
    _require(len(rows) == len(steps), "número de pasos incompatible")
    phase, rank, world = (identity[key] for key in ("phase", "rank", "world_size"))
    workload, measurement = config["workload"], config["measurement"]
    for number, row in zip(steps, rows):
        ids = sft_protocol.step_ids(number - 1, rank, world)
        _expect(row, {"step": number, "ids": ids, "rank": rank, "node_id": identity["node_id"], "phase": phase,
                      "loss_definition": "global_supervised_token_mean",
                      "measured": phase == "measure" and number > measurement["warmup_steps"],
                      "audit_performed": phase == "pilot" and number == measurement["pilot_steps"],
                      "global_processed_tokens": config["training"]["global_batch"] * workload["sequence_length"],
                      "global_supervised_tokens": config["training"]["global_batch"] * workload["supervised_tokens"]},
                f"paso {number}")
        processed = _count(row["processed_tokens"], "tokens procesados")
        supervised = _count(row["supervised_tokens"], "tokens supervisados")
        _require(processed == len(ids) * workload["sequence_length"]
                 and supervised == len(ids) * workload["supervised_tokens"],
                 f"conteos de tokens incompatibles en paso {number}")
        _number(row["seconds"], "tiempo local del paso", positive=True)
        audit_seconds = _number(row["parameter_audit_seconds"], "tiempo de auditoría")
        _require((audit_seconds > 0) == row["audit_performed"], "tiempo de auditoría incompatible con el paso")
        _number(row["loss"], "pérdida")
        _cuda_memory(row["cuda_memory"])
    return rows


def _summarize_steps(ranks):
    steps = []
    for rows in zip(*(rank["steps"] for rank in ranks)):
        first = rows[0]
        _require(all(row["loss"] == first["loss"] for row in rows), "pérdida global difiere entre ranks")
        steps.append({
            "step": first["step"],
            "seconds": max(row["seconds"] for row in rows),
            "parameter_audit_seconds": max(row["parameter_audit_seconds"] for row in rows)
            if first["audit_performed"] else None,
            "samples": len({sample for row in rows for sample in row["ids"]}),
            "processed_tokens": sum(row["processed_tokens"] for row in rows),
            "supervised_tokens": sum(row["supervised_tokens"] for row in rows),
            "loss": first["loss"],
        })
    return steps


def _phase(root, job, config, campaign_id, phase, steps, config_sha256, model_records):
    ranks = []
    for rank, node in enumerate(job["nodes"]):
        base = f"raw/nodes/{node}/{phase}"
        identity = _object(root, f"{base}/identity.json")
        _validate_identity(identity, node, phase, rank, job, root.name, campaign_id)
        receipt = _object(root, f"{base}/receipt.json")
        _receipt_evidence(root, base, receipt, identity, config, config_sha256, model_records[node])
        rows = _phase_steps(_rows(root, f"{base}/steps.jsonl"), steps, identity, config)
        _require(receipt["startup_seconds"] + math.fsum(row["seconds"] + row["parameter_audit_seconds"] for row in rows)
                 <= receipt["phase_seconds"] + 1e-6, "arranque y pasos exceden la duración local")
        telemetry = _telemetry(_rows(root, f"{base}/telemetry.jsonl"), identity, receipt)
        ranks.append({"identity": identity, "receipt": receipt, "steps": rows, "telemetry": telemetry})
    return ranks


def _phase_window(ranks, steps):
    windows = []
    for rank in ranks:
        receipt = rank["receipt"]
        window = _number(receipt["window_seconds"], "ventana medida", positive=True)
        local_steps = [row["seconds"] for row in rank["steps"] if row["step"] in steps]
        _require(math.fsum(local_steps) <= window + 1e-6 and window <= receipt["phase_seconds"],
                 "ventana medida fuera de fase o más corta que la suma de pasos locales")
        windows.append(window)
    return max(windows)


def _telemetry(rows, identity, receipt):
    _require(bool(rows), "telemetría ausente")
    expected = {key: identity[key] for key in _IDENTITY}
    phase = identity["phase"]
    stages = {"startup", "validation"} | ({"warmup", "measured"} if phase == "measure" else {phase})
    if phase == "pilot":
        stages.add("checkpoint")
    indexes = []
    by_stage = {}
    for row in rows:
        _expect(row, expected, "identidad de telemetría")
        stage = row["training_stage"]
        _require(stage in stages, f"etapa de telemetría inválida: {stage}")
        instant = _count(row["monotonic_ns"], "reloj monotónico")
        _require(not indexes or instant > indexes[-1], "reloj monotónico de telemetría desordenado")
        indexes.append(instant)
        _timestamp(row["timestamp_utc"])
        gpu = [device for device in row["gpus"] if device["uuid"] == identity["gpu_uuid"]]
        _require(len(gpu) == 1, "telemetría sin GPU efectiva o con UUID duplicado")
        signals = {key: gpu[0][key] for key in _GPU_FIELDS}
        signals.update(_cuda_memory(row["cuda_memory"]))
        signals["MemAvailable_kib"] = row["memory"]["MemAvailable_kib"]
        by_stage.setdefault(stage, []).append((instant, signals))
    _require((indexes[-1] - indexes[0]) / 1e9 <= receipt["phase_seconds"] + 1e-6,
             "telemetría excede la duración de fase")
    if phase == "measure":
        _check_telemetry_window(by_stage, receipt)
    return {"samples": len(rows), "first_timestamp_utc": rows[0]["timestamp_utc"],
            "last_timestamp_utc": rows[-1]["timestamp_utc"],
            "stages": {stage: _telemetry_summary(samples) for stage, samples in by_stage.items()}}


def _telemetry_summary(samples):
    summary = {}
    for field in (*_GPU_FIELDS, *_MEMORY_FIELDS, "MemAvailable_kib"):
        values = [_number(row[field], f"telemetría {field}") for _, row in samples if row[field] is not None]
        summary[field] = {"samples": len(values), "max": max(values) if values else None, **_spread(values)}
    return {"samples": len(samples), "first_monotonic_ns": samples[0][0],
            "last_monotonic_ns": samples[-1][0], "summary": summary}


def _check_telemetry_window(by_stage, receipt):
    measured = by_stage.get("measured", [])
    _require(bool(measured), "sin muestras de telemetría en la ventana medida")
    start = _count(receipt["window_start_monotonic_ns"], "inicio de ventana")
    end = _count(receipt["window_end_monotonic_ns"], "fin de ventana")
    _require(start < end and math.isclose((end - start) / 1e9, receipt["window_seconds"],
                                         rel_tol=0, abs_tol=1e-6),
             "marcas monotónicas de ventana incompatibles con su duración")
    _require(all((stage == "measured") == (start <= instant < end)
                 for stage, samples in by_stage.items() for instant, _ in samples),
             "etiquetas de telemetría incompatibles con la ventana monotónica")


# Checkpoint y prueba de actualización


def _checkpoint_file(record):
    path = record["path"]
    manifest._validate_component(path, "archivo de checkpoint")
    _require(_digest(record.get("sha256")) and _count(record["bytes"], "bytes de checkpoint") > 0,
             "huella o tamaño de checkpoint inválido")
    return path


def _shape_size(shape):
    _require(isinstance(shape, list), "forma de tensor inválida")
    return math.prod(_count(value, "dimensión") for value in shape)


def _checkpoint_indexes(pilot, resume, config, rank, world):
    step = config["measurement"]["checkpoint_step"]
    expected = {"step": step, "next_step": step + 1, "hashes_verified": True}
    _expect(pilot, {**expected, "restored": False, "rng_restored": False}, "checkpoint guardado")
    _expect(resume, {**expected, "restored": True, "rng_restored": True,
                     "metadata_sha256": pilot["metadata_sha256"], "files": pilot["files"]},
            "checkpoint restaurado")
    _require(_digest(pilot["metadata_sha256"]), "SHA-256 del índice de checkpoint inválido")
    files = pilot["files"]
    _require(isinstance(files, list) and len(files) == config["model"]["tensor_count"] + 1,
             "inventario de archivos de checkpoint incompleto")
    names = [_checkpoint_file(row) for row in files]
    _require(len(set(names)) == len(names) and names[-1] == "state.pt",
             "archivos repetidos o estado ausente en checkpoint")
    tensors = {}
    for row in files[:-1]:
        name = row["name"]
        _require(isinstance(name, str) and bool(name) and name not in tensors
                 and row["dtype"] == "torch.float32", "tensor de checkpoint inválido o repetido")
        local_size = _shape_size(row["shape"])
        global_size = _shape_size(row["global_shape"])
        first_dimension, *tail = row["global_shape"]
        width = (first_dimension + world - 1) // world
        local_first_dimension = min(width, max(0, first_dimension - rank * width))
        _expect(row, {"shape": [local_first_dimension, *tail]}, f"forma local de {name}")
        _require(local_size <= global_size and row["bytes"] >= 4 * local_size,
                 "tamaño de shard de checkpoint incompatible")
        tensors[name] = {"global_shape": row["global_shape"], "local_size": local_size}
    return tensors


def _tensor_index(rows, expected, label):
    _require(isinstance(rows, list), f"{label}: se requiere una lista de tensores")
    by_name = {row["name"]: row for row in rows}
    _require(len(by_name) == len(rows) and set(by_name) == set(expected),
             f"{label}: tensores incompletos o duplicados")
    return by_name


def _pilot_proof(phases, config):
    model = config["model"]
    global_shapes = {}
    totals = {}
    global_counts = None
    for pilot, resume in zip(phases["pilot"], phases["resume"]):
        pilot_receipt, resume_receipt = pilot["receipt"], resume["receipt"]
        _expect(resume_receipt, {"model_source": pilot_receipt["model_source"]}, "snapshot restaurado")
        tensors = _checkpoint_indexes(pilot_receipt["checkpoint"], resume_receipt["checkpoint"], config,
                                      pilot_receipt["rank"], pilot_receipt["world_size"])
        shapes = {name: row["global_shape"] for name, row in tensors.items()}
        if global_shapes:
            _require(shapes == global_shapes, "formas lógicas diferentes entre ranks")
        else:
            global_shapes = shapes
            _require(sft_protocol.tensor_shapes_sha256(shapes) == model["tensor_shapes_sha256"],
                     "nombres o formas de los tensores distintos del layout BF16 fijado")
            totals = {name: dict(numel=0, finite_grad_elements=0, nonzero_grad_elements=0, changed_shards=0)
                      for name in shapes}
        proof = pilot_receipt["proof"]
        _expect(proof, {"audit_performed": True, "audited_step": config["measurement"]["pilot_steps"],
                        "tensor_count": model["tensor_count"], "parameter_count": model["parameter_count"],
                        "gradient_finite_tensor_count": model["tensor_count"],
                        "gradient_nonzero_tensor_count": model["tensor_count"],
                        "updated_tensor_count": model["tensor_count"],
                        "method": "sha256-fp32-local-before-after-sgd", "hash_chunk_bytes": 1048576},
                "auditoría de parámetros")
        counts = _tensor_index(proof["tensors"], tensors, "auditoría global")
        weights = _tensor_index(proof["local_weight_hashes"], tensors, "hashes de shard")
        if global_counts is None:
            global_counts = counts
        else:
            _require(_same(counts, global_counts), "conteos de auditoría global distintos entre ranks")
        for name, row in tensors.items():
            size = row["local_size"]
            weight = weights[name]
            _expect(weight, {"local_numel": size}, f"auditoría local de {name}")
            finite = _count(weight["finite_grad_elements"], "gradientes finitos locales")
            nonzero = _count(weight["nonzero_grad_elements"], "gradientes no nulos locales")
            _require(nonzero <= finite <= size, f"conteos locales de gradiente incompatibles para {name}")
            before, after = (weight[key] for key in ("before_sha256", "after_sha256"))
            _require(_digest(before) and _digest(after) and (size > 0 or before == after),
                     f"hashes de actualización inválidos para {name}")
            claims = {"gradient_finite": finite == size, "gradient_nonzero": nonzero > 0,
                      "weight_changed": before != after}
            _expect(weight, {key: value for key, value in claims.items() if key in weight}, f"auditoría local de {name}")
            total = totals[name]
            total["numel"] += size
            total["finite_grad_elements"] += finite
            total["nonzero_grad_elements"] += nonzero
            total["changed_shards"] += int(before != after)
    expected_sizes = {name: _shape_size(shape) for name, shape in global_shapes.items()}
    _require(sum(expected_sizes.values()) == model["parameter_count"],
             "los shards no reconstruyen el tamaño de los tensores lógicos")
    for name, size in expected_sizes.items():
        total = totals[name]
        _expect(global_counts[name], total, f"auditoría global de {name}")
        _require(total["numel"] == total["finite_grad_elements"] == size
                 and total["nonzero_grad_elements"] > 0 and total["changed_shards"] > 0,
                 f"prueba de gradiente o actualización insuficiente para {name}")
    return {"audited_step": config["measurement"]["pilot_steps"], "tensor_count": len(expected_sizes),
            "parameter_count": sum(expected_sizes.values()), "tensor_shapes_sha256": model["tensor_shapes_sha256"]}


def _loss_comparison(original, resumed, measurement):
    result = []
    for reference, restored in zip(original, resumed):
        delta = abs(restored["loss"] - reference["loss"])
        tolerance = measurement["loss_atol"] + measurement["loss_rtol"] * abs(reference["loss"])
        fields = ("step", "samples", "processed_tokens", "supervised_tokens")
        _expect(restored, {key: reference[key] for key in fields}, "restauración")
        _require(delta <= tolerance, f"pérdida tras restaurar fuera de tolerancia en paso {restored['step']}")
        result.append({"step": restored["step"], "reference_loss": reference["loss"],
                       "restored_loss": restored["loss"], "absolute_delta": delta, "tolerance": tolerance})
    return result


# Reconstrucción de corridas y campañas


def _assess(root, job, config, run, *, cluster=None, preflight=None):
    campaign_id = _validate_manifest(root, run, job, config)
    if run["status"] != "running":
        manifest.verify_artifacts(root, run)
    hashes = _input_hashes(root, run)
    hardware, model_records = _validate_workload(root, hashes, config, preflight)
    archived_cluster = _object(root, "raw/inputs/cluster.json")
    cluster_inventory.validate_config(archived_cluster)
    if cluster is not None:
        _require(_same(archived_cluster, cluster), "inventario archivado incompatible")
    result = {"outcome": "indeterminate", "eligible": False, "issues": [], "input_hashes": hashes,
              "window_seconds": None, "step_seconds": [], "pilot_step_seconds": [],
              "oom_evidence": [], "removal_confirmed": False}
    try:
        pilot = job["block"] == 0
        stages = ("pilot", "resume") if pilot else ("measure",)
        oom = _oom_evidence(root, job, campaign_id, stages, hardware, result["issues"])
        result["oom_evidence"] = oom
        if result["issues"]:
            return result
        _require(not oom or len({row["phase"] for row in oom}) == 1,
                 "evidencia OOM en más de una fase detenida")
        completed = stages[: stages.index(oom[0]["phase"]) + 1] if oom else stages
        result["removal_seconds"] = _cleanup(root, job, campaign_id, completed)
        result["removal_confirmed"] = True
        result["participants"] = _groups_observed(root, run, job, campaign_id, completed, archived_cluster, hardware)
        setup = _workspace(root, job, campaign_id, config, hashes)
        controls = {phase: _phase_control(root, job, config, campaign_id, phase,
                                          {row["node_id"] for row in oom if row["phase"] == phase})
                    for phase in completed}
        if oom:
            _require(run["status"] != "complete", "OOM contradice el estado completo")
            if run["status"] == "failed":
                _require(bool(run["errors"]) and all(row.get("type") == "cuda_oom" for row in run["errors"]),
                         "OOM contradice los errores del manifiesto")
            result["outcome"] = "oom"
            return result
        measurement = config["measurement"]
        if pilot:
            containers, launch_duration = controls["pilot"]
            _require(set(containers.values()).isdisjoint(controls["resume"][0].values()),
                     "restauración sin contenedores nuevos")
            steps = {"pilot": range(1, measurement["pilot_steps"] + 1),
                     "resume": range(measurement["checkpoint_step"] + 1, measurement["pilot_steps"] + 1)}
        else:
            steps = {"measure": range(1, measurement["warmup_steps"] + measurement["measured_steps"] + 1)}
        phases = {phase: _phase(root, job, config, campaign_id, phase, steps[phase],
                                hashes["requested_config"], model_records) for phase in stages}
        _check_receipt_rosters(phases)
        summaries = {phase: _summarize_steps(ranks) for phase, ranks in phases.items()}
        result["telemetry"] = {phase: {rank["identity"]["node_id"]: rank["telemetry"] for rank in ranks}
                               for phase, ranks in phases.items()}
        result["peak_allocated_bytes"] = {
            phase: {rank["identity"]["node_id"]: rank["receipt"]["peak_allocated_bytes"] for rank in ranks}
            for phase, ranks in phases.items()
        }
        result["phase_steps"] = summaries
        _require(run["status"] != "failed", "manifiesto cerrado con fallo")
        if pilot:
            result["parameter_audit"] = _pilot_proof(phases, config)
            result["restoration"] = _loss_comparison(summaries["pilot"][measurement["checkpoint_step"]:],
                                                      summaries["resume"], measurement)
            worker_phase = max(rank["receipt"]["phase_seconds"] for rank in phases["pilot"])
            startup_components = {
                "setup_before_launch_seconds": setup,
                "launcher_overhead_seconds": max(0.0, launch_duration - worker_phase),
                "worker_startup_seconds": max(rank["receipt"]["startup_seconds"] for rank in phases["pilot"]),
            }
            result.update(startup_seconds=sum(startup_components.values()), startup_components=startup_components,
                          launch_to_terminal_seconds=launch_duration, longest_worker_phase_seconds=worker_phase)
            result["pilot_step_seconds"] = [row["seconds"] for row in summaries["pilot"]]
        else:
            measured = summaries["measure"][measurement["warmup_steps"]:]
            window = _phase_window(phases["measure"], {row["step"] for row in measured})
            samples, processed, supervised = (sum(row[field] for row in measured)
                                               for field in ("samples", "processed_tokens", "supervised_tokens"))
            result.update(window_seconds=window, step_seconds=[row["seconds"] for row in measured],
                          measured_steps=measured, samples=samples, processed_tokens=processed,
                          supervised_tokens=supervised, samples_per_second=samples / window,
                          processed_tokens_per_second=processed / window,
                          supervised_tokens_per_second=supervised / window)
            result["step_seconds_summary"] = _spread(result["step_seconds"])
        result["outcome"] = "qualified" if pilot else "measured"
        result["eligible"] = True
    except _EVIDENCE_ERRORS as exc:
        result["issues"].append(str(exc))
        if run["status"] == "complete":
            result["outcome"] = "indeterminate"
    return result


def assess_run(run_dir, job, config):
    """Valida una corrida abierta para su cierre o vuelve a verificar una cerrada."""
    directory = Path(run_dir).resolve()
    sft_protocol.validate_config(config)
    return _assess(directory, job, config, manifest.load_run(directory))


def _campaign_close(directory):
    markers = [name for name in ("COMPLETE", "PARTIAL") if (directory / name).exists()]
    _require(len(markers) == 1, "se requiere un marcador de cierre de campaña")
    status = markers[0].lower()
    marker_bytes = _file(directory, markers[0]).read_bytes()
    closed = _json_object(json.loads(marker_bytes))
    publication_bytes = _file(directory, "publication.json").read_bytes()
    publication = _json_object(json.loads(publication_bytes))
    _require(set(publication) == {"campaign_id", "status", "marker_sha256", "observed_at", "duration_seconds"},
             "campos inválidos en publication.json")
    _expect(publication, {"campaign_id": directory.name, "status": status,
                          "marker_sha256": hashlib.sha256(marker_bytes).hexdigest()}, "publication.json")
    expected = {f"{name}_sha256" for name in _CAMPAIGN_FILES} | {"cleanup_sha256", "sealed_at", "duration_seconds"}
    _require(set(closed) <= expected, "índice de cierre inválido")
    documents = {}
    for name in _CAMPAIGN_FILES:
        path = _file(directory, f"{name}.json")
        _require(closed.get(f"{name}_sha256") == manifest.sha256_file(path),
                 f"SHA-256 del cierre incompatible para {name}.json")
        documents[name] = _read(path)
    return status, closed, documents, publication, hashlib.sha256(publication_bytes).hexdigest()


def _ordered_groups(value):
    _require(isinstance(value, list) and value == [group for group in _GROUPS if group in value],
             "lista de topologías calificadas inválida")
    return value


def _timing(campaign, config, seal, publication):
    stage = campaign["stage"]
    started, ended = (_timestamp(campaign[field]) for field in ("started_at", "ended_at"))
    elapsed = _number(campaign["duration_seconds"], "duración de campaña")
    limit = _number(campaign["limit_seconds"], "límite de campaña", positive=True)
    _require(started <= ended, "intervalo de campaña invertido")
    _require(config["measurement"]["cleanup_reserve_seconds"] < limit
             <= config["measurement"][f"{stage}_limit_seconds"], "límite de campaña fuera del protocolo")
    sealed_at = _timestamp(seal["sealed_at"])
    observed_at = _timestamp(publication["observed_at"])
    _require(all(instant.utcoffset().total_seconds() == 0 for instant in (started, ended, sealed_at, observed_at)),
             "las marcas de índice, sello y publicación deben estar en UTC")
    sealed_elapsed = _number(seal["duration_seconds"], "duración hasta el sello")
    wall_index = (ended - started).total_seconds()
    wall_seal = (sealed_at - started).total_seconds()
    _require(abs(wall_index - elapsed) <= 2, "duración UTC y monotónica distintas al escribir el índice")
    _require(ended <= sealed_at and elapsed <= sealed_elapsed and
             abs(wall_seal - sealed_elapsed) <= 2 and abs((wall_seal - wall_index) - (sealed_elapsed - elapsed)) <= 2,
             "intervalo del índice y duración UTC y monotónica incompatibles con el sello")
    _require(max(wall_seal, sealed_elapsed) <= limit, "duración hasta el sello superior al límite")
    observed_elapsed = _number(publication["duration_seconds"], "duración hasta observar la publicación")
    wall_publication = (observed_at - started).total_seconds()
    _require(sealed_at <= observed_at and sealed_elapsed <= observed_elapsed,
             "la observación de publicación precede a la muestra del sello")
    _require(abs(wall_publication - observed_elapsed) <= 2
             and abs((observed_at - sealed_at).total_seconds() - (observed_elapsed - sealed_elapsed)) <= 2,
             "relojes UTC y monotónico incompatibles al observar la publicación")
    _require(max(wall_publication, observed_elapsed) <= limit, "duración hasta la publicación superior al límite")


def _campaign_cleanup(directory, seal, campaign, preflight, nodes):
    path = _file(directory, "cleanup.json")
    _require(seal.get("cleanup_sha256") == manifest.sha256_file(path),
             "SHA-256 del cierre incompatible para cleanup.json")
    cleanup = _read(path)
    _expect(cleanup, {"removal_confirmed": campaign["removal_confirmed"]}, "retirada de campaña")
    if cleanup.get("remote_started") is False:
        _expect(cleanup, {"nodes": {}, "removal_confirmed": True}, "cierre sin operación remota")
        _require(not campaign["runs"] and not preflight.get("nodes"), "cierre sin operación con participantes")
    elif campaign["removal_confirmed"]:
        _removal(cleanup, nodes)
    else:
        _require(set(cleanup["nodes"]) == set(nodes) and
                 any(row.get("confirmed") is False for row in cleanup["nodes"].values()),
                 "retirada no confirmada sin evidencia por nodo")
    _require(_timestamp(campaign["started_at"]) <= _timestamp(cleanup["ended_at"])
             <= _timestamp(campaign["ended_at"]), "retirada de campaña fuera del intervalo del índice")


def _reference_path(directory, category, ident):
    manifest._validate_component(ident, "identificador")
    _require(ident not in (".", ".."), "identificador no canónico")
    reference = directory.parent.parent / category / ident
    _require(reference == reference.resolve(), "referencia de archivo no canónica")
    return reference


def _run_reference(directory, ref, job, campaign, config, cluster, preflight):
    expected = {"job": job["name"], "group": job["group"], "block": job["block"]}
    _expect(ref, expected, "referencia de corrida")
    _require(ref["status"] in ("complete", "failed"), "referencia de corrida sin cierre")
    duration = _number(ref["duration_seconds"], "duración de corrida", positive=True)
    run_dir = _reference_path(directory, "full-sft", ref["run_id"])
    _require(manifest.sha256_file(_file(run_dir, "manifest.json")) == ref["manifest_sha256"],
             "SHA-256 del manifiesto de la corrida incompatible")
    run = manifest.load_run(run_dir)
    _expect(run, {"status": ref["status"]}, "manifiesto")
    _expect(run["git"], {"commit": campaign["git"]["commit"], "dirty": False}, "manifiesto Git")
    started, ended = (_timestamp(run[key]) for key in ("started_at", "ended_at"))
    _expect(run["requested_config"], {"campaign_id": campaign["campaign_id"]}, "solicitud")
    result = _assess(run_dir, job, config, run, cluster=cluster, preflight=preflight)
    result["manifest_errors"] = run["errors"]
    _require(result["input_hashes"] == campaign["input_hashes"], "entradas de corrida distintas del cierre")
    if result["outcome"] != ref["outcome"]:
        result["issues"].append("el resultado reconstruido contradice el índice de campaña")
        result.update(indexed_outcome=ref["outcome"], observed_outcome=result["outcome"],
                      outcome="indeterminate", eligible=False)
    try:
        _require(_timestamp(campaign["started_at"]) <= started <= ended <= _timestamp(campaign["ended_at"]),
                 "corrida fuera del intervalo de campaña")
    except _EVIDENCE_ERRORS as exc:
        result["issues"].append(str(exc))
        result.setdefault("observed_outcome", result["outcome"])
        result.update(outcome="indeterminate", eligible=False)
    return {**result, **expected, "duration_seconds": duration, "run_id": ref["run_id"],
            "manifest_sha256": ref["manifest_sha256"], "status": ref["status"],
            "started_at": run["started_at"], "ended_at": run["ended_at"]}


def _inspect_runs(directory, campaign, jobs, config, cluster, preflight):
    planned = {job["name"]: job for job in jobs}
    refs = campaign["runs"]
    _require(isinstance(refs, list), "índice de corridas inválido")
    names = [ref["job"] for ref in refs]
    ids = [ref["run_id"] for ref in refs]
    _require(len(ids) == len(set(ids)) and len(names) == len(set(names)), "referencia duplicada en el cierre")
    _require(all(name in planned for name in names) and names == [name for name in planned if name in names],
             "referencias fuera del orden o del plan")
    runs = []
    for ref in refs:
        job = planned[ref["job"]]
        try:
            result = _run_reference(directory, ref, job, campaign, config, cluster, preflight)
        except _EVIDENCE_ERRORS as exc:
            result = {"job": job["name"], "group": job["group"], "block": job["block"],
                      "run_id": ref["run_id"], "outcome": "indeterminate", "eligible": False,
                      "issues": [str(exc)], "indexed_outcome": ref.get("outcome")}
        runs.append(result)
    return runs


def _pilot_report(directory, reference, matrix_commit):
    reference = _json_object(reference)
    pilot_dir = _reference_path(directory, "sft-campaign", reference["campaign_id"])
    evidence = _campaign_close(pilot_dir)
    _, seal, documents, _, _ = evidence
    _require(documents["campaign"]["stage"] == "pilot", "referencia de calificación fuera de etapa")
    _require(seal["campaign_sha256"] == reference["campaign_sha256"], "SHA-256 del piloto incompatible")
    pilot = _build(pilot_dir, *evidence)
    _expect(reference, {"git_commit": pilot["git"]["commit"],
                        "estimated_matrix_seconds": pilot["estimated_matrix_seconds"]}, "referencia al piloto")
    sft_protocol.qualified_pilot_groups(pilot)
    published = _json_object(reference["published_report"])
    path = Path(published["path"])
    _require(bool(path.name) and not path.is_absolute() and ".." not in path.parts
             and path.as_posix() == published["path"], "ruta del derivado publicado del piloto inválida")
    _expect(published, {"git_commit": matrix_commit}, "commit del derivado publicado del piloto")
    archive = _file(directory, "published-pilot.json")
    _expect(published, {"sha256": manifest.sha256_file(archive)}, "SHA-256 del derivado publicado del piloto")
    _require(_same(_read(archive), pilot), "el derivado publicado difiere del piloto reconstruido")
    return pilot


def _campaign_inputs(directory, hashes, campaign_hashes, config):
    _require(_REQUIRED_INPUTS <= hashes.keys(),
             f"fuentes de campaña ausentes: {sorted(_REQUIRED_INPUTS - hashes.keys())}")
    _expect(hashes, {"requested_config": campaign_hashes["config_sha256"],
                     "cluster.json": campaign_hashes["cluster_sha256"],
                     f"code/{config['cluster_config']}": campaign_hashes["cluster_sha256"],
                     "corpus.jsonl": config["workload"]["corpus_sha256"],
                     "reference/preflight.json": config["reference_preflight_sha256"]}, "entradas de campaña")
    for name, digest in hashes.items():
        if name == "requested_config":
            continue
        _require(manifest.sha256_file(_file(directory, f"inputs/{name}")) == digest,
                 f"integridad de entrada de campaña inválida: {name}")


def _build(directory, status, seal, documents, publication, publication_sha256):
    campaign, config, cluster, preflight, plan = (documents[name] for name in
                                                  ("campaign", "config", "cluster", "preflight", "plan"))
    _expect(campaign, {"campaign_id": directory.name, "status": status}, "campaña")
    stage = campaign["stage"]
    _require(stage in ("pilot", "run"), "etapa inválida")
    sft_protocol.validate_config(config)
    # Reproducir el plan de trabajo incluso cuando el archivo se haya trasladado.
    qualified = _ordered_groups(campaign["qualified_groups"])
    _require(isinstance(campaign["errors"], list) and type(campaign["removal_confirmed"]) is bool,
             "errores o retirada de campaña inválidos")
    _json_object(campaign["git"])
    _require(campaign["git"].get("dirty") is False and isinstance(campaign["git"].get("commit"), str),
             "campaña fuera de política Git")
    input_hashes = _json_object(campaign["input_hashes"])
    _require(all(_digest(digest) for digest in input_hashes.values()), "hash de entrada inválido")
    _campaign_inputs(directory, input_hashes, seal, config)
    closure_issues = []
    try:
        _timing(campaign, config, seal, publication)
    except _EVIDENCE_ERRORS as exc:
        closure_issues.append(f"presupuesto de campaña no acreditado: {exc}")
    removal_valid = False
    try:
        _campaign_cleanup(directory, seal, campaign, preflight, cluster["groups"]["four"])
        removal_valid = campaign["removal_confirmed"]
    except _EVIDENCE_ERRORS as exc:
        closure_issues.append(str(exc))
    pilot = None
    if stage == "pilot":
        _require(campaign["pilot_reference"] is None, "piloto con referencia a otro piloto")
        expected_plan = sft_protocol.make_plan(config, stage)
    else:
        expected_plan = sft_protocol.make_plan(config, stage, qualified)
        try:
            pilot = _pilot_report(directory, campaign["pilot_reference"], campaign["git"]["commit"])
            _require(qualified == pilot["qualified_groups"] and pilot["config_sha256"] == seal["config_sha256"]
                     and pilot["cluster_sha256"] == seal["cluster_sha256"]
                     and pilot["input_hashes"] == input_hashes,
                     "calificación del piloto incompatible con esta campaña")
            _require(pilot["estimated_matrix_seconds"] <= campaign["limit_seconds"]
                     - config["measurement"]["cleanup_reserve_seconds"],
                     "proyección del piloto fuera del plazo de la matriz")
        except _EVIDENCE_ERRORS as exc:
            closure_issues.append(str(exc))
            pilot = None
    _expect(plan, expected_plan, "plan")
    cluster_inventory.validate_config(cluster)
    jobs = plan["jobs"]
    runs = _inspect_runs(directory, campaign, jobs, config, cluster, preflight)
    environment = None
    if preflight.get("status") == "ready" or status == "complete":
        try:
            environment = _preflight_environment(preflight, config)
        except _EVIDENCE_ERRORS as exc:
            closure_issues.append(str(exc))
    if pilot and environment is not None and not _same(pilot["environment"], environment):
        closure_issues.append("entorno del piloto diferente de la matriz")
    if closure_issues:
        for run in runs:
            if run["eligible"] or run["outcome"] == "oom":
                run.update(observed_outcome=run["outcome"], outcome="indeterminate", eligible=False)
                run["issues"].append("el cierre de campaña no permite acreditar este resultado")
    errors = [*campaign["errors"], *closure_issues]
    if not campaign["removal_confirmed"]:
        errors.append("retirada final de recursos sin confirmar")
    if stage == "pilot":
        reconstructed = [run["group"] for run in runs if run["eligible"]]
        if qualified != reconstructed:
            errors.append("las topologías calificadas del cierre difieren de la evidencia verificable")
        qualified = reconstructed
        estimate, projection = _pilot_estimate(runs)
    else:
        if closure_issues:
            qualified = []
        estimate, projection = (pilot["estimated_matrix_seconds"], pilot["matrix_estimate_groups"]) if pilot else (None, {})
    failures = [run for run in runs if not run["eligible"]]
    unexpected = [run for run in failures if stage != "pilot" or run["outcome"] != "oom"]
    if status == "complete":
        _require(not closure_issues, "marcador COMPLETE inválido: " + "; ".join(closure_issues))
        _require(len(runs) == len(jobs) and not errors and not unexpected,
                 "marcador COMPLETE incompatible con la evidencia reconstruida")
    return {
        "campaign_id": campaign["campaign_id"], "stage": stage, "status": status,
        "git": campaign["git"], "input_hashes": input_hashes,
        **{field: digest for field, digest in seal.items() if field.endswith("_sha256")},
        "publication_sha256": publication_sha256,
        "environment": environment,
        "started_at": campaign["started_at"], "ended_at": publication["observed_at"],
        "duration_seconds": publication["duration_seconds"], "limit_seconds": campaign.get("limit_seconds"),
        "index_ended_at": campaign["ended_at"], "index_duration_seconds": campaign.get("duration_seconds"),
        "sealed_at": seal.get("sealed_at"), "sealed_duration_seconds": seal.get("duration_seconds"),
        "removal_confirmed": removal_valid, "pilot_reference": campaign["pilot_reference"],
        "qualified_groups": qualified, "estimated_matrix_seconds": estimate, "matrix_estimate_groups": projection,
        "runs": runs, "errors": errors, "failures": failures,
        "pending_jobs": [job["name"] for job in jobs if all(run["job"] != job["name"] for run in runs)],
        "comparisons": _comparisons(runs, jobs) if stage == "run" else {},
        "drift": _drift(runs) if stage == "run" else {},
        "capacity": _capacity(runs),
        "pilot_capacity": pilot["capacity"] if pilot else {},
    }


def build_report(campaign_dir):
    """Reconstruye un informe a partir de un cierre local y sus corridas hermanas."""
    directory = Path(campaign_dir).resolve()
    _require(directory.parent.name == "sft-campaign", "directorio fuera del archivo de campañas SFT")
    return _build(directory, *_campaign_close(directory))


# Cálculo y publicación de derivados


def _comparisons(runs, planned):
    indexed = {(run["block"], run["group"]): run for run in runs}
    expected = {(job["block"], job["group"]) for job in planned}
    comparisons = {}
    for larger, smaller, ratio in _PAIRS:
        name = f"{_PAIR_NAMES[larger]}/{_PAIR_NAMES[smaller]}"
        pairs = []
        exclusions = []
        eligible = 0
        for block in range(1, 4):
            keys = [(block, group) for group in (larger, smaller)]
            eligible += all(key in expected for key in keys)
            candidates = [indexed.get(key) for key in keys]
            reasons = {}
            for (_, group), candidate in zip(keys, candidates):
                if (block, group) not in expected:
                    reasons[group] = "topología sin calificar para el plan"
                elif candidate is None:
                    reasons[group] = "corrida planificada sin evidencia"
                elif not candidate["eligible"]:
                    reasons[group] = candidate["outcome"]
            if reasons:
                exclusions.append({"block": block, "reasons": reasons})
                continue
            fast, slow = candidates
            work_fields = ("step", "samples", "processed_tokens", "supervised_tokens")
            if [tuple(step[key] for key in work_fields) for step in fast["measured_steps"]] != [
                    tuple(step[key] for key in work_fields) for step in slow["measured_steps"]]:
                exclusions.append({"block": block, "reasons": {"pair": "trabajo global distinto"}})
                continue
            speedup = slow["window_seconds"] / fast["window_seconds"]
            pairs.append({"block": block, "numerator_run_id": slow["run_id"],
                          "denominator_run_id": fast["run_id"], "speedup": speedup,
                          "efficiency": speedup / ratio})
        comparisons[name] = {
            "pairs": pairs, "exclusions": exclusions,
            "coverage": {"paired_blocks": len(pairs), "planned_blocks": eligible, "design_blocks": 3},
            "speedup": _spread([row["speedup"] for row in pairs]),
            "efficiency": _spread([row["efficiency"] for row in pairs]),
        }
    return comparisons


def _drift(runs):
    drift = {}
    for group in _GROUPS:
        trend = [{"block": row["block"], "window_seconds": row["window_seconds"],
                  "started_at": row["started_at"], "ended_at": row["ended_at"],
                  "telemetry": row["telemetry"]} for row in runs
                 if row["group"] == group and row["eligible"]]
        windows = [row["window_seconds"] for row in trend]
        drift[group] = {
            "blocks": trend,
            "window_seconds": _spread(windows),
            "first_to_last_relative_change": windows[-1] / windows[0] - 1 if len(windows) > 1 else None,
        }
    return drift


def _capacity(runs):
    return {
        group: {outcome: [run["run_id"] for run in runs if run["group"] == group and run["outcome"] == outcome]
                for outcome in ("qualified", "measured", "oom", "indeterminate")}
        for group in _GROUPS
    }


def _pilot_estimate(runs):
    groups = {}
    for run in runs:
        if not run["eligible"]:
            continue
        steps = run["pilot_step_seconds"]
        times = {
            "startup_seconds": run["startup_seconds"],
            "first_step_seconds": steps[0],
            "projected_remaining_step_seconds": 24 * max(steps[1:]),
            "removal_seconds": run["removal_seconds"],
        }
        groups[run["group"]] = {**times, "startup_components": run["startup_components"],
                                "projected_three_blocks_seconds": 3 * 1.25 * sum(times.values())}
    estimate = sum(group["projected_three_blocks_seconds"] for group in groups.values()) if groups else None
    return estimate, groups


def _display(value):
    return "—" if value is None else f"{value:.4g}"


def _pilot_markdown(report):
    lines = ["## Calificación y restauración", "",
             "| Trabajo | Corrida | Tensores con gradiente finito no nulo y cambio | Parámetros auditados | Δ absoluto de pérdida paso 3 | Δ absoluto de pérdida paso 4 |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for run in report["runs"]:
        if run["outcome"] != "qualified":
            continue
        audit = run["parameter_audit"]
        loss = run["restoration"]
        lines.append(f"| {run['job']} | {run['run_id']} | {audit['tensor_count']} | {audit['parameter_count']} | "
                     f"{_display(loss[0]['absolute_delta'])} | {_display(loss[1]['absolute_delta'])} |")
    if report["matrix_estimate_groups"]:
        lines.extend(["", "## Proyección por topología", "",
                      "El arranque reúne la preparación, el tiempo del lanzador fuera de la fase del worker "
                      "y el arranque interno hasta el primer paso. Cada bloque añade el primer paso, "
                      "24 veces el máximo de los pasos 2 a 4, la retirada y un margen del 25 %. "
                      "El checkpoint y la auditoría se cronometran fuera de las actualizaciones.", "",
                      "| Topología | Arranque total (s) | Primer paso (s) | 24 pasos (s) | Retirada (s) | Tres bloques con margen (s) |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for group, timings in report["matrix_estimate_groups"].items():
            fields = ("startup_seconds", "first_step_seconds", "projected_remaining_step_seconds",
                      "removal_seconds", "projected_three_blocks_seconds")
            values = [_display(timings[field]) for field in fields]
            lines.append(f"| {group} | {' | '.join(values)} |")
    return lines


def _matrix_markdown(report):
    lines = ["## Corridas medidas", "",
             "| Trabajo | Corrida | Ventana (s) | s/actualización (mediana) | Ejemplos/s | Tokens procesados/s | Tokens supervisados/s | Pérdida final | Pico CUDA asignado por nodo (GiB) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for run in report["runs"]:
        if run["outcome"] != "measured":
            continue
        peaks = ", ".join(f"{node} {_display(used / 1024**3)}"
                          for node, used in run["peak_allocated_bytes"]["measure"].items())
        values = [run["window_seconds"], run["step_seconds_summary"]["median"], run["samples_per_second"],
                  run["processed_tokens_per_second"], run["supervised_tokens_per_second"],
                  run["measured_steps"][-1]["loss"]]
        lines.append(f"| {run['job']} | {run['run_id']} | "
                     f"{' | '.join(_display(value) for value in values)} | {peaks} |")
    return lines


def _markdown(report):
    lines = [
        f"# SFT completo · {report['campaign_id']}", "",
        f"Etapa `{report['stage']}` · Cierre `{report['status']}`", "",
        f"Topologías calificadas · {', '.join(report['qualified_groups']) or 'ninguna'}", "",
        f"Presupuesto estimado para tres bloques · {_display(report['estimated_matrix_seconds'])} s", "",
    ]
    lines.extend(_pilot_markdown(report) if report["stage"] == "pilot" else _matrix_markdown(report))
    if report["comparisons"]:
        lines.extend(["", "## Escalamiento fuerte", "",
                      "El speedup divide la ventana de la topología menor por la de la mayor dentro del mismo bloque.",
                      "", "| Par | Bloques válidos / planificados / diseño | Speedup mediano [rango] | Eficiencia mediana [rango] |",
                      "| --- | --- | --- | --- |"])
        for name, comparison in report["comparisons"].items():
            coverage = comparison["coverage"]
            values = []
            for field in ("speedup", "efficiency"):
                summary = comparison[field]
                bounds = summary["range"]
                values.append(f"{_display(summary['median'])} [{_display(bounds[0])}, {_display(bounds[1])}]"
                              if bounds else "—")
            lines.append(f"| {name} | {coverage['paired_blocks']} / {coverage['planned_blocks']} / "
                         f"{coverage['design_blocks']} | {' | '.join(values)} |")
        exclusions = []
        for name, comparison in report["comparisons"].items():
            for row in comparison["exclusions"]:
                reasons = ", ".join(f"`{key}` · {value}" for key, value in row["reasons"].items())
                exclusions.append(f"- {name}, bloque {row['block']}. {reasons}.")
        if exclusions:
            lines.extend(["", "### Exclusiones", "", *exclusions])
        lines.extend(["", "### Deriva de la ventana por topología", "",
                      "El cambio relativo usa la primera y la última ventana verificadas. El JSON conserva la telemetría por nodo.", ""])
        for group, row in report["drift"].items():
            change = row["first_to_last_relative_change"]
            lines.append(f"- `{group}` · {_display(100 * change) + ' %' if change is not None else '—'}.")
        unavailable = [group for group, outcomes in report["pilot_capacity"].items() if outcomes["oom"]]
        if unavailable:
            lines.extend(["", f"OOM de CUDA verificado en el piloto para {', '.join(unavailable)}. "
                          "La frontera corresponde al modelo y la receta de esta campaña."])
    if report["errors"] or report["failures"]:
        lines.extend(["", "## Fallos y observaciones del cierre", ""])
        for error in report["errors"]:
            lines.append(f"- {error}")
        for run in report["failures"]:
            lines.append(f"- `{run['job']}` (`{run['run_id']}`) · {run['outcome']}.")
            lines.extend(f"  - {item['node_id']}, fase `{item['phase']}`, etapa `{item['stage']}`."
                         for item in run.get("oom_evidence", []))
            lines.extend(f"  - {issue}" for issue in run["issues"])
    return "\n".join(lines) + "\n"


def write_report(path, report):
    """Escribe ambos derivados de manera exclusiva y revierte una escritura incompleta."""
    base = Path(path)
    if base.suffix in (".json", ".md"):
        base = base.with_suffix("")
    destinations = [base.with_name(base.name + suffix) for suffix in (".json", ".md")]
    for destination in destinations:
        resolved = destination.resolve()
        protected = (REPO_ROOT / "resultados/raw", REPO_ROOT / "datos")
        inside_run = any(
            ((parent / "manifest.json").is_file() and (parent / "raw/context.json").is_file())
            or (parent / "campaign.json").is_file() and parent.parent.name == "sft-campaign"
            for parent in resolved.parents
        )
        _require(not inside_run and not any(resolved.is_relative_to(root.resolve()) for root in protected),
                 "los derivados se escriben fuera de raw y datos")
    payloads = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", _markdown(report))
    destinations[0].parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with ExitStack() as stack:
            streams = []
            for destination in destinations:
                stream = stack.enter_context(destination.open("x", encoding="utf-8"))
                created.append(destination)
                streams.append(stream)
            for stream, payload in zip(streams, payloads):
                stream.write(payload)
    except BaseException:
        for destination in created:
            destination.unlink()
        raise
