"""Worker CUDA de SFT completo; se lanza con torchrun --module.

Importar o consultar --help no importa torch/transformers ni toca GPU. El
launcher es responsable de la autorización operacional y de cerrar la campaña;
este worker solo produce evidencia local nueva, nunca un marcador COMPLETE.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import errno
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import socket
import threading
import time
import uuid

from infraestructura import inference_observation as observation
from infraestructura import sft_checkpoint as checkpoint
from infraestructura.manifest import sha256_file


TENSOR_COUNT = 291
PARAMETER_COUNT = 8030261248
HASH_CHUNK_ELEMENTS = 262144  # Un MiB FP32, sin clonar el modelo en GPU.
DETERMINISM = {"algorithms": True, "warn_only": False, "cudnn_benchmark": False,
               "cudnn_deterministic": True, "cublas_workspace_config": ":4096:8"}


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def framework_versions(config, torch, transformers):
    """Coteja por separado la distribución instalada y el runtime importado."""
    versions = {"torch_version": importlib.metadata.version("torch"),
                "torch_runtime_version": str(torch.__version__),
                "transformers_version": str(transformers.__version__)}
    for key, observed in versions.items():
        if observed != config["image"][key]:
            raise ValueError(f"{key} diferente del entorno aprobado: {observed}")
    return versions


def validate_determinism_environment(config):
    """El launcher fija cuBLAS antes de importar torch; v1 conserva su semántica."""
    if "determinism" not in config["training"]:
        return None
    policy = config["training"]["determinism"]
    if (not isinstance(policy, dict) or set(policy) != set(DETERMINISM)
            or any(type(policy[key]) is not type(value) or policy[key] != value
                   for key, value in DETERMINISM.items())):
        raise ValueError("se requieren los cinco controles deterministas de la receta v2")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != policy["cublas_workspace_config"]:
        raise ValueError("el launcher debe fijar CUBLAS_WORKSPACE_CONFIG=:4096:8 antes de torchrun")
    return policy


def configure_determinism(config, torch):
    policy = validate_determinism_environment(config)
    if policy is None:
        return None
    torch.use_deterministic_algorithms(policy["algorithms"], warn_only=policy["warn_only"])
    torch.backends.cudnn.benchmark = policy["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = policy["cudnn_deterministic"]
    observed = {"algorithms": torch.are_deterministic_algorithms_enabled(),
                "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}
    if observed != policy:
        raise RuntimeError("los controles deterministas efectivos difieren de la receta")
    return observed


def verify_tensor_shapes(model, protocol, config):
    """Fija nombres y formas lógicas antes de que FSDP materialice los shards."""
    shapes = {name: list(parameter.shape) for name, parameter in model.named_parameters()}
    digest = protocol.tensor_shapes_sha256(shapes)
    if digest != config["model"]["tensor_shapes_sha256"]:
        raise ValueError("SHA-256 de formas de tensores lógicos incompatible")
    return digest


def resume_digest(args, identity, config):
    """Ancla obligatoria del checkpoint en el recibo original del piloto."""
    pilot = Path(args.out).parent / "pilot"
    receipt = json.loads((pilot / "receipt.json").read_text(encoding="utf-8"))
    expected = {key: identity[key] for key in checkpoint.IDENTITY_FIELDS}
    expected.update(phase="pilot", status="ok", config_sha256=sha256_file(Path(args.config)),
                    model_revision=config["model"]["revision"],
                    **{key: config["workload"][key] for key in ("corpus_sha256", "tokens_sha256")})
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("recibo original del piloto de otra identidad, fase o configuración")
    record = receipt["checkpoint"]
    digest = record["metadata_sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or record.get("restored") is not False or record.get("hashes_verified") is not True
            or record.get("step") != config["measurement"]["checkpoint_step"]
            or record.get("next_step") != config["measurement"]["checkpoint_step"] + 1):
        raise ValueError("prueba de checkpoint del piloto incompleta")
    if args.checkpoint_sha256 is not None and args.checkpoint_sha256 != digest:
        raise ValueError("--checkpoint-sha256 difiere del recibo original")
    for name in ("identity.json", "group.json", "steps.jsonl", "telemetry.jsonl"):
        artifact = receipt["artifacts"][name]
        path = pilot / name
        if artifact["sha256"] != sha256_file(path) or artifact["bytes"] != path.stat().st_size:
            raise ValueError("evidencia original del piloto modificada")
    return digest


def phase_steps(phase, config):
    measurement = config["measurement"]
    if phase == "pilot":
        return list(range(1, measurement["pilot_steps"] + 1))
    if phase == "resume":
        return list(range(measurement["checkpoint_step"] + 1,
                          measurement["pilot_steps"] + 1))
    if phase == "measure":
        return list(range(1, measurement["warmup_steps"] + measurement["measured_steps"] + 1))
    raise ValueError("fase de SFT desconocida")


def validate_work(records, protocol, config, world_size):
    """Comprueba los 100 ejemplos y la alineación de los 25 batches globales."""
    workload = config["workload"]
    if len(records) != workload["examples"] or len(records) != 100:
        raise ValueError("se requieren exactamente 100 ejemplos")
    if [row["id"] for row in records] != list(range(100)):
        raise ValueError("IDs de ejemplos no contiguos")
    if world_size not in (1, 2, 4):
        raise ValueError("world size fuera de 1/2/4")
    for row in records:
        inputs, labels = row["input_ids"], row["labels"]
        if len(inputs) != workload["sequence_length"] or len(labels) != len(inputs):
            raise ValueError("longitud de secuencia incompatible")
        if any(type(value) is not int or value < 0 for value in inputs):
            raise ValueError("input_ids inválidos")
        if any(label != -100 and label != token for token, label in zip(inputs, labels)):
            raise ValueError("labels incompatibles con input_ids")
        if sum(value != -100 for value in labels[1:]) != workload["supervised_tokens"]:
            raise ValueError("cantidad de tokens supervisados incompatible")
    observed = []
    for step in range(25):
        batch = []
        for rank in range(world_size):
            ids = protocol.step_ids(step, rank, world_size)
            if len(ids) != 4 // world_size or any(type(i) is not int or i not in range(100) for i in ids):
                raise ValueError("microbatches o IDs de rank incompatibles")
            batch.extend(ids)
        if len(set(batch)) != 4 or set(batch) != set(range(4 * step, 4 * step + 4)):
            raise ValueError("alineación global de datos incompatible")
        observed.extend(batch)
    if sorted(observed) != list(range(100)):
        raise ValueError("cobertura global de datos incompatible")


def _snapshot_index(directory, expected_names):
    directory = Path(directory)
    index_path = directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    mapping = index["weight_map"]
    if set(mapping) != set(expected_names):
        raise ValueError("cobertura del snapshot diferente del modelo lógico")
    shards = []
    for name in sorted(set(mapping.values())):
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("nombre de shard safetensors inválido")
        path = directory / name
        if not path.is_file():
            raise ValueError("shard safetensors local ausente")
        # Los symlinks de snapshots HF hacia blobs son válidos y de solo lectura.
        shards.append({"path": name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    metadata_names = ("config.json", "tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "tokenizer.model", "added_tokens.json")
    metadata = {name: sha256_file(directory / name) for name in metadata_names
                if (directory / name).is_file()}
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"}.issubset(metadata):
        raise ValueError("metadatos locales de modelo/tokenizer incompletos")
    return mapping, {"index_sha256": sha256_file(index_path), "shards": shards, "metadata": metadata}


def _load_snapshot(model, directory, mapping, *, torch, safe_open, rank, world_size):
    """Lee una porción local por tensor, sin materializar pesos globales."""
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for shard in sorted(set(mapping.values())):
            with safe_open(str(Path(directory) / shard), framework="pt", device="cpu") as handle:
                names = {name for name, file in mapping.items() if file == shard}
                if set(handle.keys()) != names:
                    raise ValueError("tensores adicionales o ausentes en shard safetensors")
                for name in sorted(names):
                    parameter = parameters[name]
                    local = checkpoint.local_tensor(parameter)
                    source = handle.get_slice(name)
                    shape = tuple(source.get_shape())
                    if shape != tuple(parameter.shape) or not shape:
                        raise ValueError("forma global del snapshot incompatible")
                    if hasattr(parameter, "placements"):
                        placements = parameter.placements
                        if len(placements) != 1 or not placements[0].is_shard(0):
                            raise ValueError("se requiere un mesh FSDP2 1D con shard en dimensión 0")
                        width = math.ceil(shape[0] / world_size)
                        start = min(rank * width, shape[0])
                        stop = min(start + width, shape[0])
                        piece = source[start:stop]
                    else:
                        if world_size != 1:
                            raise ValueError("parámetro distribuido sin DTensor")
                        piece = source[:]
                    if piece.shape != local.shape or local.dtype != torch.float32:
                        raise ValueError("forma local o master FP32 incompatible")
                    local.copy_(piece)
                    del piece


def _tensor_hash(tensor):
    """SHA-256 de bytes FP32 locales con copia CPU acotada a un MiB."""
    tensor = tensor.detach()
    if not tensor.is_contiguous():
        raise ValueError("el shard de auditoría debe ser contiguo")
    flattened = tensor.view(-1)
    digest = hashlib.sha256()
    for start in range(0, flattened.numel(), HASH_CHUNK_ELEMENTS):
        cpu = flattened[start:start + HASH_CHUNK_ELEMENTS].cpu()
        digest.update(cpu.numpy().tobytes())
        del cpu
    return digest.hexdigest()


def _rng_hashes(torch):
    """Lee los estados sin consumir muestras ni alterar los generadores."""
    return {"python_sha256": _canonical_hash(random.getstate()),
            "torch_cpu_sha256": _tensor_hash(torch.get_rng_state()),
            "torch_cuda_sha256": [_tensor_hash(state) for state in torch.cuda.get_rng_state_all()]}


def begin_recovery_diagnostic(model, torch, step):
    tensors = []
    for name, parameter in model.named_parameters():
        local = checkpoint.local_tensor(parameter)
        if local.dtype != torch.float32:
            raise ValueError("el diagnóstico de recuperación requiere maestros FP32")
        tensors.append({"name": name, "local_numel": local.numel(),
                        "weight_before_forward_sha256": _tensor_hash(local)})
    return {"method": "sha256-full-local-fp32-v1", "step": step,
            "hash_chunk_bytes": 4 * HASH_CHUNK_ELEMENTS, "tensor_count": len(tensors),
            "local_parameter_count": sum(row["local_numel"] for row in tensors),
            "rng_before_forward": _rng_hashes(torch), "tensors": tensors}


def _diagnostic_parameters(model, diagnostic):
    parameters = list(model.named_parameters())
    if [name for name, _ in parameters] != [row["name"] for row in diagnostic["tensors"]]:
        raise ValueError("cobertura de parámetros cambió durante el diagnóstico de recuperación")
    return zip(parameters, diagnostic["tensors"])


def recovery_gradient_diagnostic(model, torch, diagnostic):
    for (name, parameter), row in _diagnostic_parameters(model, diagnostic):
        grad = parameter.grad
        if grad is not None:
            grad = checkpoint.local_tensor(grad)
            if grad.dtype != torch.float32 or grad.numel() != row["local_numel"]:
                raise ValueError(f"gradiente local FP32 incompatible en diagnóstico: {name}")
        row["gradient_sha256"] = _tensor_hash(grad) if grad is not None else None
    diagnostic["rng_before_optimizer"] = _rng_hashes(torch)


def finish_recovery_diagnostic(model, torch, diagnostic):
    for (_, parameter), row in _diagnostic_parameters(model, diagnostic):
        row["weight_after_optimizer_sha256"] = _tensor_hash(checkpoint.local_tensor(parameter))
    diagnostic["rng_after_optimizer"] = _rng_hashes(torch)
    diagnostic["local_updated_tensor_count"] = sum(
        row["weight_before_forward_sha256"] != row["weight_after_optimizer_sha256"]
        for row in diagnostic["tensors"])


def compare_recovery_diagnostic(current, reference):
    """Comparación informativa; el umbral de loss del protocolo no cambia."""
    actual = {row["name"]: row for row in current["tensors"]}
    original = {row["name"]: row for row in reference["tensors"]}
    layout_equal = (current["method"] == reference["method"] and current["step"] == reference["step"]
                    and actual.keys() == original.keys()
                    and all(row["local_numel"] == original[name]["local_numel"] for name, row in actual.items()))
    result = {"reference_phase": "pilot", "layout_equal": layout_equal}
    for label, key in (("weights_before_forward_equal", "weight_before_forward_sha256"),
                       ("gradients_equal", "gradient_sha256"),
                       ("weights_after_optimizer_equal", "weight_after_optimizer_sha256")):
        result[label] = layout_equal and all(row[key] == original[name][key] for name, row in actual.items())
    for key in ("rng_before_forward", "rng_before_optimizer", "rng_after_optimizer"):
        result[key + "_equal"] = current[key] == reference[key]
    return result


def _audit_before(model, torch):
    rows = []
    for name, parameter in model.named_parameters():
        local = checkpoint.local_tensor(parameter)
        grad = parameter.grad
        if grad is None:
            raise ValueError(f"gradiente ausente: {name}")
        grad = checkpoint.local_tensor(grad).detach()
        if grad.dtype != torch.float32 or local.dtype != torch.float32:
            raise ValueError("master y gradiente deben ser FP32")
        flat = grad.view(-1)
        finite = torch.zeros((), device=grad.device, dtype=torch.int64)
        nonzero = torch.zeros_like(finite)
        for start in range(0, flat.numel(), HASH_CHUNK_ELEMENTS):
            piece = flat[start:start + HASH_CHUNK_ELEMENTS]
            finite += torch.isfinite(piece).sum()
            nonzero += torch.count_nonzero(piece)
        rows.append({"name": name, "numel": parameter.numel(), "local_numel": local.numel(),
                     "finite_grad_elements": int(finite.item()),
                     "nonzero_grad_elements": int(nonzero.item()),
                     "before_sha256": _tensor_hash(local)})
    return rows


def _audit_after(model, rows, *, torch, dist, device, step):
    values = []
    local_hashes = []
    for (name, parameter), row in zip(model.named_parameters(), rows):
        if name != row["name"]:
            raise ValueError("orden de tensores cambiado durante auditoría")
        after = _tensor_hash(checkpoint.local_tensor(parameter))
        values.append([row["finite_grad_elements"], row["nonzero_grad_elements"],
                       int(after != row["before_sha256"]), row["local_numel"]])
        local_hashes.append({"name": name, "before_sha256": row["before_sha256"],
                             "after_sha256": after, "local_numel": row["local_numel"],
                             "finite_grad_elements": row["finite_grad_elements"],
                             "nonzero_grad_elements": row["nonzero_grad_elements"],
                             "gradient_finite": row["finite_grad_elements"] == row["local_numel"],
                             "gradient_nonzero": row["nonzero_grad_elements"] > 0,
                             "weight_changed": after != row["before_sha256"]})
    totals = torch.tensor(values, dtype=torch.int64, device=device)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    tensors = []
    for row, (finite, nonzero, changed, owned) in zip(rows, totals.cpu().tolist()):
        if owned != row["numel"]:
            raise ValueError("cobertura de shards del tensor lógico incompleta")
        tensors.append({"name": row["name"], "numel": owned,
                        "finite_grad_elements": finite, "nonzero_grad_elements": nonzero,
                        "changed_shards": changed})
    return coverage_proof(tensors, local_hashes, step)


def coverage_proof(tensors, local_hashes, step):
    """Exige evidencia agregada de cada tensor lógico, incluidos shards vacíos."""
    if len({row["name"] for row in tensors}) != len(tensors):
        raise ValueError("tensor lógico duplicado en auditoría")
    proof = {"audit_performed": True, "audited_step": step, "tensor_count": len(tensors),
             "parameter_count": sum(row["numel"] for row in tensors),
             "gradient_finite_tensor_count": sum(row["finite_grad_elements"] == row["numel"]
                                                  for row in tensors),
             "gradient_nonzero_tensor_count": sum(row["nonzero_grad_elements"] > 0 for row in tensors),
             "updated_tensor_count": sum(row["changed_shards"] > 0 for row in tensors),
             "tensors": tensors, "local_weight_hashes": local_hashes,
             "method": "sha256-fp32-local-before-after-sgd", "hash_chunk_bytes": 4 * HASH_CHUNK_ELEMENTS}
    if (proof["tensor_count"] != TENSOR_COUNT or proof["parameter_count"] != PARAMETER_COUNT
            or any(proof[key] != TENSOR_COUNT for key in ("gradient_finite_tensor_count",
                                                         "gradient_nonzero_tensor_count",
                                                         "updated_tensor_count"))):
        raise ValueError("no se acreditó gradiente finito no nulo y actualización de todos los tensores")
    return proof


def cuda_memory(torch, device):
    free, total = torch.cuda.mem_get_info(device)
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "device_free_bytes": free, "device_total_bytes": total}


class Telemetry:
    """Muestreo de host en segundo plano; los fallos impiden emitir recibo OK."""

    def __init__(self, path, identity, memory, interval=1.0):
        self.path, self.identity, self.memory = Path(path), identity, memory
        self.interval = interval
        self.stage_lock = threading.Lock()
        self.transitions = [(0, "startup")]
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="sft-observation", daemon=True)
        self.error = None
        self.samples = 0

    def start(self):
        self.thread.start()

    @property
    def stage(self):
        with self.stage_lock:
            return self.transitions[-1][1]

    @stage.setter
    def stage(self, value):
        self.set_stage(value)

    def set_stage(self, value):
        """La transición y las marcas del recibo usan el mismo reloj local."""
        with self.stage_lock:
            instant = time.monotonic_ns()
            self.transitions.append((instant, value))
            return instant

    def stage_at(self, instant):
        with self.stage_lock:
            return next(stage for start, stage in reversed(self.transitions) if start <= instant)

    def _run(self):
        try:
            with self.path.open("x", encoding="utf-8") as output:
                while True:
                    sample = observation.collect_sample()
                    if sample.get("hostname") != self.identity["hostname"]:
                        raise ValueError("hostname de telemetría diferente de la identidad efectiva")
                    sample.update(self.identity)
                    sample.update(training_stage=self.stage_at(sample["monotonic_ns"]),
                                  cuda_memory=self.memory())
                    output.write(json.dumps(sample, sort_keys=True, allow_nan=False) + "\n")
                    output.flush()
                    self.samples += 1
                    if self.stop_event.wait(self.interval):
                        break
        except Exception as exc:
            self.error = exc

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError("el muestreador no terminó dentro de 15 s")
        if self.error is not None:
            raise self.error
        if self.samples == 0:
            raise RuntimeError("telemetría vacía")


def _gpu_uuid(torch, device, sample):
    value = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    if value:
        try:
            parsed = uuid.UUID(bytes=value) if isinstance(value, bytes) else uuid.UUID(str(value).removeprefix("GPU-"))
            return "GPU-" + str(parsed)
        except ValueError:
            pass
    observed = {row["gpu_uuid"] for row in sample["gpu_apps"] if row.get("pid") == os.getpid()}
    if len(observed) != 1 and torch.cuda.device_count() == 1 and len(sample["gpus"]) == 1:
        observed = {sample["gpus"][0]["uuid"]}
    if len(observed) != 1:
        raise ValueError("GPU UUID del proceso no observable")
    return "GPU-" + str(uuid.UUID(observed.pop().removeprefix("GPU-")))


def _check_participants(identity, inventory, *, dist):
    roster = [None] * identity["world_size"]
    dist.all_gather_object(roster, identity)
    groups = inventory["groups"]
    expected_nodes = {1: groups["single"], 2: groups["pair"], 4: groups["four"]}[identity["world_size"]]
    nodes = {row["node_id"]: row for row in inventory["nodes"]}
    for rank, row in enumerate(roster):
        if (row["rank"] != rank or row["world_size"] != len(roster)
                or row["node_id"] != expected_nodes[rank]
                or row["hostname"] != nodes[expected_nodes[rank]]["expected_hostname"]
                or any(row[key] != identity[key] for key in ("phase", "run_id", "campaign_id"))):
            raise ValueError("identidad efectiva de host/rank incompatible con el inventario")
    if any(len({row[key] for row in roster}) != len(roster) for key in ("hostname", "gpu_uuid")):
        raise ValueError("host o GPU duplicado entre ranks")
    return roster


def record_group(out, identity, inventory, *, dist):
    """Publica el grupo efectivo de esta fase solo después de comprobarlo."""
    roster = _check_participants(identity, inventory, dist=dist)
    backend = str(dist.get_backend())
    if backend != "nccl":
        raise ValueError("backend efectivo del grupo diferente de NCCL")
    checkpoint.write_json(Path(out) / "group.json",
                          {**identity, "backend": backend, "participants": roster})
    return roster


def _prepared_steps(records, protocol, config, steps, *, torch, rank, world_size):
    prepared = []
    for step in steps:
        ids = protocol.step_ids(step - 1, rank, world_size)
        global_ids = [i for r in range(world_size) for i in protocol.step_ids(step - 1, r, world_size)]
        batches = [{key: torch.tensor([records[i][key]], dtype=torch.long)
                    for key in ("input_ids", "labels")} for i in ids]
        processed = sum(len(records[i]["input_ids"]) for i in ids)
        supervised = sum(sum(v != -100 for v in records[i]["labels"][1:]) for i in ids)
        total_processed = sum(len(records[i]["input_ids"]) for i in global_ids)
        total_supervised = sum(sum(v != -100 for v in records[i]["labels"][1:]) for i in global_ids)
        prepared.append({"step": step, "ids": ids, "batches": batches,
                         "processed_tokens": processed, "supervised_tokens": supervised,
                         "global_processed_tokens": total_processed,
                         "global_supervised_tokens": total_supervised})
    return prepared


def fsdp_loss_scale(microbatch_targets, global_targets, world_size):
    """Scale a token-mean loss before FSDP averages gradients across ranks."""
    return microbatch_targets * world_size / global_targets


def _recovery_barrier(torch, dist, device):
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.synchronize(device)


def _train_step(work, model, optimizer, *, torch, dist, device, world_size, audit,
                set_stage=None, recovery_probe=False, recovery_reference=None):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    audit_seconds = 0.0
    recovery_seconds = 0.0
    recovery = None
    if recovery_probe:
        if set_stage is not None:
            set_stage("parameter_audit")
        audit_start = time.perf_counter()
        recovery = begin_recovery_diagnostic(model, torch, work["step"])
        _recovery_barrier(torch, dist, device)
        recovery_seconds += time.perf_counter() - audit_start
        if set_stage is not None:
            set_stage("train_step")
    weighted_loss = torch.zeros((), device=device, dtype=torch.float32)
    batches = work["batches"]
    for index, batch in enumerate(batches):
        last = index == len(batches) - 1
        model.set_requires_gradient_sync(last)
        model.set_is_last_backward(last)
        inputs = {key: value.to(device) for key, value in batch.items()}
        count = (inputs["labels"][:, 1:] != -100).sum()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model(**inputs, use_cache=False)
            loss = result.loss
        weighted_loss.add_(loss.detach().float() * count)
        (loss * fsdp_loss_scale(count, work["global_supervised_tokens"], world_size)).backward()
        del result, loss, inputs
    dist.all_reduce(weighted_loss, op=dist.ReduceOp.SUM)
    before = None
    if audit or recovery_probe:
        # Finish outstanding training kernels before pausing the step clock.
        torch.cuda.synchronize(device)
        if set_stage is not None:
            set_stage("parameter_audit")
        if audit:
            audit_start = time.perf_counter()
            before = _audit_before(model, torch)
            audit_seconds += time.perf_counter() - audit_start
        if recovery_probe:
            recovery_start = time.perf_counter()
            recovery_gradient_diagnostic(model, torch, recovery)
            _recovery_barrier(torch, dist, device)
            recovery_seconds += time.perf_counter() - recovery_start
        if set_stage is not None:
            set_stage("train_step")
    optimizer.step()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start - audit_seconds - recovery_seconds
    value = float(weighted_loss.item()) / work["global_supervised_tokens"]
    proof = None
    if audit or recovery_probe:
        if set_stage is not None:
            set_stage("parameter_audit")
        if audit:
            audit_start = time.perf_counter()
            proof = _audit_after(model, before, torch=torch, dist=dist, device=device, step=work["step"])
            audit_seconds += time.perf_counter() - audit_start
        if recovery_probe:
            recovery_start = time.perf_counter()
            finish_recovery_diagnostic(model, torch, recovery)
            if recovery_reference is not None:
                recovery["comparison_to_pilot"] = (
                    {"reference_phase": "pilot", "reference_missing": True}
                    if recovery_reference.get("reference_missing") else
                    compare_recovery_diagnostic(recovery, recovery_reference))
            _recovery_barrier(torch, dist, device)
            recovery["rank_barriers"] = 3
            recovery_seconds += time.perf_counter() - recovery_start
        if set_stage is not None:
            set_stage("train_step")
    row = {key: value for key, value in work.items() if key != "batches"}
    row.update(loss=value, seconds=seconds,
               loss_definition="global_supervised_token_mean",
               cuda_memory=cuda_memory(torch, device), audit_performed=audit, parameter_audit_seconds=audit_seconds)
    if recovery is not None:
        row["recovery_diagnostic"] = recovery
        row["recovery_diagnostic_seconds"] = recovery_seconds
    return row, proof


def _failure(out, identity, exc, torch, stage):
    typed_oom = torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError)
    # torch.OutOfMemoryError puede ser el mismo tipo en CPU y CUDA. La clase
    # sola no acredita frontera GPU; exigimos además mensaje y origen CUDA.
    message = str(exc)
    cuda_message = "cuda" in message.lower() and "out of memory" in message.lower()
    cuda_stages = {"cuda_init", "identity", "group_init", "group_identity", "model_materialize",
                   "snapshot_load", "restore", "checkpoint_write", "train_step", "parameter_audit"}
    oom = typed_oom and cuda_message and stage in cuda_stages
    memory_message = any(text in message.lower() for text in (
        "out of memory", "cannot allocate memory", "can't allocate memory", "defaultcpuallocator"))
    ambiguous_memory = not oom and (
        typed_oom or isinstance(exc, MemoryError) or memory_message
        or isinstance(exc, OSError) and exc.errno == errno.ENOMEM)
    status = "cuda_oom" if oom else "indeterminate" if ambiguous_memory else "error"
    checkpoint.write_json(Path(out) / "failure.json", {
        **identity, "status": status,
        "type": "cuda_oom" if oom else "indeterminate" if ambiguous_memory else type(exc).__name__,
        "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "oom_origin": "cuda" if oom else "cpu_or_unknown" if ambiguous_memory else None,
        "message": message,
        "stage": stage, "timestamp_utc": datetime.now(timezone.utc).isoformat()})


def execute(args):
    """Único punto que importa los frameworks o inicia CUDA/distributed."""
    phase_start = time.perf_counter()
    from infraestructura import sft_protocol as protocol

    config = protocol.load_config(path=args.config)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world not in (1, 2, 4) or rank not in range(world):
        raise ValueError("RANK/WORLD_SIZE inválidos")
    if int(os.environ.get("LOCAL_RANK", "0")) != 0 or int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != 1:
        raise ValueError("se requiere exactamente un proceso por nodo")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("el directorio de evidencia debe estar vacío")
    if Path(args.checkpoint_dir).resolve().is_relative_to(out.resolve()):
        raise ValueError("el checkpoint debe estar fuera de la evidencia raw")
    identity = {"node_id": args.node_id, "hostname": socket.gethostname(), "gpu_uuid": None,
                "rank": rank, "world_size": world, "phase": args.phase,
                "run_id": args.run_id, "campaign_id": args.campaign_id}
    torch = None
    telemetry = None
    initialized = False
    stage = "imports"
    try:
        validate_determinism_environment(config)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        determinism = configure_determinism(config, torch)
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        import transformers
        from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
        from safetensors import safe_open

        versions = framework_versions(config, torch, transformers)
        stage = "cuda_init"
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("se requiere una GPU CUDA visible por nodo")
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        stage = "identity"
        identity["gpu_uuid"] = _gpu_uuid(torch, device, observation.collect_sample())
        identity_record = dict(identity)
        if determinism is not None:
            identity_record["determinism"] = determinism
        checkpoint.write_json(out / "identity.json", identity_record)
        original_checkpoint_sha256 = resume_digest(args, identity, config) if args.phase == "resume" else None
        stage = "group_init"
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=300), device_id=device)
        initialized = True
        stage = "group_identity"
        roster = record_group(out, identity, protocol.cluster_config(config), dist=dist)
        stage = "group_ready"
        random.seed(config["training"]["seed"])
        torch.manual_seed(config["training"]["seed"])
        torch.cuda.manual_seed_all(config["training"]["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        telemetry = Telemetry(out / "telemetry.jsonl", identity, lambda: cuda_memory(torch, device))
        telemetry.start()
        limit = args.session_limit_seconds or config["measurement"]["pilot_limit_seconds" if args.phase != "measure" else "run_limit_seconds"]
        deadline = phase_start + limit - config["measurement"]["cleanup_reserve_seconds"]

        def budget():
            if time.perf_counter() >= deadline:
                raise TimeoutError("presupuesto de fase agotado; reserva de limpieza conservada")

        stage = "workload"
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
        records = protocol.encode_corpus(tokenizer)
        if hashlib.sha256(protocol.corpus_bytes()).hexdigest() != config["workload"]["corpus_sha256"]:
            raise ValueError("SHA-256 del corpus incompatible")
        if protocol.tokens_sha256(records) != config["workload"]["tokens_sha256"]:
            raise ValueError("SHA-256 de tokens/labels incompatible")
        validate_work(records, protocol, config, world)
        prepared = _prepared_steps(records, protocol, config, phase_steps(args.phase, config),
                                   torch=torch, rank=rank, world_size=world)
        budget()
        stage = "model_meta"
        model_config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
        if model_config.model_type != "llama" or model_config.tie_word_embeddings:
            raise ValueError("el worker requiere LlamaForCausalLM con pesos no atados")
        model_config.use_cache = False
        model_config._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = LlamaForCausalLM(model_config).float()
        logical = [(name, parameter.numel()) for name, parameter in model.named_parameters()]
        if len(logical) != TENSOR_COUNT or sum(n for _, n in logical) != PARAMETER_COUNT:
            raise ValueError("modelo distinto de los 291 tensores / 8030261248 parámetros aprobados")
        tensor_shapes_sha256 = verify_tensor_shapes(model, protocol, config)
        model.requires_grad_(True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("fsdp",))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        stage = "model_materialize"
        model.to_empty(device=device)
        # inv_freq no es persistente en safetensors; to_empty lo deja sin inicializar.
        model.model.rotary_emb = type(model.model.rotary_emb)(config=model_config, device=device)
        model.train()
        stage = "snapshot"
        if Path(args.model_dir).name != config["model"]["revision"]:
            raise ValueError("la ruta del snapshot debe terminar en la revisión fijada")
        mapping, source = _snapshot_index(args.model_dir, [name for name, _ in logical])
        contract = {"config_sha256": sha256_file(Path(args.config)), "model": config["model"],
                    "workload": config["workload"], "training": config["training"],
                    "model_source": source, "participants": [{k: row[k] for k in checkpoint.IDENTITY_FIELDS}
                                                               for row in roster]}
        digests = [None] * world
        dist.all_gather_object(digests, _canonical_hash(contract))
        if len(set(digests)) != 1:
            raise ValueError("hashes de snapshot/config/carga diferentes entre ranks")
        optimizer = torch.optim.SGD(model.parameters(), lr=config["training"]["learning_rate"],
                                    momentum=0, weight_decay=0, foreach=False, fused=False)
        sampling = {"algorithm": "sft_protocol.step_ids", "seed": config["training"]["seed"],
                    "zero_based_next_step": config["measurement"]["checkpoint_step"],
                    "next_ids": protocol.step_ids(config["measurement"]["checkpoint_step"], rank, world)}
        checkpoint_receipt = None
        if args.phase == "resume":
            stage = "restore"
            checkpoint_receipt = checkpoint.restore_checkpoint(
                args.checkpoint_dir, torch=torch, model=model, optimizer=optimizer,
                identity=identity, contract=contract, expected_step=config["measurement"]["checkpoint_step"],
                expected_sampling=sampling, learning_rate=config["training"]["learning_rate"],
                expected_sha256=original_checkpoint_sha256)
        else:
            stage = "snapshot_load"
            _load_snapshot(model, args.model_dir, mapping, torch=torch, safe_open=safe_open,
                           rank=rank, world_size=world)
        budget()
        proof = {"audit_performed": False, "tensor_count": TENSOR_COUNT,
                 "parameter_count": PARAMETER_COUNT}
        rows = []
        startup_seconds = None
        window_start = window_end = None
        warmup = config["measurement"]["warmup_steps"]
        torch.cuda.reset_peak_memory_stats(device)
        recovery_step = config["measurement"]["checkpoint_step"] + 1
        reference_recovery = None
        if determinism is not None and args.phase == "resume":
            reference_rows = [json.loads(line) for line in
                              (out.parent / "pilot" / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
            reference_recovery = next((row.get("recovery_diagnostic") for row in reference_rows
                                       if row["step"] == recovery_step), None) or {"reference_missing": True}

        def set_step_stage(value):
            nonlocal stage
            stage = value

        for work in prepared:
            budget()
            measured = args.phase == "measure" and work["step"] > warmup
            if measured and window_start is None:
                dist.barrier()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                window_start = telemetry.set_stage("measured")
            elif not measured:
                telemetry.stage = "warmup" if args.phase == "measure" else args.phase
            stage = "train_step"
            if startup_seconds is None:
                startup_seconds = time.perf_counter() - phase_start
            row, audit_proof = _train_step(work, model, optimizer, torch=torch, dist=dist,
                                           device=device, world_size=world,
                                           audit=args.phase == "pilot" and work["step"] == config["measurement"]["pilot_steps"],
                                           set_stage=set_step_stage,
                                           recovery_probe=determinism is not None and args.phase in ("pilot", "resume")
                                           and work["step"] == recovery_step,
                                           recovery_reference=reference_recovery)
            row.update(measured=measured, rank=rank, node_id=args.node_id, phase=args.phase)
            rows.append(row)
            if audit_proof is not None:
                proof = audit_proof
            if args.phase == "pilot" and work["step"] == config["measurement"]["checkpoint_step"]:
                stage = "checkpoint_write"
                telemetry.stage = "checkpoint"
                budget()
                checkpoint_receipt = checkpoint.save_checkpoint(
                    args.checkpoint_dir, torch=torch, model=model, optimizer=optimizer,
                    identity=identity, contract=contract, step=work["step"], sampling=sampling,
                    learning_rate=config["training"]["learning_rate"])
                dist.barrier()
        if window_start is not None:
            torch.cuda.synchronize(device)
            window_end = telemetry.set_stage("validation")
        telemetry.stage = "validation"
        stage = "validation"
        if any(not math.isfinite(row["loss"]) or row["seconds"] <= 0 for row in rows):
            raise ValueError("loss no finita o tiempo de paso inválido")
        if args.phase == "measure" and sum(row["measured"] for row in rows) != config["measurement"]["measured_steps"]:
            raise ValueError("ventana medida incompleta")
        with (out / "steps.jsonl").open("x", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        dist.barrier()
        telemetry.stop()
        telemetry = None
        receipt = {**identity_record, "schema_version": "1.0", "status": "ok",
                   "corpus_sha256": config["workload"]["corpus_sha256"],
                   "tokens_sha256": config["workload"]["tokens_sha256"],
                   "model_revision": config["model"]["revision"],
                   "tokenizer_revision": config["model"]["tokenizer_revision"],
                   **versions,
                   "phase_seconds": time.perf_counter() - phase_start,
                   "startup_seconds": startup_seconds,
                   "startup_scope": "worker_execute_to_first_step",
                   "window_seconds": (window_end - window_start) / 1e9 if window_start is not None else None,
                   "window_start_monotonic_ns": window_start,
                   "window_end_monotonic_ns": window_end,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                   "tensor_count": TENSOR_COUNT, "parameter_count": PARAMETER_COUNT,
                   "tensor_shapes_sha256": tensor_shapes_sha256,
                   "global_batch": 4, "microbatch": 1, "gradient_accumulation_steps": 4 // world,
                   "compute_dtype": "bfloat16", "master_dtype": "float32", "gradient_dtype": "float32",
                   "reduce_dtype": "float32", "attention": "sdpa", "use_cache": False,
                   "activation_checkpointing": "non-reentrant", "optimizer": "SGD",
                   "learning_rate": config["training"]["learning_rate"],
                   "participants": roster, "proof": proof, "checkpoint": checkpoint_receipt,
                   "model_source": source, "config_sha256": contract["config_sha256"],
                   "artifacts": {name: {"sha256": sha256_file(out / name), "bytes": (out / name).stat().st_size}
                                 for name in ("identity.json", "group.json", "steps.jsonl", "telemetry.jsonl")}}
        checkpoint.write_json(out / "receipt.json", receipt)
        return receipt
    except (Exception, KeyboardInterrupt) as exc:
        _failure(out, identity, exc, torch, stage)
        raise
    finally:
        if telemetry is not None:
            try:
                telemetry.stop()
            except Exception:
                pass  # Se conserva la excepción original y nunca se publica recibo OK.
        if initialized:
            dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--phase", required=True, choices=("pilot", "resume", "measure"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--session-limit-seconds", type=float)
    args = parser.parse_args(argv)
    if args.session_limit_seconds is not None and (not math.isfinite(args.session_limit_seconds) or args.session_limit_seconds <= 0):
        parser.error("--session-limit-seconds debe ser finito y positivo")
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
