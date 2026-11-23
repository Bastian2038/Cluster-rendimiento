"""Checkpoints locales de FSDP2, verificados antes de deserializar.

Cada disco conserva únicamente el shard de su rank. Los tensores se copian a
CPU de uno en uno; nunca se reúne el modelo completo. Este módulo no importa
PyTorch: el worker pasa su instancia durante una ejecución autorizada.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random

from infraestructura.manifest import sha256_file


FORMAT = "ipre-fsdp2-local-v1"
IDENTITY_FIELDS = ("node_id", "hostname", "gpu_uuid", "rank", "world_size",
                   "run_id", "campaign_id")


def write_json(path, value):
    """Crea evidencia nueva; nunca reemplaza un archivo existente."""
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, sort_keys=True, allow_nan=False)
        output.write("\n")


def local_tensor(parameter):
    return parameter.to_local() if hasattr(parameter, "to_local") else parameter


def _file(directory, name):
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("nombre de archivo de checkpoint inválido")
    candidate = directory / name
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("archivo local de checkpoint ausente o simbólico")
    return candidate


def _record(path):
    return {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _identity(identity):
    return {key: identity[key] for key in IDENTITY_FIELDS}


def _sgd_contract(optimizer, learning_rate):
    for group in optimizer.param_groups:
        if (group["lr"] != learning_rate or group.get("momentum", 0) != 0
                or group.get("weight_decay", 0) != 0 or group.get("dampening", 0) != 0
                or group.get("nesterov", False) or group.get("maximize", False)):
            raise ValueError("el checkpoint requiere SGD fijo sin momentum ni decay")


def save_checkpoint(directory, *, torch, model, optimizer, identity, contract,
                    step, sampling, learning_rate):
    """Escribe archivos nuevos y publica el índice solo al terminar todos."""
    _sgd_contract(optimizer, learning_rate)
    root = Path(directory) / f"rank-{identity['rank']:05d}"
    root.mkdir(parents=True, exist_ok=False)
    tensors = []
    for index, (name, parameter) in enumerate(model.named_parameters()):
        local = local_tensor(parameter).detach()
        if local.dtype != torch.float32:
            raise ValueError("el master local debe ser FP32")
        path = root / f"tensor-{index:04d}.pt"
        # La mayor copia CPU es un tensor lógico local, nunca 32 GiB de modelo.
        cpu = local.cpu()
        torch.save(cpu, path)
        del cpu
        tensors.append({"name": name, "shape": list(local.shape),
                        "global_shape": list(parameter.shape), "dtype": str(local.dtype),
                        **_record(path)})
    state_path = root / "state.pt"
    state = {"step": step, "next_step": step + 1, "learning_rate": learning_rate,
             "sampling": sampling, "optimizer": optimizer.state_dict(),
             "rng": {"python": random.getstate(), "cpu": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state_all()}}
    torch.save(state, state_path)
    metadata = {"format": FORMAT, "identity": _identity(identity), "contract": contract,
                "step": step, "next_step": step + 1, "learning_rate": learning_rate,
                "sampling": sampling, "tensors": tensors, "state": _record(state_path)}
    path = root / "checkpoint.json"
    write_json(path, metadata)
    return {"metadata_sha256": sha256_file(path), "restored": False, "step": step,
            "next_step": step + 1, "rng_restored": False, "hashes_verified": True,
            "files": [*tensors, metadata["state"]]}


def verify_checkpoint(directory, *, identity, contract, expected_step,
                      expected_sampling, expected_sha256=None):
    """Verifica identidad, cursor y TODOS los hashes antes de torch.load."""
    root = Path(directory) / f"rank-{identity['rank']:05d}"
    if root.is_symlink():
        raise ValueError("directorio de checkpoint simbólico")
    path = _file(root, "checkpoint.json")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("SHA-256 del índice de checkpoint diferente del esperado")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("format") != FORMAT or metadata.get("identity") != _identity(identity):
        raise ValueError("checkpoint de otra identidad o topología")
    if metadata.get("contract") != contract:
        raise ValueError("contrato del checkpoint incompatible")
    if (type(metadata.get("step")) is not int or metadata["step"] != expected_step
            or metadata.get("next_step") != expected_step + 1
            or metadata.get("sampling") != expected_sampling):
        raise ValueError("cursor o muestreo de checkpoint incompatible")
    tensors = metadata.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ValueError("checkpoint sin tensores")
    records = [*tensors, metadata["state"]]
    paths = [record["path"] for record in records]
    names = [record["name"] for record in tensors]
    if len(paths) != len(set(paths)) or len(names) != len(set(names)):
        raise ValueError("archivos o tensores duplicados en checkpoint")
    if set(item.name for item in root.iterdir()) != {*paths, "checkpoint.json"}:
        raise ValueError("archivos ausentes o adicionales en checkpoint")
    for record in records:
        candidate = _file(root, record["path"])
        if (candidate.stat().st_size != record["bytes"]
                or sha256_file(candidate) != record["sha256"]):
            raise ValueError(f"SHA-256 o tamaño de checkpoint inválido: {record['path']}")
    return root, metadata, digest


def restore_checkpoint(directory, *, torch, model, optimizer, identity, contract,
                       expected_step, expected_sampling, learning_rate,
                       expected_sha256=None):
    root, metadata, digest = verify_checkpoint(
        directory, identity=identity, contract=contract, expected_step=expected_step,
        expected_sampling=expected_sampling, expected_sha256=expected_sha256)
    parameters = dict(model.named_parameters())
    if set(parameters) != {row["name"] for row in metadata["tensors"]}:
        raise ValueError("cobertura de tensores del checkpoint incompatible")
    if metadata["learning_rate"] != learning_rate:
        raise ValueError("LR de checkpoint incompatible")
    for record in metadata["tensors"]:
        parameter = parameters[record["name"]]
        local = local_tensor(parameter)
        if (list(local.shape) != record["shape"]
                or list(parameter.shape) != record["global_shape"]
                or str(local.dtype) != record["dtype"]):
            raise ValueError("forma o dtype del shard incompatible")
        saved = torch.load(root / record["path"], map_location="cpu", weights_only=True)
        if saved.shape != local.shape or saved.dtype != torch.float32:
            raise ValueError("tensor local guardado incompatible")
        with torch.no_grad():
            local.copy_(saved)
        del saved
    state = torch.load(root / metadata["state"]["path"], map_location="cpu", weights_only=True)
    if (state["step"] != expected_step or state["next_step"] != expected_step + 1
            or state["sampling"] != expected_sampling
            or not math.isfinite(state["learning_rate"])
            or state["learning_rate"] != learning_rate):
        raise ValueError("cursor, LR o muestreo del estado incompatible")
    optimizer.load_state_dict(state["optimizer"])
    _sgd_contract(optimizer, learning_rate)
    # Restaurar al final: la materialización y el setup ya no consumen RNG.
    random.setstate(state["rng"]["python"])
    torch.set_rng_state(state["rng"]["cpu"])
    if len(state["rng"]["cuda"]) != torch.cuda.device_count():
        raise ValueError("cantidad de estados RNG CUDA incompatible")
    torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return {"metadata_sha256": digest, "restored": True, "step": expected_step,
            "next_step": expected_step + 1, "rng_restored": True, "hashes_verified": True,
            "files": [*metadata["tensors"], metadata["state"]]}
