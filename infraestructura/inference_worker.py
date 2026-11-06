"""Evidence-only vLLM 0.24 GPU Worker extension.

The parent methods remain authoritative; evidence is written only after each
parent method succeeds. Import this module inside the vLLM container only.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid as uuid_module

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.v1.worker.gpu_worker import Worker


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_uuid(device) -> str:
    properties = torch.cuda.get_device_properties(device)
    uuid = getattr(properties, "uuid", None)
    if uuid:
        raw = str(uuid).strip()
        try:
            canonical = str(uuid_module.UUID(raw.removeprefix("GPU-")))
            return "GPU-" + canonical
        except ValueError:
            # An unexpected property format is not evidence of a GPU identity.
            pass
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    result = subprocess.run(
        ["nvidia-smi", "-i", str(index), "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=5,
    )
    uuid = result.stdout.strip()
    try:
        canonical = str(uuid_module.UUID(uuid.removeprefix("GPU-")))
    except ValueError as exc:
        raise RuntimeError("GPU UUID no observable") from exc
    if not uuid.startswith("GPU-") or "\n" in uuid:
        raise RuntimeError("GPU UUID no observable")
    return "GPU-" + canonical


def _group_size(group) -> int:
    size = getattr(group, "world_size", None)
    if not isinstance(size, int) or size < 1:
        raise RuntimeError("world size de grupo vLLM no observable")
    return size


class _OncePerRecord(logging.Filter):
    """Avoid duplicate writes when a named logger propagates to root."""

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "_ipre_worker_file_seen", False):
            return False
        record._ipre_worker_file_seen = True
        return True


def _nccl_version() -> tuple[str | None, str | None]:
    try:
        return ".".join(map(str, torch.cuda.nccl.version())), None
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return None, type(exc).__name__


class ObservedWorker(Worker):
    """Write one immutable JSON file per successful lifecycle stage and rank."""

    def _attach_worker_log(self) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed no inicializado")
        evidence_dir = os.environ.get("IPRE_EVIDENCE_DIR")
        if not evidence_dir:
            raise RuntimeError("IPRE_EVIDENCE_DIR requerido")
        log_dir = Path(evidence_dir).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"worker-rank{dist.get_rank():02d}.log"
        handler = logging.FileHandler(log_path, mode="x", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.addFilter(_OncePerRecord())
        formatter = logging.Formatter(
            "%(asctime)s UTC %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        for name in ("", "vllm", "torch"):
            logging.getLogger(name).addHandler(handler)
        self._ipre_worker_log_path = str(log_path)
        handler.emit(logging.LogRecord("ipre.worker", logging.INFO, __file__, 0,
                                       "IPRE_WORKER_LOG_READY", (), None))
        handler.flush()
        os.environ["IPRE_WORKER_LOG_READY"] = "1"

    def _write_evidence(self, stage: str) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed no inicializado")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank != self.rank:
            raise RuntimeError("rank vLLM difiere de torch.distributed")
        node_id = os.environ.get("IPRE_NODE_ID")
        evidence_dir = os.environ.get("IPRE_EVIDENCE_DIR")
        if not node_id or not evidence_dir:
            raise RuntimeError("IPRE_NODE_ID/IPRE_EVIDENCE_DIR requeridos")
        config = self.vllm_config.model_config
        nccl_version, nccl_error = _nccl_version()
        record = {
            "stage": stage,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            "node_id": node_id,
            "hostname": socket.gethostname(),
            "gpu_uuid": _gpu_uuid(self.device),
            "rank": rank,
            "world_size": world_size,
            "device": str(self.device),
            "model_path": str(config.model),
            "tokenizer_path": str(config.tokenizer),
            "dtype": str(config.dtype),
            "effective_tp": _group_size(get_tp_group()),
            "effective_pp": _group_size(get_pp_group()),
            "versions": {**{name: _version(name) for name in ("vllm", "ray", "torch")},
                         "nccl": nccl_version},
            "nccl_version_error": nccl_error,
            "torch_cuda_version": torch.version.cuda,
            "worker_log_path": getattr(self, "_ipre_worker_log_path", None),
        }
        path = Path(evidence_dir) / f"{stage}-rank{rank:02d}.json"
        # Existing raw evidence must never be replaced, including a failed run.
        with path.open("x", encoding="utf-8") as output:
            json.dump(record, output, sort_keys=True)
            output.write("\n")

    def init_device(self):
        result = super().init_device()
        self._attach_worker_log()
        self._write_evidence("init_device")
        return result

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        result = super().load_model(load_dummy_weights=load_dummy_weights)
        self._write_evidence("load_model")
        return result
