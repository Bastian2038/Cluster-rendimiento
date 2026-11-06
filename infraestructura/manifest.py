"""Creación y cierre de manifiestos reproducibles para experimentos.

El módulo usa únicamente la biblioteca estándar. Puede importarse desde un
runner o utilizarse como CLI con los subcomandos ``start`` y ``finalize``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "2.1"
SUPPORTED_VERSIONS = frozenset({"2.0", SCHEMA_VERSION})
MODEL_FREE_KINDS = frozenset({"system-baseline", "cluster-inventory", "nccl-test"})
DISTRIBUTED_KINDS = frozenset({"cluster-inventory", "nccl-test", "inference-serving"})
_CONTEXT_PATH = "raw/context.json"
_CONTEXT_FIELDS = frozenset({
    "policy", "git", "system", "packages", "accelerator", "requested_config",
    "requested_config_file", "inputs", "input_verification", "services", "artifacts",
    "participants",
})
DEFAULT_RESULTS_ROOT = Path("resultados/raw")
PACKAGE_DISTRIBUTIONS = (
    "vllm",
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "accelerate",
    "datasets",
    "peft",
    "trl",
    "bitsandbytes",
    "numpy",
    "pandas",
    "scipy",
)
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PINNED_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_MARKERS = frozenset({"COMPLETE", "FAILED"})


class ManifestError(RuntimeError):
    """Error de validación o estado de un manifiesto."""


def utc_now() -> str:
    """Devuelve un instante UTC ISO-8601 con sufijo Z."""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def generate_run_id() -> str:
    """Genera un identificador legible, ordenable y resistente a colisiones."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcula SHA-256 leyendo el archivo por bloques."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_command(command: Sequence[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    # ``git status --porcelain`` usa el primer carácter como parte del estado
    # (por ejemplo, `` M``). No eliminar whitespace inicial aquí.
    return completed.stdout.rstrip()


def capture_git(repo_path: Path) -> dict[str, Any]:
    """Captura el commit y la suciedad del worktree sin modificar Git."""

    result: dict[str, Any] = {
        "commit": None,
        "dirty": None,
        "root": None,
        "branch": None,
        "changed_paths": [],
        "error": None,
    }
    try:
        result["root"] = _run_command(
            ["git", "rev-parse", "--show-toplevel"], cwd=repo_path
        )
        result["commit"] = _run_command(
            ["git", "rev-parse", "HEAD"], cwd=repo_path
        )
        result["branch"] = _run_command(
            ["git", "branch", "--show-current"], cwd=repo_path
        ) or None
        status = _run_command(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo_path,
        )
        result["dirty"] = bool(status)
        changed_paths: list[str] = []
        for line in status.splitlines():
            path = line[3:] if len(line) >= 4 else line
            if " -> " in path:
                old_path, new_path = path.split(" -> ", 1)
                changed_paths.extend((old_path, new_path))
            elif path:
                changed_paths.append(path)
        result["changed_paths"] = sorted(set(changed_paths))
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        result["error"] = str(exc)
    return result


def capture_system() -> dict[str, Any]:
    """Captura información portable del host y del intérprete."""

    return {
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def capture_packages(
    distributions: Iterable[str] = PACKAGE_DISTRIBUTIONS,
) -> dict[str, str | None]:
    """Registra versiones instaladas; ``null`` significa no instalado."""

    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def capture_nvidia() -> dict[str, Any]:
    """Consulta GPU y driver con nvidia-smi cuando esté disponible."""

    executable = shutil.which("nvidia-smi")
    result: dict[str, Any] = {
        "nvidia_smi_available": executable is not None,
        "gpus": [],
        "error": None,
    }
    if executable is None:
        return result

    fields = ("name", "uuid", "driver_version", "memory.total")
    try:
        output = _run_command(
            [
                executable,
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ]
        )
        for index, line in enumerate(output.splitlines()):
            values = [value.strip() for value in line.split(",")]
            if len(values) != len(fields):
                raise ManifestError(f"salida inesperada de nvidia-smi: {line!r}")
            name, gpu_uuid, driver, memory_mib = values
            try:
                parsed_memory: int | str = int(memory_mib)
            except ValueError:
                parsed_memory = memory_mib
            result["gpus"].append(
                {
                    "index": index,
                    "name": name,
                    "uuid": gpu_uuid,
                    "driver_version": driver,
                    "memory_total_mib": parsed_memory,
                }
            )
    except (subprocess.SubprocessError, ManifestError, OSError) as exc:
        result["error"] = str(exc)
    return result


def _input_record(name: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ManifestError(f"input inexistente o no regular: {path}")
    return {
        "name": name,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_component(value: str, field: str) -> None:
    if not _SAFE_COMPONENT.fullmatch(value):
        raise ManifestError(
            f"{field} inválido {value!r}; use letras, números, '.', '_' o '-'"
        )


def _validate_revision(revision: str | None, field: str) -> None:
    if not revision or not _PINNED_REVISION.fullmatch(revision):
        raise ManifestError(
            f"{field} debe ser un commit SHA completo (40 hexadecimales) "
            "en modo estricto"
        )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_write_text(path: Path, value: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def start_run(
    *,
    kind: str,
    model_id: str | None,
    model_revision: str | None,
    tokenizer_id: str | None,
    tokenizer_revision: str | None,
    requested_config: Mapping[str, Any],
    requested_config_path: str | os.PathLike[str] | None = None,
    inputs: Mapping[str, str | os.PathLike[str]] | None = None,
    results_root: str | os.PathLike[str] = DEFAULT_RESULTS_ROOT,
    run_id: str | None = None,
    repo_path: str | os.PathLike[str] = ".",
    allow_dirty: bool = False,
    allow_floating_revision: bool = False,
    allow_missing_git: bool = False,
) -> Path:
    """Inicia una corrida y crea su ``manifest.json`` atómicamente.

    Por defecto aplica el modo estricto. Las excepciones destinadas a smoke
    tests son explícitas y quedan registradas en ``policy.exceptions``.
    Los inventarios y ``nccl-test`` registran ``model: null``: sus
    cuatro argumentos de modelo/tokenizer deben ser None. No omite la política
    Git ni requiere una excepción de revisión flotante.
    """

    _validate_component(kind, "kind")
    selected_run_id = run_id or generate_run_id()
    _validate_component(selected_run_id, "run_id")
    system_baseline = kind in MODEL_FREE_KINDS
    if system_baseline:
        if any(value is not None for value in (
            model_id, model_revision, tokenizer_id, tokenizer_revision
        )):
            raise ManifestError(f"{kind} requiere modelo y tokenizer null")
        if allow_floating_revision:
            raise ManifestError(f"floating_revision no aplica a {kind}")
    elif not model_id or not tokenizer_id:
        raise ManifestError("model_id y tokenizer_id son obligatorios")
    if not isinstance(requested_config, Mapping):
        raise ManifestError("requested_config debe ser un objeto JSON")
    try:
        json.dumps(requested_config, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"requested_config no es serializable como JSON: {exc}") from exc

    git = capture_git(Path(repo_path).resolve())
    if git["commit"] is None and not allow_missing_git:
        raise ManifestError(
            "no se pudo determinar el commit Git; use --allow-missing-git solo para smoke"
        )
    if git["dirty"] is True and not allow_dirty:
        raise ManifestError(
            "el worktree está dirty; confirme los cambios o use --allow-dirty solo para smoke"
        )
    if not system_baseline and (not allow_floating_revision or kind == "inference-serving"):
        _validate_revision(model_revision, "model_revision")
        _validate_revision(tokenizer_revision, "tokenizer_revision")

    input_records: list[dict[str, Any]] = []
    for name, input_path in sorted((inputs or {}).items()):
        if not name:
            raise ManifestError("cada input debe tener un nombre no vacío")
        input_records.append(_input_record(name, Path(input_path)))
    config_file_record = (
        _input_record("requested_config", Path(requested_config_path))
        if requested_config_path is not None
        else None
    )

    run_directory = Path(results_root).resolve() / kind / selected_run_id
    try:
        run_directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ManifestError(f"la corrida ya existe: {run_directory}") from exc
    for subdirectory in ("raw", "logs", "metrics"):
        (run_directory / subdirectory).mkdir()

    started_at = utc_now()
    exceptions = []
    if allow_dirty:
        exceptions.append("dirty_worktree")
    if allow_floating_revision:
        exceptions.append("floating_revision")
    if allow_missing_git:
        exceptions.append("missing_git")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": selected_run_id,
        "kind": kind,
        "status": "running",
        "started_at": started_at,
        "ended_at": None,
        "policy": {"strict": not exceptions, "exceptions": exceptions},
        "git": git,
        "system": capture_system(),
        "packages": {} if system_baseline else capture_packages(),
        "accelerator": capture_nvidia(),
        "model": None if system_baseline else {
            "id": model_id,
            "revision": model_revision,
            "tokenizer": {"id": tokenizer_id, "revision": tokenizer_revision},
        },
        "requested_config": dict(requested_config),
        "requested_config_file": config_file_record,
        "inputs": input_records,
        "input_verification": [],
        "services": {},
        "artifacts": [],
        "errors": [],
    }
    if kind in DISTRIBUTED_KINDS:
        manifest["participants"] = []
    try:
        _write_compact_run(run_directory, manifest)
    except BaseException:
        # Intenta dejar libre el run_id si todavía no se escribió ningún dato.
        try:
            for subdirectory in ("raw", "logs", "metrics"):
                (run_directory / subdirectory).rmdir()
            run_directory.rmdir()
        except OSError:
            pass
        raise
    return run_directory


def _observed_hosts(run: Mapping[str, Any]) -> list[str]:
    if run.get("schema_version") == "2.1" and run.get("kind") in DISTRIBUTED_KINDS:
        return [item["hostname"] for item in run.get("participants", [])]
    return [run["system"].get("hostname")]


def _internal_file(directory: Path, relative: str) -> Path:
    """Resuelve evidencia interna sin admitir escapes, enlaces externos ni metadatos."""
    if not isinstance(relative, str) or not relative:
        raise ManifestError("ruta de evidencia inválida")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ManifestError("evidencia fuera del directorio de corrida")
    if relative in {"manifest.json", _CONTEXT_PATH, *_MARKERS}:
        raise ManifestError("metadatos reservados no son evidencia de participante")
    path = directory / candidate
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ManifestError("evidencia fuera del directorio de corrida")
    if not path.is_file():
        raise ManifestError(f"evidencia ausente: {relative}")
    return path


def _validate_participants(directory: Path, participants: Any, kind: str) -> None:
    if not isinstance(participants, list):
        raise ManifestError("participants debe ser una lista")
    identity = {"node_id", "hostname", "gpu_uuid", "rank"}
    fields = identity | {"evidence_path", "evidence_sha256"}
    seen: dict[str, set[Any]] = {key: set() for key in identity}
    for item in participants:
        if not isinstance(item, dict) or set(item) != fields:
            raise ManifestError("campos de participante inválidos")
        for key in ("node_id", "hostname"):
            if not isinstance(item[key], str) or not _SAFE_COMPONENT.fullmatch(item[key]):
                raise ManifestError(f"participante {key} inválido")
        gpu, rank = item["gpu_uuid"], item["rank"]
        if gpu is not None and (not isinstance(gpu, str) or not _SAFE_COMPONENT.fullmatch(gpu)):
            raise ManifestError("gpu_uuid inválido")
        if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank < 0):
            raise ManifestError("rank inválido")
        if kind in {"nccl-test", "inference-serving"} and (gpu is None or rank is None):
            raise ManifestError(f"{kind} requiere GPU y rank observados")
        for key in identity:
            value = item[key]
            if value is not None:
                if value in seen[key]:
                    raise ManifestError(f"participante duplicado: {key}={value}")
                seen[key].add(value)
        path = _internal_file(directory, item["evidence_path"])
        if sha256_file(path) != item["evidence_sha256"]:
            raise ManifestError("SHA-256 de evidencia de participante no coincide")
        observed = _load_json_object(path)
        if any(key not in observed or observed[key] != item[key] for key in identity):
            raise ManifestError("identidad de participante no coincide con evidencia")


def register_participants(
    run_directory: str | os.PathLike[str], participants: list[dict[str, Any]]
) -> None:
    """Registra un roster completo corroborado por evidencia JSON interna.

    Solo modifica manifiestos abiertos 2.1. El coordinador no se convierte en
    participante por iniciar la corrida. El runner valida identidad esperada y
    registra por separado cambios de ranks entre trabajos.
    """
    directory = Path(run_directory).resolve()
    run = load_run(directory)
    if run["status"] != "running":
        raise ManifestError("no se puede registrar en corrida cerrada")
    if run["schema_version"] != "2.1" or run["kind"] not in DISTRIBUTED_KINDS:
        raise ManifestError("participantes requieren corrida distribuida 2.1")
    if not isinstance(participants, list):
        raise ManifestError("participants debe ser una lista")
    records = []
    for participant in participants:
        if not isinstance(participant, dict):
            raise ManifestError("participante debe ser un objeto")
        item = dict(participant)
        path = _internal_file(directory, item.get("evidence_path"))
        item["evidence_sha256"] = sha256_file(path)
        records.append(item)
    _validate_participants(directory, records, run["kind"])
    run["participants"] = records
    _write_compact_run(directory, run)


def verify_artifacts(
    run_directory: str | os.PathLike[str], run: Mapping[str, Any] | None = None
) -> None:
    """Verifica el índice de artefactos antes de derivar informes de un cierre."""
    directory = Path(run_directory).resolve()
    # Aunque el llamante tenga un objeto, comprobar siempre el cierre actual.
    loaded = load_run(directory)
    if loaded["status"] == "running":
        raise ManifestError("el informe requiere una corrida cerrada")
    if run is not None and dict(run) != loaded:
        raise ManifestError("manifiesto aportado no coincide con la corrida")
    seen = set()
    for record in loaded["artifacts"]:
        relative = record["path"]
        if relative in seen:
            raise ManifestError("artefacto duplicado")
        seen.add(relative)
        path = _internal_file(directory, relative)
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise ManifestError(f"integridad de artefacto inválida: {relative}")
    # Detectar también archivos añadidos después del cierre.
    actual = {item["path"] for item in _artifact_records(directory)}
    if actual != seen:
        raise ManifestError("índice de artefactos no coincide con archivos de corrida")


def _write_compact_run(directory: Path, run: Mapping[str, Any]) -> None:
    """Publica primero el contexto, luego su referencia; el cierre pone el marcador."""

    context = {key: run[key] for key in _CONTEXT_FIELDS if key in run}
    if run["kind"] in MODEL_FREE_KINDS:
        context.pop("packages", None)
        context.pop("services", None)
    elif not context.get("services"):
        context.pop("services", None)
    context.update(schema_version=run["schema_version"], run_id=run["run_id"])
    _atomic_write_json(directory / _CONTEXT_PATH, context)
    compact = {key: run[key] for key in (
        "schema_version", "run_id", "kind", "status", "started_at", "ended_at",
        "model", "errors",
    )}
    compact["git"] = {key: run["git"][key] for key in ("commit", "dirty")}
    compact["hosts"] = _observed_hosts(run)
    compact["context"] = {
        "path": _CONTEXT_PATH, "sha256": sha256_file(directory / _CONTEXT_PATH)
    }
    _atomic_write_json(directory / "manifest.json", compact)


def load_run(run_directory: str | os.PathLike[str]) -> dict[str, Any]:
    """Lee una corrida 2.0/2.1 sin migrarla ni reescribir evidencia.

    Verifica consistencia del cierre, identidad y SHA-256 del contexto.
    Valida evidencia de participantes 2.1; el informe verifica además todos los
    artefactos mediante ``verify_artifacts``. No revalida entradas externas.
    """

    directory = Path(run_directory).resolve()
    core = _load_json_object(directory / "manifest.json")
    version = core.get("schema_version")
    if version not in SUPPORTED_VERSIONS:
        raise ManifestError(f"schema_version no soportada: {version!r}")
    reference = core.get("context")
    if not isinstance(reference, dict) or reference.get("path") != _CONTEXT_PATH:
        raise ManifestError("referencia de contexto inválida: se requiere raw/context.json")
    context_path = directory / _CONTEXT_PATH
    if not context_path.resolve().is_relative_to(directory):
        raise ManifestError("contexto fuera del directorio de corrida")
    try:
        digest = sha256_file(context_path)
    except OSError as exc:
        raise ManifestError(f"contexto ausente o ilegible: {exc}") from exc
    if digest != reference.get("sha256"):
        raise ManifestError("SHA-256 del contexto no coincide con manifest.json")
    context = _load_json_object(context_path)
    if (context.get("schema_version") != version
            or context.get("run_id") != core.get("run_id")):
        raise ManifestError("identidad del contexto no coincide con manifest.json")
    allowed = _CONTEXT_FIELDS if version == "2.1" else _CONTEXT_FIELDS - {"participants"}
    unexpected = set(context) - allowed - {"schema_version", "run_id"}
    if unexpected:
        raise ManifestError(f"campos inesperados en contexto: {sorted(unexpected)}")
    required = _CONTEXT_FIELDS - {"packages", "services", "participants"}
    if version == "2.1" and core.get("kind") in DISTRIBUTED_KINDS:
        required = required | {"participants"}
    if not required.issubset(context):
        raise ManifestError(f"contexto incompleto: faltan {sorted(required - set(context))}")
    if not isinstance(context["git"], dict) or core.get("git") != {
        key: context["git"].get(key) for key in ("commit", "dirty")
    }:
        raise ManifestError("Git del contexto no coincide con manifest.json")
    if "participants" in context:
        if core.get("kind") not in DISTRIBUTED_KINDS:
            raise ManifestError("participants solo aplica a corridas distribuidas")
        _validate_participants(directory, context["participants"], core["kind"])
    if not isinstance(context["system"], dict) or core.get("hosts") != _observed_hosts({**context, "kind": core.get("kind")}):
        raise ManifestError("hosts observados no coinciden con el contexto")
    policy = context["policy"]
    if not isinstance(policy, dict) or not isinstance(policy.get("exceptions"), list):
        raise ManifestError("política de contexto inválida")
    if policy.get("strict") != (not policy["exceptions"]):
        raise ManifestError("política de contexto inconsistente")
    model = core.get("model")
    model_free = {"system-baseline"} if version == "2.0" else MODEL_FREE_KINDS
    if core.get("kind") in model_free:
        if model is not None or "floating_revision" in policy["exceptions"]:
            raise ManifestError(f"{core['kind']} requiere modelo null sin floating_revision")
    elif not isinstance(model, dict) or not isinstance(model.get("tokenizer"), dict):
        raise ManifestError("metadatos de modelo/tokenizer inválidos")
    elif (version == "2.1" and core.get("kind") == "inference-serving") or "floating_revision" not in policy["exceptions"]:
        _validate_revision(model.get("revision"), "model_revision")
        _validate_revision(model["tokenizer"].get("revision"), "tokenizer_revision")
    if (version == "2.1" and core.get("kind") == "inference-serving"
            and (not isinstance(model.get("id"), str) or not model["id"]
                 or not isinstance(model["tokenizer"].get("id"), str)
                 or not model["tokenizer"]["id"])):
        raise ManifestError("model_id y tokenizer_id son obligatorios")
    if (version == "2.1" and core.get("kind") in DISTRIBUTED_KINDS
            and core.get("status") == "complete" and not context.get("participants")):
        raise ManifestError("corrida distribuida sin participantes observados no puede completar")
    _validate_completion(directory, core)
    normalized = dict(core)
    normalized.update({key: context[key] for key in _CONTEXT_FIELDS if key in context})
    normalized.setdefault("packages", {})
    normalized.setdefault("services", {})
    return normalized


def _validate_completion(directory: Path, core: Mapping[str, Any]) -> None:
    """Un manifiesto terminal sin su marcador no demuestra un cierre completo."""

    status = core.get("status")
    markers = {marker for marker in _MARKERS if (directory / marker).exists()}
    if status == "running":
        if core.get("ended_at") is not None or markers:
            raise ManifestError("cierre inconsistente: corrida running con fecha o marcador")
    elif status in {"complete", "failed"}:
        expected_marker = "COMPLETE" if status == "complete" else "FAILED"
        if not core.get("ended_at") or markers != {expected_marker}:
            raise ManifestError("cierre incompleto o inconsistente: falta marcador válido")
        try:
            marker_time = (directory / expected_marker).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ManifestError(f"marcador ilegible: {exc}") from exc
        if marker_time != core["ended_at"]:
            raise ManifestError("fecha del marcador no coincide con manifest.json")
    else:
        raise ManifestError(f"estado inválido: {status!r}")


def _artifact_records(run_directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_directory.rglob("*")):
        relative = path.relative_to(run_directory).as_posix()
        if (relative == "manifest.json" or relative in _MARKERS
                or relative == _CONTEXT_PATH):
            continue
        if not path.resolve().is_relative_to(run_directory.resolve()):
            raise ManifestError(f"artefacto fuera del directorio de corrida: {relative}")
        if path.is_file():
            records.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def _verify_record(record: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(record["path"]))
    expected_size = record.get("size_bytes")
    expected_hash = record.get("sha256")
    if path.is_file():
        observed_size: int | None = path.stat().st_size
        observed_hash: str | None = sha256_file(path)
    else:
        observed_size = None
        observed_hash = None
    return {
        "name": record.get("name"),
        "path": str(path),
        "expected_size_bytes": expected_size,
        "observed_size_bytes": observed_size,
        "expected_sha256": expected_hash,
        "observed_sha256": observed_hash,
        "matches": observed_size == expected_size and observed_hash == expected_hash,
    }


def finalize_run(
    run_directory: str | os.PathLike[str],
    *,
    status: str = "complete",
    errors: Iterable[str | Mapping[str, Any]] | None = None,
) -> Path:
    """Cierra una corrida sin modificar ni eliminar sus artefactos raw."""

    if status not in {"complete", "failed"}:
        raise ManifestError("status debe ser 'complete' o 'failed'")
    directory = Path(run_directory).resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ManifestError(f"no existe manifest.json en {directory}")
    if any((directory / marker).exists() for marker in _MARKERS):
        raise ManifestError(f"la corrida ya fue finalizada: {directory}")

    manifest = load_run(directory)
    if manifest.get("status") != "running":
        raise ManifestError(
            f"estado inesperado del manifiesto: {manifest.get('status')!r}"
        )
    if (status == "complete" and manifest.get("schema_version") == "2.1"
            and manifest.get("kind") in DISTRIBUTED_KINDS and not manifest.get("participants")):
        raise ManifestError("corrida distribuida sin participantes observados no puede completar")

    ended_at = utc_now()
    verification_records = [
        _verify_record(record) for record in manifest.get("inputs", [])
    ]
    config_file = manifest.get("requested_config_file")
    if config_file:
        verification_records.append(_verify_record(config_file))
    inputs_changed = any(not record["matches"] for record in verification_records)

    manifest["artifacts"] = _artifact_records(directory)
    manifest["input_verification"] = verification_records
    manifest["status"] = "failed" if inputs_changed else status
    manifest["ended_at"] = ended_at
    recorded_errors = list(manifest.get("errors") or [])
    for error in errors or ():
        if isinstance(error, Mapping):
            recorded_errors.append(dict(error))
        else:
            recorded_errors.append({"message": str(error)})
    if inputs_changed:
        recorded_errors.append(
            {
                "type": "input_integrity_error",
                "message": "uno o más inputs cambiaron desde el inicio de la corrida",
                "paths": [
                    record["path"] for record in verification_records if not record["matches"]
                ],
            }
        )
    manifest["errors"] = recorded_errors
    _write_compact_run(directory, manifest)

    marker = "COMPLETE" if manifest["status"] == "complete" else "FAILED"
    _atomic_write_text(directory / marker, f"{ended_at}\n")
    if inputs_changed:
        raise ManifestError(
            "la corrida se marcó FAILED porque uno o más inputs cambiaron"
        )
    return manifest_path


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"no se pudo leer configuración JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError("la configuración debe contener un objeto JSON")
    return value


def _parse_inputs(values: Sequence[str]) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = Path(path).name
        if not name or not path:
            raise ManifestError(f"input inválido: {value!r}")
        if name in inputs:
            raise ManifestError(f"nombre de input duplicado: {name!r}")
        inputs[name] = path
    return inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="crear una corrida y su manifiesto")
    start.add_argument("--kind", required=True)
    start.add_argument("--model-id", help="obligatorio salvo inventarios y nccl-test")
    start.add_argument("--model-revision")
    start.add_argument("--tokenizer-id", help="obligatorio salvo inventarios y nccl-test")
    start.add_argument("--tokenizer-revision")
    start.add_argument("--config", required=True, type=Path, metavar="CONFIG.json")
    start.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="[NOMBRE=]RUTA",
        help="archivo de entrada; repetible",
    )
    start.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    start.add_argument("--run-id")
    start.add_argument("--repo-path", type=Path, default=Path("."))
    start.add_argument("--allow-dirty", action="store_true")
    start.add_argument("--allow-floating-revision", action="store_true")
    start.add_argument("--allow-missing-git", action="store_true")

    finalize = subparsers.add_parser("finalize", help="cerrar una corrida")
    finalize.add_argument("run_directory", type=Path)
    finalize.add_argument(
        "--status", choices=("complete", "failed"), default="complete"
    )
    finalize.add_argument("--error", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "start":
            config = _load_json_object(args.config)
            inputs = _parse_inputs(args.input)
            run_directory = start_run(
                kind=args.kind,
                model_id=args.model_id,
                model_revision=args.model_revision,
                tokenizer_id=args.tokenizer_id,
                tokenizer_revision=args.tokenizer_revision,
                requested_config=config,
                requested_config_path=args.config,
                inputs=inputs,
                results_root=args.results_root,
                run_id=args.run_id,
                repo_path=args.repo_path,
                allow_dirty=args.allow_dirty,
                allow_floating_revision=args.allow_floating_revision,
                allow_missing_git=args.allow_missing_git,
            )
            print(run_directory)
        else:
            manifest_path = finalize_run(
                args.run_directory, status=args.status, errors=args.error
            )
            print(manifest_path)
    except ManifestError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
