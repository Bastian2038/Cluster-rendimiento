"""Reconstruct serving capacity from the sealed campaign and its run archives."""

import json
import math
import os
import re
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from infraestructura import (
    inference,
    inference_diag_phases,
    inference_diag_quality,
    inference_evidence,
    inference_telemetry_report,
    manifest,
    serving_capacity_client as client,
    serving_capacity_metrics as metrics,
    serving_capacity_protocol as protocol,
    serving_capacity_runner as runner,
)

PHASES = protocol.PHASES
PROFILE_FILES = ("warmup.jsonl", "measured.jsonl", "metadata.json", "reader-validation.json")
TOTAL_POINTS = 126
REFERENCE_INDEX = "reference/campaign.json"
REFERENCE_PREFLIGHT = "reference/preflight.json"
SHA256 = re.compile(r"[0-9a-f]{64}")
EVIDENCE_ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, OverflowError)


def _same_json(left, right):
    # JSON equality distinguishes bool from numbers and integer from decimal rates.
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _read(directory, name):
    return inference.read_json(manifest._internal_file(directory, name))


def _timestamp(value):
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() is None:
        raise ValueError("marca temporal sin zona horaria")
    return stamp


def _layout(plan, config, expected):
    if not _same_json(plan["protocol"], config) or len(plan["jobs"]) != len(expected):
        raise ValueError("protocolo o matriz de plan adulterados")
    for saved, job in zip(plan["jobs"], expected):
        if (any(not _same_json(saved[key], job[key]) for key in ("name", "group", "block", "model"))
                or saved["nodes"] != inference.TOPOLOGIES[job["group"]]
                or len(saved["cells"]) != len(job["cells"])):
            raise ValueError(f"identidad o matriz incorrecta en el plan de {job['name']}")
        nodes = saved["nodes"]
        runtime = inference.runtime
        if (saved["server_command"] != runtime.server_command(config, job["model"], len(nodes))
                or saved["ray_commands"] != {node: runtime.ray_command({"node_id": node}, config)
                                              for node in nodes}):
            raise ValueError(f"comandos de servicio incorrectos en el plan de {job['name']}")
        for cell, coordinate in zip(saved["cells"], job["cells"]):
            rate = coordinate["rate_rps"]
            commands = {phase: protocol.client_command(config, job["model"], rate, phase)
                        for phase in PHASES}
            if not _same_json(cell["rate_rps"], rate) or cell["commands"] != commands:
                raise ValueError(f"tasa o comandos incorrectos en el plan de {job['name']}")


def _preflight_probes(preflight):
    nodes = preflight["nodes"]
    by_node = {node["node_id"]: node for node in nodes}
    if (len(nodes) != 4 or set(by_node) != set(inference.TOPOLOGIES["four"])
            or preflight.get("status") != "ready_for_smoke" or preflight.get("errors") != []):
        raise ValueError("preflight sin verificación limpia de los cuatro nodos")
    for node in nodes:
        gpu_apps = node["gpu_apps"]
        if (gpu_apps["returncode"] != 0 or gpu_apps["stdout"].strip()
                or node["occupied_ports"] or node["ray_pids"]):
            raise ValueError(f"{node['node_id']}: entorno ocupado o desconocido en preflight")
    uuids = [node["gpu"]["stdout"].split(",")[0].strip() for node in nodes]
    if len(set(uuids)) != len(nodes) or len({node["hostname"] for node in nodes}) != len(nodes):
        raise ValueError("preflight con GPU u host duplicado")
    return by_node


def _run_path(directory, reference):
    # The archived reference supplies the run ID. Resolve the run in this
    # archive, even if the original archive still exists elsewhere.
    run_id = Path(reference).name
    manifest._validate_component(run_id, "run_id")
    candidate = directory.parent.parent / "inference-serving" / run_id
    if not candidate.is_dir() or candidate != candidate.resolve():
        raise ValueError("ruta local de corrida ausente o no canónica")
    return candidate


def _input_records(run):
    records = defaultdict(list)
    for record in run["inputs"]:
        name = record["name"]
        if Path(name).as_posix() != name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"nombre de entrada inválido: {name}")
        records[name].append(record)
    if any(len(values) != 1 for values in records.values()):
        raise ValueError("nombres de entrada duplicados en el manifiesto")
    return records


def _archived_inputs(path, run, config, job, input_hashes, preflight):
    records = _input_records(run)
    sources = inference_evidence.runner_sources(protocol.PROTOCOL)
    lengths = config["workload"]["input_tokens"]
    workload_files = [f"workload/{model['key']}/{length}/{name}"
                      for model in config["models"] for length in lengths for name in PROFILE_FILES]
    names = ["cluster", "protocol", REFERENCE_INDEX, REFERENCE_PREFLIGHT, *sources, *workload_files]
    expected_names = set(names)
    differences = (expected_names ^ records.keys()) | (expected_names ^ input_hashes.keys())
    if differences:
        raise ValueError(f"índice de entradas incompatible con el protocolo: {', '.join(sorted(differences))}")
    issues = []
    hashes = inference_evidence._archived_hashes(path, records, names, issues)
    if issues:
        raise ValueError(" | ".join(issues))
    if any(hashes[name] != input_hashes.get(name) for name in names):
        raise ValueError("las entradas archivadas difieren de input_hashes")
    if not _same_json(_read(path, "raw/inputs/protocol"), config):
        raise ValueError("el protocolo archivado difiere del cierre")
    historical = protocol.validate_reference_index(
        config, manifest._internal_file(path, f"raw/inputs/{REFERENCE_INDEX}"))
    hardware = protocol.verify_hardware(
        config, manifest._internal_file(path, f"raw/inputs/{REFERENCE_PREFLIGHT}"), preflight)
    historical_hashes = historical["input_hashes"]
    if any(hashes[name] != historical_hashes.get(name) for name in workload_files):
        raise ValueError("SHA-256 de cargas o validaciones distinto del índice histórico")
    generator_hash = historical_hashes["infraestructura/inference_workload.py"]
    for length in lengths:
        prefix = f"workload/{job['model']['key']}/{length}"
        inference_evidence._verify_profile(
            path, prefix, length, job["model"], historical["protocol"], generator_hash, hashes)
        metadata = _read(path, f"raw/inputs/{prefix}/metadata.json")
        reader = _read(path, f"raw/inputs/{prefix}/reader-validation.json")
        for phase in PHASES:
            rows = [row for row in metadata["rows"] if row["phase"] == phase]
            samples = reader["phases"][phase]["samples"]
            if [(row["id"], row["prompt_sha256"]) for row in rows] != [
                    (sample["id"], sample["prompt_sha256"]) for sample in samples]:
                raise ValueError(f"{prefix}/{phase}: identidad del lector incompatible")
    return hashes, hardware


def _job_identity(run, ref, expected, campaign):
    request = run["requested_config"]
    model = expected["model"]
    if (run["kind"] != "inference-serving" or run["schema_version"] != "2.1"
            or run["status"] not in ("complete", "failed") or ref["status"] != run["status"]
            or ref["job"] != expected["name"] or ref["group"] != expected["group"]
            or not _same_json(ref["block"], expected["block"]) or ref["model_key"] != model["key"]
            or request["campaign_id"] != campaign["campaign_id"]
            or not _same_json(request["protocol"], campaign["protocol"])
            or not _same_json(request["job"], expected)
            or run["model"] != {"id": model["id"], "revision": model["revision"],
                                "tokenizer": {"id": model["tokenizer_id"],
                                              "revision": model["tokenizer_revision"]}}):
        raise ValueError("identidad de la corrida contradice el cierre o el plan")
    if (run["git"]["dirty"] is not False or run["git"]["commit"] != campaign["git"]["commit"]
            or run["policy"]["strict"] is not True
            or run["status"] == "complete" and (run["errors"] or ref["errors"])):
        raise ValueError("commit, política o errores incompatibles con el cierre de la corrida")


def _assessment_cells(assessment, job, complete):
    cells = assessment["cells"]
    expected = [point["rate_rps"] for point in job["cells"]]
    rates = [cell["rate_rps"] for cell in cells]
    if (not _same_json(rates, expected[:len(rates)])
            or any(cell["cell"] != protocol.point_name(cell["rate_rps"]) for cell in cells)):
        raise ValueError("coordenadas, orden o nombres de puntos incompatibles con el plan")
    if complete and (assessment["valid"] is not True or assessment["errors"] or rates != expected
                     or any(cell["status"] != "valid" for cell in cells)):
        raise ValueError("corrida COMPLETE con matriz de puntos incompleta o inválida")
    return cells


def _observed_run(path, run, job, config, campaign_probes):
    preflight = _read(path, "raw/preflight.json")
    _preflight_probes(preflight)
    observed, _, comparable = inference._observed_topology(
        path, run, job, preflight, config, campaign_probes)
    if not comparable:
        raise ValueError("identidad observada difiere del entorno fijado para la campaña")
    ranks = _read(path, "raw/startup.json")["ranks"]
    indexed = [_read(path, participant["evidence_path"]) for participant in run["participants"]]
    if not _same_json(sorted(ranks, key=lambda rank: rank["rank"]),
                      sorted(indexed, key=lambda rank: rank["rank"])):
        raise ValueError("ranks de arranque distintos de la evidencia de participantes")
    return observed, ranks


def _point_telemetry(path, ranks, start, end):
    evidence = []
    for rank in ranks:
        node_id = rank["node_id"]
        source = manifest._internal_file(path, f"raw/nodes/{node_id}/logs/telemetry.jsonl")
        samples = inference_diag_quality.telemetry_window(source, start, end, rank)
        evidence.append({
            **{key: rank[key] for key in ("node_id", "hostname", "gpu_uuid", "rank")},
            "sha256": manifest.sha256_file(source),
            **inference_telemetry_report.summarize(samples, start.timestamp(), end.timestamp()),
        })
    return evidence


def _point(path, job, config, cell, ranks, run_start, previous_point):
    rate, name = cell["rate_rps"], cell["cell"]
    start = _timestamp(cell["measurement_started_at"])
    end = _timestamp(cell["measurement_ended_at"])
    phases, source_hashes = {}, {}
    for phase in PHASES:
        source = manifest._internal_file(path, f"raw/nodes/pegasus/metrics/{name}/{phase}.json")
        phases[phase] = runner._phase_evidence(path, config, job, rate, phase)
        source_hashes[phase] = manifest.sha256_file(source)
    warmup, warm_window, warm_before, warm_after = phases["warmup"]
    measured, measured_window, measured_before, measured_after = phases["measured"]
    warm_start = _timestamp(warm_window["measurement_started_at"])
    warm_end = _timestamp(warm_window["measurement_ended_at"])
    if warm_start < run_start:
        raise ValueError("warmup comienza antes del inicio de la corrida")
    if previous_point is not None:
        previous_end, previous_drain = previous_point
        if warm_start < previous_end or warm_before["started_monotonic_s"] < previous_drain:
            raise ValueError("warmup comienza antes del cierre del punto anterior")
    if warm_end > start:
        raise ValueError("warmup se solapa con la medición del punto")
    for snapshot in (warm_before, warm_after, measured_before, measured_after):
        running, waiting = inference_diag_phases._gauges(
            *inference_diag_phases._parse(snapshot["body"])[:2])
        expected = job["model"]["key"]
        if (not inference_diag_phases._model_series(running, expected)
                or any(running.values()) or any(waiting.values())):
            raise ValueError("las métricas originales muestran un motor activo o sin identidad")
    identity = runner._check_health(warm_before, config, job["model"])
    if (identity != runner._check_health(measured_before, config, job["model"])
            or warm_after["finished_monotonic_s"] > measured_before["started_monotonic_s"]):
        raise ValueError("servidor o secuencia de salud difiere entre las fases del punto")
    for key in ("measurement_started_at", "measurement_ended_at"):
        if cell[key] != measured_window[key]:
            raise ValueError("intervalo de evaluación distinto del recibo medido")
    if any(not _same_json(cell[key], value) for key, value in measured.items() if key != "rate_rps"):
        raise ValueError("evaluación distinta del resumen reconstruido")
    if (end - start).total_seconds() < measured["last_terminal_s"]:
        raise ValueError("la ventana de telemetría no abarca la última respuesta del cliente")
    log_issues = inference_diag_quality._log_issues(
        path, name, start, end, inference_diag_quality._participants(ranks))
    if log_issues:
        raise ValueError(" | ".join(log_issues))
    point = {
        "model_key": job["model"]["key"], "group": job["group"], "block": job["block"],
        "rate_rps": rate, "cell": name, "run_id": path.name,
        "measurement_started_at": cell["measurement_started_at"],
        "measurement_ended_at": cell["measurement_ended_at"],
        "summary": measured, "warmup": warmup, "listener_identity": list(identity),
        "receipts_sha256": source_hashes,
        "telemetry": _point_telemetry(path, ranks, start, end),
        "warmup_started_at": warm_window["measurement_started_at"],
        "warmup_ended_at": warm_window["measurement_ended_at"],
        "warmup_telemetry": _point_telemetry(path, ranks, warm_start, warm_end),
    }
    return point, (end, measured_after["finished_monotonic_s"])


def _cleanup_issues(path, job, campaign_id):
    cleanup = _read(path, "raw/cleanup.json")
    names = [item["name"] for item in cleanup]
    if len(set(names)) != len(names):
        return ["recibos de limpieza duplicados"]
    expected = {f"ipre-{campaign_id[-8:]}-{job['name']}-{node}"
                for node in inference.TOPOLOGIES[job["group"]]}
    if (not expected.issubset(names)
            or any(item["status"] != "confirmed_absent" or item.get("errors") for item in cleanup)):
        return ["limpieza de los recursos de la corrida sin confirmar"]
    return []


def _verified_run(path, run, job, config, campaign_probes, hashes):
    complete_run = run["status"] == "complete"
    name = "metrics/assessment.json"
    if not (path / name).exists() and not complete_run:
        return [], [], {}
    assessment = _read(path, name)
    cells = _assessment_cells(assessment, job, complete_run)
    if not cells and not complete_run:
        return [], [], {}

    runner.require_tokenization(path, config)
    observed, ranks = _observed_run(path, run, job, config, campaign_probes)
    cleanup_issues = _cleanup_issues(path, job, run["requested_config"]["campaign_id"])
    if cleanup_issues and complete_run:
        raise ValueError(" | ".join(cleanup_issues))
    verified, excluded = [], []
    run_start, run_end = (_timestamp(run[key]) for key in ("started_at", "ended_at"))
    previous_point, identity = None, None
    for cell in cells:
        try:
            if cell["status"] != "valid":
                raise ValueError(f"estado del punto registrado como {cell['status']}")
            if (_timestamp(cell["measurement_started_at"]) < run_start
                    or _timestamp(cell["measurement_ended_at"]) > run_end):
                raise ValueError("la ventana de medición queda fuera de los límites de la corrida")
            runner.require_point_quality(path, cell["cell"])
            point, closure = _point(path, job, config, cell, ranks, run_start, previous_point)
            if identity is not None and point["listener_identity"] != identity:
                raise ValueError("el proceso del servidor cambió entre puntos de la corrida")
            identity = point["listener_identity"]
            previous_point = closure
        except EVIDENCE_ERRORS as exc:
            excluded.append({"job": job["name"], "rate_rps": cell["rate_rps"], "reason": str(exc)})
        else:
            verified.append(point)
    return verified, excluded, {"observed": observed, "ranks": ranks, "input_hashes": hashes,
                                "cleanup_issues": cleanup_issues}


def _seconds(value, field):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field}: segundos inválidos")
    return value


def _run_occupation(path, ref, run_start, run_end, *, allow_unknown=False):
    name = "raw/occupation.json"
    indexed = ref.get("occupation")
    if indexed is None and not os.path.lexists(path / name) and (ref["status"] == "failed" or allow_unknown):
        return None, "Falta raw/occupation.json. Retirada sin confirmar."
    try:
        receipt = runner.require_occupation(path)
    except (json.JSONDecodeError, UnicodeError):
        if not allow_unknown or indexed is not None:
            raise
        return None, "raw/occupation.json ilegible. Retirada sin confirmar."
    if "occupation" in ref and not _same_json(receipt, indexed):
        raise ValueError("la ocupación archivada difiere del índice")
    start, observed_until = (_timestamp(receipt[key]) for key in ("started_at", "observed_until_at"))
    for field in ("started_monotonic_s", "observed_until_monotonic_s", "duration_seconds"):
        _seconds(receipt[field], field)
    if not run_start <= start <= observed_until <= run_end:
        raise ValueError("cronología de ocupación incompatible con el subrun")
    return receipt, None


def _campaign_timing(campaign, config):
    start, end = (_timestamp(campaign[key]) for key in ("started_at", "ended_at"))
    mono_start, mono_end, duration = (
        _seconds(campaign[key], key) for key in
        ("started_monotonic_s", "ended_monotonic_s", "duration_seconds")
    )
    limit = campaign["effective_session_limit_seconds"]
    measurement = config["measurement"]
    if type(limit) is not int or not measurement["cleanup_reserve_seconds"] < limit <= measurement["session_limit_seconds"]:
        raise ValueError("límite efectivo de sesión fuera del protocolo")
    deadline = mono_start + limit
    session_exceeded = mono_end > deadline
    if start > end or duration != mono_end - mono_start or campaign["session_limit_exceeded"] is not session_exceeded:
        raise ValueError("cronología o exceso de sesión contradictorios")
    return start, end, mono_start, mono_end, deadline


def _receipt_json(directory, record, kind, expected_path):
    if record[f"{kind}_path"] != expected_path:
        raise ValueError(f"ruta de {kind} incompatible con el job")
    source = manifest._internal_file(directory, expected_path)
    if manifest.sha256_file(source) != record[f"{kind}_sha256"]:
        raise ValueError(f"SHA-256 de {kind} incompatible")
    return inference.read_json(source)


def _attempt_matrix(directory, campaign, planned, timing):
    attempts = campaign["attempts"]
    attempted_jobs = [attempt["job"] for attempt in attempts]
    if attempted_jobs != planned[:len(attempts)] or campaign["pending_jobs"] != planned[len(attempts):]:
        raise ValueError("intentos o pendientes incompatibles con el orden del plan")
    start, end, previous_request, _, deadline = timing
    sealed, indexed = [], []
    seen = set()
    for index, attempt in enumerate(attempts):
        job, status, run_id = (attempt[key] for key in ("job", "status", "run_id"))
        if (status not in ("sealed", "open", "not_started", "verification_pending")
                or status != "sealed" and (campaign["status"] != "partial" or index != len(attempts) - 1)):
            raise ValueError("estado u orden de intentos incompatible con el plan")
        manifest._validate_component(run_id, "run_id")
        if run_id != Path(attempt["run_dir"]).name or run_id in seen:
            raise ValueError("identidad de intento duplicada o incompatible con su ruta")
        seen.add(run_id)
        intent = _receipt_json(directory, attempt, "intent", f"attempts/{job}.json")
        if intent.get("schema_version") != 1 or intent["campaign_id"] != campaign["campaign_id"] or any(
                intent[key] != attempt[key] for key in ("job", "run_id", "run_dir")):
            raise ValueError("intención incompatible con el índice de la campaña")
        requested_at = _timestamp(intent["requested_at"])
        requested_mono = _seconds(intent["requested_monotonic_s"], "solicitud de subrun")
        if not start <= requested_at <= end or not previous_request <= requested_mono <= deadline:
            raise ValueError("solicitud de subrun fuera de la cronología de campaña")
        previous_request = requested_mono
        state_fields = {"state_path", "state_sha256"}
        has_state = state_fields <= attempt.keys()
        if has_state:
            state = _receipt_json(directory, attempt, "state", f"attempts/{job}-state.json")
            if not _same_json(state, {key: value for key, value in attempt.items() if key not in state_fields}):
                raise ValueError("estado archivado del intento contradice el índice")
        elif (state_fields & attempt.keys() or campaign["status"] == "complete"
              or index != len(attempts) - 1 or not campaign["errors"]):
            raise ValueError("estado del intento sin cierre verificable")
        if status == "sealed":
            sealed.append((job, run_id, attempt["manifest_sha256"]))
        indexed.append((attempt, intent, has_state))

    refs = campaign["runs"]
    if [(ref["job"], Path(ref["run_dir"]).name, ref["manifest_sha256"]) for ref in refs] != sealed:
        raise ValueError("intentos sellados contradicen el índice de corridas")
    return indexed


def _inspect_unindexed_attempt(directory, attempt, intent, job, campaign, campaign_end):
    candidate = directory.parent.parent / "inference-serving" / attempt["run_id"]
    status = attempt["status"]
    if status == "not_started":
        if os.path.lexists(candidate) or attempt.get("occupation") is not None:
            raise ValueError("intento marcado no iniciado con directorio local u ocupación")
        return None, None, status, "el directorio reservado no se creó"
    if not os.path.lexists(candidate):
        return None, None, status, "el directorio del intento no está disponible"
    path = _run_path(directory, attempt["run_dir"])
    source = path / "manifest.json"
    if not os.path.lexists(source):
        return path, None, status, "falta el manifiesto del intento"
    if source.is_symlink() or not source.is_file():
        raise ValueError("manifiesto local del intento no canónico")
    digest = attempt.get("manifest_sha256")
    if digest is not None and manifest.sha256_file(source) != digest:
        raise ValueError("SHA-256 del manifiesto del intento incompatible")
    try:
        run = manifest.load_run(path)
    except manifest.ManifestError as exc:
        if not isinstance(exc.__cause__, (OSError, json.JSONDecodeError)):
            raise
        return path, None, status, f"manifiesto o contexto del intento ilegible: {exc}"
    except UnicodeError as exc:
        return path, None, status, f"manifiesto o contexto del intento ilegible: {exc}"

    terminal = run["status"] in ("complete", "failed")
    if terminal:
        ref = {"job": job["name"], "group": job["group"], "block": job["block"],
               "model_key": job["model"]["key"], "status": run["status"], "errors": run["errors"]}
        _job_identity(run, ref, job, campaign)
    else:
        request = {"job": job, "campaign_id": campaign["campaign_id"], "protocol": campaign["protocol"]}
        if (status != "open" or run["kind"] != "inference-serving"
                or not _same_json(run["requested_config"], request)
                or run["git"]["commit"] != campaign["git"]["commit"] or run["git"]["dirty"] is not False):
            raise ValueError("manifiesto abierto contradice la identidad del intento")
    started = _timestamp(run["started_at"])
    ended = _timestamp(run["ended_at"]) if terminal else started
    if (run["run_id"] != attempt["run_id"]
            or not _timestamp(intent["requested_at"]) <= started <= ended <= campaign_end):
        raise ValueError("identidad o cronología del manifiesto contradice el intento")
    if terminal:
        return path, run, "verification_pending", "subrun sellado con verificación pendiente"
    return path, run, status, "corrida abierta excluida de la matriz de puntos"


def _pending_occupation(path, run, attempt, start, end, allow_unknown):
    name = "raw/occupation.json"
    indexed = attempt.get("occupation")
    if path is None or not os.path.lexists(path / name):
        if indexed is not None:
            raise ValueError("falta el recibo de ocupación indexado del intento")
        return None, "Falta raw/occupation.json. Retirada sin confirmar."
    source = manifest._internal_file(path, name)
    if run is not None and run["status"] in ("complete", "failed"):
        records = [item for item in run["artifacts"] if item["path"] == name]
        if (len(records) != 1 or source.stat().st_size != records[0]["size_bytes"]
                or manifest.sha256_file(source) != records[0]["sha256"]):
            raise ValueError("SHA-256 o índice del recibo de ocupación del intento incompatible")
    if indexed is not None:
        return _run_occupation(path, attempt, start, end)
    if not allow_unknown:
        raise ValueError("recibo de ocupación del intento sin correspondencia en el índice")
    try:
        runner.require_occupation(path)
    except (json.JSONDecodeError, UnicodeError):
        return None, "raw/occupation.json ilegible. Retirada sin confirmar."
    return None, "Recibo de ocupación del intento sin verificar. Retirada sin confirmar."


def _campaign_occupation(campaign, timing, indexed, run_occupations):
    start, end, mono_start, mono_end, deadline = timing
    previous_end, previous_mono = start, mono_start
    observed_at, observed_mono = campaign["started_at"], mono_start
    confirmed = True
    for attempt, intent, _ in indexed:
        status, job = attempt["status"], attempt["job"]
        if (_timestamp(intent["requested_at"]) < previous_end
                or intent["requested_monotonic_s"] < previous_mono):
            raise ValueError(f"{job}: intención anterior al fin de la ocupación previa")
        if status == "not_started":
            continue
        receipt = run_occupations.get(job)
        if receipt is None:
            confirmed = False
            continue
        begin, finish = (_timestamp(receipt[key]) for key in ("started_at", "observed_until_at"))
        observed = receipt["observed_until_monotonic_s"]
        if (not _timestamp(intent["requested_at"]) <= begin <= finish <= end
                or not intent["requested_monotonic_s"] <= receipt["started_monotonic_s"] <= observed <= mono_end
                or receipt["limit_exceeded"] is not (observed > deadline)):
            raise ValueError(f"{job}: ocupación o exceso fuera de la cronología de campaña o intento")
        if not receipt["removal_confirmed"]:
            confirmed = False
            continue
        previous_end, previous_mono = finish, observed
        observed_at, observed_mono = receipt["ended_at"], previous_mono
    if not indexed or not confirmed:
        observed_at, observed_mono = campaign["ended_at"], mono_end
    expected = {
        "started_at": campaign["started_at"], "started_monotonic_s": mono_start,
        "ended_at": observed_at if confirmed else None,
        "ended_monotonic_s": observed_mono if confirmed else None,
        "observed_until_at": observed_at, "observed_until_monotonic_s": observed_mono,
        "duration_seconds": observed_mono - mono_start,
        "removal_confirmed": confirmed, "limit_exceeded": observed_mono > deadline,
    }
    if campaign["occupation"] != expected:
        raise ValueError("ocupación de campaña contradice los recibos de retirada")
    if campaign["status"] == "complete" and (
            not confirmed or expected["limit_exceeded"] or campaign["session_limit_exceeded"]):
        raise ValueError("COMPLETE requiere retirada confirmada y cumplimiento del límite efectivo")
    return expected


def _check_plan(plan, config, cluster_path, hashes):
    if manifest.sha256_file(cluster_path) != hashes.get("cluster"):
        raise ValueError("inventario archivado difiere de input_hashes")
    expected = list(protocol.planned_jobs(config, inference.read_json(cluster_path)))
    if not _same_json(plan["jobs"], expected):
        raise ValueError("comandos o recursos del plan difieren del inventario y protocolo archivados")


def build_report(directory) -> dict:
    """Verify the campaign closure and derive capacity from comparable terminal receipts."""
    directory = Path(directory).resolve()
    markers = [name for name in ("COMPLETE", "PARTIAL")
               if os.path.lexists(directory / name)]
    if len(markers) != 1:
        raise ValueError("campaña abierta o cierre ambiguo")
    marker_path = directory / markers[0]
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError("marcador de campaña inválido")
    marker = inference.read_json(marker_path)
    files = {}
    for name in ("campaign", "plan", "preflight"):
        source = manifest._internal_file(directory, f"{name}.json")
        if manifest.sha256_file(source) != marker[f"{name}_sha256"]:
            raise ValueError(f"SHA-256 de cierre incompatible: {name}.json")
        files[name] = inference.read_json(source)

    campaign, plan = files["campaign"], files["plan"]
    status, config = campaign["status"], campaign["protocol"]
    if {"complete": "COMPLETE", "partial": "PARTIAL"}.get(status) != markers[0]:
        raise ValueError("marcador y estado de campaña incompatibles")
    if config["protocol_id"] != protocol.PROTOCOL:
        raise ValueError("protocolo de serving desconocido")
    protocol.validate_protocol(config)
    commit = campaign["git"]["commit"]
    if campaign["git"]["dirty"] is not False or not inference.SHA40.fullmatch(commit or ""):
        raise ValueError("la campaña requiere un commit limpio e identificable")

    expected_jobs = list(protocol.jobs(config))
    _layout(plan, config, expected_jobs)
    expected_names = [job["name"] for job in expected_jobs]
    refs, pending = campaign["runs"], campaign["pending_jobs"]
    complete = status == "complete"
    if complete and (len(refs) != len(expected_jobs) or pending or campaign["errors"]
                     or any(ref["status"] != "complete" for ref in refs)):
        raise ValueError("COMPLETE requiere 126 puntos y todas las corridas cerradas sin errores")
    timing = _campaign_timing(campaign, config)
    indexed = _attempt_matrix(directory, campaign, expected_names, timing)

    campaign_probes = {}
    if refs or complete or not files["preflight"]["errors"]:
        campaign_probes = _preflight_probes(files["preflight"])
    hashes = campaign["input_hashes"]
    reference = config["reference"]
    pins = {**reference["historical_jsonl_sha256"],
            REFERENCE_INDEX: reference["historical_campaign_sha256"],
            REFERENCE_PREFLIGHT: reference["historical_preflight_sha256"]}
    if (any(not isinstance(digest, str) or not SHA256.fullmatch(digest) for digest in hashes.values())
            or any(hashes.get(name) != digest for name, digest in pins.items())):
        raise ValueError("input_hashes inválidos o distintos de los pins históricos")
    paths, run_ids, manifest_hashes, receipts = set(), set(), set(), set()
    cells, excluded, runs = [], [], []
    historical = manifest._internal_file(directory, REFERENCE_PREFLIGHT)
    hardware_issues = []
    try:
        hardware = protocol.verify_hardware(config, historical, files["preflight"] if campaign_probes else None)
        if campaign_probes and campaign["hardware"] != hardware:
            raise ValueError("hardware archivado contradice la campaña")
    except EVIDENCE_ERRORS as exc:
        if complete:
            raise
        hardware_issues.append(str(exc))
    plan_commands_verified = False
    previous_run_end = None
    if (directory / "cluster.json").exists():
        _check_plan(plan, config, manifest._internal_file(directory, "cluster.json"), hashes)
        plan_commands_verified = True
    run_occupations = {}
    attempts = []
    for job, (attempt, intent, has_state) in zip(expected_jobs, indexed):
        requested_at = _timestamp(intent["requested_at"])
        if previous_run_end is not None and requested_at < previous_run_end:
            raise ValueError(f"{job['name']}: intención anterior al cierre del subrun previo")
        evidence = {key: attempt[key] for key in attempt if key != "run_dir"}
        evidence.update(requested_at=intent["requested_at"], state_verified=has_state)
        attempts.append(evidence)
        allow_unknown = (not complete and campaign["occupation"]["removal_confirmed"] is False
                         and attempt.get("occupation") is None)
        if attempt["status"] != "sealed":
            path, local_run, local_status, issue = _inspect_unindexed_attempt(
                directory, attempt, intent, job, campaign, timing[1])
            evidence.update(status=local_status, diagnostic=issue)
            if local_status != attempt["status"]:
                evidence["indexed_status"] = attempt["status"]
            if local_run is not None:
                evidence.update(started_at=local_run["started_at"], ended_at=local_run["ended_at"],
                                manifest_status=local_run["status"])
            if local_status != "not_started":
                bounds = (requested_at, timing[1])
                if local_run is not None:
                    bounds = (_timestamp(local_run["started_at"]),
                              _timestamp(local_run["ended_at"]) if local_run["ended_at"] else timing[1])
                occupation, diagnostic = _pending_occupation(path, local_run, attempt, *bounds, allow_unknown)
                run_occupations[job["name"]] = occupation
                if diagnostic:
                    evidence["occupation_diagnostic"] = diagnostic
            continue
        ref = refs[len(runs)]
        path = _run_path(directory, ref["run_dir"])
        digest = manifest.sha256_file(path / "manifest.json")
        run = manifest.load_run(path)
        if (path in paths or run["run_id"] in run_ids or digest in manifest_hashes
                or path.name != run["run_id"] or digest != ref["manifest_sha256"]):
            raise ValueError("corrida duplicada o SHA-256 del manifiesto incompatible")
        paths.add(path)
        run_ids.add(run["run_id"])
        manifest_hashes.add(digest)
        manifest.verify_artifacts(path, run)
        _job_identity(run, ref, job, campaign)
        run_start, run_end = (_timestamp(run[key]) for key in ("started_at", "ended_at"))
        if (not requested_at <= run_start <= run_end <= timing[1]
                or (previous_run_end is not None and run_start < previous_run_end)):
            raise ValueError(f"{job['name']}: cronología de subruns solapada o invertida")
        previous_run_end = run_end
        cluster_path = path / "raw/inputs/cluster"
        if cluster_path.is_file():
            _check_plan(plan, config, manifest._internal_file(path, "raw/inputs/cluster"), hashes)
            plan_commands_verified = True
        information = {key: ref[key] for key in ("job", "model_key", "group", "block", "status",
                                                "manifest_sha256", "errors")}
        information.update(run_id=path.name, started_at=run["started_at"], ended_at=run["ended_at"])
        occupation, occupation_diagnostic = _run_occupation(
            path, ref, run_start, run_end, allow_unknown=allow_unknown and run["status"] == "failed")
        if "occupation" in attempt and not _same_json(attempt["occupation"], occupation):
            raise ValueError(f"{job['name']}: ocupación archivada contradice el estado del intento")
        run_occupations[job["name"]] = occupation
        information["occupation"] = occupation
        if occupation_diagnostic:
            information["occupation_diagnostic"] = occupation_diagnostic
        assessment = path / "metrics/assessment.json"
        try:
            if assessment.is_file() or run["status"] == "complete":
                input_hashes, hardware = _archived_inputs(path, run, config, job, hashes, files["preflight"])
                information["historical_hardware"] = hardware
                verified, rejected, evidence = _verified_run(
                    path, run, job, config, campaign_probes, input_hashes)
                for point in verified:
                    receipt_hashes = set(point["receipts_sha256"].values())
                    if receipts & receipt_hashes:
                        raise ValueError("recibo del cliente reutilizado en dos puntos")
                    receipts.update(receipt_hashes)
                    cells.append(point)
                excluded.extend(rejected)
                information.update(evidence)
        except EVIDENCE_ERRORS as exc:
            if complete:
                raise ValueError(f"{job['name']}: {exc}") from exc
            excluded.append({"job": job["name"], "reason": str(exc)})
        runs.append(information)

    campaign_occupation = _campaign_occupation(campaign, timing, indexed, run_occupations)
    if complete and (len(cells) != TOTAL_POINTS or excluded or not plan_commands_verified):
        raise ValueError("COMPLETE requiere exactamente 126 puntos verificables")
    measured = {(cell["model_key"], cell["group"], cell["block"], cell["rate_rps"]) for cell in cells}
    missing = [
        {"job": job["name"], "model_key": job["model"]["key"], "group": job["group"],
         "block": job["block"], "rate_rps": point["rate_rps"]}
        for job in expected_jobs for point in job["cells"]
        if (job["model"]["key"], job["group"], job["block"], point["rate_rps"]) not in measured
    ]
    return {
        "schema_version": 1, "campaign_id": campaign["campaign_id"], "status": status,
        "protocol_id": protocol.PROTOCOL, "git": campaign["git"],
        "closure_sha256": marker, "protocol": config, "input_hashes": hashes,
        "plan_commands_verified": plan_commands_verified,
        "campaign_errors": campaign["errors"], "hardware_issues": hardware_issues,
        "pending_jobs": pending, "attempts": attempts,
        "started_at": campaign["started_at"], "ended_at": campaign["ended_at"],
        "effective_session_limit_seconds": campaign["effective_session_limit_seconds"],
        "duration_seconds": campaign["duration_seconds"],
        "session_limit_exceeded": campaign["session_limit_exceeded"], "occupation": campaign_occupation,
        "post_occupation_seconds": (campaign["ended_monotonic_s"] - campaign_occupation["ended_monotonic_s"]
                                    if campaign_occupation["removal_confirmed"] else None),
        "runs": runs, "cells": cells, "excluded_cells": excluded, "missing_cells": missing,
        "frontiers": metrics.capacity_frontiers(cells, config),
        "analyzers_sha256": {
            Path(module.__file__).name: manifest.sha256_file(Path(module.__file__))
            for module in (client, metrics, protocol, manifest, inference, inference_evidence,
                           inference_diag_phases, inference_diag_quality, inference_telemetry_report, runner)
        } | {Path(__file__).name: manifest.sha256_file(Path(__file__))},
    }


def _decimal(value):
    return "—" if value is None else f"{value:.4g}"


def _latency(summary, key):
    return " / ".join(_decimal(summary[key][f"p{p}_s"]) for p in (50, 95))


def _markdown(data):
    config = data["protocol"]
    capacity = config["capacity"]
    lines = [
        f"# Capacidad de serving · {data['campaign_id']}", "",
        f"Estado **{data['status'].upper()}**, commit `{data['git']['commit']}`. "
        f"Puntos verificados {len(data['cells'])}/{TOTAL_POINTS}.",
        "",
        f"Llegadas regulares a {', '.join(map(str, capacity['rates_rps']))} solicitudes/s. "
        f"La mezcla alterna entradas de 512 y 2048 tokens y solicita "
        f"{config['workload']['output_tokens']} tokens de salida por solicitud.",
        "",
        f"El SLO conjunto exige que al menos {capacity['compliance']:.0%} de todas las solicitudes "
        f"ofrecidas terminen con TTFT ≤ {capacity['ttft_s']} s y latencia total ≤ "
        f"{capacity['e2e_s']} s. Las latencias parten de la llegada programada. Los percentiles "
        "describen solicitudes completadas. Las tasas de finalización y cumplimiento incluyen "
        "el drenaje en su denominador.",
        "",
        "## Frontera observada", "",
        "| Modelo | Grupo | Mayor tasa 3/3 | Primera tasa superior que falla | Estado | Limitaciones |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    frontier_status = {"left_censored": "Censura izquierda", "right_censored": "Censura derecha",
                       "bracketed": "Acotada", "indeterminate": "Indeterminada"}
    for result in data["frontiers"]:
        maximum = _decimal(result["highest_passing_rate_rps"])
        failure = _decimal(result["first_higher_failing_rate_rps"])
        reason = ", ".join(result["limitations"]) or "—"
        status = frontier_status[result["status"]]
        lines.append(f"| {result['model_key']} | {result['group']} | {maximum} | {failure} | "
                     f"{status} | {reason} |")
    lines += [
        "",
        "Las tasas de la frontera están en solicitudes/s y corresponden a la ventana ensayada. "
        "Los bloques ausentes y las inversiones de cumplimiento dejan la frontera indeterminada. "
        "Cuando ningún punto pasa 3/3, el fallo indicado corresponde a la primera tasa con "
        "un fallo observado. El JSON conserva los bloques observados y ausentes por tasa.",
        "",
        "## Puntos y desglose por longitud", "",
        f"Los cuartos muestran solicitudes pendientes al final de cada tramo de la oferta. "
        f"El límite de p95 del retraso de envío es {capacity['dispatch_lag_limit_s']} s. "
        f"El plazo por solicitud es {capacity['request_timeout_s']} s.",
        "",
        "| Modelo | Grupo | Bloque | Entrada | Ofrecida / completada / conforme (sol/s) | "
        "TTFT p50 / p95 (s) | Total p50 / p95 (s) | SLO conjunto | Pendientes Q1–Q4 | "
        "En vuelo Q1–Q4 | Limitaciones |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in data["cells"]:
        summary = cell["summary"]
        for length, row in [("Mezcla", summary), *summary["by_input_tokens"].items()]:
            rates = " / ".join(_decimal(row[key]) for key in ("offered_rps", "completed_rps", "compliant_rps"))
            counts = f"{row['compliant']}/{row['offered']} ({row['compliance_fraction']:.1%})"
            result = {"pass": "Cumple", "fail": "Falla"}[row["slo_status"]]
            pending = " / ".join(str(q["pending"]) for q in row["queue_quarters"])
            in_flight = " / ".join(str(q["in_flight"]) for q in row["queue_quarters"])
            limitations = ", ".join(row["limitations"]) or "—"
            lines.append(f"| {cell['model_key']} | {cell['group']} | {cell['block']} | {length} | "
                         f"{rates} | {_latency(row, 'ttft')} | {_latency(row, 'e2e')} | {counts} · "
                         f"{result} | {pending} | {in_flight} | {limitations} |")
    if data["missing_cells"]:
        lines += ["", "## Evidencia parcial", "",
                  f"Puntos sin evidencia comparable {len(data['missing_cells'])}. "
                  "El JSON detalla las coordenadas, los motivos disponibles y las corridas pendientes."]
        if data["pending_jobs"]:
            lines += ["", f"Corridas pendientes · {', '.join(data['pending_jobs'])}."]
        for attempt in data["attempts"]:
            if "diagnostic" in attempt:
                lines += ["", f"`{attempt['job']}` · {attempt['diagnostic']}."]
        for issue in data["hardware_issues"]:
            lines += ["", f"Hardware · {issue}."]
        if data["campaign_errors"]:
            lines += ["", *[f"- {error}" for error in data["campaign_errors"]]]
    occupation = data["occupation"]
    lines += ["", "## Tiempo de sesión y ocupación", "",
              f"Límite efectivo {data['effective_session_limit_seconds']} s. "
              f"Sesión {data['duration_seconds']:.3f} s. "
              f"Ocupación observada {occupation['duration_seconds']:.3f} s."]
    if data["session_limit_exceeded"]:
        lines.append("La sesión superó su límite efectivo.")
    if occupation["limit_exceeded"]:
        lines.append("La ocupación observada superó el límite efectivo.")
    if occupation["removal_confirmed"]:
        lines.append(f"Procesamiento posterior a la retirada {data['post_occupation_seconds']:.3f} s.")
    else:
        lines.append("La retirada de recursos quedó sin confirmar.")
    diagnostics = [f"- `{run['job']}` · {run['occupation_diagnostic']}"
                   for run in [*data["runs"], *data["attempts"]] if "occupation_diagnostic" in run]
    if diagnostics:
        lines.extend(["", *diagnostics])
    if not data["plan_commands_verified"]:
        lines += ["", "Los comandos se cotejaron con el protocolo. Falta un inventario archivado para contrastar el plan con el clúster."]
    return "\n".join(lines) + "\n"


def write_report(output_path, report) -> None:
    """Write a new JSON/Markdown pair and leave existing files intact."""
    output_path = Path(output_path)
    destinations = [output_path.with_suffix(ext) for ext in (".json", ".md")]
    for path in destinations:
        if {"raw", "datos"} & set(path.resolve().parts):
            raise ValueError("los derivados se escriben fuera de raw y datos")
    payloads = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", _markdown(report))
    destinations[0].parent.mkdir(parents=True, exist_ok=True)
    # Claim both destinations exclusively before writing. Roll back only our
    # own files if a destination already exists or either write fails.
    created = []
    try:
        with ExitStack() as stack:
            streams = []
            for path in destinations:
                stream = stack.enter_context(path.open("x", encoding="utf-8"))
                created.append(path)
                streams.append(stream)
            for stream, payload in zip(streams, payloads):
                stream.write(payload)
    except BaseException:
        for path in created:
            path.unlink()
        raise
