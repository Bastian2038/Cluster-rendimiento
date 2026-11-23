"""Operación autorizada del piloto y de la matriz de SFT completo."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from infraestructura import cluster, inference, manifest, sft_protocol as protocol
from infraestructura import sft_report, sft_runtime as runtime
from infraestructura.rutas import REPO_ROOT
from infraestructura.serving_capacity_runner import reservation


RESULTS_ROOT = REPO_ROOT / "resultados/raw"
CAMPAIGN_FILES = ("plan", "config", "cluster", "preflight", "campaign", "cleanup")
RANK_IDENTITY = ("node_id", "rank", "hostname", "gpu_uuid", "world_size", "phase", "run_id", "campaign_id")


def _hashes(files):
    return {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}


def _bytes_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def _read_inputs(config_path, config):
    source = {f"code/infraestructura/{name}.py": REPO_ROOT / "infraestructura" / f"{name}.py"
              for name in protocol.SOURCE_MODULES}
    paths = {**source, "requested_config": config_path, "cluster.json": REPO_ROOT / config["cluster_config"],
             "reference/preflight.json": REPO_ROOT / config["reference_preflight"]}
    paths[f"code/{config['cluster_config']}"] = paths["cluster.json"]
    files = {name: path.read_bytes() for name, path in paths.items()}
    files["corpus.jsonl"] = protocol.corpus_bytes()
    return paths, files


def _git_read(*args):
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=REPO_ROOT,
                          check=True, capture_output=True, timeout=10).stdout


def _verify_tracked(paths, files, commit):
    for name, path in paths.items():
        relative = path.resolve().relative_to(REPO_ROOT)
        if _git_read("show", f"{commit}:{relative.as_posix()}") != files[name]:
            raise ValueError(f"entrada distinta del commit fijado: {name}")


def _verify_unchanged(git, paths, hashes):
    current = manifest.capture_git(REPO_ROOT)
    if current["commit"] != git["commit"] or current["dirty"] is not False:
        raise ValueError("commit o estado Git cambió durante la sesión")
    if {name: manifest.sha256_file(path) for name, path in paths.items()} != {
        name: hashes[name] for name in paths
    }:
        raise ValueError("fuentes de la campaña cambiaron durante la sesión")


def _qualify(pilot_dir, results_root, hashes, limit, reserve, *, git_commit, pilot_report=None):
    directory = Path(pilot_dir).resolve()
    if directory.parent != results_root / "sft-campaign":
        raise ValueError("el piloto debe compartir el archivo de campañas de la matriz")
    pilot = sft_report.build_report(directory)
    groups = protocol.qualified_pilot_groups(pilot)
    estimate = pilot["estimated_matrix_seconds"]
    if (pilot["input_hashes"] != hashes or pilot["config_sha256"] != hashes["requested_config"]
            or pilot["cluster_sha256"] != hashes["cluster.json"]):
        raise ValueError("huellas de entradas, protocolo, inventario, modelo o corpus distintas del piloto")
    if type(estimate) not in (float, int) or not math.isfinite(estimate) or not 0 < estimate <= limit - reserve:
        raise ValueError("proyección de la matriz fuera del plazo operativo disponible")
    report_path = Path(pilot_report) if pilot_report is not None else Path(
        f"docs/resultados/sft-piloto-{pilot['campaign_id']}.json")
    try:
        relative = (REPO_ROOT / report_path).resolve().relative_to(REPO_ROOT).as_posix()
        if Path(relative).suffix != ".json":
            raise ValueError("el informe publicado debe ser un archivo JSON")
        published = _git_read("show", f"{git_commit}:{relative}")
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"falta un informe del piloto publicado en el commit de la matriz: {report_path}") from exc
    if json.dumps(json.loads(published), sort_keys=True, allow_nan=False) != json.dumps(
        pilot, sort_keys=True, allow_nan=False
    ):
        raise ValueError("el informe publicado difiere del piloto cerrado")
    reference = {"campaign_id": pilot["campaign_id"], "campaign_sha256": pilot["campaign_sha256"],
                 "git_commit": pilot["git"]["commit"], "estimated_matrix_seconds": estimate,
                 "published_report": {"path": relative, "sha256": hashlib.sha256(published).hexdigest(),
                                      "git_commit": git_commit}}
    return groups, reference, published


def _event(directory, event, **fields):
    record = {"event": event, "at": manifest.utc_now(), **fields}
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
    with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        print(line, flush=True)
    except BrokenPipeError:
        pass


def live_preflight(config) -> dict:
    """Consulta los cuatro hosts. El llamador debe tener autorización."""
    before = time.monotonic()
    try:
        protocol.validate_config(config)
        cluster_config = protocol.cluster_config(config)
        total = config["measurement"]["pilot_limit_seconds"]
        reserve = config["measurement"]["cleanup_reserve_seconds"]
        with reservation(cluster_config):
            return runtime.preflight(config, cluster_config, protocol.make_plan(config, "pilot")["jobs"],
                                     manifest.generate_run_id(), uuid.uuid4().hex,
                                     before + total - reserve, before + total)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        return {"status": "blocked", "nodes": [], "errors": [f"{type(exc).__name__}: {exc}"]}


def _node_errors(errors):
    return [f"{node}: {error}" for node, error in errors.items()]


def _worker_files(files):
    return {"config.json": files["requested_config"], "corpus.jsonl": files["corpus.jsonl"],
            **{name: data for name, data in files.items() if name.startswith("code/")}}


def _verify_archive(hashes, resolve):
    changed = [name for name, digest in hashes.items() if manifest.sha256_file(resolve(name)) != digest]
    if changed:
        raise ValueError(f"entradas archivadas cambiaron: {', '.join(changed)}")


def _seal_campaign(directory, campaign, started, complete):
    """Prepara el índice y el marcador antes de publicar un cierre exclusivo."""
    publication_path = directory / "publication.json"
    index_destination = directory / "campaign.json"
    if any(path.exists() for path in (index_destination, publication_path, directory / "COMPLETE", directory / "PARTIAL")):
        raise FileExistsError("la campaña ya tiene un índice o marcador de cierre")
    start_utc = datetime.fromisoformat(campaign["started_at"])
    errors = list(campaign["errors"])
    limit = campaign["limit_seconds"]

    def observe(*snapshots):
        elapsed = time.monotonic() - started
        utc = manifest.utc_now()
        observed_utc = datetime.fromisoformat(utc)
        wall_elapsed = (observed_utc - start_utc).total_seconds()
        causes = []
        if elapsed > limit or wall_elapsed > limit:
            causes.append("presupuesto total de campaña excedido al publicar el marcador")
        diverged = elapsed < 0 or wall_elapsed < 0 or abs(elapsed - wall_elapsed) > 2
        for prior_utc, prior_elapsed in snapshots:
            wall_delta = (observed_utc - datetime.fromisoformat(prior_utc)).total_seconds()
            mono_delta = elapsed - prior_elapsed
            diverged = diverged or wall_delta < 0 or mono_delta < 0 or abs(wall_delta - mono_delta) > 2
        if diverged:
            causes.append("discrepancia entre reloj UTC y monotónico al sellar la campaña")
        return utc, elapsed, causes

    def add_causes(causes):
        new = [cause for cause in dict.fromkeys(causes) if cause not in errors]
        errors.extend(new)
        return bool(new)

    static_hashes = {name + "_sha256": manifest.sha256_file(directory / (name + ".json"))
                     for name in CAMPAIGN_FILES if name != "campaign"}
    with tempfile.TemporaryDirectory(prefix=".sft-seal-", dir=directory.parent) as temporary:
        pending = Path(temporary)
        index_path = pending / "campaign.json"
        marker_path = pending / "marker.json"
        while True:
            utc, elapsed, causes = observe()
            add_causes(causes)
            status = "complete" if complete and not errors else "partial"
            index_snapshot = (utc, elapsed)
            inference.write_new(index_path, {**campaign, "status": status, "errors": errors,
                                            "ended_at": utc, "duration_seconds": elapsed})
            hashes = {**static_hashes, "campaign_sha256": manifest.sha256_file(index_path)}
            sealed_at, seal_seconds, causes = observe(index_snapshot)
            if add_causes(causes):
                index_path.unlink()
                continue
            inference.write_new(marker_path, {**hashes, "sealed_at": sealed_at, "duration_seconds": seal_seconds})
            os.link(index_path, index_destination)
            final_utc, final_seconds, causes = observe(index_snapshot, (sealed_at, seal_seconds))
            if not causes and final_seconds - seal_seconds > 0.01:
                marker_path.unlink()
                sealed_at, seal_seconds = final_utc, final_seconds
                _bytes_new(marker_path, (json.dumps({**hashes, "sealed_at": final_utc,
                    "duration_seconds": final_seconds}, sort_keys=True, allow_nan=False) + "\n").encode())
            marker_sha = manifest.sha256_file(marker_path)
            before_utc, before_elapsed, before_causes = observe(index_snapshot, (sealed_at, seal_seconds),
                                                                (final_utc, final_seconds))
            marker_destination = directory / status.upper()
            linked = False
            if not add_causes([*causes, *before_causes]):
                os.link(marker_path, marker_destination)
                observed_at, duration, after_causes = observe(index_snapshot, (sealed_at, seal_seconds),
                                                               (before_utc, before_elapsed))
                linked = True
                if not add_causes(after_causes):
                    inference.write_new(publication_path, {"campaign_id": campaign["campaign_id"], "status": status,
                        "marker_sha256": marker_sha, "observed_at": observed_at, "duration_seconds": duration})
                    return
            # El candidato queda sin recibo. Retirar solo los enlaces propios.
            if not index_destination.samefile(index_path) or (linked and not marker_destination.samefile(marker_path)):
                raise FileExistsError("el cierre provisional cambió antes de poder descartarlo")
            if linked:
                marker_destination.unlink()
            index_destination.unlink()
            index_path.unlink()
            marker_path.unlink()


def _participant(directory, node, phases, job, campaign_id, probe):
    for phase in phases:
        relative = f"raw/nodes/{node}/{phase}/identity.json"
        path = directory / relative
        if not path.exists():
            continue
        row = inference.read_json(path)
        expected = {"node_id": node, "rank": job["nodes"].index(node), "hostname": probe["hostname"],
                    "gpu_uuid": probe["gpu_uuid"], "world_size": len(job["nodes"]), "phase": phase,
                    "run_id": directory.name, "campaign_id": campaign_id}
        if any(type(row[key]) is not type(value) or row[key] != value for key, value in expected.items()):
            raise ValueError(f"identidad efectiva contradictoria: {node}/{phase}")
        return {**{key: row[key] for key in ("hostname", "gpu_uuid", "rank", "node_id")},
                "evidence_path": relative}
    return None


def _wait(nodes, ids, deadline):
    while True:
        states, errors = runtime.parallel(nodes, lambda node: runtime.state(node, ids[node["node_id"]], deadline))
        failed = any(state["Status"] != "running" and (
            state["Status"] != "exited" or state["ExitCode"] != 0 or state["OOMKilled"] or state["Error"]
        ) for state in states.values())
        if errors or failed or all(state["Status"] == "exited" for state in states.values()):
            return states, errors
        time.sleep(min(1, runtime.remaining(deadline, 15)))


def _cleanup_receipt(all_nodes, rows, campaign_id, run_id, phases, seconds):
    return {"campaign_id": campaign_id, "run_id": run_id, "removal_confirmed": all(
                row["status"] == "confirmed_absent" for row in rows),
            "removal_seconds": seconds, "nodes": rows, "phases": phases,
            "coverage": [node["node_id"] for node in all_nodes]}


def _phase(config, nodes, all_nodes, probes, workspace, directory, job, phase, campaign_id, owner,
           deadline, hard_deadline):
    phase_started = time.monotonic()
    ids, states, errors = {}, {}, []
    issued = False
    cleanup_elapsed = 0
    try:
        resources = runtime.check_ready(nodes, config, probes, deadline, job=job, phase=phase,
                                        workspace=workspace)
        inference.write_new(directory / "logs" / (phase + "-resources.json"), resources)
        # Reservar el máximo de create, verificación de etiquetas y start
        # antes de armar el watchdog que correrá dentro del contenedor.
        seconds = int(runtime.remaining(deadline, config["measurement"]["run_limit_seconds"] + 90)) - 80
        if seconds < 5:
            raise TimeoutError("plazo insuficiente para lanzar y retirar todos los ranks")
        commands = {}
        for rank, node in enumerate(nodes):
            name = node["node_id"]
            worker = protocol.worker_argv(config, job, rank, phase, config_path="/work/config.json",
                model_dir=Path("/models") / protocol.model_subdir(config), out=f"/work/out/{phase}",
                checkpoint_dir="/work/checkpoint", campaign_id=campaign_id, run_id=directory.name)
            commands[name] = runtime.phase_command(config, probes[name], workspace, campaign_id, owner,
                f"ipre-sft-{owner}-{phase}-{name}", worker, seconds, phase=phase)
        inference.write_new(directory / "logs" / (phase + "-commands.json"), commands)
        issued = True
        ids, failures = runtime.parallel(nodes, lambda node: runtime.create(
            node, commands[node["node_id"]], campaign_id, owner, deadline))
        errors.extend(_node_errors(failures))
        if not errors:
            inference.write_new(directory / "logs" / (phase + "-containers.json"), ids)
            _, failures = runtime.parallel(nodes, lambda node: runtime.start(node, ids[node["node_id"]], deadline))
            errors.extend(_node_errors(failures))
        if not errors:
            states, failures = _wait(nodes, ids, deadline)
            errors.extend(_node_errors(failures))
    except (Exception, KeyboardInterrupt) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        launch_to_terminal_seconds = time.monotonic() - phase_started
        if issued:
            cleanup_started = time.monotonic()
            active = [node for node in nodes if node["node_id"] in ids]
            try:
                logs, failures = runtime.parallel(active, lambda node: runtime.logs(
                    node, ids[node["node_id"]], hard_deadline)) if active else ({}, {})
                for name, log in logs.items():
                    inference.write_new(directory / "logs" / phase / (name + ".json"), log)
                errors.extend(_node_errors(failures))
            except (Exception, KeyboardInterrupt) as exc:
                errors.append(f"logs: {type(exc).__name__}: {exc}")
            finally:
                removal = runtime.cleanup(all_nodes, campaign_id, owner, hard_deadline)
                cleanup_elapsed = time.monotonic() - cleanup_started
        else:
            removal = runtime.cleanup(all_nodes, campaign_id, owner, hard_deadline)
    if not removal["removal_confirmed"]:
        errors.append(f"{phase}: retirada no confirmada en los cuatro nodos")
    collected, failures = runtime.parallel(nodes, lambda node: runtime.collect(node, workspace, phase, hard_deadline))
    errors.extend(_node_errors(failures))
    for node, payload in collected.items():
        for name, data in payload.items():
            _bytes_new(directory / "raw" / "nodes" / node / phase / name, data)
    inference.write_new(directory / "logs" / (phase + "-terminal.json"),
                        {"states": states, "removal": removal, "errors": errors,
                         "launch_to_terminal_seconds": launch_to_terminal_seconds})
    return states, errors, removal, cleanup_elapsed


def _phase_outcome(directory, phase, states, nodes):
    failed = set()
    signaled = set()
    for node in nodes:
        state = states.get(node, {})
        if state.get("Status") == "running":
            signaled.add(node)
        elif (state.get("Status") != "exited" or state.get("ExitCode") != 0
                or state.get("OOMKilled") or state.get("Error")):
            failed.add(node)
    oom = set()
    for node in nodes:
        path = directory / "raw" / "nodes" / node / phase / "failure.json"
        if path.exists():
            reason = inference.read_json(path).get("type")
            if reason != "cuda_oom":
                raise ValueError(f"{phase}: error de proceso en {node}: {reason}")
            oom.add(node)
    if oom and failed and failed <= oom and not any(
        state.get("OOMKilled") or state.get("Error") for state in states.values()
    ):
        return "oom"
    if failed or signaled or oom:
        raise ValueError(f"{phase}: terminación inesperada, ranks fallidos={sorted(failed)}")
    return "ok"


def _require_oom_group(directory, phase, job, campaign_id, probes):
    roster, groups = [], []
    for node in job["nodes"]:
        evidence = directory / "raw" / "nodes" / node / phase
        participant = _participant(directory, node, (phase,), job, campaign_id, probes[node])
        if participant is None:
            raise ValueError(f"{phase}: falta identidad del rank {node} para atribuir el OOM")
        identity = inference.read_json(evidence / "identity.json")
        roster.append({key: identity[key] for key in RANK_IDENTITY})
        groups.append(inference.read_json(evidence / "group.json"))
    for identity, group in zip(roster, groups):
        expected = {**identity, "backend": "nccl", "participants": roster}
        if any(json.dumps(group.get(key), sort_keys=True) != json.dumps(value, sort_keys=True)
               for key, value in expected.items()):
            raise ValueError(f"{phase}: rendezvous sin corroboración en {identity['node_id']}")


def _job(config, inventory, preflight, directory, job, campaign_id, owner, paths, files, git,
         results_root, deadline, hard_deadline):
    run_id = manifest.generate_run_id()
    run = results_root / "full-sft" / run_id
    start = time.monotonic()
    _event(directory, "job_started", job=job["name"], run_id=run_id)
    actual = cluster.selected_nodes(inventory, job["group"])
    all_nodes = cluster.selected_nodes(inventory, "four")
    if [node["node_id"] for node in actual] != job["nodes"]:
        raise ValueError("orden de ranks no coincide con el inventario")
    probes = {row["node_id"]: row for row in preflight["nodes"]}
    workspace = Path(config["paths"]["workspace"]) / campaign_id / run_id
    model = config["model"]
    candidate, errors, removals = "indeterminate", [], {}
    removal_seconds = 0
    phases = ("pilot", "resume") if job["block"] == 0 else ("measure",)
    try:
        run = manifest.start_run(kind="full-sft", model_id=model["id"], model_revision=model["revision"],
            tokenizer_id=model["tokenizer_id"], tokenizer_revision=model["tokenizer_revision"],
            requested_config={"job": job, "campaign_id": campaign_id, "protocol": config},
            requested_config_path=directory / "config.json",
            inputs={name: directory / "inputs" / name for name in files if name != "requested_config"},
            results_root=results_root, run_id=run_id, repo_path=REPO_ROOT)
        for name, data in files.items():
            _bytes_new(run / "raw" / "inputs" / name, data)
        opened = manifest.load_run(run)
        if (opened["schema_version"] != "2.2" or opened["kind"] != "full-sft"
                or opened["git"]["commit"] != git["commit"]):
            raise ValueError("versión SFT del manifiesto o commit de corrida incompatible")
        remote_files = _worker_files(files)
        remote_hashes = _hashes(remote_files)
        staging, failures = runtime.parallel(actual, lambda node: runtime.stage(
            node, workspace, remote_files, remote_hashes, probes[node["node_id"]], deadline))
        inference.write_new(run / "raw" / "workspace.json", {"retained": True, "path": str(workspace),
                             "staged": staging, "errors": failures,
                             "setup_before_launch_seconds": time.monotonic() - start})
        if failures:
            raise RuntimeError("\n".join(_node_errors(failures)))
        for phase in phases:
            _event(directory, "phase_started", job=job["name"], run_id=run_id, phase=phase)
            states, failures, removal, elapsed = _phase(config, actual, all_nodes, probes, workspace, run,
                job, phase, campaign_id, owner, deadline, hard_deadline)
            removals[phase] = removal
            removal_seconds += elapsed
            if failures:
                raise RuntimeError("\n".join(failures))
            result = _phase_outcome(run, phase, states, job["nodes"])
            if result == "oom":
                _require_oom_group(run, phase, job, campaign_id, probes)
            _event(directory, "phase_retired", job=job["name"], run_id=run_id, phase=phase, outcome=result)
            if result == "oom":
                candidate = result
                break
        else:
            candidate = "qualified" if job["block"] == 0 else "measured"
    except (Exception, KeyboardInterrupt) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        cleanup_started = time.monotonic()
        final = runtime.cleanup(all_nodes, campaign_id, owner, hard_deadline)
        removal_seconds += time.monotonic() - cleanup_started
    if not run.is_dir():
        raise RuntimeError(f"{job['name']}: intento sin manifiesto: {errors}")
    removals["final"] = final
    rows = []
    for node in all_nodes:
        name = node["node_id"]
        failures = [entry["nodes"][name].get("error", "retirada sin verificar") for entry in removals.values()
                    if not entry["nodes"][name]["confirmed"]]
        rows.append({"node_id": name, "status": "unconfirmed" if failures else "confirmed_absent", "errors": failures})
    # El lector requiere una fila por rank. La sección phases conserva también
    # la prueba de ausencia en los hosts que no participan de esa topología.
    receipt = _cleanup_receipt(all_nodes, rows, campaign_id, run_id, removals, removal_seconds)
    receipt["nodes"] = [row for row in rows if row["node_id"] in job["nodes"]]
    inference.write_new(run / "raw" / "cleanup.json", receipt)
    if not receipt["removal_confirmed"]:
        errors.append("retirada sin verificar")
    try:
        participants = [_participant(run, node, phases, job, campaign_id, probes[node]) for node in job["nodes"]]
        manifest.register_participants(run, [row for row in participants if row is not None])
        hashes = _hashes(files)
        _verify_unchanged(git, paths, hashes)
        _verify_archive(hashes, lambda name: run / "raw" / "inputs" / name)
        if not errors:
            assessment = sft_report.assess_run(run, job, config)
            if (assessment["outcome"] != candidate or assessment["issues"]
                    or assessment["eligible"] != (candidate in ("qualified", "measured"))
                    or assessment["input_hashes"] != hashes):
                raise ValueError(f"evidencia SFT incompatible: {assessment['issues']}")
    except (Exception, KeyboardInterrupt) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    outcome = "indeterminate" if errors else candidate
    status = "complete" if outcome in ("qualified", "measured") else "failed"
    if outcome == "oom":
        errors_for_manifest = [{"type": "cuda_oom", "message": "OOM CUDA documentado por un rank"}]
    else:
        errors_for_manifest = [{"type": "operation_error", "message": error} for error in errors]
    try:
        manifest.finalize_run(run, status=status, errors=errors_for_manifest)
    except (manifest.ManifestError, KeyboardInterrupt) as exc:
        # finalize_run puede anunciar un error después de sellar FAILED.
        # Recuperar la referencia del disco evita un segundo cierre.
        if not any((run / marker).is_file() for marker in ("COMPLETE", "FAILED")):
            raise
        errors.append(f"cierre de corrida: {type(exc).__name__}: {exc}")
    closed = manifest.load_run(run)
    manifest.verify_artifacts(run, closed)
    if closed["status"] != status:
        outcome = "indeterminate"
    ref = {"job": job["name"], "group": job["group"], "block": job["block"], "run_id": run_id,
           "manifest_sha256": manifest.sha256_file(run / "manifest.json"), "status": closed["status"],
           "outcome": outcome, "duration_seconds": time.monotonic() - start}
    return ref, receipt["removal_confirmed"], errors


def run_campaign(config_path, stage, session_limit_seconds, pilot_dir=None, *,
                 results_root=RESULTS_ROOT, pilot_report=None) -> Path:
    """Ejecuta una identidad nueva bajo reserva del coordinador Pegasus."""
    started = time.monotonic()
    started_at = manifest.utc_now()
    config_path = Path(config_path).resolve()
    results_root = Path(results_root).resolve()
    config = protocol.load_config(config_path)
    if stage not in ("pilot", "run") or (stage == "run") != (pilot_dir is not None):
        raise ValueError("usar --pilot-dir únicamente para ejecutar la matriz")
    if pilot_report is not None and stage != "run":
        raise ValueError("usar el informe publicado únicamente para ejecutar la matriz")
    limit = session_limit_seconds
    reserve = config["measurement"]["cleanup_reserve_seconds"]
    if type(limit) is not int or not reserve < limit <= config["measurement"][f"{stage}_limit_seconds"]:
        raise ValueError("límite de sesión fuera del protocolo o sin reserva de retirada")
    workspace = Path(config["paths"]["workspace"])
    if any(root.is_relative_to(workspace) or workspace.is_relative_to(root)
           for root in (results_root, RESULTS_ROOT, REPO_ROOT / "datos")):
        raise ValueError("workspace debe quedar fuera de los resultados raw y datos")
    git = manifest.capture_git(REPO_ROOT)
    if not git["commit"] or git["dirty"] is not False:
        raise ValueError("el SFT exige commit identificable y árbol limpio")
    paths, files = _read_inputs(config_path, config)
    if json.loads(files["requested_config"]) != config:
        raise ValueError("config cambió al capturar las entradas")
    inventory = protocol.cluster_config(config)
    if json.loads(files["cluster.json"]) != inventory:
        raise ValueError("inventario cambió al capturar las entradas")
    _verify_tracked({name: path for name, path in paths.items() if name != "reference/preflight.json"},
                    files, git["commit"])
    hashes = _hashes(files)
    if (hashes["reference/preflight.json"] != config["reference_preflight_sha256"]
            or hashes["corpus.jsonl"] != config["workload"]["corpus_sha256"]):
        raise ValueError("huellas fijadas del preflight o corpus no coinciden")
    qualified, pilot_reference, published = ([], None, None)
    if pilot_dir:
        qualified, pilot_reference, published = _qualify(
            pilot_dir, results_root, hashes, limit, reserve, git_commit=git["commit"], pilot_report=pilot_report)
    plan = protocol.make_plan(config, stage, qualified if stage == "run" else None)
    deadline, hard_deadline = started + limit - reserve, started + limit
    runtime.remaining(deadline, 10)
    with reservation(inventory):
        campaign_id, owner = manifest.generate_run_id(), uuid.uuid4().hex
        directory = results_root / "sft-campaign" / campaign_id
        directory.mkdir(parents=True, exist_ok=False)
        refs, errors, preflight = [], [], None
        removal = True
        remote_started = False
        try:
            inference.write_new(directory / "plan.json", plan)
            _bytes_new(directory / "config.json", files["requested_config"])
            _bytes_new(directory / "cluster.json", files["cluster.json"])
            if published is not None:
                _bytes_new(directory / "published-pilot.json", published)
            for name, data in files.items():
                if name != "requested_config":
                    _bytes_new(directory / "inputs" / name, data)
            _verify_unchanged(git, paths, hashes)
            _event(directory, "campaign_started", campaign_id=campaign_id, stage=stage, limit_seconds=limit)
            remote_started = True
            preflight = runtime.preflight(config, inventory, plan["jobs"], campaign_id, owner, deadline, hard_deadline)
            removal = preflight["removal"]["removal_confirmed"]
            inference.write_new(directory / "preflight.json", preflight)
            if preflight["errors"]:
                errors.extend(preflight["errors"])
            else:
                if pilot_reference and pilot_reference["estimated_matrix_seconds"] > deadline - time.monotonic():
                    raise TimeoutError("la proyección ya no cabe después del preflight")
                for job in plan["jobs"]:
                    runtime.remaining(deadline, 20)
                    _verify_unchanged(git, paths, hashes)
                    reference, retired, issues = _job(config, inventory, preflight, directory, job, campaign_id,
                        owner, paths, files, git, results_root, deadline, hard_deadline)
                    refs.append(reference)
                    removal = removal and retired
                    if stage == "pilot" and reference["outcome"] == "qualified":
                        qualified.append(job["group"])
                    _event(directory, "job_closed", **reference)
                    if issues or not retired or reference["outcome"] == "indeterminate":
                        errors.extend(f"{job['name']}: {issue}" for issue in issues)
                        break
        except (Exception, KeyboardInterrupt) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            if remote_started:
                nodes = cluster.selected_nodes(inventory, "four")
                try:
                    final = runtime.cleanup(nodes, campaign_id, owner, hard_deadline)
                except (Exception, KeyboardInterrupt) as exc:
                    errors.append(f"retirada final interrumpida: {type(exc).__name__}: {exc}")
                    try:
                        final = runtime.cleanup(nodes, campaign_id, owner, hard_deadline)
                    except (Exception, KeyboardInterrupt) as retry:
                        final = {"removal_confirmed": False, "nodes": {}, "ended_at": manifest.utc_now(),
                                 "error": f"{type(retry).__name__}: {retry}"}
                removal = removal and final["removal_confirmed"]
            else:
                final = {"removal_confirmed": True, "nodes": {}, "ended_at": manifest.utc_now(),
                         "remote_started": False}
            inference.write_new(directory / "cleanup.json", final)
        try:
            _verify_unchanged(git, paths, hashes)
            _verify_archive(hashes, lambda name: directory / "config.json" if name == "requested_config"
                            else directory / "inputs" / name)
            _verify_archive({"cluster.json": hashes["cluster.json"]}, lambda name: directory / name)
            if pilot_reference is not None:
                _verify_archive({"published-pilot.json": pilot_reference["published_report"]["sha256"]},
                                lambda name: directory / name)
        except (Exception, KeyboardInterrupt) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        if not (directory / "preflight.json").exists():
            inference.write_new(directory / "preflight.json", preflight or {"errors": errors, "nodes": []})
        if not removal:
            errors.append("retirada final no confirmada")
        complete = (len(refs) == len(plan["jobs"]) and not errors
                    and all(ref["outcome"] in (("qualified", "oom") if stage == "pilot" else ("measured",))
                            for ref in refs))
        campaign = {"campaign_id": campaign_id, "stage": stage, "git": git, "started_at": started_at,
                    "limit_seconds": limit, "removal_confirmed": removal, "runs": refs, "errors": errors,
                    "qualified_groups": qualified, "input_hashes": hashes, "pilot_reference": pilot_reference}
        _seal_campaign(directory, campaign, started, complete)
        return directory
