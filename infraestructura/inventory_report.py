"""Genera un informe breve verificando hashes, sin modificar la corrida raw."""

import argparse
import csv
import io
import json
import os
from pathlib import Path
from urllib.parse import quote

from infraestructura.manifest import load_run, sha256_file
from infraestructura.rutas import REPO_ROOT


UNAVAILABLE = "no disponible"


def _fmt(value, scale=1):
    return UNAVAILABLE if value is None else f"{value / scale:.3f}"


def _text(value):
    if value is None:
        return UNAVAILABLE
    return str(value).replace("\n", " ").replace("|", "\\|")


def _hardware(probes, samples):
    """Read observed identities only; retain heterogeneous CPU model entries."""
    cpu = probes.get("cpu", {})
    rows = []
    if cpu.get("status") == "ok":
        try:
            rows = json.loads(cpu.get("stdout", ""))["lscpu"]
        except (ValueError, KeyError, TypeError):
            pass
    def fields(name):
        found = []
        def visit(entries):
            for entry in entries:
                if entry.get("field", "").rstrip(":") == name and entry.get("data"):
                    found.append(str(entry["data"]))
                visit(entry.get("children", []))
        visit(rows)
        return list(dict.fromkeys(found))
    models, cpus = fields("Model name"), fields("CPU(s)")
    cpu_name = " + ".join(models) if models else UNAVAILABLE
    if cpus:
        cpu_name += f" ({cpus[0]} CPU lógicas)"
    gpu = probes.get("gpu", {})
    names = []
    if gpu.get("status") == "ok":
        names = [row[0].strip() for row in csv.reader(io.StringIO(gpu.get("stdout", "")))
                 if len(row) >= 4 and row[0].strip()]
    gpu_name = ", ".join(names) if names else UNAVAILABLE
    total = next((s.get("memory", {}).get("MemTotal") for s in samples
                  if s.get("memory", {}).get("MemTotal") is not None), None)
    if total is None and probes.get("memory", {}).get("status") == "ok":
        for line in probes["memory"].get("stdout", "").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "MemTotal:":
                try:
                    total = int(parts[1])
                except ValueError:
                    pass
    return cpu_name, gpu_name, total


def render(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.is_relative_to(run):
        raise ValueError("el resumen debe quedar fuera de raw")
    source = run / "manifest.json"
    data = load_run(run)
    if data["kind"] != "system-baseline" or data["status"] not in {"complete", "failed"}:
        raise ValueError("se requiere inventario finalizado")
    indexed = set()
    for artifact in data["artifacts"]:
        path = (run / artifact["path"]).resolve()
        if artifact["path"] in indexed:
            raise ValueError(f"integridad inválida: artefacto duplicado: {artifact['path']}")
        if (not path.is_relative_to(run) or not path.is_file()
                or path.stat().st_size != artifact["size_bytes"]
                or sha256_file(path) != artifact["sha256"]):
            raise ValueError(f"integridad inválida: {artifact['path']}")
        indexed.add(artifact["path"])

    for relative in ("raw/assessment.json", "raw/inventory.json", "raw/after.json", "metrics/idle.jsonl"):
        if (run / relative).exists() and relative not in indexed:
            raise ValueError(f"integridad no verificable: evidencia no registrada: {relative}")

    incidents = []
    def read_json(relative):
        path = run / relative
        if not path.exists():
            incidents.append(f"Evidencia ausente: `{relative}`.")
            return {}
        return json.loads(path.read_text())
    assessment = read_json("raw/assessment.json")
    inventory = read_json("raw/inventory.json")
    after = read_json("raw/after.json")
    samples_path = run / "metrics/idle.jsonl"
    if samples_path.exists():
        samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
    else:
        samples = []
        incidents.append("Evidencia ausente: `metrics/idle.jsonl`.")
    probes = inventory.get("probes", {})
    cpu, gpu, total = _hardware(probes, samples)
    for name, value in (("CPU", cpu), ("GPU", gpu), ("memoria total", total)):
        if value is None or value == UNAVAILABLE:
            incidents.append(f"Dato esencial no disponible: {name}.")
    for error in data.get("errors", []):
        message = error.get("message", json.dumps(error, ensure_ascii=False)) if isinstance(error, dict) else error
        incidents.append(f"Error de captura: {_text(message)}.")
    for exception in data.get("policy", {}).get("exceptions", []):
        incidents.append(f"Excepción registrada: `{_text(exception)}`.")
    if data.get("git", {}).get("dirty"):
        incidents.append("El código tenía cambios locales; consultar las fuentes preservadas.")
    for label, key in (("Evidencia faltante o inválida", "missing_or_invalid"),
                       ("Criterio incumplido", "violations")):
        incidents.extend(f"{label}: `{_text(item)}`." for item in assessment.get(key, []))
    for stage, records in (("inicial", probes), ("final", after)):
        failures = [f"{name} ({record.get('status', UNAVAILABLE)})" for name, record in records.items()
                    if record.get("status") != "ok"]
        if failures:
            incidents.append(f"Sondas {stage}es no exitosas: {_text(', '.join(failures))}.")
    sample_errors = list(dict.fromkeys(str(error) for sample in samples for error in sample.get("errors", [])))
    incidents.extend(f"Error de muestreo: {_text(error)}." for error in sample_errors)
    def metric(name, stat):
        value = assessment.get(name, {}).get(stat)
        if value is None:
            incidents.append(f"Métrica esencial no disponible: `{name}.{stat}`.")
        return value
    cpu_p95, cpu_max = metric("cpu_busy_pct", "p95"), metric("cpu_busy_pct", "max")
    gpu_p95, gpu_max = metric("gpu_utilization_pct", "p95"), metric("gpu_utilization_pct", "max")
    available = metric("memory_available_kib", "median")
    drift = assessment.get("memory_available_drift_mib")
    if drift is None:
        incidents.append("Métrica esencial no disponible: `memory_available_drift_mib`.")
    process_visibility = bool(samples) and all(s.get("gpu_processes") is not None for s in samples)
    processes_present = any(s.get("gpu_processes") for s in samples)
    if processes_present:
        processes = "presentes" if process_visibility else "presentes (cobertura incompleta)"
    else:
        processes = "ausentes" if process_visibility else UNAVAILABLE
    if not process_visibility:
        incidents.append("Visibilidad de procesos GPU incompleta; su ausencia no está verificada.")
    def link(label, path):
        return f"[{label}]({quote(os.path.relpath(path, output.parent), safe='/')})"
    system, git = data.get("system", {}), data.get("git", {})
    count = assessment.get("sample_count", len(samples) if samples_path.exists() else None)
    expected = assessment.get("expected_sample_count")
    lines = [
        "# Pegasus — inventario y referencia de reposo", "",
        f"- Corrida: `{data['run_id']}`.",
        f"- Fecha UTC: {_text(data.get('started_at'))} → {_text(data.get('ended_at'))}.",
        f"- Host: `{_text(system.get('hostname'))}`; arquitectura: `{_text(system.get('machine'))}`.",
        f"- Commit: `{_text(git.get('commit'))}`; cambios locales: `{_text(git.get('dirty'))}`.",
        f"- Captura: **{data['status']}**; criterios de reposo: **{assessment.get('idle_status', 'unverified')}**.",
        f"- Muestras: {_text(count)}/{_text(expected)}; ventana real: {_fmt(assessment.get('elapsed_seconds'))} s.",
        "", "## Hardware observado", "",
        "| Recurso | Observación |", "| --- | --- |",
        f"| CPU | {_text(cpu)} |", f"| GPU | {_text(gpu)} |",
        f"| Memoria total del sistema, GiB | {_fmt(total, 1048576)} |",
        "", "## Reposo observado", "",
        "| Métrica | Resultado |", "| --- | ---: |",
        f"| CPU ocupada p95 / máxima, % | {_fmt(cpu_p95)} / {_fmt(cpu_max)} |",
        f"| Utilización GPU p95 / máxima, % | {_fmt(gpu_p95)} / {_fmt(gpu_max)} |",
        f"| Memoria disponible mediana, GiB | {_fmt(available, 1048576)} |",
        f"| Variación de memoria disponible entre extremos, MiB | {_fmt(drift)} |",
        f"| Procesos de cómputo GPU (criterio: ausentes) | {processes} |",
        "", "## Incidencias", "",
    ]
    lines.extend(f"- {item}" for item in dict.fromkeys(incidents))
    if not incidents:
        lines.append("Sin incidencias registradas.")
    lines += ["", "## Alcance y evidencia", "",
              "Referencia exploratoria con umbrales provisionales; no garantiza reposo térmico ni ausencia de procesos entre muestras.",
              "Esta captura no mide rendimiento de modelos, SFT ni comunicación entre nodos. Las cifras son observaciones locales.", "",
              f"- {link('Protocolo y limitaciones', REPO_ROOT / 'docs/inventario-pegasus.md')}.",
              f"- {link('Manifiesto', source)} · {link('Evidencia raw local', run / 'raw')} (ignorada por Git; requiere respaldo)."]
    for label, relative in (("Contexto técnico", "raw/context.json"),
                            ("Evaluación y umbrales", "raw/assessment.json"),
                            ("Muestras", "metrics/idle.jsonl"),
                            ("Configuración preservada", "raw/sources/pegasus-inventory.json"),
                            ("Fuentes preservadas", "raw/sources")):
        if (run / relative).exists():
            lines.append(f"- {link(label, run / relative)}.")
    lines += ["- Integridad de los artefactos registrados verificada al generar este informe.", "",
              f"<!-- manifest-sha256: {sha256_file(source)}; generator-sha256: {sha256_file(Path(__file__))} -->", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(render(args.run_directory, args.output))


if __name__ == "__main__":
    main()
