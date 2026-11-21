"""Diagnose the three 2048/C8 study loads using closed, verified local evidence."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from statistics import fmean, median
import sys

from infraestructura import inference, inference_diag_protocol, inference_diag_quality, inference_evidence, inference_metrics
from infraestructura import inference_telemetry_report as telemetry
from infraestructura import inference_timeline as timeline, manifest
from infraestructura.rutas import REPO_ROOT


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


PHASES = ("queue", "prefill", "decode")
GROUPS = {"four": 4, "pair": 2, "single": 1}


def diagnostic_instruments(cell, record, run):
    """Read only instrument receipts independently revalidated by build_report."""
    issues = cell["diagnostic_issues"]
    record["diagnostic_issues"] = list(issues)
    name = cell["cell"]
    prefix = f"raw/diagnostic/{name}"

    def source(key, relative):
        path = manifest._internal_file(run, relative)
        record["source_sha256"][key] = manifest.sha256_file(path)
        return read(path)

    def reason(kind):
        return [issue for issue in issues if issue.startswith(kind + ":")]

    record["server_phases"] = {"valid": False, "errors": reason("phases")}
    if not reason("phases"):
        receipt = source("phases", f"{prefix}/phases.json")
        for tag in ("before", "after"):
            path = manifest._internal_file(run, f"{prefix}/{tag}.json")
            record["source_sha256"][tag] = manifest.sha256_file(path)
        record["server_phases"] = {
            "valid": True, "process_identity_source": receipt["process_identity_source"],
            "phases": {phase: {key: receipt["phases"][phase][key] for key in
                               ("count_delta", "inf_bucket_delta", "sum_delta_s", "mean_s")}
                       for phase in PHASES},
        }
    if "mode" not in cell:
        return
    record["client_admission"] = {"valid": False, "errors": reason("client events")}
    if not reason("client events"):
        events = source("client_events", f"raw/nodes/pegasus/metrics/{name}/client-events.json")
        source("client_validation", f"{prefix}/client-validation.json")
        rows = events["rows"]
        waits = [r["semaphore_acquired_offset_s"] - r["task_created_offset_s"] for r in rows]
        record["client_admission"] = {
            "valid": True, "mean_wait_before_admission_s": fmean(waits),
            "max_wait_before_admission_s": max(waits),
            "clock": events["clock"], "requests": rows,
        }
    if cell["mode"] == "on":
        record["profile_control"] = {"valid": False, "errors": reason("profile control")}
        if not reason("profile control"):
            record["profile_control"] = source("profile_control", f"{prefix}/profile-control.json")
        record["profile_coverage"] = {"valid": False, "errors": reason("profile coverage")}
        if not reason("profile coverage"):
            record["profile_coverage"] = source("profile_coverage", f"{prefix}/profile-coverage.json")
        record["trace"] = {"valid": False, "errors": reason("traces")}
        if not reason("traces"):
            receipt = source("traces", f"{prefix}/traces.json")
            record["trace"] = {"valid": True, "files": receipt["files"], "ranks": receipt["summary"]}


def phase_contrasts(rows, comparisons):
    indexed = {(r["model_key"], r["block"], r["input_tokens"], r["concurrency"], r["group"]): r
               for r in rows}
    result = []
    for pair in comparisons:
        if "s4_2" not in pair:
            continue
        key = (pair["model"], pair["block"], pair["input_tokens"], pair["concurrency"])
        four, two = (indexed[(*key, g)]["server_phases"] for g in ("four", "pair"))
        if four["valid"] and two["valid"]:
            result.append({"model_key": key[0], "block": key[1], "input_tokens": key[2],
                           "concurrency": key[3], "four_minus_two_mean_s": {
                               p: four["phases"][p]["mean_s"] - two["phases"][p]["mean_s"]
                               for p in PHASES}})
    return result


def calibrate(auxiliary):
    """Match OFF and ON from one server lifetime and one client load."""
    groups = {}
    for row in auxiliary:
        key = tuple(row[k] for k in ("run_id", "model_key", "group", "block", "input_tokens", "concurrency"))
        observed = groups.setdefault(key, {})
        if row["mode"] in observed:
            raise ValueError("modo auxiliar duplicado en una misma corrida y carga")
        observed[row["mode"]] = row
    result = []
    for (run_id, model, group, block, length, concurrency), modes in groups.items():
        item = {"run_id": run_id, "model_key": model, "group": group, "block": block,
                "input_tokens": length, "concurrency": concurrency,
                "t32_off_s": modes.get("off", {}).get("t32_s"),
                "t32_on_s": modes.get("on", {}).get("t32_s"),
                "rank_traces_verified": modes.get("on", {}).get("trace", {}).get("valid", False)}
        if set(modes) != {"off", "on"}:
            item.update(valid=False, reason="Falta una de las dos mediciones en el mismo servidor.")
        elif not modes["on"]["profile_control"]["valid"]:
            item.update(valid=False, reason="El servidor no confirmó la ventana ON dentro del límite previsto.")
        elif not modes["on"]["profile_coverage"]["valid"]:
            item.update(valid=False, reason="No se verificó solapamiento con una solicitud de este lote.")
        elif not all(r["client_admission"]["valid"] for r in modes.values()):
            item.update(valid=False, reason="Faltan eventos de admisión verificados en uno de los lotes.")
        else:
            item.update(valid=True, delta_percent=100 * (item["t32_on_s"] / item["t32_off_s"] - 1))
        result.append(item)
    return result


def compare_profile_order(pairs, timeline_pairs):
    """Describe whether profiling changes the observed four-minus-two direction."""
    results = []
    for model, length, concurrency in sorted({
        (p["model_key"], p["input_tokens"], p["concurrency"]) for p in pairs
    }):
        key = (model, length, concurrency)
        matched = [p for p in pairs if p["valid"] and p["block"] == 4 and
                   (p["model_key"], p["input_tokens"], p["concurrency"]) == key]
        by_group = {p["group"]: p for p in matched}
        if set(by_group) != {"four", "pair"} or len(matched) != 2:
            continue
        four, two = by_group["four"], by_group["pair"]
        deltas = {mode: four[f"t32_{mode}_s"] - two[f"t32_{mode}_s"] for mode in ("off", "on")}
        primary = [p["four_minus_two_s"]["t32_s"] for p in timeline_pairs
                   if (p["model_key"], p["input_tokens"], p["concurrency"]) == key]
        results.append({"model_key": model, "input_tokens": length, "concurrency": concurrency,
                        "primary_blocks": len(primary), "primary_delta_s": median(primary) if primary else None,
                        "aux_off_delta_s": deltas["off"], "aux_on_delta_s": deltas["on"],
                        "rank_sets_verified": {g: by_group[g]["rank_traces_verified"] for g in ("four", "pair")},
                        "order_reversed_when_on_requested": deltas["off"] * deltas["on"] < 0})
    return results


def primary_telemetry(cells):
    result = []
    by_run = {}
    for cell in cells:
        by_run.setdefault(Path(cell["run_dir"]), []).append(cell)
    for run, rows in by_run.items():
        ranks = manifest.load_run(run)["participants"]
        for rank in ranks:
            identity = {k: rank[k] for k in ("hostname", "rank", "node_id", "gpu_uuid")}
            try:
                path = manifest._internal_file(run, f"raw/nodes/{rank['node_id']}/logs/telemetry.jsonl")
                digest = manifest.sha256_file(path)
            except (OSError, manifest.ManifestError):
                path, digest = None, None
            for cell in rows:
                row = {**identity, **{k: cell[k] for k in ("model_key", "group", "block", "cell")},
                       "run_id": run.name, "source_sha256": digest}
                try:
                    if path is None:
                        raise ValueError("telemetría del participante ausente")
                    start, end = (datetime.fromisoformat(cell[key]) for key in
                                  ("measurement_started_at", "measurement_ended_at"))
                    samples = inference_diag_quality.telemetry_window(path, start, end, rank)
                    row.update(telemetry.summarize(samples, start.timestamp(), end.timestamp()))
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    row["error"] = str(exc)
                result.append(row)
    return result


def load_cell(cell, auxiliary=False):
    run = Path(cell["run_dir"])
    profile = (cell["input_tokens"], cell["concurrency"])
    record = {k: cell[k] for k in ("block", "model_key", "group", "cell", "input_tokens", "concurrency")}
    if auxiliary:
        record["mode"] = cell["mode"]
    record["run_id"] = run.name
    name = cell["cell"]
    client = manifest._internal_file(run, f"raw/nodes/pegasus/metrics/{name}/measured.json")
    receipt = manifest._internal_file(run,
        f"raw/inputs/workload/{cell['model_key']}/{profile[0]}/reader-validation.json")
    result = read(client)
    if result["duration"] != cell["t32_seconds"]:
        raise ValueError("T32 contradice el JSON original del cliente")
    record["source_sha256"] = {"measured": manifest.sha256_file(client),
                               "reader_validation": manifest.sha256_file(receipt)}
    record["t32_s"] = cell["t32_seconds"]
    try:
        samples = read(receipt)["phases"]["measured"]["samples"]
        record["timeline"] = {"valid": True, **timeline.client_timeline(result, samples, *profile)}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        record["timeline"] = {"valid": False, "reason": str(exc)}
    if "diagnostic_issues" in cell:
        diagnostic_instruments(cell, record, run)
    return record


def analyze(directory):
    directory = Path(directory).resolve()
    report = inference.build_report(directory)
    campaign = read(directory / "campaign.json")
    selected = [c for c in report["cells"] if (c["input_tokens"], c["concurrency"]) in timeline.PROFILES]
    rows = [load_cell(c) for c in selected]
    pairs = [p for p in report["paired_comparisons"]
             if (p["input_tokens"], p["concurrency"]) in timeline.PROFILES]
    result = {
        "schema_version": 1, "campaign_id": report["campaign_id"], "status": report["status"],
        "protocol_id": campaign["protocol"]["protocol_id"], "capture_commit": campaign["git"]["commit"],
        "campaign_sha256": manifest.sha256_file(directory / "campaign.json"),
        "analyzers_sha256": {p.name: manifest.sha256_file(p) for p in map(Path, (
            __file__, timeline.__file__, inference.__file__, inference_evidence.__file__,
            inference_metrics.__file__, manifest.__file__))},
        "runs": [{"run_id": Path(r["run_dir"]).name,
                  **{k: r[k] for k in ("job", "status", "manifest_sha256", "errors",
                                       "series", "failure_scope") if k in r}}
                 for r in report["runs"]],
        "campaign_errors": campaign["errors"],
        "pending_jobs": report["pending_jobs"], "excluded_cells": report.get("excluded_cells", []),
        "primary_cells": rows, "paired_comparisons": pairs,
        "timeline_contrasts_4_2": timeline.paired_timelines(rows, pairs),
        "timeline_method": {
            "unit": "s", "origin": "Primer inicio HTTP observado en cada lote, en el reloj del cliente.",
            "end": "Inicio + TTFT + suma de los 127 intervalos entre tokens recibidos.",
            "prompt_id": "ID de la fila congelada, en orden validado. El servidor puede asignar otro ID.",
            "components": "Suma de TTFT/C, suma de intervalos posteriores al primer token/C y el "
                          "resto hasta T32, donde C es la concurrencia máxima. Las tres magnitudes "
                          "describen la ocupación de las plazas del cliente.",
            "residual": "El resto incluye plazas libres durante el llenado y vaciado y los bordes del "
                        "cronómetro. La espera individual antes de adquirir plaza requiere marcas propias.",
        },
    }
    if result["protocol_id"] == inference_diag_protocol.PROTOCOL:
        aux = [load_cell(c, auxiliary=True) for c in report["auxiliary_cells"]]
        result.update(auxiliary_cells=aux, off_on_pairs=calibrate(aux),
                      phase_contrasts_4_2=phase_contrasts(rows, pairs))
        result["profile_order_4_2"] = compare_profile_order(result["off_on_pairs"],
                                                            result["timeline_contrasts_4_2"])
        result["primary_telemetry"] = primary_telemetry(selected)
        path = Path(telemetry.__file__)
        result["analyzers_sha256"][path.name] = manifest.sha256_file(path)
        for name in inference_diag_protocol.DIAGNOSTIC_SOURCES:
            result["analyzers_sha256"][Path(name).name] = manifest.sha256_file(REPO_ROOT / name)
    return result


def instrument_tables(data):
    primary, auxiliary = data["primary_cells"], data["auxiliary_cells"]
    pairs = data["off_on_pairs"]
    valid_phases = sum(c["server_phases"]["valid"] for c in primary)
    admissions = [c for c in auxiliary if c["client_admission"]["valid"]]
    controlled = [c for c in auxiliary if c["mode"] == "on" and c["profile_control"]["valid"]]
    covered = [c for c in controlled if c["profile_coverage"]["valid"]]
    on = [c for c in covered if c["trace"]["valid"]]
    rank_count = sum(len(c["trace"]["ranks"]) for c in on)
    lines = [
        "## Fases observadas por el servidor", "",
        f"Se validaron los histogramas de {valid_phases}/54 celdas primarias previstas. "
        "Cada valor usa la diferencia de sumas entre dos lecturas del mismo proceso con el motor "
        "vacío, dividida por 32 solicitudes. Las tres fases deben incrementar en 32 tanto el "
        "conteo como el último bucket. La tabla da diferencias cuatro menos dos en milisegundos "
        "por solicitud, con mediana y rango entre bloques pareados.", "",
        "| Modelo | Entrada / concurrencia | Pares válidos | Cola | Prefill | Decode |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for model in dict.fromkeys(c["model_key"] for c in primary):
        for n, concurrency in timeline.PROFILES:
            group = [p["four_minus_two_mean_s"] for p in data["phase_contrasts_4_2"]
                     if (p["model_key"], p["input_tokens"], p["concurrency"]) == (model, n, concurrency)]
            cols = [timeline.typical([p[phase] * 1000 for p in group], 1) for phase in PHASES]
            lines.append(f"| {model} | {n} / {concurrency} | {len(group)} | " + " | ".join(cols) + " |")
    lines += ["", "El JSON conserva también las medias y conteos por topología, incluido Pegasus. "
              "Cada histograma agrega solicitudes. La comparación con TTFT es a nivel de lote y "
              "carece de emparejamiento individual entre la cola del motor y cada primer token.", "",
              "## Perturbación y espera de admisión", "",
              f"Hay {len(admissions)}/24 lotes auxiliares con marcas de admisión válidas. "
              f"Se verificaron {sum(p['valid'] for p in pairs)}/12 pares OFF/ON previstos. "
              "Cada par usa un mismo arranque y la misma carga, con tres lotes OFF seguidos de "
              "tres ON. El porcentaje expresa 100 × (T32 ON / T32 OFF − 1). "
              "Mide el cambio al solicitar ON con acuses válidos y actividad del lote dentro de la ventana. La cobertura por rank figura "
              "en la siguiente sección. El orden fijo también puede influir.", "",
              "| Modelo | Nodos | Entrada / concurrencia | T32 OFF (s) | T32 ON (s) | Cambio | "
              "Espera media antes de admisión OFF / ON (ms) |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    def decimal(value, digits=2):
        return "—" if value is None else f"{value:.{digits}f}".replace(".", ",")
    waits = {(c["run_id"], c["input_tokens"], c["concurrency"], c["mode"]):
             1000 * c["client_admission"]["mean_wait_before_admission_s"] for c in admissions}
    for p in pairs:
        key = (p["run_id"], p["input_tokens"], p["concurrency"])
        wait = " / ".join(decimal(waits.get((*key, m)), 1) for m in ("off", "on"))
        change = decimal(p.get("delta_percent"), 1) + " %" if p["valid"] else "—"
        lines.append(f"| {p['model_key']} | {GROUPS[p['group']]} | {key[1]} / {key[2]} | "
                     f"{decimal(p['t32_off_s'])} | {decimal(p['t32_on_s'])} | {change} | {wait} |")
    lines += ["", "La espera de admisión se mide desde la creación de la tarea hasta que adquiere "
              "el semáforo del cliente. Sus marcas, el ID de la solicitud y las llegadas de tokens "
              "comparten un reloj monotónico en Pegasus.", ""]
    order = data["profile_order_4_2"]
    if order:
        lines += ["Los valores siguientes son segundos de T32 en cuatro menos dos nodos. "
                  "El signo describe el orden observado al solicitar ON. Una sola repetición "
                  "en orden fijo deja abierta la causa de cualquier cambio.", "",
                  "| Modelo | Entrada / concurrencia | Primaria (mediana) | Auxiliar OFF | Auxiliar ON | "
                  "Trazas completas TP4 / TP2 |",
                  "| --- | --- | ---: | ---: | ---: | --- |"]
        for p in order:
            change = " (cambio de signo)" if p["order_reversed_when_on_requested"] else ""
            rank_status = " / ".join("sí" if p["rank_sets_verified"][g] else "no" for g in ("four", "pair"))
            lines.append(f"| {p['model_key']} | {p['input_tokens']} / {p['concurrency']} | "
                         f"{decimal(p['primary_delta_s'])} | {decimal(p['aux_off_delta_s'])} | "
                         f"{decimal(p['aux_on_delta_s'])}{change} | {rank_status} |")
        lines.append("")
    lines += [
              "## Trabajo GPU por rank", "",
              f"El servidor confirmó {len(controlled)}/12 ventanas ON. En {len(covered)} hubo "
              "solapamiento con solicitudes medidas. "
              f"En {len(on)} se verificó el "
              f"conjunto de ranks, con {rank_count}/36 trazas previstas. "
              "Cada tiempo acumula la unión de intervalos de kernels dentro del reloj de un rank. "
              "El solapamiento es la intersección temporal de los kernels NCCL reconocidos y los "
              "clasificados como cómputo. Se muestra «—» cuando faltan etiquetas NCCL verificables.", "",
              "| Modelo | Nodos | Entrada / concurrencia | Nodo (rank) | Cómputo (ms) | "
              "NCCL (ms) | Solapamiento (ms) | Kernels ligados a iteración / total |",
              "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |"]
    for c in on:
        for r in c["trace"]["ranks"]:
            activity = r["activity"]
            classified = sum(i["kernel_count"] for i in r["iteration_activity"]
                             if i["category"] != "insufficient")
            nccl = r["nccl_label_status"] == "observed"
            compute = decimal(activity["compute_us"] / 1000, 1)
            comm = decimal(activity["nccl_us"] / 1000, 1) if nccl else "—"
            overlap = decimal(activity["overlap_us"] / 1000, 1) if nccl else "—"
            lines.append(f"| {c['model_key']} | {GROUPS[c['group']]} | "
                         f"{c['input_tokens']} / {c['concurrency']} | {r['node_id']} ({r['rank']}) | "
                         f"{compute} | {comm} | {overlap} | {classified}/{r['kernel_count']} |")
    if covered:
        lines += ["", "La cobertura siguiente cuenta solicitudes que se solapan con la ventana de "
                  "control y solicitudes enteramente contenidas en ella. El profiler puede "
                  "alcanzar el límite de iteraciones antes del STOP.", "",
                  "| Modelo | Nodos | Entrada / concurrencia | Solapadas / 32 | Completas / 32 |",
                  "| --- | ---: | --- | ---: | ---: |"]
        for c in covered:
            coverage = c["profile_coverage"]
            lines.append(f"| {c['model_key']} | {GROUPS[c['group']]} | "
                         f"{c['input_tokens']} / {c['concurrency']} | "
                         f"{coverage['count']} | {coverage['fully_covered_count']} |")
    lines += ["", "Los intervalos cubren como máximo 32 iteraciones y 60 s por ventana. "
              "La asociación a contexto, generación o iteraciones mixtas requiere enlazar el "
              "lanzamiento CPU con el kernel. El JSON indica cuántos kernels quedan sin enlace "
              "y conserva el detalle por stream y rank. Los relojes de los distintos hosts y "
              "del cliente tienen orígenes propios. Una coincidencia de duraciones entre relojes "
              "solo permite formular una hipótesis sobre la causa de una espera.", ""]
    summaries = data["primary_telemetry"]
    observed = sum(row.get("sample_count", 0) > 0 for row in summaries)
    lines += ["## Telemetría de la serie primaria", "",
              f"El JSON conserva muestras resumidas para {observed}/{len(summaries)} combinaciones "
              "de celda y nodo admitidas, con hostname, UUID, rank y hash de la fuente. "
              "La ventana UTC incluye el arranque y salida del proceso cliente. La cadencia de "
              "muestreo y el desfase entre relojes limitan las comparaciones temporales. Las "
              "lecturas de memoria de CPU y GPU en UMA comparten capacidad física. La interfaz "
              "puede omitir tráfico RoCE descargado en hardware.", ""]
    return lines


def markdown(data):
    cells = data["primary_cells"]
    valid = [c["timeline"] for c in cells if c["timeline"]["valid"]]
    lines = [f"# Diagnóstico de inferencia {data['campaign_id']}", "",
             f"Captura `{data['capture_commit']}` con estado **{data['status'].upper()}**. "
             f"Se reconstruyeron {len(valid)} de {len(cells)} cronologías primarias seleccionadas.",
             ""]
    if valid:
        closure = max(abs(t["t32_minus_span_s"]) for t in valid)
        diff_ms = f"{closure * 1000:.3f}".replace(".", ",")
        lines += ["La mayor diferencia absoluta entre T32 y el intervalo desde el primer inicio "
                  f"hasta el último token fue de {diff_ms} ms.", ""]
    lines += [
        "## Dónde cambia el tiempo del lote", "",
        "Cada columna muestra la mediana [mínimo–máximo] de las diferencias cuatro menos dos "
        "del mismo bloque, en segundos. Un valor positivo indica más tiempo en cuatro nodos. "
        "Las tres contribuciones suman el cambio de T32 dentro de cada bloque. Las medianas se "
        "calculan por columna.", "",
        "| Modelo | Entrada | Concurrencia | Bloques | Δ T32 | Δ hasta el primer token / plazas | "
        "Δ después del primer token / plazas | Δ restante |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    contrasts = data["timeline_contrasts_4_2"]
    for model in dict.fromkeys(row["model_key"] for row in cells):
        for length, concurrency in timeline.PROFILES:
            paired = [item["four_minus_two_s"] for item in contrasts
                      if (item["model_key"], item["input_tokens"], item["concurrency"])
                      == (model, length, concurrency)]
            cols = [timeline.typical([x[k] for x in paired]) for k in
                    ("t32_s", "ttft_slot_seconds_div_c", "stream_slot_seconds_div_c",
                     "remaining_slot_seconds_div_c")]
            lines.append(f"| {model} | {length} | {concurrency} | {len(paired)} | "
                         + " | ".join(cols) + " |")
    lines += ["", "Las contribuciones previas y posteriores al primer token se obtienen sumando "
              "los tiempos individuales y dividiéndolos por el número máximo de plazas. "
              "El resto recoge las plazas libres y los bordes de la medición. Las solicitudes "
              "concurrentes se solapan. El tiempo de kernel y la espera en la cola del servidor "
              "requieren evidencia del motor.", "",
              "El JSON conserva inicio, primer y último token y la mayor pausa de cada solicitud "
              "en el reloj del cliente, los tiempos por bloque y los hashes de sus fuentes.", ""]
    if valid:
        lines += ["## Ocupación de plazas y fin del lote", "",
                  "Todos los tiempos parten del primer envío HTTP de cada lote. Para concurrencia C, "
                  "el envío número C+1 indica cuándo el cliente pudo reutilizar una plaza. "
                  "Se muestran medianas y rangos entre bloques. La última columna identifica la "
                  "fila del JSONL cuyo último token llegó al final de cada lote, en orden de bloque.", "",
                  "| Modelo | Nodos | Entrada / C | Envío C+1 (s) | Último envío (s) | "
                  "Inicio de la mayor pausa (s) | Duración de la pausa (ms) | Última respuesta B1 / B2 / B3 |",
                  "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |"]
        groups = {}
        for cell in cells:
            if cell["timeline"]["valid"]:
                key = (cell["model_key"], cell["group"], cell["input_tokens"], cell["concurrency"])
                groups.setdefault(key, {})[cell["block"]] = cell["timeline"]
        for (model, group, n, concurrency), blocks in groups.items():
            values = list(blocks.values())
            closing = " / ".join(blocks[b]["last_prompt_id"] if b in blocks else "—" for b in (1, 2, 3))
            cols = [timeline.typical([t["first_start_beyond_initial_slots_s"] for t in values]),
                    timeline.typical([t["last_request_start_s"] for t in values]),
                    timeline.typical([t["longest_pause"]["start_s"] for t in values]),
                    timeline.typical([1000 * t["longest_pause"]["duration_s"] for t in values], 1)]
            lines.append(f"| {model} | {GROUPS[group]} | {n} / {concurrency} | "
                         + " | ".join(cols) + f" | {closing} |")
        lines.append("")
    if "auxiliary_cells" in data:
        lines += instrument_tables(data)
    return "\n".join(lines)


def write_outputs(output, data, *, pdf=False):
    payloads = {".json": (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(),
                ".md": markdown(data).encode()}
    if pdf:
        payloads[".pdf"] = timeline.timeline_pdf(data)
    paths = [Path(output).with_suffix(ext) for ext in payloads]
    protected = (REPO_ROOT / "resultados/raw", REPO_ROOT / "datos")
    if any(path.resolve().is_relative_to(root.resolve()) for path in paths for root in protected):
        raise ValueError("los derivados se escriben fuera de raw y datos")
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path, contents in zip(paths, payloads.values()):
            with path.open("xb") as stream:
                created.append(path)
                stream.write(contents)
    except BaseException:
        for path in created:
            path.unlink()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeline-pdf", action="store_true", help="dibujar cronologías con ReportLab")
    args = parser.parse_args(argv)
    try:
        write_outputs(args.output, analyze(args.campaign), pdf=args.timeline_pdf)
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as exc:
        print(f"diagnóstico: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
