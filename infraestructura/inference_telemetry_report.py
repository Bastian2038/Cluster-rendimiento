#!/usr/bin/env python3
"""Descriptive telemetry summary for verified, closed inference campaign runs.

Run as ``python3 -m infraestructura.inference_telemetry_report`` with a JSON
report produced by ``python3 -m infraestructura.inference report``. Writes new
JSON and Markdown outside raw; never touches a run. No GPU, Docker, SSH, or
sampling is invoked.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from infraestructura import manifest


INTERFACES = ("enp1s0f0np0", "enP2p1s0f0np0")
GPU_FIELDS = {
    "utilization_gpu": "%", "power_draw": "W", "temperature_gpu": "°C",
    "clocks_sm": "MHz", "clocks_mem": "MHz", "memory_used": "MiB",
    "memory_total": "MiB",
}


def stamp(value):
    point = datetime.fromisoformat(value)
    if point.tzinfo is None:
        raise ValueError("UTC timestamp without offset")
    return point.timestamp()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def describe(values, unit):
    valid = [float(v) for v in values if numeric(v)]
    return {"unit": unit, "valid_samples": len(valid), "null_samples": len(values) - len(valid),
            "min": min(valid) if valid else None,
            "median": statistics.median(valid) if valid else None,
            "max": max(valid) if valid else None}


def rates(samples, interface, direction):
    result = []
    for before, after in zip(samples, samples[1:]):
        dt = (after["monotonic_ns"] - before["monotonic_ns"]) / 1e9
        a = before.get("nic_f0", {}).get(interface, {}).get(direction + "_bytes")
        b = after.get("nic_f0", {}).get(interface, {}).get(direction + "_bytes")
        if dt > 0 and numeric(a) and numeric(b) and b >= a:
            result.append((b - a) / dt)
        else:
            result.append(None)
    return describe(result, "B/s")


def cpu_busy(samples):
    result = []
    for before, after in zip(samples, samples[1:]):
        a, b = before.get("cpu_ticks", {}), after.get("cpu_ticks", {})
        keys = set(a) & set(b)
        if not keys or any(not numeric(a[k]) or not numeric(b[k]) for k in keys):
            result.append(None)
            continue
        delta = {k: b[k] - a[k] for k in keys}
        total = sum(delta.values())
        if total <= 0 or any(v < 0 for v in delta.values()):
            result.append(None)
            continue
        result.append(100 * (total - delta.get("idle", 0) - delta.get("iowait", 0)) / total)
    return describe(result, "% of aggregate CPU ticks, iowait excluded")


def load_telemetry(path, hostname):
    samples = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if line.strip():
                sample = json.loads(line)
                if sample.get("hostname") != hostname:
                    raise ValueError(f"{path}:{line_no}: hostname mismatch")
                sample["_utc_seconds"] = stamp(sample["timestamp_utc"])
                if not numeric(sample.get("monotonic_ns")):
                    raise ValueError(f"{path}:{line_no}: missing monotonic time")
                samples.append(sample)
    if any(b["monotonic_ns"] <= a["monotonic_ns"] or
           b["_utc_seconds"] < a["_utc_seconds"] for a, b in zip(samples, samples[1:])):
        raise ValueError(f"{path}: non-monotonic telemetry")
    return samples


def summarize(samples, start, end):
    if end <= start:
        raise ValueError("invalid measurement window")
    window = [s for s in samples if start <= s["_utc_seconds"] <= end]
    gaps = [(b["monotonic_ns"] - a["monotonic_ns"]) / 1e9
            for a, b in zip(window, window[1:])]
    edge_start = window[0]["_utc_seconds"] - start if window else None
    edge_end = end - window[-1]["_utc_seconds"] if window else None
    result = {
        "window_seconds": end - start,
        "sample_count": len(window),
        "nominal_expected_samples_at_1s": round(end - start) + 1,
        "first_sample_utc": window[0]["timestamp_utc"] if window else None,
        "last_sample_utc": window[-1]["timestamp_utc"] if window else None,
        "edge_gap_start_seconds": edge_start,
        "edge_gap_end_seconds": edge_end,
        "max_intersample_seconds": max(gaps) if gaps else None,
        "mem_available_gib": describe(
            [s.get("memory", {}).get("MemAvailable_kib") / 1048576
             if numeric(s.get("memory", {}).get("MemAvailable_kib")) else None for s in window], "GiB"),
        "cpu_busy": cpu_busy(window),
        "psi_memory_some_avg10": describe(
            [s.get("memory_pressure", {}).get("some", {}).get("avg10") for s in window], "%"),
        "psi_memory_full_avg10": describe(
            [s.get("memory_pressure", {}).get("full", {}).get("avg10") for s in window], "%"),
        "gpus": {}, "nic_f0": {},
    }
    uuids = sorted({gpu.get("uuid") for s in window for gpu in s.get("gpus", []) if gpu.get("uuid")})
    for uuid in uuids:
        per_sample = [next((g for g in s.get("gpus", []) if g.get("uuid") == uuid), {})
                      for s in window]
        result["gpus"][uuid] = {
            field: describe([g.get(field) for g in per_sample], unit)
            for field, unit in GPU_FIELDS.items()
        }
    for interface in INTERFACES:
        result["nic_f0"][interface] = {
            direction + "_bytes_per_second": rates(window, interface, direction)
            for direction in ("rx", "tx")
        }
    return result


def analyze(report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    refs = {Path(r["run_dir"]).resolve(): r for r in report["runs"]}
    cache = {}
    rows = []
    seen = set()
    for cell in report["cells"]:
        run_dir = Path(cell["run_dir"]).resolve()
        ref = refs.get(run_dir)
        if ref is None:
            raise ValueError("cell has no campaign run reference")
        if run_dir not in cache:
            run = manifest.load_run(run_dir)
            if run["status"] != "complete" or not (run_dir / "COMPLETE").is_file():
                raise ValueError(f"run is not COMPLETE: {run_dir}")
            manifest.verify_artifacts(run_dir, run)
            manifest_hash = digest(run_dir / "manifest.json")
            if ref["manifest_sha256"] != manifest_hash:
                raise ValueError("run manifest changed since campaign report")
            if run.get("kind") != "inference-serving":
                raise ValueError("unexpected run kind")
            assessment_path = run_dir / "metrics" / "assessment.json"
            assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
            if not assessment.get("valid"):
                raise ValueError("run assessment invalid")
            cache[run_dir] = (run, assessment, manifest_hash, {})
        run, assessment, manifest_hash, telemetry_cache = cache[run_dir]
        request = run.get("requested_config", {})
        job = request.get("job", {})
        model = job.get("model", {})
        if (request.get("campaign_id") != report["campaign_id"] or
            cell.get("model_key") != model.get("key") or
            cell.get("group") != job.get("group") or
            cell.get("block") != job.get("block") or
            ref.get("model_key") != cell.get("model_key") or
            ref.get("group") != cell.get("group") or
            ref.get("block") != cell.get("block")):
            raise ValueError("cell identity differs from verified run")
        key = (run_dir, cell["cell"])
        if key in seen:
            raise ValueError("duplicate cell in report")
        seen.add(key)
        source_cell = next((item for item in assessment["cells"]
                            if item.get("cell") == cell["cell"]), None)
        if source_cell is None or source_cell.get("status") != "valid" or any(
            source_cell.get(k) != cell.get(k) for k in
            ("measurement_started_at", "measurement_ended_at", "input_tokens", "concurrency",
             "t32_seconds")
        ):
            raise ValueError("cell differs from verified assessment")
        start, end = stamp(cell["measurement_started_at"]), stamp(cell["measurement_ended_at"])
        for participant in run["participants"]:
            node_id, hostname = participant["node_id"], participant["hostname"]
            path = run_dir / "raw" / "nodes" / node_id / "logs" / "telemetry.jsonl"
            if not path.is_file():
                raise ValueError(f"missing telemetry: {path}")
            if node_id not in telemetry_cache:
                telemetry_cache[node_id] = (load_telemetry(path, hostname), digest(path))
            samples, telemetry_hash = telemetry_cache[node_id]
            row = {"model_key": cell["model_key"], "group": cell["group"],
                   "block": cell["block"], "cell": cell["cell"], "node_id": node_id,
                   "hostname": hostname, "run_dir": str(run_dir),
                   "measurement_started_at": cell["measurement_started_at"],
                   "measurement_ended_at": cell["measurement_ended_at"],
                   "t32_seconds": cell.get("t32_seconds"),
                   "external_minus_t32_seconds": (
                       end - start - cell["t32_seconds"]
                       if numeric(cell.get("t32_seconds")) else None),
                   "source_sha256": {"manifest": manifest_hash,
                                     "assessment": digest(run_dir / "metrics" / "assessment.json"),
                                     "telemetry": telemetry_hash}}
            row.update(summarize(samples, start, end))
            rows.append(row)
    return {"campaign_id": report["campaign_id"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "analyzer_source": str(Path(__file__).resolve()),
            "analyzer_sha256": digest(Path(__file__).resolve()),
            "report_source": str(report_path.resolve()), "report_sha256": digest(report_path),
            "rows": rows,
            "method": {
                "selection": "UTC timestamps inside [measurement_started_at, measurement_ended_at], inclusive",
                "rates": "Observed sysfs netdev rx_bytes/tx_bytes differences between adjacent in-window samples divided by monotonic seconds; resets omitted",
                "temporal_precision": "Nominal 1 s polling plus query, SSH, and clock jitter; edge gaps and maximum in-window gap shown",
                "t32_relation": "External measurement markers bracket client process; T32 is client benchmark duration. Their difference includes launch/collection and is not GPU time.",
                "limits": ["Descriptive per-node values only; no causal attribution or cross-node counter sum.",
                           "GPU N/A stays null. UMA MemAvailable is host memory, not additional GPU capacity.",
                           "PSI avg10 is a rolling 10 s average and overlaps time outside the cell.",
                           "CPU busy is a host-wide tick fraction, not process utilization.",
                           "The two f0 interface names remain separate; whether they are independent physical links is not inferred.",
                           "Netdev counters may omit offloaded RoCE traffic; rates do not measure NCCL/RDMA bandwidth or link saturation.",
                           "High GPU utilization does not distinguish useful model compute from NCCL kernels or waiting.",
                           "gpu_apps memory is another device view; it is not added to UMA host memory.",
                           "No exact inter-node clock offset was measured; UTC coverage alone does not establish clock alignment."]}}


def fmt(stat):
    if stat["median"] is None:
        return f"N/A (0/{stat['valid_samples'] + stat['null_samples']})"
    return (f"{stat['min']:.3g}/{stat['median']:.3g}/{stat['max']:.3g} "
            f"{stat['unit']} ({stat['valid_samples']}/{stat['valid_samples'] + stat['null_samples']})")


def markdown(data):
    lines = [f"# Telemetría descriptiva: {data['campaign_id']}", "",
             "Cada terna es mínimo/mediana/máximo; entre paréntesis figuran muestras válidas/totales.", "",
             "| Modelo | Grupo | Bloque | Celda | Nodo | Muestras | Bordes s | Hueco máx s | Externa−T32 s | UMA disponible GiB | GPU util % | GPU potencia W | GPU temp °C | CPU activo % |",
             "|---|---|---:|---|---|---:|---|---:|---:|---|---|---|---|---|"]
    for row in data["rows"]:
        gpu = next(iter(row["gpus"].values()), {})
        field = lambda name: fmt(gpu[name]) if name in gpu else "N/A"
        lines.append("| " + " | ".join(map(str, [row["model_key"], row["group"], row["block"],
            row["cell"], row["node_id"], row["sample_count"],
            f"{row['edge_gap_start_seconds']:.2f}/{row['edge_gap_end_seconds']:.2f}" if row["sample_count"] else "N/A",
            f"{row['max_intersample_seconds']:.2f}" if row["max_intersample_seconds"] is not None else "N/A",
            f"{row['external_minus_t32_seconds']:.3f}" if row["external_minus_t32_seconds"] is not None else "N/A",
            fmt(row["mem_available_gib"]), field("utilization_gpu"), field("power_draw"),
            field("temperature_gpu"), fmt(row["cpu_busy"])])) + " |")
    lines += ["", "## Relojes, memoria GPU y presión", "",
              "Memoria GPU N/A permanece ausente; UMA disponible es memoria del host.", "",
              "| Modelo | Grupo | Bloque | Celda | Nodo | GPU | SM MHz | Mem MHz | GPU usada MiB | PSI some avg10 % | PSI full avg10 % |",
              "|---|---|---:|---|---|---|---|---|---|---|---|"]
    for row in data["rows"]:
        if not row["gpus"]:
            lines.append(f"| {row['model_key']} | {row['group']} | {row['block']} | {row['cell']} | {row['node_id']} | N/A | N/A | N/A | N/A | "
                         f"{fmt(row['psi_memory_some_avg10'])} | {fmt(row['psi_memory_full_avg10'])} |")
        for uuid, gpu in row["gpus"].items():
            lines.append(f"| {row['model_key']} | {row['group']} | {row['block']} | {row['cell']} | {row['node_id']} | {uuid} | "
                         f"{fmt(gpu['clocks_sm'])} | {fmt(gpu['clocks_mem'])} | {fmt(gpu['memory_used'])} | "
                         f"{fmt(row['psi_memory_some_avg10'])} | {fmt(row['psi_memory_full_avg10'])} |")
    lines += ["", "## Interfaces f0", "",
              "Tasas observadas de sysfs netdev en bytes por segundo, por interfaz y dirección. No se suman interfaces.", "",
              "| Modelo | Grupo | Bloque | Celda | Nodo | Interfaz | RX B/s | TX B/s |",
              "|---|---|---:|---|---|---|---|---|"]
    for row in data["rows"]:
        for interface, values in row["nic_f0"].items():
            lines.append(f"| {row['model_key']} | {row['group']} | {row['block']} | {row['cell']} | {row['node_id']} | {interface} | "
                         f"{fmt(values['rx_bytes_per_second'])} | {fmt(values['tx_bytes_per_second'])} |")
    lines += ["", "## Método y límites", ""]
    for key in ("selection", "rates", "temporal_precision", "t32_relation"):
        lines.append(f"- {data['method'][key]}")
    lines += [f"- {limit}" for limit in data["method"]["limits"]]
    lines += ["", "Hashes SHA-256 de reporte, manifiesto, assessment y telemetría por fila: ver JSON adjunto.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new JSON path outside raw")
    args = parser.parse_args()
    output = args.output.resolve()
    md = output.with_suffix(".md")
    if output.suffix != ".json" or output.exists() or md.exists():
        parser.error("output must be a new .json path with new .md sibling")
    if "resultados/raw" in str(output) or "resultados/raw" in str(md):
        parser.error("output may not be inside raw")
    result = analyze(args.report.resolve())
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    with md.open("x", encoding="utf-8") as stream:
        stream.write(markdown(result))


if __name__ == "__main__":
    main()
