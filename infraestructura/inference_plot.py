"""Gráfico derivado de una campaña de inferencia; nunca escribe evidencia raw."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _summary(values: list[float]) -> dict[str, float | int]:
    return {"median": statistics.median(values), "min": min(values),
            "max": max(values), "blocks": len(values)}


def prepare_plot_data(report: dict) -> dict:
    """Selecciona solo celdas válidas y pares ya verificados por el informe."""
    cells: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in report.get("cells", []):
        if row.get("status", "valid") != "valid" or row.get("group") not in {"four", "pair"}:
            continue
        try:
            key = (str(row["model_key"]), int(row["input_tokens"]),
                   int(row["concurrency"]), row["group"])
            block = int(row["block"])
            throughput = row["output_tokens_per_second"]
            ttft = row["ttft"]["p50_ms"]
            duration = row["t32_seconds"]
        except (KeyError, TypeError, ValueError):
            continue
        if block < 1 or key[1] < 1 or key[2] < 1 or not all(map(_positive, (throughput, ttft, duration))):
            continue
        if block in cells[key]:
            raise ValueError(f"celda duplicada: {key}, bloque {block}")
        cells[key][block] = {"throughput": float(throughput), "ttft": float(ttft)}

    speedups: dict[tuple, dict[int, float]] = defaultdict(dict)
    for row in report.get("paired_comparisons", []):
        try:
            key = (str(row["model"]), int(row["input_tokens"]), int(row["concurrency"]))
            block, speedup = int(row["block"]), row["s4_2"]
        except (KeyError, TypeError, ValueError):
            continue
        if block < 1 or not _positive(speedup):
            continue
        if block in speedups[key]:
            raise ValueError(f"par duplicado: {key}, bloque {block}")
        speedups[key][block] = float(speedup)

    result = []
    coordinates = sorted({(model, length, concurrency) for model, length, concurrency, _ in cells})
    for model, length, concurrency in coordinates:
        group_data = {}
        for group in ("four", "pair"):
            blocks = cells.get((model, length, concurrency, group), {})
            if blocks:
                group_data[group] = {
                    "output_tokens_per_second": _summary([v["throughput"] for v in blocks.values()]),
                    "ttft_p50_ms": _summary([v["ttft"] for v in blocks.values()]),
                }
        paired = speedups.get((model, length, concurrency), {})
        result.append({"model_key": model, "input_tokens": length,
                       "concurrency": concurrency, "groups": group_data,
                       "s4_2": _summary(list(paired.values())) if paired else None})
    return {"points": result, "models": sorted({row["model_key"] for row in result})}


def plot_report(report: dict, output: Path) -> dict:
    """Escribe una figura SVG/PNG nueva con medianas y rangos entre bloques."""
    output = Path(output).resolve()
    if "raw" in output.parts:
        raise ValueError("el gráfico debe escribirse fuera de raw")
    if output.suffix.lower() not in {".svg", ".png"}:
        raise ValueError("el gráfico requiere salida .svg o .png")
    if output.exists():
        raise FileExistsError(output)
    data = prepare_plot_data(report)
    if not data["points"]:
        return {"status": "skipped", "reason": "sin celdas válidas", "output": None}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib no está instalado; no se generó el gráfico") from exc

    panels = sorted({(p["model_key"], p["input_tokens"]) for p in data["points"]})
    fig, axes = plt.subplots(len(panels), 3, figsize=(15, 3.5 * len(panels)),
                             squeeze=False, constrained_layout=True)
    colors = {"four": "#1769aa", "pair": "#c66b17"}
    try:
        for row_index, (model, length) in enumerate(panels):
            points = [p for p in data["points"] if (p["model_key"], p["input_tokens"]) == (model, length)]
            for column, (field, title) in enumerate((
                ("output_tokens_per_second", "Output tokens/s"),
                ("ttft_p50_ms", "TTFT p50 (ms)"),
                ("s4_2", "Speedup 4/2 (T₂/T₄)"),
            )):
                ax = axes[row_index][column]
                if field == "s4_2":
                    for point in points:
                        value = point["s4_2"]
                        if value:
                            ax.errorbar(point["concurrency"], value["median"],
                                        yerr=[[value["median"] - value["min"]],
                                              [value["max"] - value["median"]]],
                                        fmt="o", color="#417d42", capsize=3)
                    ax.axhline(1, color="0.6", linestyle=":", linewidth=1)
                else:
                    for group in ("four", "pair"):
                        for point in points:
                            value = point["groups"].get(group, {}).get(field)
                            if value:
                                ax.errorbar(point["concurrency"] + (0.08 if group == "four" else -0.08),
                                            value["median"], yerr=[[value["median"] - value["min"]],
                                                                   [value["max"] - value["median"]]],
                                            fmt="o", color=colors[group], capsize=3,
                                            label=group if point is points[0] else None)
                    if row_index == 0:
                        ax.legend(title="Nodos", labels=["4", "2"])
                ax.set(title=f"{model} · entrada {length} · {title}", xlabel="Concurrencia")
                ax.set_xticks(sorted({point["concurrency"] for point in points}))
                ax.grid(alpha=0.25)
        fig.suptitle("Inferencia: mediana y rango entre bloques; pares 4/2 verificados")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            try:
                fig.savefig(stream, format=output.suffix[1:].lower(), dpi=160)
            except Exception:
                stream.close()
                output.unlink(missing_ok=True)
                raise
    finally:
        plt.close(fig)
    return {"status": "created", "output": str(output),
            "panels": len(panels), "points": len(data["points"])}
