"""Reconstruct the client-visible timeline of a closed vLLM request batch."""

import math
from statistics import median

from infraestructura.inference_metrics import validate_bench_result


# Target, concurrency control, length control.
PROFILES = ((2048, 8), (2048, 4), (512, 8))


def client_timeline(raw, samples, input_tokens, concurrency):
    """Use detailed client times and the frozen reader order, all in seconds.

    Starts are recorded inside the concurrency semaphore. The last token time
    is reconstructed from streamed arrivals, independently of the reported T32.
    """
    validate_bench_result(raw, expected_prompt_tokens=input_tokens, expected_count=32)
    if raw.get("max_concurrency") != concurrency or type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrencia de la cronología incompatible con el cliente")
    starts = raw.get("start_times")
    if not isinstance(starts, list) or len(starts) != 32 or any(
        type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in starts
    ):
        raise ValueError("faltan 32 tiempos de inicio finitos del mismo cliente")
    if not isinstance(samples, list) or len(samples) != 32:
        raise ValueError("se requieren los 32 prompts medidos en el orden validado")
    ids = [sample.get("id") if isinstance(sample, dict) else None for sample in samples]
    if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != 32:
        raise ValueError("IDs de prompt ausentes o duplicados")
    first_start = min(starts)
    requests = []
    boundaries = []
    for index, (prompt_id, start, ttft, intervals, tokens) in enumerate(
        zip(ids, starts, raw["ttfts"], raw["itls"], raw["output_lens"])
    ):
        if len(intervals) != tokens - 1:
            raise ValueError(f"{prompt_id}: número de intervalos incompatible con la salida")
        start -= first_start
        first = start + ttft
        after_first = math.fsum(intervals)
        last = first + after_first
        peak_index = max(range(len(intervals)), key=intervals.__getitem__)
        requests.append({
            "row": index, "prompt_id": prompt_id, "start_s": start,
            "first_token_s": first, "last_token_s": last, "ttft_s": ttft,
            "after_first_s": after_first, "max_itl_s": intervals[peak_index],
            "max_itl_start_s": first + math.fsum(intervals[:peak_index]),
        })
        boundaries.extend(((start, 1), (last, -1)))
    active = peak = 0
    for _, change in sorted(boundaries):
        active += change
        peak = max(peak, active)
    if peak > concurrency:
        raise ValueError("la cronología reconstruida excede las plazas del cliente")
    last = max(requests, key=lambda entry: entry["last_token_s"])
    pause = max(requests, key=lambda entry: entry["max_itl_s"])
    ordered_starts = sorted(r["start_s"] for r in requests)
    duration = raw["duration"]
    before = math.fsum(raw["ttfts"]) / concurrency
    after = math.fsum(r["after_first_s"] for r in requests) / concurrency
    span = last["last_token_s"]
    if span > duration + max(1e-6, 1e-9 * duration):
        raise ValueError("la cronología reconstruida supera la duración del lote")
    return {
        "t32_s": duration, "span_s": span, "t32_minus_span_s": duration - span,
        "peak_in_flight": peak, "last_prompt_id": last["prompt_id"],
        "first_start_beyond_initial_slots_s": ordered_starts[concurrency] if concurrency < len(requests) else None,
        "last_request_start_s": ordered_starts[-1],
        "longest_pause": {"prompt_id": pause["prompt_id"], "start_s": pause["max_itl_start_s"],
                          "duration_s": pause["max_itl_s"]},
        "last_admission_to_last_token_s": span - max(r["start_s"] for r in requests),
        "ttft_slot_seconds_div_c": before, "stream_slot_seconds_div_c": after,
        "remaining_slot_seconds_div_c": duration - before - after,
        "requests": requests,
    }


def paired_timelines(cells, comparisons):
    """Return contrasts only when the verified campaign has that exact 4/2 pair."""
    indexed = {(c["model_key"], c["block"], c["input_tokens"], c["concurrency"], c["group"]): c
               for c in cells}
    if len(indexed) != len(cells):
        raise ValueError("cronología primaria duplicada")
    result = []
    for pair in comparisons:
        key = (pair["model"], pair["block"], pair["input_tokens"], pair["concurrency"])
        if key[2:] not in PROFILES or "s4_2" not in pair:
            continue
        a, b = indexed.get((*key, "four")), indexed.get((*key, "pair"))
        if a is None or b is None or not (a["timeline"]["valid"] and b["timeline"]["valid"]):
            continue
        four, two = a["timeline"], b["timeline"]
        if not math.isclose(two["t32_s"] / four["t32_s"], pair["s4_2"], rel_tol=1e-12):
            raise ValueError("la cronología contradice el cociente verificado")
        result.append({
            "model_key": key[0], "block": key[1], "input_tokens": key[2],
            "concurrency": key[3], "s4_2": pair["s4_2"],
            "four_minus_two_s": {f: four[f] - two[f] for f in (
                "t32_s", "ttft_slot_seconds_div_c", "stream_slot_seconds_div_c",
                "remaining_slot_seconds_div_c", "last_admission_to_last_token_s")},
        })
    return result


def typical(values, digits=3):
    """Spanish display of a median and observed range among paired blocks."""
    if not values:
        return "—"
    def number(value):
        return format(value, f".{digits}f").replace(".", ",")
    return f"{number(median(values))} [{number(min(values))}–{number(max(values))}]"


def timeline_pdf(data):
    """Render one page per model, load and block with common time axes."""
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen.canvas import Canvas

    buffer = BytesIO()
    pdf = Canvas(buffer, pagesize=A4, invariant=1, pageCompression=1)
    pdf.setTitle("Cronologías de las solicitudes de inferencia")
    width, height = A4
    left, right = 58, width - 35
    panels = (("four", "Cuatro nodos"), ("pair", "Pegasus + Octopus"), ("single", "Pegasus"))
    groups = {}
    for cell in data["primary_cells"]:
        key = (cell["model_key"], cell["input_tokens"], cell["concurrency"], cell["block"])
        groups.setdefault(key, {})[cell["group"]] = cell
    for (model, length, concurrency, block), cells in sorted(groups.items()):
        pdf.setFillColorRGB(0.08, 0.12, 0.18)
        pdf.setFont("Helvetica-Bold", 17)
        pdf.drawString(left, height - 43, f"{model}   /   {length} tokens   /   concurrencia {concurrency}")
        pdf.setFont("Helvetica", 10)
        pdf.drawString(left, height - 62, f"Bloque {block}. Cada fila sigue una solicitud en el orden del JSONL.")
        pdf.setFillColorRGB(0.1, 0.32, 0.58)
        pdf.drawString(left, height - 82, "En azul, desde el inicio HTTP hasta el primer token")
        pdf.setFillColorRGB(0.08, 0.42, 0.31)
        pdf.drawString(left, height - 97, "En verde, desde el primer token hasta el último")
        pdf.setFillColorRGB(0.65, 0.26, 0.01)
        pdf.drawString(left, height - 111, "En naranja, la mayor pausa entre tokens de cada lote")
        maximum = max(c["t32_s"] for c in cells.values())
        tick = max(1, math.ceil(maximum / 5))
        scale = (right - left) / (tick * 5)
        for panel_index, (group, label) in enumerate(panels):
            cell = cells.get(group)
            top = height - 127 - 218 * panel_index
            pdf.setFillColorRGB(0.08, 0.12, 0.18)
            pdf.setFont("Helvetica-Bold", 11)
            pdf.drawString(left, top, label)
            if not cell or not cell["timeline"]["valid"]:
                pdf.setFont("Helvetica", 9)
                pdf.drawString(left, top - 20, "Cronología sin evidencia completa")
                continue
            pdf.drawRightString(right, top, f"T32 = {cell['t32_s']:.2f} s".replace(".", ","))
            for i in range(6):
                x = left + scale * tick * i
                pdf.setStrokeColorRGB(0.84, 0.87, 0.89)
                pdf.setLineWidth(0.4)
                pdf.line(x, top - 179, x, top - 9)
                pdf.setFont("Helvetica", 8)
                pdf.drawCentredString(x, top - 192, str(i * tick))
            for req in cell["timeline"]["requests"]:
                y = top - 12 - req["row"] * 5.25
                first = left + scale * req["first_token_s"]
                pdf.setLineWidth(2.8)
                pdf.setStrokeColorRGB(0.1, 0.32, 0.58)
                pdf.line(left + scale * req["start_s"], y, first, y)
                pdf.setStrokeColorRGB(0.08, 0.42, 0.31)
                pdf.line(first, y, left + scale * req["last_token_s"], y)
                pause = cell["timeline"]["longest_pause"]
                if req["prompt_id"] == pause["prompt_id"]:
                    x = left + scale * pause["start_s"]
                    pdf.setStrokeColorRGB(0.75, 0.29, 0.01)
                    pdf.line(x, y, x + scale * pause["duration_s"], y)
                if req["row"] in (0, 7, 15, 23, 31):
                    pdf.setFont("Helvetica", 7)
                    pdf.drawRightString(left - 6, y - 2, str(req["row"] + 1))
        pdf.setFont("Helvetica", 9)
        pdf.drawString(left, 53, "Segundos desde el primer inicio HTTP observado en cada lote.")
        pdf.drawString(left, 36, data["campaign_id"])
        pdf.showPage()
    if not groups:
        pdf.drawString(left, height - 60, "Campaña cerrada sin cronologías primarias.")
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()
