"""Cliente de llegadas abiertas para serving-slo-4-2-1-v1.

Las marcas del lote son segundos desde el origen monotónico de la fase.
El plazo y las latencias parten de scheduled_s. sent_s marca la escritura
local del cuerpo HTTP completo. first_token_s y last_token_s marcan el primer
y último delta con IDs de salida de vLLM, incluso con texto vacío. stream_done_s
marca [DONE] y response_end_s registra el EOF HTTP validado. end_s coincide
con response_end_s para un éxito y marca el desenlace en los demás casos.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path


INPUT_LENGTHS = (512, 2048)
OUTPUT_TOKENS = 128
PHASE_COUNTS = {"warmup": 16, "measured": 128}
SEED = 20261106
IDENTITY_FIELDS = ("model_id", "model_revision", "tokenizer_id", "tokenizer_revision")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _verify_metadata(path, phase, payload, rows, model_key, length):
    metadata = json.loads(path.read_bytes())
    if (metadata.get("schema_version") != 1 or metadata.get("input_tokens") != length
            or metadata.get("output_tokens") != OUTPUT_TOKENS
            or metadata.get("model_key", model_key) != model_key):
        raise ValueError(f"{path}: identidad o longitud de carga incorrecta")
    identity = tuple(metadata.get(field) for field in IDENTITY_FIELDS)
    if any(not isinstance(value, str) or not value for value in identity):
        raise ValueError(f"{path}: falta identidad de modelo o tokenizer")
    if metadata["files_sha256"].get(phase) != _sha256(payload):
        raise ValueError(f"{path}: SHA-256 de {phase} incorrecto")
    recorded = [row for row in metadata["rows"] if row["phase"] == phase]
    if len(recorded) != len(rows):
        raise ValueError(f"{path}: cantidad de filas de {phase} incorrecta")
    for source, row in zip(recorded, rows):
        if (source.get("prompt_sha256") != row["prompt_sha256"]
                or source.get("prompt_tokens") != length
                or source.get("output_tokens") != OUTPUT_TOKENS):
            raise ValueError(f"{path}: identidad o longitud de fila incorrecta")
    return identity


def expected_prompts(workloads_model_dir, phase, count):
    """Reconstruye la secuencia exacta de prompts para el lote y su auditoría."""
    if phase not in PHASE_COUNTS or type(count) is not int or not 1 <= count <= PHASE_COUNTS[phase]:
        raise ValueError("fase o cantidad de solicitudes inválida")
    root = Path(workloads_model_dir)
    profiles = {}
    identities = set()
    for length in INPUT_LENGTHS:
        path = root / str(length) / f"{phase}.jsonl"
        payload = path.read_bytes()
        rows = []
        for line in payload.decode("utf-8").split("\n"):
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get("prompt")
            if (not isinstance(prompt, str) or not prompt
                    or item.get("output_tokens") != OUTPUT_TOKENS):
                raise ValueError(f"{path}: prompt o longitud de salida inválidos")
            digest = _sha256(prompt.encode("utf-8"))
            if (item.get("prompt_sha256", digest) != digest
                    or item.get("prompt_tokens", length) != length
                    or item.get("input_tokens", length) != length):
                raise ValueError(f"{path}: identidad o longitud de prompt incorrecta")
            rows.append({"prompt": prompt, "prompt_sha256": digest, "input_tokens": length})
        if not rows:
            raise ValueError(f"{path}: carga vacía")
        metadata_path = path.with_name("metadata.json")
        if metadata_path.exists():
            identities.add(_verify_metadata(metadata_path, phase, payload, rows, root.name, length))
        profiles[length] = rows
    if len(identities) > 1:
        raise ValueError("los perfiles difieren en modelo o tokenizer")
    selected = []
    for index in range(count):
        length = INPUT_LENGTHS[index % 2]
        rows = profiles[length]
        offset = index // 2
        if phase == "warmup" and offset >= len(rows):
            raise ValueError(f"faltan filas de warmup para {length} tokens")
        selected.append(rows[offset % len(rows)].copy())
    return selected


async def _sse_events(lines):
    """Agrupa líneas SSE completas, incluido data multilínea."""
    data = []
    async for line in lines:
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
            continue
        field, separator, value = line.partition(":")
        if field == "data":
            data.append(value.removeprefix(" ") if separator else "")
    if data:
        raise ValueError("evento SSE truncado")


async def _completion(response, row, stamp, model):
    finished = False
    done = False
    usage_seen = False
    token_count = 0
    async for data in _sse_events(response.aiter_lines()):
        received = stamp()
        if done:
            raise ValueError("datos SSE después de [DONE]")
        if data == "[DONE]":
            if not (finished and usage_seen):
                raise ValueError("[DONE] sin terminación y uso verificados")
            row["stream_done_s"] = received
            done = True
            continue
        chunk = json.loads(data)
        if not isinstance(chunk, dict) or "error" in chunk:
            raise ValueError("error SSE del servidor")
        if chunk.get("model") != model:
            raise ValueError("modelo SSE distinto del solicitado")
        choices = chunk.get("choices")
        usage = chunk.get("usage")
        if choices == [] and usage is not None:
            if not finished or usage_seen or not isinstance(usage, dict):
                raise ValueError("uso SSE fuera de secuencia")
            completion_tokens = usage.get("completion_tokens")
            prompt_tokens = usage.get("prompt_tokens")
            if type(completion_tokens) is int and completion_tokens >= 0:
                row["output_tokens"] = completion_tokens
            if type(prompt_tokens) is int and prompt_tokens >= 0:
                row["server_input_tokens"] = prompt_tokens
            if (type(completion_tokens) is not int or completion_tokens != OUTPUT_TOKENS
                    or type(prompt_tokens) is not int or prompt_tokens != row["input_tokens"]
                    or usage.get("total_tokens", prompt_tokens + completion_tokens)
                    != prompt_tokens + completion_tokens
                    or token_count != completion_tokens):
                raise ValueError("conteo de tokens SSE incorrecto")
            usage_seen = True
            continue
        if (finished or usage is not None or not isinstance(choices, list)
                or len(choices) != 1 or not isinstance(choices[0], dict)):
            raise ValueError("secuencia de choices SSE inválida")
        choice = choices[0]
        content = choice.get("text")
        if type(choice.get("index")) is not int or choice["index"] != 0 or not isinstance(content, str):
            raise ValueError("choice SSE inválida")
        token_ids = choice.get("token_ids")
        if token_ids is not None:
            if not isinstance(token_ids, list) or any(type(token) is not int or token < 0 for token in token_ids):
                raise ValueError("delta de IDs inválido")
            token_count += len(token_ids)
            row["output_tokens"] = token_count
            if token_ids:
                if row["first_token_s"] is None:
                    row["first_token_s"] = received
                row["last_token_s"] = received
        reason = choice.get("finish_reason")
        if reason is not None:
            row["finish_reason"] = reason
            if reason != "length" or choice.get("stop_reason") is not None:
                raise ValueError("respuesta terminada antes del límite de longitud")
            finished = True
    if not done or row["last_token_s"] is None:
        raise ValueError("stream SSE incompleto o sin IDs de salida")


async def _request(transport, model, prompt, row, origin, clock, request_timeout):
    def stamp():
        return clock() - origin

    deadline = row["scheduled_s"] + request_timeout

    def sent():
        row["sent_s"] = stamp()
        row["send_lag_s"] = row["sent_s"] - row["scheduled_s"]

    payload = {
        "model": model, "prompt": prompt, "max_tokens": OUTPUT_TOKENS,
        "n": 1, "temperature": 0, "seed": SEED, "ignore_eos": True,
        "add_special_tokens": True, "stream": True,
        "return_token_ids": True, "stream_options": {"include_usage": True},
    }
    try:
        remaining = deadline - stamp()
        if remaining <= 0:
            raise TimeoutError
        async with asyncio.timeout(remaining):
            async with transport(payload, sent) as response:
                row["http_status"] = response.status_code
                if stamp() >= deadline:
                    raise TimeoutError
                if row["sent_s"] is None:
                    raise ValueError("falta la marca de envío")
                if response.status_code == 429:
                    row.update(status="rejected", error="HTTP 429")
                    return
                if response.status_code != 200:
                    row.update(status="error", error=f"HTTP {response.status_code}")
                    return
                if response.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "text/event-stream":
                    raise ValueError("tipo HTTP distinto de text/event-stream")
                await _completion(response, row, stamp, model)
                row["response_end_s"] = stamp()
        if row["response_end_s"] >= deadline:
            raise TimeoutError
        row["status"] = "ok"
    except TimeoutError:
        row.update(status="timeout", error="plazo agotado desde la llegada programada")
    except json.JSONDecodeError:
        row.update(status="error", error="JSON SSE inválido")
    except ValueError as exc:
        row.update(status="error", error=str(exc))
    except Exception as exc:
        row.update(status="error", error=type(exc).__name__)
    finally:
        row["end_s"] = row["response_end_s"] if row["status"] == "ok" else stamp()


async def run_phase(model, phase, rate, prompts, request_timeout, transport, *, clock=None, sleep=asyncio.sleep):
    """Ejecuta un lote con reloj y transporte sustituibles por fixtures CPU."""
    if not math.isfinite(rate) or rate <= 0 or not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("la tasa y el timeout deben ser finitos y positivos")
    clock = clock if clock is not None else asyncio.get_running_loop().time
    rows = [{
        "index": index, "input_tokens": prompt["input_tokens"],
        "prompt_sha256": prompt["prompt_sha256"], "scheduled_s": index / rate,
        "sent_s": None, "send_lag_s": None, "first_token_s": None,
        "last_token_s": None, "stream_done_s": None, "response_end_s": None, "end_s": None,
        "status": "error", "output_tokens": 0,
    } for index, prompt in enumerate(prompts)]
    origin = clock()
    async with asyncio.TaskGroup() as tasks:
        for prompt, row in zip(prompts, rows):
            delay = row["scheduled_s"] - (clock() - origin)
            while delay > 0:
                await sleep(delay)
                delay = row["scheduled_s"] - (clock() - origin)
            tasks.create_task(_request(transport, model, prompt["prompt"], row,
                                       origin, clock, request_timeout))
    return {"schema_version": 1, "model": model, "phase": phase, "rate_rps": rate, "rows": rows}


async def _run_http(model, port, phase, rate, prompts, request_timeout):
    import httpx

    limits = httpx.Limits(max_connections=None)
    async with httpx.AsyncClient(limits=limits, timeout=None, trust_env=False) as client:
        @asynccontextmanager
        async def transport(payload, sent):
            async def trace(event, _info):
                # httpcore emits this after handing the full HTTP body to TCP.
                if event == "http11.send_request_body.complete":
                    sent()

            try:
                url = f"http://127.0.0.1:{port}/v1/completions"
                async with client.stream("POST", url, json=payload, extensions={"trace": trace}) as response:
                    yield response
            except httpx.TimeoutException as exc:
                raise TimeoutError from exc

        return await run_phase(model, phase, rate, prompts, request_timeout, transport)


def write_new(path, payload):
    """Publica el JSON completo mediante un enlace exclusivo en el mismo directorio."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--workloads", required=True, type=Path)
    parser.add_argument("--rate", required=True, type=float)
    parser.add_argument("--phase", required=True, choices=PHASE_COUNTS)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--request-timeout", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port debe pertenecer a [1, 65535]")
    if args.workloads.name != args.model:
        parser.error("--workloads debe apuntar al directorio del modelo solicitado")
    if os.path.lexists(args.output):
        raise FileExistsError(args.output)
    prompts = expected_prompts(args.workloads, args.phase, args.count)
    payload = asyncio.run(_run_http(args.model, args.port, args.phase, args.rate,
                                    prompts, args.request_timeout))
    write_new(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
