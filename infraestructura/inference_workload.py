"""Deterministic, tokenizer-verified custom workloads for vLLM 0.24.0.

No model, GPU, network, or filesystem access occurs on import. The caller owns
the tokenizer and the destination. JSONL prompts already include one chat
template and require ``vllm bench serve --skip-chat-template``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path
from typing import Any


SEED = 20261106
WORDS = (
    "archivo sensor mapa vector serie tabla resumen registro señal puerto nodo "
    "muestra lectura valor fecha cambio enlace sistema memoria tarea proceso "
    "entrada salida usuario texto pregunta respuesta informe prueba bloque "
    "secuencia código cálculo detalle evento intervalo medida fuente destino "
    "estado condición número criterio operación recurso dato parámetro índice"
).split()


def load_local_tokenizer(path: str | Path, revision: str):
    """Load a pinned tokenizer from local cache at execution time only."""
    from transformers import AutoTokenizer  # optional runtime dependency

    return AutoTokenizer.from_pretrained(
        str(path), revision=revision, local_files_only=True, trust_remote_code=False
    )


def _ids(tokenizer: Any, prompt: str) -> list[int]:
    ids = tokenizer(prompt).input_ids  # matches vLLM CustomDataset.sample
    if not isinstance(ids, (list, tuple)) or not all(isinstance(x, int) for x in ids):
        raise ValueError("tokenizer did not return a flat integer input_ids list")
    return list(ids)


def _render(tokenizer: Any, content: str, *, thinking_off: bool) -> str:
    kwargs = {"enable_thinking": False} if thinking_off else {}
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=False,
        **kwargs,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("chat template did not return a nonempty string")
    return prompt


def _exact_prompt(tokenizer: Any, rng: random.Random, target: int, index: int,
                  *, thinking_off: bool) -> tuple[str, list[int]]:
    # Search a stable pseudorandom lexical stream by word count, then scan the
    # boundary word character by character. A bounded failure remains explicit.
    prefix = f"Registro {index:03d}. Resume las observaciones: "
    count = len(_ids(tokenizer, _render(tokenizer, prefix, thinking_off=thinking_off)))
    if count > target:
        raise ValueError(f"template and prefix exceed target {target}: {count}")
    body = " ".join(rng.choice(WORDS) for _ in range(target * 2))
    def candidate(n: int) -> tuple[str, list[int]]:
        prompt = _render(tokenizer, prefix + body[:n], thinking_off=thinking_off)
        return prompt, _ids(tokenizer, prompt)
    _, high_ids = candidate(len(body))
    if len(high_ids) < target:
        raise ValueError(f"token target {target} not reached for prompt {index}")
    low, high = 0, len(body)
    while low + 1 < high:
        middle = (low + high) // 2
        prompt, ids = candidate(middle)
        if len(ids) == target:
            return prompt, ids
        if len(ids) < target:
            low = middle
        else:
            high = middle
    prompt, ids = candidate(low)
    if len(ids) == target:
        return prompt, ids
    for n in range(max(0, low - 32), min(len(body), high + 32) + 1):
        prompt, ids = candidate(n)
        if len(ids) == target:
            return prompt, ids
    # Unicode boundary changes may skip a count. Try an ASCII suffix whose
    # tokenization is checked at every step; never assume one char is one token.
    base = prefix + body[:low]
    for n in range(1, 65):
        prompt = _render(tokenizer, base + ("a" * n), thinking_off=thinking_off)
        ids = _ids(tokenizer, prompt)
        if len(ids) == target:
            return prompt, ids
        if len(ids) > target + 8:
            break
    raise ValueError(f"cannot construct exactly {target} tokens for prompt {index}")


def generate_workload(
    tokenizer: Any, *, model_id: str, model_revision: str,
    tokenizer_id: str, tokenizer_revision: str, input_tokens: int,
    count: int = 48, warmup_count: int = 16, output_tokens: int = 128,
    seed: int = SEED, enable_thinking: bool = False,
) -> dict[str, Any]:
    """Build one reusable profile, with disjoint warmup and measured prompts.

    ``enable_thinking`` must be false for Qwen. It is recorded explicitly; the
    Qwen template receives ``enable_thinking=False``. Other models receive no
    Qwen-specific template argument.
    """
    if input_tokens <= 0 or output_tokens <= 0 or count <= 0 or not 0 <= warmup_count < count:
        raise ValueError("invalid token or sample counts")
    if enable_thinking:
        raise ValueError("thinking must be disabled for this campaign")
    for name, value in (("model_id", model_id), ("model_revision", model_revision),
                        ("tokenizer_id", tokenizer_id), ("tokenizer_revision", tokenizer_revision)):
        if not value:
            raise ValueError(f"{name} is required")
    thinking_off = model_id.lower().startswith("qwen/")
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        prompt, ids = _exact_prompt(tokenizer, rng, input_tokens, index,
                                    thinking_off=thinking_off)
        rows.append({
            "id": f"{input_tokens}-{index:03d}",
            "phase": "warmup" if index < warmup_count else "measured",
            "prompt": prompt,
            "prompt_tokens": len(ids),
            "token_ids": ids,
            "output_tokens": output_tokens,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "token_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        })
    if len({row["prompt_sha256"] for row in rows}) != count:
        raise ValueError("generated prompts are not distinct")
    return {
        "schema_version": 1, "seed": seed, "model_id": model_id,
        "model_revision": model_revision, "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision, "input_tokens": input_tokens,
        "output_tokens": output_tokens, "thinking_enabled": False,
        "template_mode": "pre_rendered_user_generation_prompt",
        "vllm_flags": ["--skip-chat-template", "--disable-shuffle", "--no-oversample",
                       "--custom-output-len", "-1", "--num-warmups", "0"],
        "rows": rows,
    }


def write_workload(workload: dict[str, Any], directory: str | Path) -> dict[str, str]:
    """Write new files exclusively; never replace prior workload evidence."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"destination is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    paths = {phase: directory / f"{phase}.jsonl" for phase in ("warmup", "measured")}
    paths["metadata"] = directory / "metadata.json"
    for phase in ("warmup", "measured"):
        with paths[phase].open("x", encoding="utf-8") as stream:
            for row in workload["rows"]:
                if row["phase"] == phase:
                    stream.write(json.dumps({"prompt": row["prompt"], "output_tokens": row["output_tokens"]},
                                            ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {key: value for key, value in workload.items() if key != "rows"}
    metadata["rows"] = [{key: value for key, value in row.items() if key != "prompt"}
                        for row in workload["rows"]]
    metadata["files_sha256"] = {
        phase: hashlib.sha256(paths[phase].read_bytes()).hexdigest()
        for phase in ("warmup", "measured")
    }
    with paths["metadata"].open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    return {name: str(path) for name, path in paths.items()}


def validate_vllm_reader(
    workload: dict[str, Any], directory: str | Path, tokenizer: Any, *,
    reader_cls: Any = None, package_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Read both JSONL files with vLLM CustomDataset and persist proof.

    The import is deliberately lazy. A reader or dependency failure raises
    before any benchmark can be launched by the caller. ``reader_cls`` and
    ``package_versions`` support small CPU fixtures without vLLM installed.
    """
    if reader_cls is None:
        from vllm.benchmarks.datasets.datasets import CustomDataset
        reader_cls = CustomDataset
    if package_versions is None:
        package_versions = {
            name: importlib.metadata.version(name)
            for name in ("vllm", "pandas", "transformers")
        }
    if str(package_versions.get("vllm", "")).split("+", 1)[0] != "0.24.0":
        raise ValueError(f"CustomDataset contract requires vLLM 0.24.0, got {package_versions.get('vllm')!r}")
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("input_tokens") != workload["input_tokens"] or metadata.get("output_tokens") != workload["output_tokens"]:
        raise ValueError("written metadata does not match workload")
    checked: dict[str, Any] = {}
    for phase in ("warmup", "measured"):
        path = directory / f"{phase}.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if metadata.get("files_sha256", {}).get(phase) != digest:
            raise ValueError(f"{phase} JSONL hash differs from metadata")
        expected = [row for row in workload["rows"] if row["phase"] == phase]
        metadata_rows = [row for row in metadata["rows"] if row["phase"] == phase]
        if len(expected) != len(metadata_rows):
            raise ValueError(f"{phase} metadata count differs")
        for row, recorded in zip(expected, metadata_rows):
            if recorded.get("id") != row["id"] or recorded.get("token_ids") != row["token_ids"] or recorded.get("prompt_sha256") != row["prompt_sha256"]:
                raise ValueError(f"{phase} metadata row differs")
        jsonl_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(jsonl_rows) != len(expected):
            raise ValueError(f"{phase} JSONL count differs")
        for index, (row, item) in enumerate(zip(expected, jsonl_rows)):
            if item != {"prompt": row["prompt"], "output_tokens": row["output_tokens"]}:
                raise ValueError(f"{phase} JSONL row {index} differs")
        dataset = reader_cls(dataset_path=str(path), random_seed=workload["seed"], disable_shuffle=True)
        samples = dataset.sample(
            tokenizer=tokenizer, num_requests=len(expected), output_len=-1,
            skip_chat_template=True, no_oversample=True,
        )
        if len(samples) != len(expected):
            raise ValueError(f"{phase} vLLM returned {len(samples)} samples, expected {len(expected)}")
        checked_rows = []
        for index, (row, sample) in enumerate(zip(expected, samples)):
            if sample.prompt != row["prompt"]:
                raise ValueError(f"{phase} vLLM changed prompt order/text at {index}")
            if sample.prompt_len != workload["input_tokens"] or sample.prompt_len != row["prompt_tokens"]:
                raise ValueError(f"{phase} vLLM prompt length differs at {index}")
            if sample.expected_output_len != workload["output_tokens"] or sample.expected_output_len != row["output_tokens"]:
                raise ValueError(f"{phase} vLLM output length differs at {index}")
            checked_rows.append({"id": row["id"], "prompt_sha256": row["prompt_sha256"],
                                 "prompt_len": sample.prompt_len,
                                 "expected_output_len": sample.expected_output_len})
        checked[phase] = {"count": len(samples), "jsonl_sha256": digest, "samples": checked_rows}
    report = {
        "status": "validated", "reader": "vllm.benchmarks.datasets.datasets.CustomDataset",
        "package_versions": package_versions, "seed": workload["seed"],
        "reader_arguments": {"disable_shuffle": True, "output_len": -1,
                             "skip_chat_template": True, "no_oversample": True},
        "phases": checked,
    }
    report_path = directory / "reader-validation.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="local tokenizer snapshot")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True, help="pinned model/tokenizer revision")
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output", required=True, help="new, empty destination directory")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    tokenizer = load_local_tokenizer(args.model_path, args.revision)
    workload = generate_workload(
        tokenizer, model_id=args.model_id, model_revision=args.revision,
        tokenizer_id=args.model_id, tokenizer_revision=args.revision,
        input_tokens=args.input_tokens, seed=args.seed,
    )
    workload["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    workload["tokenizer_local_path"] = str(Path(args.model_path).resolve())
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("tokenizer has no auditable chat_template string")
    workload["chat_template_sha256"] = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    paths = write_workload(workload, args.output)
    validate_vllm_reader(workload, args.output, tokenizer)
    paths["reader_validation"] = str(Path(args.output) / "reader-validation.json")
    print(json.dumps(paths, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
