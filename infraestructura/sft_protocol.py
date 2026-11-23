"""Frozen workload, topology and local checks for Llama full-parameter SFT."""

import hashlib
import json
from pathlib import Path
import struct

from infraestructura import cluster, manifest
from infraestructura.rutas import REPO_ROOT


DEFAULT_CONFIG = REPO_ROOT / "experimentos/configs/sft-llama31-8b.json"
PROTOCOL = "llama31-full-sft-4-2-1-v2"
_V1_PROTOCOL = "llama31-full-sft-4-2-1-v1"
GROUPS = ("four", "pair", "single")
SOURCE_MODULES = (
    "__init__", "cluster", "manifest", "rutas", "sft_protocol", "sft_train", "sft_checkpoint",
    "sft_report", "sft_runner", "sft_runtime", "sft", "inference", "inference_evidence",
    "inference_metrics", "inference_runtime", "inference_observation", "serving_capacity_protocol",
    "serving_capacity_runner",
)
CORPUS_SHA256 = "c9229d2e8ad8ec84da91d2f30b9cf2d0f18213f405995e053fd81588c2eacaaf"
TOKENS_SHA256 = "33ed6a6646bc673718436893862133523c07f0269790f426d72776f88db61907"
_RECIPE_FIELDS = ("schema_version", "protocol_id", "reference_preflight_sha256", "model",
                  "image", "network", "training", "workload", "measurement")
_V1_RECIPE_SHA256 = "04909adb6412482b3f74f83dd6e386440db70e900a025c8eedaabe50d8e64a9a"
_RECIPE_SHA256 = "636e5ed133301167c73e2909efa708ab1efb43a1847eb6832413334be999254d"
_WORDS = "pan sal agua coco libro papel mesa bola foto flor disco carta mapa radio metal oro".split()
_INSTRUCTION = "Devuelve en orden inverso los 127 productos finales del inventario."
# Tokenization of the fixed instruction, word vocabulary and response delimiter
# under the pinned Llama tokenizer. Each worker also encodes every actual string.
_PREFIX = [128000, 14934, 47956, 665, 39488, 65683, 78, 2537, 220, 6804, 42425,
           1913, 3916, 1624, 17459, 3370, 13]
_WORD_IDS = [7363, 4371, 56562, 83450, 52111, 70257, 71544, 98287,
             26656, 70240, 54969, 76647, 69210, 9063, 9501, 95993]
_DELIMITER = [198, 37655, 20006, 25]
_EOT = 128009


def validate_config(config):
    """A different numerical recipe needs a distinct protocol and fresh pilot."""
    recipe = {key: config[key] for key in _RECIPE_FIELDS}
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    approved = {PROTOCOL: _RECIPE_SHA256, _V1_PROTOCOL: _V1_RECIPE_SHA256}
    if hashlib.sha256(encoded).hexdigest() != approved.get(config["protocol_id"]):
        raise ValueError("la receta, imagen, red o referencia difiere del protocolo SFT declarado")
    for field in ("cluster_config", "reference_preflight"):
        path = Path(config[field])
        if path.is_absolute() or ".." in path.parts or not path.name:
            raise ValueError(f"{field} debe ser una ruta relativa al repositorio")
    roots = [Path(config["paths"][field]) for field in ("workspace", "hf_hub")]
    if any(not p.is_absolute() or ".." in p.parts for p in roots):
        raise ValueError("se requieren rutas absolutas de trabajo y caché")
    if roots[0].is_relative_to(roots[1]) or roots[1].is_relative_to(roots[0]):
        raise ValueError("workspace y caché de pesos deben estar en directorios independientes")
    if any(roots[0].resolve().is_relative_to(path.resolve())
           for path in (REPO_ROOT / "datos", REPO_ROOT / "resultados/raw")):
        raise ValueError("el workspace SFT debe estar fuera de datos y resultados raw")


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_bytes())
    validate_config(config)
    return config


def cluster_config(config):
    return cluster.load_config(REPO_ROOT / config["cluster_config"])


def tensor_shapes_sha256(shapes):
    encoded = json.dumps(shapes, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def llama_tensor_shapes():
    """Logical parameter layout of the pinned 32-layer Llama checkpoint."""
    hidden, intermediate, key_value, vocabulary = 4096, 14336, 1024, 128256
    shapes = {"model.embed_tokens.weight": [vocabulary, hidden], "model.norm.weight": [hidden],
              "lm_head.weight": [vocabulary, hidden]}
    layer_shapes = {
        "self_attn.q_proj.weight": [hidden, hidden], "self_attn.k_proj.weight": [key_value, hidden],
        "self_attn.v_proj.weight": [key_value, hidden], "self_attn.o_proj.weight": [hidden, hidden],
        "mlp.gate_proj.weight": [intermediate, hidden], "mlp.up_proj.weight": [intermediate, hidden],
        "mlp.down_proj.weight": [hidden, intermediate], "input_layernorm.weight": [hidden],
        "post_attention_layernorm.weight": [hidden],
    }
    for layer in range(32):
        shapes.update({f"model.layers.{layer}.{name}": shape.copy() for name, shape in layer_shapes.items()})
    return shapes


def snapshot_tensor_shapes(directory, weight_map):
    """Read only the length and JSON header of each safetensors shard."""
    shapes = {}
    for shard in sorted(set(weight_map.values())):
        if Path(shard).name != shard:
            raise ValueError("nombre de shard de pesos inválido")
        with (directory / shard).open("rb") as stream:
            prefix = stream.read(8)
            size = int.from_bytes(prefix, "little")
            if len(prefix) != 8 or not 0 < size <= 4 * 1024 * 1024:
                raise ValueError("longitud inválida de cabecera safetensors")
            data = stream.read(size)
            if len(data) != size:
                raise ValueError("cabecera safetensors truncada")
            header = json.loads(data)
        if not isinstance(header, dict):
            raise ValueError("cabecera safetensors inválida")
        for name, record in header.items():
            if name == "__metadata__":
                continue
            if name in shapes or weight_map.get(name) != shard or record["dtype"] != "BF16":
                raise ValueError("nombre, shard o dtype incompatible con el índice del modelo")
            shapes[name] = record["shape"]
    if set(shapes) != set(weight_map):
        raise ValueError("cabeceras de pesos incompletas")
    return shapes


def _word_indices(index):
    key = f"ipre-sft-v1/17/{index}".encode("ascii")
    return [byte % len(_WORDS) for byte in hashlib.shake_256(key).digest(363)]


def _example(index):
    words = [_WORDS[n] for n in _word_indices(index)]
    return {"id": index, "input": _INSTRUCTION + " " + " ".join(words) + "\nRespuesta:",
            "output": " " + " ".join(reversed(words[-127:]))}


def corpus_bytes():
    rows = (json.dumps(_example(i), ensure_ascii=False, separators=(",", ":")) for i in range(100))
    data = "".join(row + "\n" for row in rows).encode("utf-8")
    if hashlib.sha256(data).hexdigest() != CORPUS_SHA256:
        raise ValueError("el generador no reproduce el JSONL SFT fijado")
    return data


def _token_row(index, context, answer):
    if len(context) != 384 or len(answer) != 128:
        raise ValueError(f"longitud SFT inesperada en ejemplo {index}: {len(context)} + {len(answer)}")
    return {"id": index, "input_ids": context + answer, "labels": [-100] * len(context) + answer}


def tokens_sha256(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(struct.pack("<512I512i", *row["input_ids"], *row["labels"]))
    return digest.hexdigest()


def frozen_tokens():
    """The offline reference is usable on a CPU host without Transformers."""
    rows = []
    for i in range(100):
        words = [_WORD_IDS[n] for n in _word_indices(i)]
        answer = words[-127:][::-1] + [_EOT]
        rows.append(_token_row(i, _PREFIX + words + _DELIMITER, answer))
    if tokens_sha256(rows) != TOKENS_SHA256:
        raise ValueError("las posiciones y etiquetas de referencia difieren del protocolo")
    return rows


def encode_corpus(tokenizer):
    """Re-encode actual text using the checkpoint tokenizer on every worker."""
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    rows = []
    for i in range(100):
        example = _example(i)
        prompt = tokenizer.encode(example["input"], add_special_tokens=False)
        answer = tokenizer.encode(example["output"], add_special_tokens=False)
        rows.append(_token_row(i, [tokenizer.bos_token_id] + prompt, answer + [eot]))
    if tokens_sha256(rows) != TOKENS_SHA256:
        raise ValueError("el tokenizer efectivo no reproduce las entradas y etiquetas fijadas")
    return rows


def step_ids(step, rank, world_size):
    """Return one rank's microbatches of a zero-based optimizer step."""
    if (type(world_size) is not int or world_size not in (1, 2, 4)
            or type(rank) is not int or not 0 <= rank < world_size
            or type(step) is not int or not 0 <= step < 25):
        raise ValueError("paso, rank o world_size SFT inválido")
    return list(range(step * 4 + rank, (step + 1) * 4, world_size))


def model_subdir(config):
    model = config["model"]
    return Path("models--" + model["id"].replace("/", "--")) / "snapshots" / model["revision"]


def qualified_pilot_groups(pilot):
    groups = pilot.get("qualified_groups")
    if (pilot.get("stage") != "pilot" or pilot.get("status") != "complete"
            or pilot.get("removal_confirmed") is not True or pilot.get("errors") != []
            or not isinstance(groups, list) or not groups
            or groups != [name for name in GROUPS if name in groups]):
        raise ValueError("la matriz requiere un piloto cerrado, sin errores y con topologías calificadas")
    return list(groups)


def worker_argv(config, job, rank, phase, *, config_path, model_dir, out, checkpoint_dir,
                campaign_id, run_id):
    return [
        "python3", "-B", "-m", "torch.distributed.run",
        "--nnodes", str(len(job["nodes"])), "--nproc-per-node", "1", "--node-rank", str(rank),
        "--max-restarts", "0", "--master-addr", config["network"]["node_ips"]["pegasus"],
        "--master-port", str(config["network"]["master_port"]), "--module", "infraestructura.sft_train",
        "--config", str(config_path), "--model-dir", str(model_dir), "--phase", phase,
        "--out", str(out), "--checkpoint-dir", str(checkpoint_dir), "--node-id", job["nodes"][rank],
        "--campaign-id", campaign_id, "--run-id", run_id,
    ]


def make_plan(config, stage, qualified=None):
    validate_config(config)
    if stage not in ("pilot", "run"):
        raise ValueError("etapa SFT desconocida")
    groups = list(GROUPS) if qualified is None else list(qualified)
    if groups != [group for group in GROUPS if group in groups] or (stage == "pilot" and groups != list(GROUPS)):
        raise ValueError("grupos SFT repetidos, desconocidos o fuera del orden 4 → 2 → 1")
    blocks = [0] if stage == "pilot" else range(1, config["measurement"]["blocks"] + 1)
    phases = ["pilot", "resume"] if stage == "pilot" else ["measure"]
    jobs, launches = [], {}
    for block in blocks:
        for group in groups:
            name = f"pilot-{group}" if stage == "pilot" else f"b{block}-{group}"
            job = {"name": name, "group": group, "block": block, "nodes": cluster.TOPOLOGIES[group]}
            jobs.append(job)
            launches[name] = {
                phase: [worker_argv(config, job, rank, phase, config_path="/work/config.json",
                                    model_dir=Path("/models") / model_subdir(config),
                                    out=f"/work/out/{phase}", checkpoint_dir="/work/checkpoint",
                                    campaign_id="<campaign_id>", run_id="<run_id>")
                        for rank in range(len(job["nodes"]))]
                for phase in phases
            }
    return {"protocol_id": config["protocol_id"], "stage": stage, "jobs": jobs, "launches": launches,
            "limit_seconds": config["measurement"][f"{stage}_limit_seconds"],
            "cleanup_reserve_seconds": config["measurement"]["cleanup_reserve_seconds"],
            "workload": config["workload"], "training": config["training"],
            "measurement": config["measurement"]}


def local_preflight(config):
    validate_config(config)
    result = {"scope": "local_cpu", "protocol_id": config["protocol_id"], "errors": [],
              "corpus_sha256": hashlib.sha256(corpus_bytes()).hexdigest(),
              "tokens_sha256": tokens_sha256(frozen_tokens()), "examples": 100,
              "tokenization_source": "pinned_tokens", "weights_content_rehashed": False}
    try:
        inventory = cluster_config(config)
        ref = REPO_ROOT / config["reference_preflight"]
        if manifest.sha256_file(ref) != config["reference_preflight_sha256"]:
            raise ValueError("el preflight histórico tiene otra huella")
        history = json.loads(ref.read_bytes())
        old_nodes = {node["node_id"]: node for node in history["nodes"]}
        if history["errors"] or set(old_nodes) != set(inventory["groups"]["four"]):
            raise ValueError("la referencia histórica no acredita los cuatro nodos")
        result["historical_hosts"] = {key: node["hostname"] for key, node in old_nodes.items()}
        record = old_nodes["pegasus"]["models"][config["model"]["key"]]
        snapshot = Path(config["paths"]["hf_hub"]) / model_subdir(config)
        for name, digest in record["metadata_sha256"].items():
            if manifest.sha256_file(snapshot / name) != digest:
                raise ValueError(f"metadatos locales distintos: {name}")
        for item in record["shards"]:
            shard = snapshot / item["name"]
            if shard.resolve(strict=True).name != item["blob_name"] or shard.stat().st_size != item["size_bytes"]:
                raise ValueError(f"shard local distinto: {shard.name}")
        index = json.loads((snapshot / "model.safetensors.index.json").read_bytes())
        if (len(index["weight_map"]) != config["model"]["tensor_count"]
                or index["metadata"]["total_size"] != 2 * config["model"]["parameter_count"]):
            raise ValueError("el índice BF16 no corresponde a los parámetros previstos")
        shapes = snapshot_tensor_shapes(snapshot, index["weight_map"])
        result["tensor_shapes_sha256"] = tensor_shapes_sha256(shapes)
        if shapes != llama_tensor_shapes() or result["tensor_shapes_sha256"] != config["model"]["tensor_shapes_sha256"]:
            raise ValueError("nombres o formas de los tensores distintos del snapshot fijado")
        result["model_snapshot"] = str(snapshot)
        result["model_metadata_sha256"] = record["metadata_sha256"]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result["errors"].append(str(exc))
    return result
