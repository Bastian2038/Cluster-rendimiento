"""Fixtures CPU de identidad, integridad, recuperación y pruebas lógicas.

Los dobles pequeños no acreditan soporte observado de CUDA/FSDP2: esa
calificación corresponde al piloto autorizado en la imagen fijada.
"""

from array import array
from contextlib import nullcontext
import copy
from itertools import product
import json
from pathlib import Path
import pickle
import random
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from infraestructura import sft_checkpoint as checkpoint
from infraestructura import sft_protocol as protocol
from infraestructura import sft_train as train


IDENTITY = {"node_id": "pegasus", "hostname": "spark-cfea", "gpu_uuid": "GPU-" + "1" * 32,
            "rank": 0, "world_size": 1, "phase": "pilot", "run_id": "pilot-single",
            "campaign_id": "fixture-cpu"}


class SmallTensor:
    """Unos pocos valores FP32 de CPU, sin importar torch ni NumPy."""

    dtype = "torch.float32"
    copied_sizes = []

    def __init__(self, values, shape=None):
        self.values = array("f", values)
        self.shape = tuple(shape or (len(self.values),))

    def detach(self):
        return self

    def cpu(self):
        self.copied_sizes.append(self.numel())
        return self

    def numpy(self):
        return self.values

    def numel(self):
        return len(self.values)

    def is_contiguous(self):
        return True

    def view(self, *shape):
        return SmallTensor(self.values)

    def copy_(self, other):
        self.values = array("f", other.values)

    def __getitem__(self, key):
        if len(self.shape) == 2:
            start, stop, stride = key.indices(self.shape[0])
            assert stride == 1
            width = self.shape[1]
            return SmallTensor(self.values[start * width:stop * width], (stop - start, width))
        return SmallTensor(self.values[key])


class DistributedTensor:
    def __init__(self, local, shape):
        self.local, self.shape = local, shape
        self.placements = [SimpleNamespace(is_shard=lambda dim: dim == 0)]

    def to_local(self):
        return self.local


class SmallModel:
    def __init__(self, values=(1, 2, 3)):
        self.parameters = {"layer.weight": SmallTensor(values), "norm.weight": SmallTensor([4])}

    def named_parameters(self):
        return iter(self.parameters.items())


class SmallOptimizer:
    def __init__(self):
        self.param_groups = [{"lr": 0.001, "momentum": 0, "weight_decay": 0}]

    def state_dict(self):
        return {"state": {}, "param_groups": copy.deepcopy(self.param_groups)}

    def load_state_dict(self, state):
        self.param_groups = copy.deepcopy(state["param_groups"])


class SmallTorch:
    float32 = "torch.float32"

    def __init__(self):
        self.cpu_rng = SmallTensor([17])
        self.cuda_rng = [SmallTensor([23])]
        self.loads = []
        self.cuda = SimpleNamespace(get_rng_state_all=lambda: self.cuda_rng,
                                    set_rng_state_all=self._cuda_rng, device_count=lambda: 1)

    def _cuda_rng(self, state):
        self.cuda_rng = state

    def get_rng_state(self):
        return self.cpu_rng

    def set_rng_state(self, state):
        self.cpu_rng = state

    def save(self, value, path):
        # Solo fixtures creadas por este test; el runtime real usa weights_only.
        Path(path).write_bytes(pickle.dumps(value))

    def load(self, path, *, map_location, weights_only):
        assert map_location == "cpu" and weights_only is True
        self.loads.append(str(path))
        return pickle.loads(Path(path).read_bytes())

    def no_grad(self):
        return nullcontext()


class SftTrainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = protocol.load_config()

    def test_import_and_help_cannot_import_frameworks(self):
        script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'safetensors'}:
        raise AssertionError('framework importado durante help')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from infraestructura import sft_train, sft_checkpoint
sft_train.main(['--help'])
"""
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True,
                                text=True, timeout=3, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint-dir", result.stdout)

    def test_phase_steps_preserve_original_cursor(self):
        self.assertEqual(train.phase_steps("pilot", self.config), [1, 2, 3, 4])
        self.assertEqual(train.phase_steps("resume", self.config), [3, 4])
        self.assertEqual(train.phase_steps("measure", self.config), list(range(1, 26)))

    def test_package_and_runtime_versions_are_distinct_and_both_required(self):
        package = "2.13.0a0+9186a08b2c.nv26.7.59513937"
        runtime = "2.13.0a0+9186a08b2c.nv26.07"
        image = {"torch_version": package, "torch_runtime_version": runtime,
                 "transformers_version": "5.6.1"}
        torch, transformers = SimpleNamespace(__version__=runtime), SimpleNamespace(__version__="5.6.1")
        with patch.object(train.importlib.metadata, "version", return_value=package) as metadata:
            versions = train.framework_versions({"image": image}, torch, transformers)
            self.assertEqual(versions, image)
            metadata.assert_called_once_with("torch")
            for field in image:
                wrong = dict(image, **{field: "otra-version"})
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                    train.framework_versions({"image": wrong}, torch, transformers)

    def test_historical_recipe_does_not_enable_new_controls_or_require_environment(self):
        with patch.dict(train.os.environ, {}, clear=True):
            self.assertIsNone(train.validate_determinism_environment({"training": {}}))
            self.assertIsNone(train.configure_determinism({"training": {}}, object()))

    def test_v2_requires_exact_policy_and_cublas_environment_before_cuda(self):
        config = {"training": {"determinism": dict(train.DETERMINISM)}}
        for environment in ({}, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
            with patch.dict(train.os.environ, environment, clear=True), self.assertRaisesRegex(ValueError, "launcher"):
                train.validate_determinism_environment(config)
        with patch.dict(train.os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            self.assertEqual(train.validate_determinism_environment(config), train.DETERMINISM)
            for policy in ({}, {**train.DETERMINISM, "warn_only": True},
                           {**train.DETERMINISM, "algorithms": 1}):
                with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "cinco controles"):
                    train.validate_determinism_environment({"training": {"determinism": policy}})

    def test_v2_records_effective_strict_controls_without_calling_cuda(self):
        calls = []
        state = {"algorithms": False, "warn_only": True}

        def set_algorithms(enabled, *, warn_only):
            calls.append((enabled, warn_only))
            state.update(algorithms=enabled, warn_only=warn_only)

        def unexpected_cuda():
            self.fail("los controles deben fijarse antes de consultar CUDA")

        torch = SimpleNamespace(use_deterministic_algorithms=set_algorithms,
            are_deterministic_algorithms_enabled=lambda: state["algorithms"],
            is_deterministic_algorithms_warn_only_enabled=lambda: state["warn_only"],
            backends=SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True)),
            cuda=SimpleNamespace(is_available=unexpected_cuda))
        config = {"training": {"determinism": dict(train.DETERMINISM)}}
        with patch.dict(train.os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            observed = train.configure_determinism(config, torch)
            self.assertEqual(observed, train.DETERMINISM)
            self.assertEqual(calls, [(True, False)])
            torch.are_deterministic_algorithms_enabled = lambda: False
            with self.assertRaisesRegex(RuntimeError, "efectivos"):
                train.configure_determinism(config, torch)

    def test_identity_records_each_ranks_controls_before_group_even_on_oom(self):
        class CudaOOM(RuntimeError):
            pass

        torch = Mock()
        dist = torch.distributed
        modules = {"torch": torch, "torch.distributed": dist, "torch.distributed.fsdp": dist.fsdp,
                   "torch.distributed.device_mesh": dist.device_mesh, "transformers": Mock(), "safetensors": Mock()}
        torch.are_deterministic_algorithms_enabled.return_value = True
        torch.is_deterministic_algorithms_warn_only_enabled.return_value = False
        torch.cuda.is_available.return_value = True
        torch.cuda.device_count.return_value = 1
        torch.cuda.OutOfMemoryError = CudaOOM
        dist.all_gather_object.side_effect = CudaOOM("CUDA out of memory")
        inventory = protocol.cluster_config(self.config)
        hosts = {row["node_id"]: row["expected_hostname"] for row in inventory["nodes"]}
        env = {"WORLD_SIZE": "4", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        with (patch.dict(sys.modules, modules), patch.dict(train.os.environ, env),
              patch.object(train.observation, "collect_sample", return_value={}),
              patch.object(train, "framework_versions"), patch.object(train, "resume_digest"),
              patch.object(train.socket, "gethostname") as hostname, patch.object(protocol, "load_config") as load):
            for phase, rank, version in product(("pilot", "resume", "measure"), range(4), ("v1", "v2")):
                with self.subTest(phase=phase, rank=rank, version=version):
                    fields = {"determinism": train.DETERMINISM} if version == "v2" else {}
                    load.return_value = {**self.config, "training": fields}
                    node = inventory["groups"]["four"][rank]
                    gpu = f"GPU-00000000-0000-0000-0000-{rank:012x}"
                    torch.cuda.get_device_properties.return_value.uuid = gpu
                    hostname.return_value = hosts[node]
                    train.os.environ["RANK"] = str(rank)
                    identity = dict(IDENTITY, phase=phase, rank=rank, world_size=4,
                                    hostname=hosts[node], node_id=node, gpu_uuid=gpu)
                    out = self.root / version / node / phase
                    args = SimpleNamespace(config=protocol.DEFAULT_CONFIG, out=out, checkpoint_dir=self.root / "ckpt",
                                           phase=phase, node_id=node, run_id=identity["run_id"],
                                           campaign_id=identity["campaign_id"])
                    dist.init_process_group.side_effect = lambda **_: self.assertEqual(
                        json.loads((out / "identity.json").read_text()), {**identity, **fields})
                    with self.assertRaises(CudaOOM):
                        train.execute(args)
                    self.assertEqual(dist.all_gather_object.call_args.args[1], identity)
                    self.assertFalse((out / "receipt.json").exists())

    def test_tensor_shapes_bind_names_and_shapes_even_at_same_parameter_count(self):
        model = SmallModel()
        config = {"model": {"tensor_shapes_sha256": protocol.tensor_shapes_sha256(
            {"layer.weight": [3], "norm.weight": [1]})}}
        digest = train.verify_tensor_shapes(model, protocol, config)
        self.assertEqual(digest, config["model"]["tensor_shapes_sha256"])
        with patch.object(protocol, "tensor_shapes_sha256", wraps=protocol.tensor_shapes_sha256) as shapes:
            train.verify_tensor_shapes(model, protocol, config)
            shapes.assert_called_once_with({"layer.weight": [3], "norm.weight": [1]})
        model.parameters["layer.weight"].shape = (1, 3)
        with self.assertRaisesRegex(ValueError, "formas de tensores"):
            train.verify_tensor_shapes(model, protocol, config)
        model.parameters["layer.weight"].shape = (3,)
        model.parameters["other.weight"] = model.parameters.pop("layer.weight")
        with self.assertRaisesRegex(ValueError, "formas de tensores"):
            train.verify_tensor_shapes(model, protocol, config)

    def test_full_logical_shape_fixture_matches_shared_pinned_fingerprint(self):
        shapes = protocol.llama_tensor_shapes()
        model = SimpleNamespace(named_parameters=lambda: (
            (name, SimpleNamespace(shape=shape)) for name, shape in shapes.items()))
        digest = train.verify_tensor_shapes(model, protocol, self.config)
        self.assertEqual(digest, "ed8c135dde19a11c0ea4201c6bc54fa36cdc42d3d219b536c8f79751b9861733")

    def test_full_work_alignment_and_shifted_supervision(self):
        rows = protocol.frozen_tokens()
        for world in (1, 2, 4):
            train.validate_work(rows, protocol, self.config, world)
        rows[0]["labels"][0] = rows[0]["input_ids"][0]
        rows[0]["labels"][1] = rows[0]["input_ids"][1]
        with self.assertRaisesRegex(ValueError, "supervisados"):
            train.validate_work(rows, protocol, self.config, 4)

    def test_misaligned_rank_ids_fail_even_with_four_unique_samples(self):
        bad = SimpleNamespace(step_ids=lambda step, rank, world: [4 * step + ((rank + 1) % 4)])
        train.validate_work(protocol.frozen_tokens(), bad, self.config, 4)
        bad.step_ids = lambda step, rank, world: [((4 * step + rank + 1) % 100)]
        with self.assertRaisesRegex(ValueError, "alineación"):
            train.validate_work(protocol.frozen_tokens(), bad, self.config, 4)

    def test_accumulated_fsdp_gradient_matches_global_supervised_mean(self):
        means = [1.5, -3.0, 7.25, 0.75]
        for counts in ([128] * 4, [2, 17, 80, 127]):
            total = sum(counts)
            expected = sum(count * mean for count, mean in zip(counts, means)) / total
            for world in (4, 2, 1):
                gradients = []
                for rank in range(world):
                    ids = protocol.step_ids(0, rank, world)
                    gradients.append(sum(means[i] * train.fsdp_loss_scale(counts[i], total, world)
                                         for i in ids))
                with self.subTest(counts=counts, world=world):
                    self.assertAlmostEqual(sum(gradients) / world, expected)

    def test_pilot_does_not_project_parameter_hashing_as_training_time(self):
        elapsed = [0.0]

        def advance(seconds):
            elapsed[0] += seconds

        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: None), float32="float32",
                                zeros=lambda *a, **kw: SimpleNamespace(item=lambda: 4.0))
        dist = SimpleNamespace(all_reduce=lambda *a, **kw: advance(2), ReduceOp=SimpleNamespace(SUM="sum"))
        optimizer = SimpleNamespace(zero_grad=lambda **kwargs: None, step=lambda: advance(1))
        work = {"step": 4, "batches": [], "global_supervised_tokens": 4}
        with (patch.object(train.time, "perf_counter", side_effect=lambda: elapsed[0]),
              patch.object(train, "_audit_before", side_effect=lambda *a: advance(10)),
              patch.object(train, "_audit_after", side_effect=lambda *a, **kw: advance(12)),
              patch.object(train, "cuda_memory", return_value={})):
            row, _ = train._train_step(work, None, optimizer, torch=torch, dist=dist, device=0, world_size=1, audit=True)
        self.assertEqual(row["seconds"], 3)
        self.assertEqual(row["parameter_audit_seconds"], 22)

    def test_recovery_hashes_comparison_and_three_barrier_waits_have_separate_cost(self):
        elapsed = [0.0]
        events = []

        def advance(name, seconds, result=None):
            events.append(name)
            elapsed[0] += seconds
            return result

        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: None), float32="float32",
                                zeros=lambda *a, **kw: SimpleNamespace(item=lambda: 4.0))
        dist = SimpleNamespace(all_reduce=lambda *a, **kw: advance("reduce", 2),
                               barrier=lambda: advance("barrier", 5), ReduceOp=SimpleNamespace(SUM="sum"))
        optimizer = SimpleNamespace(zero_grad=lambda **kwargs: None, step=lambda: advance("sgd", 1))
        work = {"step": 3, "batches": [], "global_supervised_tokens": 4}
        with (patch.object(train.time, "perf_counter", side_effect=lambda: elapsed[0]),
              patch.object(train, "begin_recovery_diagnostic", side_effect=lambda *a: advance("before", 2, {})),
              patch.object(train, "recovery_gradient_diagnostic", side_effect=lambda *a: advance("gradient", 3)),
              patch.object(train, "finish_recovery_diagnostic", side_effect=lambda *a: advance("after", 4)),
              patch.object(train, "compare_recovery_diagnostic", side_effect=lambda *a:
                           advance("compare", 7, {"gradients_equal": False})),
              patch.object(train, "cuda_memory", return_value={})):
            row, proof = train._train_step(work, None, optimizer, torch=torch, dist=dist, device=0,
                world_size=1, audit=False, recovery_probe=True, recovery_reference={"method": "fixture"})
        self.assertEqual(row["seconds"], 3)
        self.assertEqual(row["recovery_diagnostic_seconds"], 31)
        self.assertEqual(row["parameter_audit_seconds"], 0)
        self.assertFalse(row["audit_performed"])
        self.assertIsNone(proof)
        self.assertEqual(row["recovery_diagnostic"]["rank_barriers"], 3)
        self.assertFalse(row["recovery_diagnostic"]["comparison_to_pilot"]["gradients_equal"])
        self.assertEqual(events, ["before", "barrier", "reduce", "gradient", "barrier", "sgd", "after", "compare", "barrier"])

    def test_measure_without_recovery_does_not_add_barriers_or_diagnostic(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: None), float32="float32",
                                zeros=lambda *a, **kw: SimpleNamespace(item=lambda: 4.0))
        dist = SimpleNamespace(all_reduce=lambda *a, **kw: None, ReduceOp=SimpleNamespace(SUM="sum"))
        optimizer = SimpleNamespace(zero_grad=lambda **kwargs: None, step=lambda: None)
        work = {"step": 6, "batches": [], "global_supervised_tokens": 4}
        with patch.object(train, "cuda_memory", return_value={}):
            row, _ = train._train_step(work, None, optimizer, torch=torch, dist=dist, device=0,
                                       world_size=1, audit=False)
        self.assertNotIn("recovery_diagnostic", row)
        self.assertNotIn("recovery_diagnostic_seconds", row)
        self.assertEqual(row["parameter_audit_seconds"], 0)

    def test_recovery_diagnostic_distinguishes_restored_weights_gradients_and_update(self):
        def update(multiplier=1):
            model, torch = SmallModel(), SmallTorch()
            for _, parameter in model.named_parameters():
                parameter.grad = SmallTensor([0.25 * multiplier] * parameter.numel())
            before_rng = copy.deepcopy(random.getstate())
            diagnostic = train.begin_recovery_diagnostic(model, torch, 3)
            train.recovery_gradient_diagnostic(model, torch, diagnostic)
            for _, parameter in model.named_parameters():
                for index, grad in enumerate(parameter.grad.values):
                    parameter.values[index] -= 0.001 * grad
            train.finish_recovery_diagnostic(model, torch, diagnostic)
            self.assertEqual(before_rng, random.getstate())
            return diagnostic

        original, resumed, different_gradient = update(), update(), update(2)
        comparison = train.compare_recovery_diagnostic(resumed, original)
        self.assertTrue(all(value for key, value in comparison.items() if key != "reference_phase"))
        self.assertEqual(resumed["local_updated_tensor_count"], 2)
        self.assertLessEqual(resumed["hash_chunk_bytes"], 1024 * 1024)
        changed = train.compare_recovery_diagnostic(different_gradient, original)
        self.assertTrue(changed["weights_before_forward_equal"])
        self.assertFalse(changed["gradients_equal"])
        self.assertFalse(changed["weights_after_optimizer_equal"])
        for row in resumed["tensors"]:
            self.assertEqual(len(row["gradient_sha256"]), 64)
            self.assertNotEqual(row["weight_before_forward_sha256"], row["weight_after_optimizer_sha256"])

    def test_recovery_diagnostic_oom_is_audit_evidence_for_all_three_reads(self):
        class CudaOOM(RuntimeError):
            pass

        def oom(*args):
            raise CudaOOM("CUDA out of memory")

        helpers = ("begin_recovery_diagnostic", "recovery_gradient_diagnostic", "finish_recovery_diagnostic")
        for failed in helpers:
            stages = ["train_step"]
            torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: None, OutOfMemoryError=CudaOOM),
                                    float32="float32", zeros=lambda *a, **kw: SimpleNamespace(item=lambda: 4.0))
            dist = SimpleNamespace(all_reduce=lambda *a, **kw: None, barrier=lambda: None,
                                   ReduceOp=SimpleNamespace(SUM="sum"))
            optimizer = SimpleNamespace(zero_grad=lambda **kwargs: None, step=lambda: None)
            with (patch.object(train, helpers[0], side_effect=oom if failed == helpers[0] else lambda *a: {}),
                  patch.object(train, helpers[1], side_effect=oom if failed == helpers[1] else lambda *a: None),
                  patch.object(train, helpers[2], side_effect=oom if failed == helpers[2] else lambda *a: None),
                  self.assertRaises(CudaOOM)):
                train._train_step({"step": 3, "batches": [], "global_supervised_tokens": 4}, None, optimizer,
                    torch=torch, dist=dist, device=0, world_size=1, audit=False, recovery_probe=True, set_stage=stages.append)
            self.assertEqual(stages[-1], "parameter_audit")

    def test_snapshot_index_hashes_actual_files_and_rejects_missing_tensor(self):
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (self.root / name).write_text("{}")
        (self.root / "weights.safetensors").write_bytes(b"fixture CPU, no son pesos reales")
        checkpoint.write_json(self.root / "model.safetensors.index.json",
                              {"weight_map": {"a": "weights.safetensors"}})
        mapping, source = train._snapshot_index(self.root, ["a"])
        self.assertEqual(mapping, {"a": "weights.safetensors"})
        self.assertEqual(source["shards"][0]["sha256"], checkpoint.sha256_file(self.root / "weights.safetensors"))
        self.assertEqual(set(source["metadata"]), {"config.json", "tokenizer.json", "tokenizer_config.json"})
        with self.assertRaisesRegex(ValueError, "cobertura"):
            train._snapshot_index(self.root, ["a", "b"])

    def test_snapshot_reads_only_local_slice_including_uneven_shard(self):
        class SafeHandle:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def keys(self):
                return ["a"]

            def get_slice(self, name):
                value = SmallTensor(range(10), (5, 2))
                value.get_shape = lambda: (5, 2)
                return value

        for rank, expected, shape in ((0, list(range(6)), (3, 2)), (1, list(range(6, 10)), (2, 2))):
            local = SmallTensor([0] * len(expected), shape)
            model = SimpleNamespace(named_parameters=lambda: [("a", DistributedTensor(local, (5, 2)))])
            train._load_snapshot(model, self.root, {"a": "weights.safetensors"}, torch=SmallTorch(),
                                 safe_open=lambda *args, **kwargs: SafeHandle(), rank=rank, world_size=2)
            self.assertEqual(list(local.values), expected)

    def test_weight_hash_detects_change_after_first_chunk_without_full_clone(self):
        values = SmallTensor([1, 2, 3, 4, 5, 6, 7])
        SmallTensor.copied_sizes.clear()
        with patch.object(train, "HASH_CHUNK_ELEMENTS", 2):
            before = train._tensor_hash(values)
            values.values[-1] -= 0.001
            after = train._tensor_hash(values)
        self.assertNotEqual(before, after)
        self.assertLessEqual(max(SmallTensor.copied_sizes), 2)

    def test_logical_coverage_requires_finite_nonzero_grad_and_real_update(self):
        tensors = [{"name": f"p{i}", "numel": 1, "finite_grad_elements": 1,
                    "nonzero_grad_elements": 1, "changed_shards": 1} for i in range(291)]
        tensors[0]["numel"] = tensors[0]["finite_grad_elements"] = train.PARAMETER_COUNT - 290
        proof = train.coverage_proof(tensors, [], 4)
        self.assertEqual(proof["updated_tensor_count"], 291)
        self.assertEqual(proof["parameter_count"], train.PARAMETER_COUNT)
        for field in ("finite_grad_elements", "nonzero_grad_elements", "changed_shards"):
            bad = copy.deepcopy(tensors)
            bad[-1][field] = 0
            with self.assertRaisesRegex(ValueError, "todos los tensores"):
                train.coverage_proof(bad, [], 4)
        with self.assertRaisesRegex(ValueError, "duplicado"):
            train.coverage_proof(tensors + [tensors[0]], [], 4)

    def test_local_proof_preserves_gradient_counts_and_actual_weight_hashes(self):
        model = SmallModel()
        rows = [{"name": name, "numel": tensor.numel(), "local_numel": tensor.numel(),
                 "finite_grad_elements": tensor.numel(), "nonzero_grad_elements": 1,
                 "before_sha256": train._tensor_hash(tensor)}
                for name, tensor in model.named_parameters()]
        for _, tensor in model.named_parameters():
            tensor.values[0] -= 0.001
        torch = SimpleNamespace(int64="int64", tensor=lambda values, **kwargs:
                                SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: values)))
        dist = SimpleNamespace(all_reduce=lambda *args, **kwargs: None, ReduceOp=SimpleNamespace(SUM="sum"))
        with patch.object(train, "TENSOR_COUNT", 2), patch.object(train, "PARAMETER_COUNT", 4):
            proof = train._audit_after(model, rows, torch=torch, dist=dist, device="cpu", step=4)
        self.assertEqual(len(proof["local_weight_hashes"]), 2)
        for row, source in zip(proof["local_weight_hashes"], rows):
            for field in ("name", "local_numel", "finite_grad_elements", "nonzero_grad_elements", "before_sha256"):
                self.assertEqual(row[field], source[field])
            self.assertEqual(row["after_sha256"], train._tensor_hash(model.parameters[row["name"]]))
            self.assertNotEqual(row["before_sha256"], row["after_sha256"])

    def test_effective_participants_require_correct_hosts_ranks_and_unique_gpu(self):
        inventory = protocol.cluster_config(self.config)
        rows = [dict(IDENTITY, world_size=2), dict(IDENTITY, rank=1, world_size=2,
                 node_id="octopus", hostname="spark-60c3", gpu_uuid="GPU-" + "2" * 32)]

        def gather(target, identity):
            target[:] = rows

        dist = SimpleNamespace(all_gather_object=gather)
        train._check_participants(rows[0], inventory, dist=dist)
        rows[1]["hostname"] = rows[0]["hostname"]
        with self.assertRaisesRegex(ValueError, "identidad"):
            train._check_participants(rows[0], inventory, dist=dist)

    def test_group_is_recorded_with_effective_backend_and_current_phase(self):
        inventory = protocol.cluster_config(self.config)
        for phase in ("pilot", "resume", "measure"):
            identity = dict(IDENTITY, phase=phase)
            out = self.root / phase
            out.mkdir()
            dist = SimpleNamespace(all_gather_object=lambda target, row: target.__setitem__(slice(None), [row]),
                                   get_backend=lambda: "nccl")
            roster = train.record_group(out, identity, inventory, dist=dist)
            group = json.loads((out / "group.json").read_text())
            self.assertEqual(group, {**identity, "backend": "nccl", "participants": [identity]})
            self.assertEqual(roster, [identity])
            with self.assertRaises(FileExistsError):
                train.record_group(out, identity, inventory, dist=dist)

    def test_invalid_phase_or_backend_cannot_publish_group(self):
        inventory = protocol.cluster_config(self.config)
        for index, (phase, backend) in enumerate((("resume", "nccl"), ("pilot", "gloo"))):
            out = self.root / str(index)
            out.mkdir()
            dist = SimpleNamespace(all_gather_object=lambda target, row: target.__setitem__(
                slice(None), [dict(row, phase=phase)]), get_backend=lambda: backend)
            with self.assertRaises(ValueError):
                train.record_group(out, IDENTITY, inventory, dist=dist)
            self.assertFalse((out / "group.json").exists())

    def test_oom_before_group_preserves_typed_cuda_evidence_without_group_file(self):
        class CudaOOM(RuntimeError):
            pass

        def failed_gather(*args):
            raise CudaOOM("CUDA out of memory during NCCL identity exchange")

        dist = SimpleNamespace(all_gather_object=failed_gather)
        torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=CudaOOM))
        try:
            train.record_group(self.root, IDENTITY, protocol.cluster_config(self.config), dist=dist)
        except CudaOOM as exc:
            train._failure(self.root, IDENTITY, exc, torch, "group_identity")
        else:
            self.fail("el intercambio debía fallar antes de publicar group.json")
        self.assertFalse((self.root / "group.json").exists())
        failure = json.loads((self.root / "failure.json").read_text())
        self.assertEqual(failure["status"], "cuda_oom")
        self.assertEqual(failure["stage"], "group_identity")
        self.assertEqual(failure["message"], "CUDA out of memory during NCCL identity exchange")

    def test_oom_audit_before_and_after_optimizer_have_separate_stage(self):
        class CudaOOM(RuntimeError):
            pass

        def oom(*args, **kwargs):
            raise CudaOOM("CUDA out of memory")

        for source in ("before", "after", "optimizer"):
            out = self.root / source
            out.mkdir()
            stages = ["train_step"]
            torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: None,
                                    OutOfMemoryError=CudaOOM), float32="float32",
                                    zeros=lambda *a, **kw: SimpleNamespace(item=lambda: 4.0))
            dist = SimpleNamespace(all_reduce=lambda *a, **kw: None, ReduceOp=SimpleNamespace(SUM="sum"))
            optimizer = SimpleNamespace(zero_grad=lambda **kw: None,
                                        step=oom if source == "optimizer" else lambda: None)
            work = {"step": 4, "batches": [], "global_supervised_tokens": 4}
            with (patch.object(train, "_audit_before", side_effect=oom if source == "before" else lambda *a: []),
                  patch.object(train, "_audit_after", side_effect=oom if source == "after" else lambda *a, **kw: {})):
                try:
                    train._train_step(work, None, optimizer, torch=torch, dist=dist, device="cpu",
                                      world_size=1, audit=True, set_stage=stages.append)
                except CudaOOM as exc:
                    train._failure(out, IDENTITY, exc, torch, stages[-1])
                else:
                    self.fail("se esperaba conservar la excepción OOM original")
            failure = json.loads((out / "failure.json").read_text())
            with self.subTest(source=source):
                self.assertEqual(failure["type"], "cuda_oom")
                self.assertEqual(failure["stage"], "train_step" if source == "optimizer" else "parameter_audit")
                self.assertEqual(failure["exception_type"], f"{CudaOOM.__module__}.{CudaOOM.__qualname__}")

    def test_uuid_single_gpu_fallback_does_not_require_host_pid(self):
        gpu = "GPU-12345678-1234-5678-1234-567812345678"
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda device: object(),
                                                    device_count=lambda: 1))
        sample = {"gpu_apps": [{"pid": -1, "gpu_uuid": gpu}], "gpus": [{"uuid": gpu}]}
        self.assertEqual(train._gpu_uuid(torch, "cuda:0", sample), gpu)
        sample["gpus"].append({"uuid": "GPU-ambigua"})
        with self.assertRaisesRegex(ValueError, "no observable"):
            train._gpu_uuid(torch, "cuda:0", sample)

    def test_only_typed_cuda_oom_has_oom_classification(self):
        class CudaOOM(RuntimeError):
            pass

        torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=CudaOOM))
        for index, (exc, expected) in enumerate(((RuntimeError("CUDA out of memory"), "indeterminate"),
                                                (CudaOOM("CUDA out of memory."), "cuda_oom"),
                                                (CudaOOM("CPUAllocator: out of memory"), "indeterminate"),
                                                (CudaOOM("origen ambiguo"), "indeterminate"),
                                                (MemoryError("sin RAM"), "indeterminate"))):
            out = self.root / str(index)
            out.mkdir()
            train._failure(out, IDENTITY, exc, torch, "train_step")
            failure = json.loads((out / "failure.json").read_text())
            self.assertEqual(failure["status"], expected)
            self.assertEqual(failure["message"], str(exc))
            self.assertEqual(failure["stage"], "train_step")
            self.assertEqual(failure["exception_type"], f"{type(exc).__module__}.{type(exc).__qualname__}")
            for field, value in IDENTITY.items():
                self.assertEqual(failure[field], value)
            if expected == "cuda_oom":
                self.assertEqual(failure["type"], "cuda_oom")

    def test_checkpoint_and_restore_keep_origin_separate_from_training(self):
        class AliasedOOM(RuntimeError):
            pass

        torch = SimpleNamespace(OutOfMemoryError=AliasedOOM,
                                cuda=SimpleNamespace(OutOfMemoryError=AliasedOOM))
        for stage in ("checkpoint_write", "restore"):
            for source, message in (("cuda", "CUDA out of memory"), ("cpu", "CPUAllocator: out of memory")):
                out = self.root / f"{stage}-{source}"
                out.mkdir()
                train._failure(out, IDENTITY, AliasedOOM(message), torch, stage)
                failure = json.loads((out / "failure.json").read_text())
                self.assertEqual(failure["stage"], stage)
                self.assertEqual(failure["status"], "cuda_oom" if source == "cuda" else "indeterminate")

    def test_background_telemetry_preserves_hardware_na_and_identity(self):
        memory = {"allocated_bytes": 4, "device_free_bytes": 1024}
        sample = {"hostname": IDENTITY["hostname"], "gpus": [{"uuid": IDENTITY["gpu_uuid"],
                  "power_draw": None}], "memory_pressure": {"some": {"avg10": 0.1}}}
        with patch.object(train.observation, "collect_sample",
                          side_effect=lambda: dict(sample, monotonic_ns=time.monotonic_ns())):
            sampler = train.Telemetry(self.root / "telemetry.jsonl", IDENTITY, lambda: memory)
            sampler.stage = "measured"
            sampler.start()
            sampler.stop()
        row = json.loads((self.root / "telemetry.jsonl").read_text())
        self.assertEqual(row["training_stage"], "measured")
        self.assertEqual(row["cuda_memory"], memory)
        self.assertIsNone(row["gpus"][0]["power_draw"])

    def test_telemetry_boundary_uses_sample_timestamp_instead_of_current_stage(self):
        sampler = train.Telemetry(self.root / "telemetry.jsonl", IDENTITY, lambda: {})
        with patch.object(train.time, "monotonic_ns", side_effect=[100, 200]):
            self.assertEqual(sampler.set_stage("measured"), 100)
            self.assertEqual(sampler.set_stage("validation"), 200)
        self.assertEqual(sampler.stage_at(99), "startup")
        self.assertEqual(sampler.stage_at(100), "measured")
        self.assertEqual(sampler.stage_at(199), "measured")
        self.assertEqual(sampler.stage_at(200), "validation")


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(random.setstate, random.getstate())
        self.root = Path(self.temp.name)
        self.torch = SmallTorch()
        self.model, self.optimizer = SmallModel(), SmallOptimizer()
        self.contract = {"model_revision": "pinned", "corpus_sha256": "b" * 64}
        self.sampling = {"next_ids": [8, 9, 10, 11], "zero_based_next_step": 2}
        self.saved = checkpoint.save_checkpoint(
            self.root / "checkpoint", torch=self.torch, model=self.model, optimizer=self.optimizer,
            identity=IDENTITY, contract=self.contract, step=2, sampling=self.sampling, learning_rate=0.001)

    def restore(self, **overrides):
        options = dict(torch=self.torch, model=self.model, optimizer=self.optimizer,
                       identity=dict(IDENTITY, phase="resume"), contract=self.contract,
                       expected_step=2, expected_sampling=self.sampling, learning_rate=0.001,
                       expected_sha256=self.saved["metadata_sha256"])
        options.update(overrides)
        return checkpoint.restore_checkpoint(self.root / "checkpoint", **options)

    def test_fresh_objects_restore_local_weights_cursor_optimizer_and_rng(self):
        expected_random = random.random()
        self.model, self.optimizer = SmallModel([99, 98, 97]), SmallOptimizer()
        self.optimizer.param_groups[0]["lr"] = 1
        self.torch.cpu_rng = SmallTensor([0])
        self.torch.cuda_rng = [SmallTensor([0])]
        restored = self.restore()
        self.assertEqual(list(self.model.parameters["layer.weight"].values), [1, 2, 3])
        self.assertEqual(self.optimizer.param_groups[0]["lr"], 0.001)
        self.assertEqual(random.random(), expected_random)
        self.assertEqual(list(self.torch.cpu_rng.values), [17])
        self.assertEqual(list(self.torch.cuda_rng[0].values), [23])
        self.assertTrue(restored["rng_restored"])
        self.assertEqual(restored["next_step"], 3)
        self.assertEqual(restored["files"], self.saved["files"])

    def test_modified_tensor_fails_before_any_deserialization(self):
        path = self.root / "checkpoint/rank-00000/tensor-0000.pt"
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.restore()
        self.assertEqual(self.torch.loads, [])

    def test_topology_cursor_and_sampling_cannot_change(self):
        for options in ({"identity": dict(IDENTITY, world_size=2)}, {"expected_step": 3},
                        {"expected_sampling": {"next_ids": [1]}}):
            with self.assertRaises(ValueError):
                self.restore(**options)
        self.assertEqual(self.torch.loads, [])

    def test_existing_checkpoint_cannot_be_overwritten(self):
        with self.assertRaises(FileExistsError):
            checkpoint.save_checkpoint(self.root / "checkpoint", torch=self.torch, model=self.model,
                                       optimizer=self.optimizer, identity=IDENTITY, contract=self.contract,
                                       step=2, sampling=self.sampling, learning_rate=0.001)

    def test_resume_requires_original_pilot_receipt_and_matching_optional_digest(self):
        config = protocol.load_config()
        args = SimpleNamespace(out=str(self.root / "out/resume"), config=str(protocol.DEFAULT_CONFIG),
                               checkpoint_sha256=None)
        pilot = self.root / "out/pilot"
        pilot.mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            train.resume_digest(args, IDENTITY, config)
        artifacts = {}
        for name in ("identity.json", "group.json", "steps.jsonl", "telemetry.jsonl"):
            (pilot / name).write_text("fixture CPU\n")
            artifacts[name] = {"sha256": checkpoint.sha256_file(pilot / name),
                               "bytes": (pilot / name).stat().st_size}
        receipt = {**IDENTITY, "status": "ok", "checkpoint": self.saved, "artifacts": artifacts,
                   "config_sha256": checkpoint.sha256_file(Path(args.config)),
                   "model_revision": config["model"]["revision"],
                   **{key: config["workload"][key] for key in ("corpus_sha256", "tokens_sha256")}}
        path = pilot / "receipt.json"
        checkpoint.write_json(path, receipt)
        digest = train.resume_digest(args, IDENTITY, config)
        self.assertEqual(digest, self.saved["metadata_sha256"])
        args.checkpoint_sha256 = "f" * 64
        with self.assertRaisesRegex(ValueError, "difiere"):
            train.resume_digest(args, IDENTITY, config)
        args.checkpoint_sha256 = None
        for field, value in (("phase", "resume"), ("hostname", "otro-host")):
            bad = dict(receipt, **{field: value})
            path.write_text(json.dumps(bad))  # Solo muta una fixture temporal.
            with self.assertRaisesRegex(ValueError, "identidad, fase"):
                train.resume_digest(args, IDENTITY, config)
        receipt["checkpoint"] = dict(self.saved, metadata_sha256="e" * 64)
        path.write_text(json.dumps(receipt))
        changed = train.resume_digest(args, IDENTITY, config)
        with self.assertRaisesRegex(ValueError, "índice"):
            self.restore(expected_sha256=changed)
        self.assertEqual(self.torch.loads, [])


if __name__ == "__main__":
    unittest.main()
