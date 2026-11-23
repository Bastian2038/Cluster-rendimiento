import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import Mock

from infraestructura import sft, sft_protocol as protocol


class SFTProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = protocol.load_config()

    def test_span_supervision_and_canonical_bytes(self):
        raw = protocol.corpus_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), protocol.CORPUS_SHA256)
        examples = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual([row["id"] for row in examples], list(range(100)))
        frozen = protocol.frozen_tokens()
        for tokens in frozen:
            self.assertEqual(tokens["labels"][:384], [-100] * 384)
            self.assertEqual(tokens["labels"][384:], tokens["input_ids"][384:])
            self.assertEqual(len(tokens["input_ids"]), 512)
            self.assertEqual(sum(x != -100 for x in tokens["labels"][1:]), 128)
        lookup = {}
        for example, tokens in zip(examples, frozen):
            lookup[example["input"]] = tokens["input_ids"][1:384]
            lookup[example["output"]] = tokens["input_ids"][384:-1]
        tokenizer = Mock(bos_token_id=128000)
        tokenizer.convert_tokens_to_ids.return_value = 128009
        tokenizer.encode.side_effect = lambda text, *, add_special_tokens: lookup[text]
        self.assertEqual(protocol.encode_corpus(tokenizer), frozen)
        lookup[examples[0]["input"]][0] += 1
        with self.assertRaisesRegex(ValueError, "tokenizer efectivo"):
            protocol.encode_corpus(tokenizer)

    def test_each_global_update_is_identical_across_topologies(self):
        for step in range(25):
            expected = list(range(4 * step, 4 * step + 4))
            for size in (4, 2, 1):
                ids_by_rank = [protocol.step_ids(step, r, size) for r in range(size)]
                self.assertEqual(sorted(i for ids in ids_by_rank for i in ids), expected)
                self.assertTrue(all(len(ids) == 4 // size for ids in ids_by_rank))
        for args in ((25, 0, 4), (0, 4, 4), (0, -1, 2), (0, 0, 3), (False, 0, 1)):
            with self.assertRaises(ValueError):
                protocol.step_ids(*args)

    def test_tensor_names_and_shapes_are_anchored_to_safetensors_headers(self):
        shapes = protocol.llama_tensor_shapes()
        model = self.config["model"]
        self.assertEqual(len(shapes), model["tensor_count"])
        self.assertEqual(sum(math.prod(shape) for shape in shapes.values()), model["parameter_count"])
        self.assertEqual(protocol.tensor_shapes_sha256(shapes), model["tensor_shapes_sha256"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "model.safetensors"
            header = json.dumps({key: {"dtype": "BF16", "shape": value} for key, value in shapes.items()}).encode()
            (root / name).write_bytes(len(header).to_bytes(8, "little") + header)
            weight_map = dict.fromkeys(shapes, name)
            self.assertEqual(protocol.snapshot_tensor_shapes(root, weight_map), shapes)
            weight_map["different.weight"] = weight_map.pop("lm_head.weight")
            with self.assertRaises(ValueError):
                protocol.snapshot_tensor_shapes(root, weight_map)

    def test_pilot_and_qualified_matrix_order_and_launcher(self):
        pilot = protocol.make_plan(self.config, "pilot")
        self.assertEqual([job["group"] for job in pilot["jobs"]], list(protocol.GROUPS))
        launch = pilot["launches"]["pilot-four"]["resume"][2]
        self.assertEqual(launch[launch.index("--node-rank") + 1], "2")
        self.assertEqual(launch[launch.index("--node-id") + 1], "griffin")
        self.assertIn("infraestructura.sft_train", launch)
        matrix = protocol.make_plan(self.config, "run", ["four", "pair"])
        self.assertEqual([j["name"] for j in matrix["jobs"]],
                         ["b1-four", "b1-pair", "b2-four", "b2-pair", "b3-four", "b3-pair"])
        with self.assertRaises(ValueError):
            protocol.make_plan(self.config, "run", ["pair", "four"])
        with self.assertRaises(ValueError):
            protocol.make_plan(self.config, "pilot", ["four"])

    def test_recipe_rejects_incompatible_numerics_and_image(self):
        for section, field, value in (("training", "global_batch", 8), ("training", "seed", True),
                                      ("model", "revision", "main"), ("image", "id", "latest"),
                                      ("measurement", "measured_steps", 19)):
            changed = copy.deepcopy(self.config)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                protocol.validate_config(changed)
        for name in ("datos", "resultados/raw"):
            changed = copy.deepcopy(self.config)
            changed["paths"]["workspace"] = str(protocol.REPO_ROOT / name / "sft")
            with self.subTest(name=name), self.assertRaises(ValueError):
                protocol.validate_config(changed)

    def test_only_closed_qualified_pilot_can_plan_a_matrix(self):
        pilot = {"stage": "pilot", "status": "complete", "removal_confirmed": True,
                 "errors": [], "qualified_groups": ["four", "pair"]}
        self.assertEqual(protocol.qualified_pilot_groups(pilot), ["four", "pair"])
        for change in ({"status": "partial"}, {"removal_confirmed": False}, {"stage": "run"},
                       {"errors": ["timeout"]}, {"qualified_groups": []},
                       {"qualified_groups": ["pair", "four"]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                protocol.qualified_pilot_groups({**pilot, **change})

    def test_cli_preview_exclusive_and_execution_explicit(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stderr(StringIO()):
            p = Path(temp) / "plan.json"
            self.assertEqual(sft.main(["plan", "--stage", "run", "--output", str(p)]), 0)
            self.assertEqual(len(json.loads(p.read_bytes())["jobs"]), 9)
            self.assertEqual(sft.main(["plan", "--output", str(p)]), 2)
            self.assertEqual(sft.main(["pilot"]), 2)
            self.assertEqual(sft.main(["run", "--pilot-dir", temp]), 2)
