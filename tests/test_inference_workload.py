import json
import tempfile
import unittest
from pathlib import Path

from infraestructura.inference_workload import generate_workload, write_workload


class FakeTokenizer:
    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
        self.template_calls.append(kwargs)
        return "<user>" + messages[0]["content"] + "<assistant>"

    def __call__(self, prompt):
        class Result:
            input_ids = list(prompt.encode("utf-8"))
        return Result()


class WorkloadTests(unittest.TestCase):
    def test_reproducible_disjoint_and_exact(self):
        tok = FakeTokenizer()
        kwargs = dict(model_id="Qwen/Qwen3.5-9B", model_revision="a" * 40,
                      tokenizer_id="Qwen/Qwen3.5-9B", tokenizer_revision="a" * 40,
                      input_tokens=90, count=6, warmup_count=2)
        first = generate_workload(tok, **kwargs)
        second = generate_workload(FakeTokenizer(), **kwargs)
        self.assertEqual(first, second)
        self.assertEqual([r["phase"] for r in first["rows"]],
                         ["warmup"] * 2 + ["measured"] * 4)
        self.assertEqual(len({r["prompt"] for r in first["rows"]}), 6)
        self.assertTrue(all(len(r["token_ids"]) == 90 for r in first["rows"]))
        self.assertTrue(all(call == {"enable_thinking": False} for call in tok.template_calls))
        with tempfile.TemporaryDirectory() as directory:
            paths = write_workload(first, Path(directory) / "profile")
            measured = [json.loads(line) for line in Path(paths["measured"]).read_text().splitlines()]
            metadata = json.loads(Path(paths["metadata"]).read_text())
            self.assertEqual(len(measured), 4)
            self.assertEqual(len(metadata["rows"][0]["token_ids"]), 90)
            with self.assertRaises(FileExistsError):
                write_workload(first, Path(directory) / "profile")

    def test_no_qwen_argument_for_llama_and_impossible_target_fails(self):
        tok = FakeTokenizer()
        args = dict(model_id="meta-llama/Llama-3.1-8B-Instruct", model_revision="b" * 40,
                    tokenizer_id="meta-llama/Llama-3.1-8B-Instruct", tokenizer_revision="b" * 40)
        generate_workload(tok, input_tokens=90, count=2, warmup_count=1, **args)
        self.assertTrue(all(call == {} for call in tok.template_calls))
        with self.assertRaises(ValueError):
            generate_workload(FakeTokenizer(), input_tokens=2, count=2, warmup_count=1, **args)


if __name__ == "__main__":
    unittest.main()
