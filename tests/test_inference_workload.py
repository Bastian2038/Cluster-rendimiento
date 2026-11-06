import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from infraestructura.inference_workload import (generate_workload, validate_vllm_reader,
                                               write_workload)


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
    def test_reader_validation_checks_both_phases_and_fails_closed(self):
        tok = FakeTokenizer()
        workload = generate_workload(
            tok, model_id="Qwen/Qwen3.5-9B", model_revision="a" * 40,
            tokenizer_id="Qwen/Qwen3.5-9B", tokenizer_revision="a" * 40,
            input_tokens=90, count=6, warmup_count=2,
        )
        calls = []

        class Reader:
            corrupt = False

            def __init__(self, **kwargs):
                calls.append(("init", kwargs))
                self.rows = [json.loads(line) for line in Path(kwargs["dataset_path"]).read_text().splitlines()]

            def sample(self, **kwargs):
                calls.append(("sample", {k: v for k, v in kwargs.items() if k != "tokenizer"}))
                samples = [SimpleNamespace(
                    prompt=row["prompt"], prompt_len=len(tok(row["prompt"]).input_ids),
                    expected_output_len=row["output_tokens"],
                ) for row in self.rows]
                if self.corrupt:
                    samples[0].prompt_len -= 1
                return samples

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "profile"
            write_workload(workload, destination)
            report = validate_vllm_reader(
                workload, destination, tok, reader_cls=Reader,
                package_versions={"vllm": "0.24.0+092c4842.nv26.7.59534043", "pandas": "fixture", "transformers": "fixture"},
            )
            self.assertEqual(report["package_versions"]["vllm"], "0.24.0+092c4842.nv26.7.59534043")
            self.assertEqual(report["phases"]["warmup"]["count"], 2)
            self.assertEqual(report["phases"]["measured"]["count"], 4)
            self.assertEqual(len([item for item in calls if item[0] == "sample"]), 2)
            self.assertTrue(all(item[1]["output_len"] == -1 and item[1]["skip_chat_template"]
                                and item[1]["no_oversample"] for item in calls if item[0] == "sample"))
            self.assertTrue((destination / "reader-validation.json").exists())
            with self.assertRaises(FileExistsError):
                validate_vllm_reader(workload, destination, tok, reader_cls=Reader,
                                     package_versions={"vllm": "0.24.0"})
            with self.assertRaisesRegex(ValueError, "requires vLLM 0.24.0"):
                validate_vllm_reader(workload, destination, tok, reader_cls=Reader,
                                     package_versions={"vllm": "0.25.0"})
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "profile"
            write_workload(workload, destination)
            Reader.corrupt = True
            with self.assertRaisesRegex(ValueError, "prompt length differs"):
                validate_vllm_reader(workload, destination, tok, reader_cls=Reader,
                                     package_versions={"vllm": "0.24.0"})
            self.assertFalse((destination / "reader-validation.json").exists())

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
