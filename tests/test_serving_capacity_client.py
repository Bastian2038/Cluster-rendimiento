import asyncio
import hashlib
import json
import tempfile
import unittest
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from infraestructura import serving_capacity_client as client


MODEL = "qwen35-9b"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fixture_prompts(count):
    return [
        {
            "prompt": str(index),
            "prompt_sha256": digest(str(index).encode()),
            "input_tokens": client.INPUT_LENGTHS[index % 2],
        }
        for index in range(count)
    ]


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        await asyncio.sleep(0)
        self.now += seconds


class Response:
    def __init__(self, lines=(), *, status=200, end=None):
        self.status_code = status
        self.headers = {"content-type": "text/event-stream; charset=utf-8"}
        self.lines = lines
        self.end = end

    async def aiter_lines(self):
        for line in self.lines:
            if callable(line):
                line()
            else:
                yield line
        if self.end:
            self.end()


def delta(text="", tokens=(), finish=None, *, model=MODEL):
    return {"model": model, "choices": [
        {"text": text, "token_ids": list(tokens), "index": 0, "finish_reason": finish},
    ]}


def usage(length, output=128):
    return {"model": MODEL, "choices": [],
            "usage": {"prompt_tokens": length, "completion_tokens": output,
                      "total_tokens": length + output}}


def event(payload):
    return ["data: " + (payload if isinstance(payload, str) else json.dumps(payload)), ""]


def stream(length):
    return [
        *event(delta("contenido", range(127))),
        *event(delta("", [127], "length")),
        *event(usage(length)),
        *event("[DONE]"),
    ]


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / MODEL
        for length in client.INPUT_LENGTHS:
            directory = self.root / str(length)
            directory.mkdir(parents=True)
            metadata = {
                "schema_version": 1,
                "model_key": MODEL,
                "model_id": "org/model",
                "model_revision": "a" * 40,
                "tokenizer_id": "org/model",
                "tokenizer_revision": "a" * 40,
                "input_tokens": length,
                "output_tokens": 128,
                "rows": [],
                "files_sha256": {},
            }
            for phase, count in (("warmup", 16), ("measured", 32)):
                prompts = [f"{phase}/{length}/{i} á" for i in range(count)]
                data = "".join(json.dumps({"prompt": prompt, "output_tokens": 128}) + "\n"
                               for prompt in prompts).encode()
                (directory / f"{phase}.jsonl").write_bytes(data)
                metadata["files_sha256"][phase] = digest(data)
                metadata["rows"].extend({
                    "phase": phase, "prompt_sha256": digest(prompt.encode()),
                    "prompt_tokens": length, "output_tokens": 128,
                } for prompt in prompts)
            (directory / "metadata.json").write_text(json.dumps(metadata))

    def test_alternates_warm_prefix_and_cycles_measured(self):
        warm = client.expected_prompts(self.root, "warmup", 16)
        measured = client.expected_prompts(self.root, "measured", 128)
        self.assertEqual([row["prompt"] for row in warm],
                         [f"warmup/{length}/{i} á" for i in range(8)
                          for length in client.INPUT_LENGTHS])
        self.assertEqual([row["prompt"] for row in measured],
                         [f"measured/{length}/{i % 32} á" for i in range(64)
                          for length in client.INPUT_LENGTHS])
        self.assertEqual(measured[1], {
            "prompt": "measured/2048/0 á", "input_tokens": 2048,
            "prompt_sha256": digest("measured/2048/0 á".encode()),
        })

    def test_checks_file_and_row_hashes_and_model_identity(self):
        path = self.root / "2048" / "metadata.json"
        original = path.read_text()
        for field, message in (("files_sha256", "SHA-256"), ("rows", "identidad o longitud de fila"),
                               ("model_revision", "difieren")):
            with self.subTest(field=field):
                metadata = json.loads(original)
                if field == "files_sha256":
                    metadata[field]["warmup"] = "0" * 64
                elif field == "rows":
                    metadata[field][0]["prompt_sha256"] = "0" * 64
                else:
                    metadata[field] = "b" * 40
                path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, message):
                    client.expected_prompts(self.root, "warmup", 2)

    def test_optional_metadata_and_required_output_length(self):
        directory = self.root / "512"
        (directory / "metadata.json").unlink()
        self.assertEqual(len(client.expected_prompts(self.root, "measured", 128)), 128)
        (directory / "measured.jsonl").write_text('{"prompt":"fixture","output_tokens":127}\n')
        with self.assertRaisesRegex(ValueError, "longitud de salida"):
            client.expected_prompts(self.root, "measured", 2)

    def test_cli_publication_is_exclusive_and_cleans_failed_temporary_file(self):
        output = self.root.parent / "metrics" / "phase.json"
        argv = ["--model", MODEL, "--port", "18000", "--workloads", str(self.root),
                "--rate", "0.75", "--phase", "warmup", "--count", "16",
                "--request-timeout", "30", "--output", str(output)]
        receipt = {"schema_version": 1, "rows": []}
        with patch.object(client, "_run_http", return_value=receipt) as run:
            self.assertEqual(client.main(argv), 0)
            model, port, phase, rate, prompts, timeout = run.call_args.args
            self.assertEqual((model, port, phase, rate, len(prompts), timeout),
                             (MODEL, 18000, "warmup", 0.75, 16, 30))
            with self.assertRaises(FileExistsError):
                client.main(argv)
            run.assert_awaited_once()
        with self.assertRaises(FileExistsError):
            client.write_new(output, {"other": "value"})
        with self.assertRaises(ValueError):
            client.write_new(output.with_name("invalid.json"), {"invalid": float("nan")})
        self.assertEqual(list(output.parent.iterdir()), [output])
        self.assertEqual(json.loads(output.read_text()), receipt)


class PhaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_adapter_uses_loopback_and_marks_completed_body_write(self):
        clock = Clock()
        transport_calls = []

        @asynccontextmanager
        async def stream_request(method, url, *, json, extensions):
            transport_calls.append((method, url, json["prompt"]))
            trace = extensions["trace"]
            await trace("http11.send_request_body.started", {})
            clock.now += 1
            await trace("http11.send_request_body.complete", {})
            clock.now += 1
            yield Response(stream(512))

        @asynccontextmanager
        async def async_client(*, limits, timeout, trust_env):
            self.assertEqual(limits, {"max_connections": None})
            self.assertIsNone(timeout)
            self.assertFalse(trust_env)
            yield SimpleNamespace(stream=stream_request)

        httpx = SimpleNamespace(AsyncClient=async_client, Limits=dict, TimeoutException=TimeoutError)
        with patch.dict("sys.modules", {"httpx": httpx}), patch.object(
            client, "run_phase", partial(client.run_phase, clock=clock)
        ):
            receipt = await client._run_http(MODEL, 18000, "warmup", 1, fixture_prompts(1), 30)
        self.assertEqual(transport_calls, [("POST", "http://127.0.0.1:18000/v1/completions", "0")])
        row, = receipt["rows"]
        self.assertEqual((row["status"], row["sent_s"], row["first_token_s"]), ("ok", 1, 2))

    async def test_regular_arrivals_continue_while_previous_streams_wait(self):
        clock = Clock()
        sent_at = []
        all_sent = asyncio.Event()
        count = 128

        @asynccontextmanager
        async def transport(payload, sent):
            index = int(payload["prompt"])
            self.assertEqual(payload["model"], MODEL)
            self.assertEqual(payload["max_tokens"], 128)
            self.assertEqual(payload["stream_options"], {"include_usage": True})
            self.assertTrue(payload["stream"] and payload["ignore_eos"] and payload["return_token_ids"])
            sent_at.append(clock() - 100)
            sent()
            if len(sent_at) == count:
                all_sent.set()
            await all_sent.wait()
            yield Response(stream(client.INPUT_LENGTHS[index % 2]))

        result = await asyncio.wait_for(client.run_phase(
            MODEL, "measured", 2.0, fixture_prompts(count), 100, transport,
            clock=clock, sleep=clock.sleep,
        ), 2)
        target = [i / 2 for i in range(count)]
        self.assertEqual(sent_at, target)
        self.assertEqual([row["scheduled_s"] for row in result["rows"]], target)
        self.assertEqual([row["sent_s"] for row in result["rows"]], target)
        self.assertEqual([row["end_s"] for row in result["rows"]], [target[-1]] * count)
        self.assertTrue(all(row["status"] == "ok" for row in result["rows"]))

    async def test_sse_content_token_ids_done_and_http_eof_have_distinct_marks(self):
        clock = Clock()

        def advance():
            clock.now += 1

        @asynccontextmanager
        async def transport(_payload, sent):
            advance()
            sent()
            yield Response([
                *event(delta("", [1])), advance,
                ": comment", "event: message",
                'data: {"model":"qwen35-9b",',
                'data: "choices":[{"index":0,"text":" ","token_ids":[2]}]}', "",
                advance, *event(delta("", range(3, 129), "length")),
                advance, *event(usage(512)), *event("[DONE]"),
            ], end=advance)

        result = await client.run_phase(MODEL, "warmup", 1, fixture_prompts(1), 30,
                                        transport, clock=clock)
        row, = result["rows"]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["output_tokens"], 128)
        self.assertEqual([row[key] for key in (
            "scheduled_s", "sent_s", "send_lag_s", "first_token_s", "last_token_s",
            "stream_done_s", "response_end_s", "end_s",
        )], [0, 1, 1, 1, 3, 4, 5, 5])

    async def test_ttft_meets_three_second_limit_before_text_arrives(self):
        for prefix in ("", "texto sin IDs"):
            with self.subTest(prefix=prefix):
                clock = Clock()

                def advance():
                    clock.now += 2

                @asynccontextmanager
                async def transport(_payload, sent):
                    sent()
                    yield Response([
                        *event(delta(prefix)), advance,
                        *event(delta("", [1])), advance,
                        *event(delta("texto visible", range(2, 129), "length")),
                        *event(usage(512)), *event("[DONE]"), advance,
                    ])

                result = await client.run_phase(MODEL, "warmup", 1, fixture_prompts(1),
                                                30, transport, clock=clock)
                row, = result["rows"]
                self.assertEqual(row["status"], "ok")
                self.assertEqual(row["first_token_s"] - row["scheduled_s"], 2)
                self.assertEqual(row["last_token_s"], 4)
                self.assertEqual(row["end_s"], 6)

    async def test_all_tokens_may_have_empty_text(self):
        @asynccontextmanager
        async def transport(_payload, sent):
            sent()
            yield Response([
                *event(delta("", range(128), "length")),
                *event(usage(512)), *event("[DONE]"),
            ])

        result = await client.run_phase(MODEL, "warmup", 1, fixture_prompts(1),
                                        30, transport, clock=Clock())
        row, = result["rows"]
        self.assertEqual((row["status"], row["first_token_s"], row["last_token_s"]),
                         ("ok", 0, 0))

    async def test_stream_failures_remain_terminal_rows(self):
        valid = stream(512)
        cases = [
            ("rejected", Response(status=429)),
            ("error", Response(status=503)),
            ("error", Response(valid[:-2])),
            ("error", Response(valid[:-1])),
            ("error", Response(valid + event(delta("después")))),
            ("error", Response(event(delta("otro modelo", [1], model="different")))),
            ("error", Response(event(delta("eos", [1], "stop")))),
            ("error", Response(valid[:4] + event(usage(512, 127)) + valid[-2:])),
            ("error", Response(valid[:4] + event(usage(2048)) + valid[-2:])),
            ("error", Response(event(delta("sin IDs", (), "length")) + valid[4:])),
            ("error", Response(event('{"error":{"message":"internal"}}'))),
        ]
        for status, response in cases:
            with self.subTest(status=status, response=response.lines):
                @asynccontextmanager
                async def transport(_payload, sent):
                    sent()
                    yield response

                result = await client.run_phase(MODEL, "warmup", 1, fixture_prompts(1), 30,
                                                transport, clock=Clock())
                row, = result["rows"]
                self.assertEqual(row["index"], 0)
                self.assertEqual(row["status"], status)
                self.assertIsNotNone(row["end_s"])
                if row["first_token_s"] is None:
                    self.assertIsNone(row["last_token_s"])

    async def test_timeout_from_scheduled_arrival_and_pre_send_transport_failure(self):
        clock = Clock()
        sent_indices = []

        async def late_sleep(seconds):
            await clock.sleep(seconds + 31)

        @asynccontextmanager
        async def transport(payload, _sent):
            sent_indices.append(payload["prompt"])
            raise ConnectionError("fixture")
            yield

        result = await client.run_phase(MODEL, "measured", 1, fixture_prompts(2), 30,
                                        transport, clock=clock, sleep=late_sleep)
        self.assertEqual(sent_indices, ["0"])
        self.assertEqual([row["status"] for row in result["rows"]], ["error", "timeout"])
        self.assertEqual([row["sent_s"] for row in result["rows"]], [None, None])
        self.assertEqual([row["end_s"] for row in result["rows"]], [0, 32])

    async def test_timeout_while_connecting_or_waiting_for_http_end(self):
        for sent_first in (False, True):
            with self.subTest(sent_first=sent_first):
                class SlowResponse(Response):
                    async def aiter_lines(self):
                        async for line in super().aiter_lines():
                            yield line
                        await asyncio.Event().wait()

                @asynccontextmanager
                async def transport(_payload, sent):
                    if not sent_first:
                        await asyncio.Event().wait()
                    sent()
                    yield SlowResponse(stream(512))

                result = await client.run_phase(MODEL, "warmup", 1, fixture_prompts(1),
                                                0.05, transport)
                row, = result["rows"]
                self.assertEqual(row["status"], "timeout")
                self.assertGreaterEqual(row["end_s"], 0.05)
                self.assertEqual(row["sent_s"] is not None, sent_first)
                self.assertEqual(row["stream_done_s"] is not None, sent_first)
                self.assertIsNone(row["response_end_s"])
