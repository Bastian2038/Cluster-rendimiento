"""Selección y renderizado local de gráficos derivados de inferencia."""
import tempfile
import unittest
from pathlib import Path

from infraestructura.inference_plot import plot_report, prepare_plot_data


def cell(model, group, block, *, status="valid", value=100.0):
    return {"model_key": model, "group": group, "block": block,
            "input_tokens": 512, "concurrency": 4, "status": status,
            "t32_seconds": 20.0, "output_tokens_per_second": value,
            "ttft": {"p50_ms": value / 2, "p95_ms": value},
            "tpot": {"p50_ms": 10}, "e2el": {"p50_ms": 100}}


class InferencePlotTests(unittest.TestCase):
    def test_summary_keeps_models_separate_and_valid_only(self):
        report = {"cells": [cell("qwen", "four", 1, value=100),
                            cell("qwen", "four", 2, value=200),
                            cell("qwen", "four", 3, value=300),
                            cell("qwen", "pair", 1, value=90),
                            cell("llama", "four", 1, value=500),
                            cell("qwen", "pair", 2, status="invalid", value=999)],
                  "paired_comparisons": [
                      {"model": "qwen", "block": b, "input_tokens": 512,
                       "concurrency": 4, "s4_2": float(b)} for b in (1, 2, 3)]}
        data = prepare_plot_data(report)
        self.assertEqual(data["models"], ["llama", "qwen"])
        qwen = next(p for p in data["points"] if p["model_key"] == "qwen")
        self.assertEqual(qwen["groups"]["four"]["output_tokens_per_second"],
                         {"median": 200, "min": 100, "max": 300, "blocks": 3})
        self.assertEqual(qwen["groups"]["pair"]["output_tokens_per_second"]["blocks"], 1)
        self.assertEqual(qwen["s4_2"]["median"], 2)
        llama = next(p for p in data["points"] if p["model_key"] == "llama")
        self.assertIsNone(llama["s4_2"])

    def test_empty_skips_without_creating_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plot.svg"
            self.assertEqual(plot_report({"cells": [], "paired_comparisons": []}, output)["status"], "skipped")
            self.assertFalse(output.exists())

    def test_rejects_raw_destination_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "fuera de raw"):
                plot_report({"cells": []}, root / "raw" / "plot.svg")
            output = root / "plot.svg"
            output.write_text("original")
            with self.assertRaises(FileExistsError):
                plot_report({"cells": []}, output)
            self.assertEqual(output.read_text(), "original")

    def test_render_svg_if_matplotlib_available(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib no instalado")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plot.svg"
            result = plot_report({"cells": [cell("qwen", "four", 1), cell("qwen", "pair", 1)],
                                  "paired_comparisons": [{"model": "qwen", "block": 1,
                                      "input_tokens": 512, "concurrency": 4, "s4_2": 1.2}]}, output)
            self.assertEqual(result["status"], "created")
            self.assertIn("<svg", output.read_text())


if __name__ == "__main__":
    unittest.main()
