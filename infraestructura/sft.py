"""Planificar, calificar, medir e informar el SFT completo de Llama."""

import argparse
import json
from pathlib import Path
import signal
import sys

from infraestructura import inference, sft_protocol as protocol
from infraestructura.rutas import REPO_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "preflight", "pilot", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, default=protocol.DEFAULT_CONFIG)
        if name in ("plan", "preflight"):
            sub.add_argument("--output", type=Path, help="archivo JSON nuevo, o stdout por defecto")
        if name == "plan":
            sub.add_argument("--stage", choices=("pilot", "run"), default="pilot")
            sub.add_argument("--pilot-dir", type=Path, help="cierre de piloto para proyectar la matriz")
        if name == "preflight":
            sub.add_argument("--live", action="store_true", help="consulta operativa de los cuatro nodos")
        if name in ("pilot", "run"):
            sub.add_argument("--execute", action="store_true", help="ejecutar la etapa autorizada")
            sub.add_argument("--session-limit-seconds", type=int, default=7200 if name == "pilot" else 28800)
        if name == "run":
            sub.add_argument("--pilot-dir", type=Path, required=True, help="campaña de piloto cerrada")
            sub.add_argument("--pilot-report", type=Path, help="JSON publicado del piloto, si tiene otra ruta")
    sub = commands.add_parser("report")
    sub.add_argument("directory", type=Path)
    sub.add_argument("--output", type=Path, required=True, help="base nueva de los archivos JSON y Markdown")
    args = parser.parse_args(argv)
    try:
        output = getattr(args, "output", None)
        if output and any(output.resolve().is_relative_to(root.resolve())
                          for root in (REPO_ROOT / "resultados/raw", REPO_ROOT / "datos")):
            raise ValueError("elegir un destino de derivados fuera de raw y datos")
        if args.command == "report":
            from infraestructura import sft_report
            sft_report.write_report(output, sft_report.build_report(args.directory))
            return 0
        config = protocol.load_config(args.config)
        if args.command in ("pilot", "run"):
            if not args.execute:
                raise ValueError(f"{args.command} requiere --execute y la autorización de esa etapa")
            from infraestructura import sft_runner

            def stop(_signum, _frame):
                raise KeyboardInterrupt("señal de parada")

            previous = signal.signal(signal.SIGTERM, stop)
            try:
                directory = sft_runner.run_campaign(args.config, args.command, args.session_limit_seconds,
                                                    pilot_dir=getattr(args, "pilot_dir", None),
                                                    pilot_report=getattr(args, "pilot_report", None))
            finally:
                signal.signal(signal.SIGTERM, previous)
            status = inference.read_json(directory / "campaign.json")["status"]
            print(json.dumps({"event": "campaign_closed", "directory": str(directory), "status": status}), flush=True)
            return 0 if status == "complete" else 2
        if args.command == "plan":
            groups = None
            if args.pilot_dir:
                if args.stage != "run":
                    raise ValueError("--pilot-dir se usa con --stage run")
                from infraestructura import sft_report
                pilot = sft_report.build_report(args.pilot_dir)
                groups = protocol.qualified_pilot_groups(pilot)
            result = protocol.make_plan(config, args.stage, groups)
        else:
            result = protocol.local_preflight(config)
            if args.live and not result["errors"]:
                from infraestructura import sft_runner
                result["scope"] = "cluster"
                result["cluster"] = sft_runner.live_preflight(config)
                result["errors"].extend(result["cluster"]["errors"])
        if output is None:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            inference.write_new(output, result)
        return 2 if result.get("errors") else 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"sft: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
