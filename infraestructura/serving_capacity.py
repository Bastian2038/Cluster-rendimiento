"""Plan, preflight, run and report the mixed-load serving capacity campaign."""

import argparse
import json
from pathlib import Path
import signal
import sys

from infraestructura import inference, inference_runtime, serving_capacity_protocol as protocol
from infraestructura.rutas import REPO_ROOT


def _check_destination(path):
    if path is not None and any(path.resolve().is_relative_to(root.resolve())
                                for root in (REPO_ROOT / "resultados/raw", REPO_ROOT / "datos")):
        raise ValueError("los derivados deben escribirse fuera de los resultados raw y datos")


def preflight(path, *, live=False):
    config, inventory, _ = protocol.load_config(path)
    workloads = protocol.frozen_workloads(config)
    hardware_path = protocol.historical_preflight_path(config)
    result = {
        "scope": "cluster" if live else "local_cpu", "protocol_id": protocol.PROTOCOL,
        "historical_campaign_sha256": config["reference"]["historical_campaign_sha256"],
        "historical_preflight_sha256": config["reference"]["historical_preflight_sha256"],
        "historical_hardware": protocol.verify_hardware(config, hardware_path),
        "historical_jsonl_sha256": config["reference"]["historical_jsonl_sha256"],
        "workload_directory": str(workloads), "errors": [],
    }
    if live:
        result["cluster"] = inference_runtime.preflight(inventory, config, container_probe=True)
        result["errors"] = list(result["cluster"]["errors"])
        try:
            protocol.verify_hardware(config, hardware_path, result["cluster"])
        except (KeyError, ValueError, TypeError) as exc:
            result["errors"].append(f"comparación con el preflight histórico: {exc}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "preflight", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, default=protocol.DEFAULT_CONFIG)
        if name in ("preflight", "plan"):
            sub.add_argument("--output", type=Path, help="archivo JSON nuevo, stdout por defecto")
        if name == "preflight":
            sub.add_argument("--live", action="store_true",
                             help="sondas operativas autorizadas en los nodos y su imagen")
        if name == "run":
            sub.add_argument("--execute", action="store_true",
                             help="ejecutar bajo la autorización específica de esta reserva")
            sub.add_argument("--session-limit-seconds", type=int,
                             help="reducir el máximo de ocupación propuesto para la sesión")
    sub = commands.add_parser("report")
    sub.add_argument("directory", type=Path, help="directorio de campaña cerrado")
    sub.add_argument("--output", type=Path, required=True, help="ruta base nueva para Markdown y JSON")
    args = parser.parse_args(argv)
    try:
        _check_destination(getattr(args, "output", None))
        if args.command == "report":
            from infraestructura import serving_capacity_report
            serving_capacity_report.write_report(args.output, serving_capacity_report.build_report(args.directory))
            return 0
        if args.command == "run":
            if not args.execute:
                raise ValueError("run requiere --execute y la autorización específica de la campaña")
            from infraestructura import serving_capacity_runner

            def stop(_signum, _frame):
                raise KeyboardInterrupt("señal de parada")

            previous = signal.signal(signal.SIGTERM, stop)
            try:
                directory = serving_capacity_runner.run_campaign(
                    args.config, session_limit_seconds=args.session_limit_seconds)
            finally:
                signal.signal(signal.SIGTERM, previous)
            status = inference.read_json(directory / "campaign.json")["status"]
            print(json.dumps({"event": "campaign_closed", "directory": str(directory), "status": status}),
                  flush=True)
            return 0 if status == "complete" else 2

        if args.command == "plan":
            result = protocol.make_plan(args.config)
        else:
            result = preflight(args.config, live=args.live)
        if args.output is None:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            inference.write_new(args.output, result)
        return 2 if result.get("errors") else 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"serving_capacity: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
