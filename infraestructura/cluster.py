"""Configuración declarada del clúster; no realiza conexiones."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path


_IDENT = re.compile(r"^[a-z][a-z0-9-]*$")
_ALIAS = re.compile(r"^[A-Za-z][A-Za-z0-9 -]*$")
_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")
_USER = re.compile(r"^[a-z_][a-z0-9_-]*$")
_ADDRESS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.:_-]*$")
_SECRET = re.compile(r"password|passwd|private.?key|secret|token|credential", re.I)


def validate_config(config: dict) -> None:
    if not isinstance(config, dict) or config.get("schema_version") != "1.0":
        raise ValueError("schema_version de cluster debe ser 1.0")
    if not _IDENT.fullmatch(str(config.get("cluster_id", ""))):
        raise ValueError("cluster_id inválido")
    if _SECRET.search(json.dumps(config, ensure_ascii=False)):
        raise ValueError("la configuración no debe incluir secretos")
    nodes = config.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 4:
        raise ValueError("se requieren cuatro nodos declarados")
    ids, aliases, hosts = set(), set(), set()
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("nodo inválido")
        if not isinstance(node.get("node_id"), str) or not _IDENT.fullmatch(node["node_id"]):
            raise ValueError("node_id inválido")
        if not isinstance(node.get("alias"), str) or not _ALIAS.fullmatch(node["alias"]):
            raise ValueError("alias inválido")
        hostname = node.get("expected_hostname")
        if not isinstance(hostname, str) or not _HOST.fullmatch(hostname):
            raise ValueError("expected_hostname inválido")
        if node["node_id"] in ids or node["alias"] in aliases or hostname in hosts:
            raise ValueError("nodo, alias o hostname duplicado")
        ids.add(node["node_id"]); aliases.add(node["alias"]); hosts.add(hostname)
        if not isinstance(node.get("local"), bool):
            raise ValueError("local debe ser bool")
        ssh = node.get("ssh")
        if not isinstance(ssh, dict):
            raise ValueError("ssh inválido")
        if not isinstance(ssh.get("host"), str) or not _ADDRESS.fullmatch(ssh["host"]):
            raise ValueError("ssh.host inválido")
        if not isinstance(ssh.get("user"), str) or not _USER.fullmatch(ssh["user"]):
            raise ValueError("ssh.user inválido")
        if type(ssh.get("port")) is not int or not 1 <= ssh["port"] <= 65535:
            raise ValueError("ssh.port inválido")
        address = node.get("management_address")
        if address is not None and (not isinstance(address, str) or not _ADDRESS.fullmatch(address)):
            raise ValueError("management_address inválida")
        data = node.get("data_addresses")
        if not isinstance(data, list) or any(not isinstance(a, str) or not _ADDRESS.fullmatch(a) for a in data):
            raise ValueError("data_addresses inválidas")
        if not isinstance(node.get("access_status"), str):
            raise ValueError("access_status inválido")
    if sum(node["local"] for node in nodes) != 1:
        raise ValueError("se requiere exactamente un nodo local")
    groups = config.get("groups")
    if not isinstance(groups, dict) or set(groups) != {"single", "pair", "four"}:
        raise ValueError("grupos inesperados")
    expected = {"single": ["pegasus"], "pair": ["pegasus", "octopus"],
                "four": ["pegasus", "octopus", "griffin", "draggon"]}
    if groups != expected or ids != set(expected["four"]):
        raise ValueError("nodos/grupos no corresponden al contrato")
    switch = config.get("switch")
    if not isinstance(switch, dict) or set(switch) != {"status", "manufacturer", "model", "management_address"}:
        raise ValueError("switch inválido")
    if switch["status"] not in {"unknown", "reported", "known"}:
        raise ValueError("estado de switch inválido")
    for key in ("manufacturer", "model"):
        if switch[key] is not None and (not isinstance(switch[key], str) or not switch[key].strip()):
            raise ValueError(f"switch.{key} inválido")
    address = switch["management_address"]
    if address is not None and (not isinstance(address, str) or not _ADDRESS.fullmatch(address)):
        raise ValueError("switch.management_address inválida")


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def selected_nodes(config: dict, group: str) -> list[dict]:
    validate_config(config)
    if group not in config["groups"]:
        raise ValueError(f"grupo desconocido: {group}")
    lookup = {node["node_id"]: node for node in config["nodes"]}
    return [lookup[node_id] for node_id in config["groups"][group]]


def ssh_command(node: dict, remote_command: list[str]) -> list[str]:
    """Construye argv, nunca invoca shell local ni modifica autenticación SSH."""
    if not isinstance(remote_command, list) or not remote_command or any(
        not isinstance(part, str) or not part for part in remote_command
    ):
        raise ValueError("comando remoto inválido")
    if node.get("local") is True:
        return list(remote_command)
    ssh = node["ssh"]
    if (not _ADDRESS.fullmatch(ssh["host"]) or not _USER.fullmatch(ssh["user"])
            or type(ssh["port"]) is not int or not 1 <= ssh["port"] <= 65535):
        raise ValueError("destino SSH inválido")
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "UpdateHostKeys=no", "-o", "ControlMaster=no", "-o", "ControlPath=none",
            "-o", "ConnectTimeout=8", "-p", str(ssh["port"]),
            f"{ssh['user']}@{ssh['host']}", shlex.join(remote_command)]
