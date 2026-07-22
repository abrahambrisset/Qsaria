#!/usr/bin/env python3
"""Check that a Qsaria checkout can launch the deterministic MCP profile."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_BACKENDS = ("chemprop", "lightgbm", "tabicl")
TRAINING_MARKER = "QSARIA_TRAINING="


def _result(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "ok": ok, "detail": detail}


def _training_readiness(payload: dict[str, Any]) -> tuple[bool, str]:
    ready: list[str] = []
    unavailable: list[str] = []
    for name in TRAINING_BACKENDS:
        snapshot = payload.get(name) or {}
        version = snapshot.get("package_version")
        if snapshot.get("available") is True and snapshot.get("importable") is True and version:
            ready.append(f"{name}={version}")
            continue
        reason = snapshot.get("import_error") or "package unavailable"
        unavailable.append(f"{name} ({reason})")

    if not unavailable:
        return True, "training ready: " + ", ".join(ready)
    return (
        False,
        "training unavailable: "
        + ", ".join(unavailable)
        + "; run `uv sync --extra mcp --extra prediction`",
    )


def run_checks(repo_root: Path, *, require_training: bool = True) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    pyproject_path = repo_root / "pyproject.toml"

    if not pyproject_path.is_file():
        results.append(_result("repository", False, f"missing {pyproject_path}"))
        return results

    try:
        project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        results.append(_result("repository", False, f"invalid pyproject.toml: {exc}"))
        return results

    results.append(
        _result(
            "repository",
            project.get("name") == "cs_copilot",
            f"project={project.get('name')!r}",
        )
    )

    uv_path = shutil.which("uv")
    results.append(_result("uv", uv_path is not None, uv_path or "uv is not on PATH"))

    venv_entrypoint = repo_root / ".venv" / "bin" / "cscopilot-mcp"
    results.append(
        _result(
            "mcp-extra",
            venv_entrypoint.is_file(),
            (
                str(venv_entrypoint)
                if venv_entrypoint.is_file()
                else "run `uv sync --extra mcp` from the repository root"
            ),
        )
    )

    if uv_path is None or not venv_entrypoint.is_file():
        return results

    command = [
        uv_path,
        "run",
        "--no-sync",
        "cscopilot-mcp",
        "--help",
    ]
    environment = os.environ.copy()
    environment.setdefault("UV_CACHE_DIR", str(Path(tempfile.gettempdir()) / "qsaria-uv-cache"))
    environment.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qsaria-matplotlib-cache")
    )
    environment.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "qsaria-xdg-cache"))
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        results.append(_result("mcp-cli", False, str(exc)))
        return results

    help_text = f"{completed.stdout}\n{completed.stderr}"
    cli_ok = completed.returncode == 0 and "--profile" in help_text
    cli_ok = cli_ok and "--llm-policy" in help_text
    detail = (
        "profile and LLM-policy options available"
        if cli_ok
        else f"help command exited {completed.returncode}; expected --profile and --llm-policy"
    )
    results.append(_result("mcp-cli", cli_ok, detail))

    if not cli_ok:
        return results

    contract_path = repo_root / "plugins" / "qsaria-codex" / "contracts" / "agent-tools.json"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        expected_tools = {
            tool for role_tools in (contract.get("roles") or {}).values() for tool in role_tools
        }
        expected_tools.update(
            {
                "qsaria_create_experiment",
                "qsaria_list_experiments",
                "qsaria_complete_experiment",
                "qsaria_curation_start_operation",
                "qsaria_training_start_operation",
                "qsaria_registry_start_operation",
                "qsaria_inference_start_operation",
                "qsaria_list_operations",
                "qsaria_get_operation_state",
                "qsaria_get_operation_result",
            }
        )
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        results.append(_result("tool-inventory", False, f"invalid agent-tools contract: {exc}"))
        return results

    inventory_code = (
        "import json; "
        "from cs_copilot.mcp.tools_registry import all_specs; "
        "print('QSARIA_TOOLS=' + json.dumps(sorted(s.mcp_name for s in all_specs('qsaria'))))"
    )
    inventory_command = [uv_path, "run", "--no-sync", "python", "-c", inventory_code]
    try:
        inventory_result = subprocess.run(
            inventory_command,
            cwd=repo_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        results.append(_result("tool-inventory", False, str(exc)))
        return results

    marker_line = next(
        (line for line in inventory_result.stdout.splitlines() if line.startswith("QSARIA_TOOLS=")),
        "",
    )
    try:
        actual_tools = set(json.loads(marker_line.removeprefix("QSARIA_TOOLS=")))
    except (json.JSONDecodeError, TypeError):
        actual_tools = set()
    inventory_ok = inventory_result.returncode == 0 and actual_tools == expected_tools
    if inventory_ok:
        inventory_detail = f"exact isolated surface ({len(actual_tools)} tools)"
    else:
        missing = sorted(expected_tools - actual_tools)
        extra = sorted(actual_tools - expected_tools)
        inventory_detail = (
            f"exit={inventory_result.returncode}; missing={missing}; extra={extra}; "
            f"stderr={inventory_result.stderr.strip()[:300]}"
        )
    results.append(_result("tool-inventory", inventory_ok, inventory_detail))

    if not require_training:
        results.append(
            _result(
                "training-backends",
                True,
                "skipped by --mcp-only; scientific training was not validated",
            )
        )
        return results

    training_code = """
import importlib
import json

from cs_copilot.tools.prediction.qsar_training_toolkit import QSARTrainingToolkit

environment = QSARTrainingToolkit().describe_qsar_training_environment()
result = {}
for name, snapshot in environment["backends"].items():
    item = {
        "available": bool(snapshot.get("available")),
        "package_version": snapshot.get("package_version"),
        "importable": False,
        "import_error": None,
    }
    try:
        importlib.import_module(name)
    except Exception as exc:
        item["import_error"] = f"{type(exc).__name__}: {exc}"
    else:
        item["importable"] = True
    result[name] = item
print("QSARIA_TRAINING=" + json.dumps(result, sort_keys=True))
"""
    training_command = [uv_path, "run", "--no-sync", "python", "-c", training_code]
    try:
        training_result = subprocess.run(
            training_command,
            cwd=repo_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        results.append(_result("training-backends", False, str(exc)))
        return results

    training_line = next(
        (line for line in training_result.stdout.splitlines() if line.startswith(TRAINING_MARKER)),
        "",
    )
    try:
        training_payload = json.loads(training_line.removeprefix(TRAINING_MARKER))
    except (json.JSONDecodeError, TypeError):
        training_payload = {}

    training_ok, training_detail = _training_readiness(training_payload)
    if training_result.returncode != 0 or not training_line:
        training_ok = False
        training_detail = (
            f"training probe exited {training_result.returncode}; "
            f"stderr={training_result.stderr.strip()[:500]}"
        )
    results.append(_result("training-backends", training_ok, training_detail))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Qsaria repository root (default: inferred from this script)",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.add_argument(
        "--mcp-only",
        action="store_true",
        help="validate the read-only MCP surface without requiring training backends",
    )
    args = parser.parse_args()

    results = run_checks(args.repo_root.resolve(), require_training=not args.mcp_only)
    ok = bool(results) and all(item["ok"] for item in results)
    if args.json:
        print(json.dumps({"ok": ok, "checks": results}, indent=2))
    else:
        for item in results:
            marker = "PASS" if item["ok"] else "FAIL"
            print(f"[{marker}] {item['check']}: {item['detail']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
