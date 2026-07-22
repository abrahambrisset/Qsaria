#!/usr/bin/env python3
"""Validate the shared deterministic Qsaria runtime without scientific writes."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]

EXPECTED_CLIENTS = ["codex_v1", "claude_code_v1", "claude_science_v1"]
EXPECTED_COORDINATOR_CONTRACT = "external_mcp_coordinator_v1"
OPERATION_TOOLS = {
    "qsaria_curation_start_operation",
    "qsaria_training_start_operation",
    "qsaria_registry_start_operation",
    "qsaria_inference_start_operation",
    "qsaria_list_operations",
    "qsaria_get_operation_state",
    "qsaria_get_operation_result",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return payload


def validate() -> list[str]:
    errors: list[str] = []

    from cs_copilot.mcp.qsaria import ExperimentManager
    from cs_copilot.mcp.tools_registry import all_specs

    with tempfile.TemporaryDirectory(prefix="qsaria-claude-runtime-") as temporary:
        state_root = Path(temporary) / "state"
        bootstrap = ExperimentManager(local_root=state_root).bootstrap()
        if state_root.exists():
            errors.append("qsaria_bootstrap wrote persistent state")

    if bootstrap.get("status") != "ok":
        errors.append("bootstrap status must be ok")
    if bootstrap.get("profile") != "qsaria":
        errors.append("bootstrap profile must be qsaria")
    if bootstrap.get("llm_policy") != "disabled" or bootstrap.get("model") is not None:
        errors.append("bootstrap must disable every MCP-side LLM")
    for key in ("auto_resume", "experiments_listed"):
        if bootstrap.get(key) is not False:
            errors.append(f"bootstrap {key} must be false")
    if bootstrap.get("active_experiment_id") is not None:
        errors.append("bootstrap must not activate an experiment")

    compatibility = bootstrap.get("compatibility") or {}
    if compatibility.get("target") != EXPECTED_COORDINATOR_CONTRACT:
        errors.append("bootstrap target must use the neutral external coordinator contract")
    if compatibility.get("coordinator_contract") != EXPECTED_COORDINATOR_CONTRACT:
        errors.append("bootstrap coordinator contract is incompatible")
    if compatibility.get("supported_clients") != EXPECTED_CLIENTS:
        errors.append("bootstrap supported client list is incompatible")
    if compatibility.get("agno_chainlit_runtime_changed") is not False:
        errors.append("bootstrap must preserve the Agno/Chainlit runtime")

    contract = _load_json(PLUGIN_ROOT / "contracts" / "agent-tools.json")
    expected_tools = {tool for tools in (contract.get("roles") or {}).values() for tool in tools}
    expected_tools.update(
        {
            "qsaria_create_experiment",
            "qsaria_list_experiments",
            "qsaria_complete_experiment",
        }
    )
    specs = list(all_specs(profile="qsaria"))
    actual_tools = {spec.mcp_name for spec in specs}
    expected_runtime_tools = expected_tools | OPERATION_TOOLS
    if len(actual_tools) != 53 or actual_tools != expected_runtime_tools:
        missing = sorted(expected_runtime_tools - actual_tools)
        extra = sorted(actual_tools - expected_runtime_tools)
        errors.append(
            f"Qsaria tool inventory differs; count={len(actual_tools)}, "
            f"missing={missing}, extra={extra}"
        )
    if any(spec.run_in_worker_process for spec in specs):
        errors.append("Qsaria must not use the ChEMBL worker process")

    return errors


def main() -> int:
    try:
        errors = validate()
    except (ImportError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Qsaria shared runtime validation: FAIL\n- {exc}", file=sys.stderr)
        return 1

    if errors:
        print("Qsaria shared runtime validation: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Qsaria shared runtime validation: PASS")
    print("- coordinator contract: external_mcp_coordinator_v1")
    print("- supported clients: codex_v1, claude_code_v1, claude_science_v1")
    print("- deterministic tool inventory: 46 synchronous + 7 durable-operation")
    print("- bootstrap persistence writes: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
