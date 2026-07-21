#!/usr/bin/env python3
"""Check or record Qsaria-to-Claude Code compatibility signatures."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
COMPATIBILITY_PATH = PLUGIN_ROOT / "contracts" / "compatibility.json"
CODEX_CONTRACT_ROOT = REPO_ROOT / "plugins" / "qsaria-codex" / "contracts"

SOURCE_CONTRACT_FILES = {
    "agent_tools": CODEX_CONTRACT_ROOT / "agent-tools.json",
    "handoff_schema": CODEX_CONTRACT_ROOT / "handoff.schema.json",
    "scientific_invariants": CODEX_CONTRACT_ROOT / "scientific-invariants.md",
}
RUNTIME_FILES = (
    "src/cs_copilot/mcp/__main__.py",
    "src/cs_copilot/mcp/server.py",
    "src/cs_copilot/mcp/tools_registry.py",
    "src/cs_copilot/mcp/tool_specs/qsaria.py",
    "src/cs_copilot/mcp/tool_specs/qsaria_lifecycle.py",
)
REPOSITORY_INTEGRATION_FILES = (
    ".claude-plugin/marketplace.json",
    ".claude/settings.json",
    ".github/ISSUE_TEMPLATE/qsaria_claude_code_bug.yml",
    ".github/workflows/claude-plugin.yml",
    "tests/unit/test_qsaria_claude_bundle.py",
)
CONTRACT_CONSTANTS = {
    "experiment": (
        "src/cs_copilot/mcp/qsaria/contracts.py",
        "EXPERIMENT_SCHEMA_VERSION",
    ),
    "handoff": (
        "src/cs_copilot/mcp/qsaria/contracts.py",
        "HANDOFF_SCHEMA_VERSION",
    ),
    "report_facts": (
        "src/cs_copilot/tools/prediction/qsar_reporting.py",
        "REPORT_FACTS_SCHEMA_VERSION",
    ),
    "training": (
        "src/cs_copilot/tools/prediction/qsar_contracts.py",
        "TRAINING_CONTRACT_VERSION",
    ),
}
SUPPORTED_CLIENTS = ["codex_v1", "claude_code_v1"]
COORDINATOR_CONTRACT = "external_mcp_coordinator_v1"


class SignatureError(RuntimeError):
    """Raised when a compatibility signature cannot be calculated safely."""


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _hash_files(root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths}):
        relative = path.relative_to(root.resolve()).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return payload


def _literal_constant(relative: str, name: str) -> str:
    path = REPO_ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        value: ast.AST | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                value = node.value
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    raise SignatureError(f"missing literal contract constant {name} in {relative}")


def _contract_versions() -> dict[str, str]:
    return {
        contract: _literal_constant(relative, constant)
        for contract, (relative, constant) in CONTRACT_CONSTANTS.items()
    }


def _bundle_files() -> list[Path]:
    files = [
        path
        for path in PLUGIN_ROOT.rglob("*")
        if path.is_file()
        and path.resolve() != COMPATIBILITY_PATH.resolve()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    ]
    return sorted(files)


def _signed_source_files() -> list[Path]:
    files = list(_bundle_files())
    files.extend(SOURCE_CONTRACT_FILES.values())
    files.extend(REPO_ROOT / relative for relative in RUNTIME_FILES)
    files.extend((REPO_ROOT / "src/cs_copilot/mcp/qsaria").rglob("*.py"))
    files.extend(REPO_ROOT / relative for relative in REPOSITORY_INTEGRATION_FILES)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise SignatureError(f"missing signed Claude source: {', '.join(missing)}")
    return sorted({path.resolve() for path in files})


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SignatureError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _source_provenance(repository: str, signed_files: list[Path]) -> dict[str, Any]:
    relative_paths = [path.relative_to(REPO_ROOT).as_posix() for path in signed_files]
    commit = _git("log", "-1", "--format=%H", "--", *relative_paths)
    if not commit:
        raise SignatureError("no committed provenance found for the Claude source tree")
    return {
        "repository": repository,
        "commit": commit,
        "tree": _git("rev-parse", f"{commit}^{{tree}}"),
        "signed_tree": _hash_files(REPO_ROOT, signed_files),
        "describe": _git("describe", "--tags", "--always", commit),
        "dirty": bool(
            _git(
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                *relative_paths,
            )
        ),
    }


def _normalize_handoff(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    normalized.pop("$id", None)
    return normalized


def _verify_copied_contracts() -> None:
    claude_tools = _load_json(PLUGIN_ROOT / "contracts" / "agent-tools.json")
    codex_tools = _load_json(SOURCE_CONTRACT_FILES["agent_tools"])
    if claude_tools != codex_tools:
        raise SignatureError("Claude agent-tools.json differs from Codex")

    claude_handoff = _load_json(PLUGIN_ROOT / "contracts" / "handoff.schema.json")
    codex_handoff = _load_json(SOURCE_CONTRACT_FILES["handoff_schema"])
    if _normalize_handoff(claude_handoff) != _normalize_handoff(codex_handoff):
        raise SignatureError("Claude handoff.schema.json differs from Codex")


def build_compatibility() -> dict[str, Any]:
    _verify_copied_contracts()
    manifest = _load_json(PLUGIN_ROOT / ".claude-plugin" / "plugin.json")
    signed_files = _signed_source_files()
    return {
        "schema_version": "1.0",
        "phase": "phase_b_integrated_bundle",
        "plugin_version": manifest["version"],
        "client_contract": "claude_code_v1",
        "coordinator_contract": COORDINATOR_CONTRACT,
        "supported_clients": SUPPORTED_CLIENTS,
        "source": _source_provenance(str(manifest["repository"]), signed_files),
        "contracts": _contract_versions(),
        "signatures": {
            "source_contracts": {
                name: _sha256(path) for name, path in SOURCE_CONTRACT_FILES.items()
            },
            "runtime": _hash_files(
                REPO_ROOT,
                [
                    *(REPO_ROOT / relative for relative in RUNTIME_FILES),
                    *((REPO_ROOT / "src/cs_copilot/mcp/qsaria").rglob("*.py")),
                ],
            ),
            "bundle": _hash_files(REPO_ROOT, _bundle_files()),
        },
        "runtime_bootstrap_validation": "automated_validate_shared_runtime",
    }


def compatibility_differences(expected: dict[str, Any], current: dict[str, Any]) -> list[str]:
    keys = (
        "schema_version",
        "phase",
        "plugin_version",
        "client_contract",
        "coordinator_contract",
        "supported_clients",
        "source",
        "contracts",
        "signatures",
        "runtime_bootstrap_validation",
    )
    return [key for key in keys if expected.get(key) != current.get(key)]


def _default_pilot() -> dict[str, Any]:
    return {
        "status": "pending",
        "workflow": "standard_lightgbm_pxr",
        "experiment_id": None,
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="check recorded signatures")
    mode.add_argument("--update", action="store_true", help="record current signatures")
    parser.add_argument(
        "--pilot-status",
        choices=("pending", "passed", "failed"),
        help="record the explicit Claude scientific pilot outcome with --update",
    )
    parser.add_argument(
        "--pilot-experiment-id",
        help="record the pilot experiment id; requires --pilot-status",
    )
    args = parser.parse_args()

    if args.pilot_status and not args.update:
        parser.error("--pilot-status requires --update")
    if args.pilot_experiment_id and not args.pilot_status:
        parser.error("--pilot-experiment-id requires --pilot-status")

    try:
        current = build_compatibility()
        recorded = _load_json(COMPATIBILITY_PATH)
    except (OSError, ValueError, SyntaxError, SignatureError, json.JSONDecodeError) as exc:
        print(f"Qsaria Claude compatibility calculation failed: {exc}", file=sys.stderr)
        return 2

    if args.update:
        pilot = recorded.get("scientific_pilot") or _default_pilot()
        if args.pilot_status:
            pilot = {
                "status": args.pilot_status,
                "workflow": "standard_lightgbm_pxr",
                "experiment_id": args.pilot_experiment_id,
            }
        current["scientific_pilot"] = pilot
        _atomic_write(COMPATIBILITY_PATH, current)
        recorded = current
        print(f"Updated {COMPATIBILITY_PATH.relative_to(REPO_ROOT)}")

    differences = compatibility_differences(recorded, current)
    if differences:
        print("Qsaria Claude compatibility: FAIL", file=sys.stderr)
        print(f"- stale fields: {', '.join(differences)}", file=sys.stderr)
        return 1

    pilot = recorded.get("scientific_pilot") or _default_pilot()
    print("Qsaria Claude compatibility: PASS")
    print(f"- phase: {recorded.get('phase')}")
    print(f"- plugin version: {recorded.get('plugin_version')}")
    print(f"- scientific pilot: {pilot.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
