#!/usr/bin/env python3
"""Check or record Qsaria-to-Claude Science compatibility signatures."""

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

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
COMPATIBILITY_PATH = BUNDLE_ROOT / "contracts" / "compatibility.json"
CODEX_CONTRACT_ROOT = REPO_ROOT / "plugins" / "qsaria-codex" / "contracts"

RUNTIME_FILES = (
    "src/cs_copilot/mcp/__main__.py",
    "src/cs_copilot/mcp/server.py",
    "src/cs_copilot/mcp/tools_registry.py",
    "src/cs_copilot/mcp/tool_specs/qsaria.py",
    "src/cs_copilot/mcp/tool_specs/qsaria_lifecycle.py",
)
LAUNCHER_FILES = (
    "plugins/qsaria-codex/.mcp.json",
    "plugins/qsaria-claude-code/.mcp.json",
)
CONTRACT_CONSTANTS = {
    "experiment": ("src/cs_copilot/mcp/qsaria/contracts.py", "EXPERIMENT_SCHEMA_VERSION"),
    "handoff": ("src/cs_copilot/mcp/qsaria/contracts.py", "HANDOFF_SCHEMA_VERSION"),
    "report_facts": (
        "src/cs_copilot/tools/prediction/qsar_reporting.py",
        "REPORT_FACTS_SCHEMA_VERSION",
    ),
    "training": (
        "src/cs_copilot/tools/prediction/qsar_contracts.py",
        "TRAINING_CONTRACT_VERSION",
    ),
}
SUPPORTED_CLIENTS = ["codex_v1", "claude_code_v1", "claude_science_v1"]


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
        raise SignatureError(f"expected JSON object at {path}")
    return payload


def _literal_constant(relative: str, name: str) -> str:
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
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
    raise SignatureError(f"missing contract constant {name} in {relative}")


def _bundle_files() -> list[Path]:
    return sorted(
        path
        for path in BUNDLE_ROOT.rglob("*")
        if path.is_file()
        and path.resolve() != COMPATIBILITY_PATH.resolve()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )


def _signed_source_files() -> list[Path]:
    files = list(_bundle_files())
    files.extend(REPO_ROOT / relative for relative in RUNTIME_FILES)
    files.extend(REPO_ROOT / relative for relative in LAUNCHER_FILES)
    files.extend((REPO_ROOT / "src/cs_copilot/mcp/qsaria").rglob("*.py"))
    files.extend(
        (
            CODEX_CONTRACT_ROOT / "agent-tools.json",
            CODEX_CONTRACT_ROOT / "handoff.schema.json",
            CODEX_CONTRACT_ROOT / "scientific-invariants.md",
        )
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise SignatureError("missing signed source: " + ", ".join(missing))
    return sorted({path.resolve() for path in files})


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise SignatureError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _normalize_handoff(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    normalized.pop("$id", None)
    return normalized


def _verify_handoff() -> None:
    science = _load_json(BUNDLE_ROOT / "contracts" / "handoff.schema.json")
    codex = _load_json(CODEX_CONTRACT_ROOT / "handoff.schema.json")
    if _normalize_handoff(science) != _normalize_handoff(codex):
        raise SignatureError("Science handoff schema differs from the canonical Codex schema")


def build_compatibility() -> dict[str, Any]:
    _verify_handoff()
    bundle = _load_json(BUNDLE_ROOT / "bundle.json")
    signed_files = _signed_source_files()
    relative = [path.relative_to(REPO_ROOT).as_posix() for path in signed_files]
    source_commit = _git("log", "-1", "--format=%H", "--", *relative)
    if not source_commit:
        raise SignatureError("no committed provenance found for the Science source tree")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=all", "--", *relative))
    return {
        "schema_version": "1.0",
        "bundle_version": bundle["version"],
        "client_contract": "claude_science_v1",
        "coordinator_contract": "external_mcp_coordinator_v1",
        "supported_clients": SUPPORTED_CLIENTS,
        "minimum_claude_science_version": bundle["minimum_claude_science_version"],
        "source": {
            "repository": bundle["repository"],
            "commit": source_commit,
            "tree": _git("rev-parse", f"{source_commit}^{{tree}}"),
            "describe": _git("describe", "--tags", "--always", source_commit),
            "signed_tree": _hash_files(REPO_ROOT, signed_files),
            "dirty": dirty,
        },
        "contracts": {
            key: _literal_constant(relative_path, constant)
            for key, (relative_path, constant) in CONTRACT_CONSTANTS.items()
        },
        "signatures": {
            "canonical_handoff": _sha256(CODEX_CONTRACT_ROOT / "handoff.schema.json"),
            "canonical_agent_tools": _sha256(CODEX_CONTRACT_ROOT / "agent-tools.json"),
            "science_agent_tools": _sha256(BUNDLE_ROOT / "contracts" / "agent-tools.json"),
            "runtime": _hash_files(
                REPO_ROOT,
                [
                    *(REPO_ROOT / relative_path for relative_path in RUNTIME_FILES),
                    *((REPO_ROOT / "src/cs_copilot/mcp/qsaria").rglob("*.py")),
                ],
            ),
            "bundle": _hash_files(REPO_ROOT, _bundle_files()),
        },
        "operation_contract": {
            "schema_version": "1.0",
            "synchronous_tools": 46,
            "durable_operation_tools": 7,
            "poll_interval_seconds": 20,
            "cancellation": False,
        },
    }


def compatibility_differences(expected: dict[str, Any], current: dict[str, Any]) -> list[str]:
    keys = (
        "schema_version",
        "bundle_version",
        "client_contract",
        "coordinator_contract",
        "supported_clients",
        "minimum_claude_science_version",
        "source",
        "contracts",
        "signatures",
        "operation_contract",
    )
    return [key for key in keys if expected.get(key) != current.get(key)]


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify recorded signatures")
    mode.add_argument("--update", action="store_true", help="record current signatures")
    args = parser.parse_args()
    try:
        current = build_compatibility()
    except (OSError, ValueError, json.JSONDecodeError, SignatureError) as exc:
        print(f"Qsaria Science compatibility calculation failed: {exc}", file=sys.stderr)
        return 2
    if args.update:
        _atomic_write(COMPATIBILITY_PATH, current)
        print(f"Updated {COMPATIBILITY_PATH}")
        return 0
    try:
        expected = _load_json(COMPATIBILITY_PATH)
    except (OSError, json.JSONDecodeError, SignatureError) as exc:
        print(f"Qsaria Science compatibility metadata missing: {exc}", file=sys.stderr)
        return 1
    differences = compatibility_differences(expected, current)
    if differences:
        print("Qsaria Science compatibility is stale: " + ", ".join(differences), file=sys.stderr)
        return 1
    print("Qsaria Science compatibility signatures match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
