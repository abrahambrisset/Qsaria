#!/usr/bin/env python3
"""Check or record Qsaria-to-Codex compatibility signatures."""

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
from typing import Iterable

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
COMPATIBILITY_PATH = PLUGIN_ROOT / "contracts" / "compatibility.json"

PROMPT_CONSTANTS = (
    "HANDLING_NEW_FILES_INSTRUCTIONS",
    "DATASET_CURATION_INSTRUCTIONS",
    "QSAR_TRAINING_INSTRUCTIONS",
    "MODEL_REGISTRY_INSTRUCTIONS",
    "MODEL_INFERENCE_INSTRUCTIONS",
    "QSAR_REPORT_INSTRUCTIONS",
)

TOOLKIT_CLASSES = {
    "src/cs_copilot/tools/curation/dataset_curation_toolkit.py": "DatasetCurationToolkit",
    "src/cs_copilot/tools/prediction/qsar_training_toolkit.py": "QSARTrainingToolkit",
    "src/cs_copilot/tools/prediction/model_registry_toolkit.py": "ModelRegistryToolkit",
    "src/cs_copilot/tools/prediction/prediction_inference_toolkit.py": "PredictionInferenceToolkit",
    "src/cs_copilot/tools/prediction/ensemble_toolkit.py": "EnsembleToolkit",
    "src/cs_copilot/tools/prediction/benchmark_toolkit.py": "BenchmarkToolkit",
    "src/cs_copilot/tools/activity_cliffs/toolkit.py": "ActivityCliffToolkit",
}

REPORTING_FILES = (
    "src/cs_copilot/tools/prediction/qsar_reporting.py",
    "src/cs_copilot/tools/prediction/qsar_response_compaction.py",
)

SUPPORTING_SCIENTIFIC_FILES = (
    "src/cs_copilot/tools/prediction/catalog.py",
    "src/cs_copilot/tools/prediction/external_evaluation.py",
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

QSARIA_PROFILE_FILES = (
    "src/cs_copilot/mcp/__main__.py",
    "src/cs_copilot/mcp/server.py",
    "src/cs_copilot/mcp/tools_registry.py",
    "src/cs_copilot/mcp/tool_specs/qsaria.py",
    "src/cs_copilot/mcp/tool_specs/qsaria_lifecycle.py",
)

REPOSITORY_INTEGRATION_FILES = (
    ".agents/plugins/marketplace.json",
    ".codex/config.toml",
    ".github/ISSUE_TEMPLATE/qsaria_codex_bug.yml",
    ".github/workflows/ci.yml",
)


class SignatureError(RuntimeError):
    """Raised when an expected Qsaria contract cannot be found."""


def _sha256(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return f"sha256:{digest.hexdigest()}"


def _hash_files(root: Path, paths: Iterable[Path]) -> str:
    chunks: list[bytes] = []
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        chunks.extend((relative, path.read_bytes()))
    return _sha256(chunks)


def _source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise SignatureError("could not recover an AST source segment")
    return segment


def _prompt_signature(repo_root: Path) -> str:
    prompts_path = repo_root / "src/cs_copilot/agents/prompts.py"
    teams_path = repo_root / "src/cs_copilot/agents/teams.py"
    prompt_source = prompts_path.read_text(encoding="utf-8")
    team_source = teams_path.read_text(encoding="utf-8")
    prompt_tree = ast.parse(prompt_source)
    team_tree = ast.parse(team_source)

    assignments: dict[str, ast.AST] = {}
    for node in prompt_tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in PROMPT_CONSTANTS:
                    assignments[target.id] = node

    missing = sorted(set(PROMPT_CONSTANTS) - assignments.keys())
    if missing:
        raise SignatureError(f"missing prompt contracts: {', '.join(missing)}")

    team_function = next(
        (
            node
            for node in team_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "get_qsar_agent_team"
        ),
        None,
    )
    if team_function is None:
        raise SignatureError("missing get_qsar_agent_team")

    chunks = [
        f"{name}\n{_source_segment(prompt_source, assignments[name])}".encode("utf-8")
        for name in PROMPT_CONSTANTS
    ]
    chunks.append(_source_segment(team_source, team_function).encode("utf-8"))
    return _sha256(chunks)


def _callable_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    returns = ast.unparse(node.returns) if node.returns is not None else ""
    decorators = ",".join(ast.unparse(item) for item in node.decorator_list)
    return f"{prefix}{node.name}({ast.unparse(node.args)})->{returns}|{decorators}"


def _toolkit_surface(repo_root: Path, relative: str, class_name: str) -> bytes:
    source = (repo_root / relative).read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name),
        None,
    )
    if class_node is None:
        raise SignatureError(f"missing {class_name} in {relative}")
    methods = [
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]
    canonical = "\n".join(_callable_signature(node) for node in methods)
    return f"{relative}:{class_name}\n{canonical}".encode("utf-8")


def _tool_surface_signature(repo_root: Path) -> str:
    chunks = [
        _toolkit_surface(repo_root, relative, class_name)
        for relative, class_name in sorted(TOOLKIT_CLASSES.items())
    ]
    qsaria_mcp_root = repo_root / "src/cs_copilot/mcp/qsaria"
    if not qsaria_mcp_root.is_dir():
        raise SignatureError(f"missing Qsaria MCP package: {qsaria_mcp_root}")
    mcp_files = sorted(qsaria_mcp_root.rglob("*.py"))
    if not mcp_files:
        raise SignatureError("Qsaria MCP package contains no Python files")
    for path in mcp_files:
        chunks.extend((path.relative_to(repo_root).as_posix().encode("utf-8"), path.read_bytes()))
    for relative in QSARIA_PROFILE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise SignatureError(f"missing Qsaria profile source: {path}")
        chunks.extend((relative.encode("utf-8"), path.read_bytes()))
    for relative in SUPPORTING_SCIENTIFIC_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise SignatureError(f"missing supporting scientific source: {path}")
        chunks.extend((relative.encode("utf-8"), path.read_bytes()))
    return _sha256(chunks)


def _reporting_signature(repo_root: Path) -> str:
    paths = [repo_root / relative for relative in REPORTING_FILES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SignatureError(f"missing reporting sources: {', '.join(missing)}")
    return _hash_files(repo_root, paths)


def _literal_constant(repo_root: Path, relative: str, name: str) -> str:
    path = repo_root / relative
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
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


def _contract_versions(repo_root: Path) -> dict[str, str]:
    return {
        contract: _literal_constant(repo_root, relative, constant)
        for contract, (relative, constant) in CONTRACT_CONSTANTS.items()
    }


def _bundle_signature() -> str:
    return _hash_files(PLUGIN_ROOT, _bundle_files())


def _bundle_files() -> list[Path]:
    paths: list[Path] = [
        PLUGIN_ROOT / ".codex-plugin" / "plugin.json",
        PLUGIN_ROOT / ".mcp.json",
        PLUGIN_ROOT / "README.md",
    ]
    paths.extend((PLUGIN_ROOT / "skills").rglob("*"))
    paths.extend((PLUGIN_ROOT / "assets").rglob("*"))
    paths.extend((PLUGIN_ROOT / "contracts").rglob("*"))
    paths.extend((PLUGIN_ROOT / "scripts").glob("*.py"))
    filtered = [
        path for path in paths if path.is_file() and path.resolve() != COMPATIBILITY_PATH.resolve()
    ]
    return sorted(set(filtered))


def _signed_source_files(repo_root: Path) -> list[Path]:
    paths = [
        repo_root / "src/cs_copilot/agents/prompts.py",
        repo_root / "src/cs_copilot/agents/teams.py",
    ]
    paths.extend(repo_root / relative for relative in TOOLKIT_CLASSES)
    paths.extend(repo_root / relative for relative in REPORTING_FILES)
    paths.extend(repo_root / relative for relative in SUPPORTING_SCIENTIFIC_FILES)
    paths.extend(repo_root / relative for relative in QSARIA_PROFILE_FILES)
    paths.extend((repo_root / "src/cs_copilot/mcp/qsaria").rglob("*.py"))
    paths.extend(_bundle_files())
    paths.extend(repo_root / relative for relative in REPOSITORY_INTEGRATION_FILES)
    paths.extend((repo_root / ".codex/agents").glob("qsaria_*.toml"))
    files = sorted({path.resolve() for path in paths if path.is_file()})
    if not files:
        raise SignatureError("signed Qsaria source tree contains no files")
    for path in files:
        try:
            path.relative_to(repo_root)
        except ValueError as exc:
            raise SignatureError(f"signed source escapes repository root: {path}") from exc
    return files


def _source_provenance(repo_root: Path, repository: str) -> dict[str, object]:
    signed_files = _signed_source_files(repo_root)
    relative_paths = [path.relative_to(repo_root).as_posix() for path in signed_files]
    commit = _git(
        repo_root,
        "log",
        "-1",
        "--format=%H",
        "--",
        *relative_paths,
    )
    if not commit:
        raise SignatureError(
            "no committed provenance found for the signed source tree; fetch Git history first"
        )
    commit_tree = _git(repo_root, "rev-parse", f"{commit}^{{tree}}")
    describe = _git(repo_root, "describe", "--tags", "--always", commit)
    status = _git(
        repo_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *relative_paths,
    )
    return {
        "repository": repository,
        "commit": commit,
        "tree": commit_tree,
        "signed_tree": _hash_files(repo_root, signed_files),
        "describe": describe,
        "dirty": bool(status),
    }


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SignatureError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def build_compatibility(repo_root: Path) -> dict[str, object]:
    manifest = json.loads(
        (PLUGIN_ROOT / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    return {
        "schema_version": "1.0",
        "plugin_version": manifest["version"],
        "source": _source_provenance(repo_root, str(manifest["repository"])),
        "contracts": _contract_versions(repo_root),
        "signatures": {
            "prompts": _prompt_signature(repo_root),
            "tool_surface": _tool_surface_signature(repo_root),
            "reporting": _reporting_signature(repo_root),
            "bundle": _bundle_signature(),
        },
    }


def compatibility_differences(expected: dict[str, object], current: dict[str, object]) -> list[str]:
    differences: list[str] = []
    for key in ("schema_version", "plugin_version", "source", "contracts", "signatures"):
        if expected.get(key) != current.get(key):
            differences.append(key)
    return differences


def _atomic_json_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify recorded signatures")
    mode.add_argument("--update", action="store_true", help="record current signatures")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Qsaria repository root (default: inferred from this script)",
    )
    args = parser.parse_args()

    try:
        current = build_compatibility(args.repo_root.resolve())
    except (OSError, ValueError, SyntaxError, SignatureError) as exc:
        print(f"compatibility calculation failed: {exc}", file=sys.stderr)
        return 2

    if args.update:
        _atomic_json_write(COMPATIBILITY_PATH, current)
        print(f"updated {COMPATIBILITY_PATH}")
        return 0

    try:
        expected = json.loads(COMPATIBILITY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not read compatibility metadata: {exc}", file=sys.stderr)
        return 2

    differences = compatibility_differences(expected, current)
    if differences:
        print(
            "Qsaria Codex compatibility is stale: " + ", ".join(differences),
            file=sys.stderr,
        )
        print("review the bundle, then run this command with --update", file=sys.stderr)
        return 1
    print("Qsaria Codex compatibility signatures match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
