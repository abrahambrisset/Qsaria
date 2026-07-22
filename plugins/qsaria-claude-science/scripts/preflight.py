#!/usr/bin/env python3
"""Print a non-mutating Claude Science installation preflight."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
MINIMUM_SCIENCE_VERSION = (0, 1, 21)
_VERSION_RE = re.compile(r"\b(\d+)\.(\d+)\.(\d+)\b")


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _version_tuple(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    match = _VERSION_RE.search(value)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def _science_version() -> str | None:
    candidates = (
        Path("/Applications/Claude Science.app/Contents/Resources/bin/claude-science"),
        Path.home() / ".local" / "bin" / "claude-science",
    )
    detected: list[tuple[tuple[int, int, int], str]] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        result = subprocess.run(
            [str(candidate), "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode != 0:
            continue
        value = result.stdout.strip() or result.stderr.strip()
        version = _version_tuple(value)
        if version is not None:
            detected.append((version, value))
    return max(detected, default=((), None), key=lambda item: item[0])[1]


def _supported_science_version(value: str | None) -> bool:
    version = _version_tuple(value)
    return bool(version and version >= MINIMUM_SCIENCE_VERSION)


def _artifact_roots() -> list[str]:
    science_root = Path.home() / ".claude-science" / "orgs"
    if not science_root.is_dir():
        return []
    candidates: list[Path] = []
    for org in science_root.iterdir():
        if not org.is_dir():
            continue
        for relative in ("artifacts", "files", "uploads"):
            candidate = org / relative
            if candidate.is_dir():
                candidates.append(candidate.resolve())
    return sorted({str(path) for path in candidates})


def build_report(artifact_root: str | None = None) -> dict[str, Any]:
    python = REPO_ROOT / ".venv" / "bin" / "python"
    launcher = PLUGIN_ROOT / "scripts" / "launch-mcp.sh"
    resolved_python = python.resolve(strict=False)
    roots = _artifact_roots()
    selected_artifact_root = artifact_root or (roots[0] if len(roots) == 1 else None)
    science_version = _science_version()
    checks = [
        _check("repository", (REPO_ROOT / "pyproject.toml").is_file(), str(REPO_ROOT)),
        _check("python", python.is_file(), f"{python} -> {resolved_python}"),
        _check("launcher", launcher.is_file(), str(launcher)),
        _check("storage", (REPO_ROOT / ".files").is_dir(), str(REPO_ROOT / ".files")),
        _check("models", (REPO_ROOT / "data").is_dir(), str(REPO_ROOT / "data")),
        _check(
            "artifact-root",
            bool(selected_artifact_root and Path(selected_artifact_root).is_dir()),
            selected_artifact_root or "set QSARIA_SCIENCE_ARTIFACT_ROOT explicitly",
        ),
        _check(
            "science-version",
            _supported_science_version(science_version),
            science_version or "Claude Science was not detected; require 0.1.21 or later",
        ),
    ]
    return {
        "status": "ok" if all(item["ok"] for item in checks) else "needs_configuration",
        "science_version": science_version,
        "minimum_claude_science_version": "0.1.21",
        "checks": checks,
        "connector": {
            "name": "qsaria",
            "type": "Local command",
            "command": "/bin/zsh",
            "arguments": [str(launcher)],
            "environment": {
                "QSARIA_SCIENCE_ARTIFACT_ROOT": selected_artifact_root
                or "<select-one-artifact-root>"
            },
        },
        "permissions": {
            "read_only": [str(REPO_ROOT), str(resolved_python.parent.parent)],
            "read_write": [str(REPO_ROOT / ".files"), str(REPO_ROOT / "data")],
            "artifact_read_only": selected_artifact_root,
        },
        "discovered_artifact_roots": roots,
        "notes": [
            "Grant the repository read-only before granting nested .files and data read/write.",
            "Restart Claude Science after changing persistent folder permissions.",
            "Do not use --dangerously-no-sandbox or --dangerously-skip-approvals.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", help="explicit Claude Science artifact root")
    args = parser.parse_args()
    report = build_report(args.artifact_root)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
