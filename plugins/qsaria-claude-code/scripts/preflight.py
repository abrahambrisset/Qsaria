#!/usr/bin/env python3
"""Run non-mutating repository checks for the Qsaria Claude Code plugin."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = PLUGIN_ROOT / "scripts"


def _run(script: str, *args: str) -> tuple[bool, str]:
    uv_path = shutil.which("uv")
    command = [uv_path, "run", "--no-sync", "python"] if uv_path is not None else [sys.executable]
    environment = os.environ.copy()
    environment.setdefault(
        "UV_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "qsaria-uv-cache"),
    )
    completed = subprocess.run(
        [*command, str(SCRIPTS_ROOT / script), *args],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    detail = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-claude",
        action="store_true",
        help="fail when the Claude Code CLI is not visible in PATH",
    )
    args = parser.parse_args()

    checks: list[tuple[str, bool, str]] = []
    checks.append(("repository", (REPO_ROOT / "pyproject.toml").is_file(), str(REPO_ROOT)))
    uv_path = shutil.which("uv")
    checks.append(("uv", uv_path is not None, uv_path or "not found"))

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    checks.append(
        (
            "MCP entrypoint",
            "cscopilot-mcp" in pyproject,
            "declared" if "cscopilot-mcp" in pyproject else "missing",
        )
    )
    valid, detail = _run("validate_claude_bundle.py")
    checks.append(("bundle", valid, detail.splitlines()[0] if detail else "no output"))
    synced, detail = _run("sync_qsaria_claude.py", "--check")
    checks.append(("contract sync", synced, detail.splitlines()[0] if detail else "no output"))
    runtime, detail = _run("validate_shared_runtime.py")
    checks.append(("shared runtime", runtime, detail.splitlines()[0] if detail else "no output"))

    claude_path = shutil.which("claude")
    claude_ok = claude_path is not None or not args.require_claude
    claude_detail = claude_path or "not in PATH; official CLI validation is unavailable"
    checks.append(("Claude CLI", claude_ok, claude_detail))

    failed = False
    for name, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        failed = failed or not passed
    print("[INFO] scientific pilot remains an explicit end-to-end validation")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
