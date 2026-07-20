#!/usr/bin/env python3
"""Install and inspect the Qsaria plugin in an isolated temporary Codex home."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]


class SmokeFailure(RuntimeError):
    """Raised when one isolated Codex CLI step fails validation."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Qsaria repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--require-codex",
        action="store_true",
        help="fail instead of reporting a skip when the Codex CLI is unavailable",
    )
    return parser.parse_args()


def _run(
    codex: str,
    arguments: list[str],
    *,
    repo_root: Path,
    environment: dict[str, str],
) -> str:
    completed = subprocess.run(
        [codex, *arguments],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SmokeFailure(f"codex {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _json_output(output: str, *, command: str) -> Any:
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise SmokeFailure(f"{command} did not return valid JSON: {output[:500]}") from exc


def run_smoke(repo_root: Path, codex: str) -> dict[str, Any]:
    marketplace_path = repo_root / ".agents" / "plugins" / "marketplace.json"
    marketplace = _json_output(
        marketplace_path.read_text(encoding="utf-8"),
        command=str(marketplace_path),
    )
    marketplace_name = marketplace.get("name") if isinstance(marketplace, dict) else None
    if not isinstance(marketplace_name, str) or not marketplace_name:
        raise SmokeFailure("repo marketplace must declare a non-empty name")

    with tempfile.TemporaryDirectory(prefix="qsaria-codex-smoke-") as temporary_name:
        temporary = Path(temporary_name)
        isolated_home = temporary / "home"
        codex_home = temporary / "codex-home"
        isolated_home.mkdir()
        codex_home.mkdir()
        environment = os.environ.copy()
        environment.update(
            {
                "HOME": str(isolated_home),
                "CODEX_HOME": str(codex_home),
                "TMPDIR": str(temporary / "tmp"),
            }
        )
        Path(environment["TMPDIR"]).mkdir()

        _run(
            codex,
            ["plugin", "marketplace", "add", str(repo_root), "--json"],
            repo_root=repo_root,
            environment=environment,
        )
        _run(
            codex,
            ["plugin", "add", f"qsaria-codex@{marketplace_name}", "--json"],
            repo_root=repo_root,
            environment=environment,
        )
        plugin_list = _json_output(
            _run(
                codex,
                ["plugin", "list", "--json"],
                repo_root=repo_root,
                environment=environment,
            ),
            command="codex plugin list --json",
        )
        mcp_list = _json_output(
            _run(
                codex,
                ["mcp", "list", "--json"],
                repo_root=repo_root,
                environment=environment,
            ),
            command="codex mcp list --json",
        )

        plugins_serialized = json.dumps(plugin_list, sort_keys=True)
        mcp_serialized = json.dumps(mcp_list, sort_keys=True)
        if "qsaria-codex" not in plugins_serialized or "installed" not in plugins_serialized:
            raise SmokeFailure("installed Qsaria plugin is absent from `codex plugin list`")
        required_mcp_tokens = {
            "qsaria",
            "cscopilot-mcp",
            "--profile",
            "--llm-policy",
            "disabled",
        }
        missing = sorted(token for token in required_mcp_tokens if token not in mcp_serialized)
        if missing:
            raise SmokeFailure(f"Qsaria MCP registration is incomplete; missing={missing}")

        return {
            "ok": True,
            "marketplace": marketplace_name,
            "plugin": "qsaria-codex",
            "isolated": True,
        }


def main() -> int:
    args = _parse_args()
    codex = shutil.which("codex")
    if codex is None:
        message = "Codex CLI unavailable; isolated plugin smoke skipped"
        if args.require_codex:
            print(message, file=sys.stderr)
            return 1
        print(message)
        return 0
    try:
        result = run_smoke(args.repo_root.resolve(), codex)
    except (OSError, subprocess.TimeoutExpired, SmokeFailure) as exc:
        print(f"Qsaria Codex smoke failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
