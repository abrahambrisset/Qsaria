#!/usr/bin/env python3
"""Run Claude's static plugin validator when the CLI is available."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-claude",
        action="store_true",
        help="fail rather than skip when claude is absent from PATH",
    )
    args = parser.parse_args()

    claude = shutil.which("claude")
    if claude is None:
        message = "Claude Code CLI not found; official plugin validation skipped"
        if args.require_claude:
            print(f"FAIL: {message}")
            return 1
        print(f"SKIP: {message}")
        return 0

    completed = subprocess.run(
        [claude, "plugin", "validate", str(PLUGIN_ROOT), "--strict"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip())
    if completed.returncode:
        print("Qsaria Claude official plugin validation: FAIL")
        return completed.returncode
    print("Qsaria Claude official plugin validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
