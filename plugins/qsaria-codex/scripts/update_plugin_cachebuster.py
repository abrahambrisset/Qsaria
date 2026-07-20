#!/usr/bin/env python3
"""Give the local Qsaria plugin a fresh Codex cachebuster version."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
CACHEBUSTER_PREFIX = "codex"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cachebuster",
        help="optional deterministic token; the default is the current UTC timestamp",
    )
    return parser.parse_args()


def _sanitize_cachebuster(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9-]+", "-", value.strip().lower())
    sanitized = re.sub(r"-{2,}", "-", sanitized).strip("-")
    if not sanitized:
        raise ValueError("cachebuster must contain at least one letter or digit")
    return sanitized


def _with_cachebuster(version: str, cachebuster: str) -> str:
    base_version = version.split("+", 1)[0]
    return f"{base_version}+{CACHEBUSTER_PREFIX}.{cachebuster}"


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = _parse_args()
    manifest_path = PLUGIN_ROOT / ".codex-plugin" / "plugin.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("plugin manifest must contain a JSON object")
        version = manifest.get("version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("plugin manifest must contain a non-empty string version")
        token = args.cachebuster or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        next_version = _with_cachebuster(version, _sanitize_cachebuster(token))
        manifest["version"] = next_version
        _atomic_json_write(manifest_path, manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cachebuster update failed: {exc}", file=sys.stderr)
        return 1

    print(f"updated plugin version: {version} -> {next_version}")
    print("run sync_qsaria_codex.py --update before reinstalling the plugin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
