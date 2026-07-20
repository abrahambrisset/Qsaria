#!/usr/bin/env python3
"""Preview, install, or check the Qsaria project-agent templates."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_ROOT = PLUGIN_ROOT / "assets" / "agents"


def _templates() -> list[Path]:
    return sorted(TEMPLATE_ROOT.glob("qsaria_*.toml"))


def _same_bytes(source: Path, destination: Path) -> bool:
    return destination.is_file() and source.read_bytes() == destination.read_bytes()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(source.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="copy templates into the project")
    mode.add_argument("--check", action="store_true", help="require installed files to match")
    parser.add_argument("--force", action="store_true", help="replace differing installed files")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Qsaria repository root (default: inferred from this script)",
    )
    args = parser.parse_args()

    templates = _templates()
    if len(templates) != 5:
        print(f"expected five agent templates, found {len(templates)}", file=sys.stderr)
        return 2

    destination_root = args.repo_root.resolve() / ".codex" / "agents"
    mismatches: list[str] = []
    blocked = False
    for source in templates:
        destination = destination_root / source.name
        if _same_bytes(source, destination):
            action = "unchanged"
        elif not destination.exists():
            action = "create"
            mismatches.append(source.name)
        else:
            action = "replace"
            mismatches.append(source.name)

        if args.check:
            print(f"{action}: {destination}")
            continue

        if args.apply:
            if action == "replace" and not args.force:
                print(f"blocked (use --force): {destination}", file=sys.stderr)
                blocked = True
                continue
            if action != "unchanged":
                _atomic_copy(source, destination)
            print(f"{action}: {destination}")
        else:
            print(f"would {action}: {destination}")

    if args.check and mismatches:
        print("project agents are not synchronized", file=sys.stderr)
        return 1
    if blocked:
        return 1

    config_template = PLUGIN_ROOT / "assets" / "project-config.toml"
    print(f"config template: {config_template}")
    if not args.apply and not args.check:
        print("preview only; rerun with --apply after reviewing the destinations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
