#!/usr/bin/env python3
"""Two-step detached-process probe for the Claude Science sandbox."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _start(root: Path, seconds: float) -> int:
    probe_id = f"probe_{uuid.uuid4().hex}"
    state_path = root.resolve() / f"{probe_id}.json"
    _atomic_write(
        state_path,
        {"probe_id": probe_id, "status": "queued", "seconds": seconds},
    )
    process = subprocess.Popen(  # noqa: S603 - this exact script is the worker
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--state",
            str(state_path),
            "--seconds",
            str(seconds),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )
    print(
        json.dumps(
            {
                "probe_id": probe_id,
                "state": str(state_path),
                "worker_pid": process.pid,
                "check_after_seconds": seconds + 2,
            },
            indent=2,
        )
    )
    return 0


def _worker(state: Path, seconds: float) -> int:
    _atomic_write(
        state,
        {"status": "running", "worker_pid": os.getpid(), "seconds": seconds},
    )
    time.sleep(seconds)
    _atomic_write(
        state,
        {"status": "completed", "worker_pid": os.getpid(), "seconds": seconds},
    )
    return 0


def _check(state: Path) -> int:
    if not state.is_file():
        print(json.dumps({"status": "missing", "state": str(state)}, indent=2))
        return 1
    payload = json.loads(state.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--root", type=Path, default=Path("/tmp"))
    start.add_argument("--seconds", type=float, default=65.0)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--state", type=Path, required=True)
    worker.add_argument("--seconds", type=float, required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "start":
        return _start(args.root, args.seconds)
    if args.command == "worker":
        return _worker(args.state, args.seconds)
    return _check(args.state)


if __name__ == "__main__":
    raise SystemExit(main())
