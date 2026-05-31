#!/usr/bin/env python
# coding: utf-8
"""
Dedicated subprocess worker for isolated TabICL training runs.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from .tabicl_toolkit import TabICLToolkit


def _read_job(job_path: Path) -> Dict[str, Any]:
    payload = json.loads(job_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("TabICL worker job payload must be a JSON object.")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if len(args) != 1:
        sys.stderr.write("Usage: python -m cs_copilot.tools.prediction.tabicl_train_worker <job.json>\n")
        return 2

    job_path = Path(args[0]).expanduser().resolve()
    job_dir = job_path.parent
    result_path = job_dir / "result.json"
    error_path = job_dir / "error.json"

    try:
        job = _read_job(job_path)
        toolkit = TabICLToolkit()
        result = toolkit._run_protocol_training(
            train_csv=str(job["train_csv"]),
            task_type=str(job["task_type"]),
            resolved_output_dir=str(Path(job["output_dir"]).expanduser().resolve()),
            target_columns=list(job["target_columns"]),
            feature_columns=job.get("feature_columns"),
            split_type=str(job.get("split_type", "random")),
            split_sizes=job.get("split_sizes"),
            random_state=int(job.get("random_state", 42)),
            extra_args=dict(job.get("extra_args") or {}),
            prediction_state=None,
            active_marker_path=Path(job["output_dir"]).expanduser().resolve() / ".training_in_progress",
            worker_pid=os.getpid(),
            worker_status="running",
        )
        _write_json(result_path, result)
        return 0
    except Exception as exc:
        _write_json(
            error_path,
            {
                "error_message": str(exc),
                "exception_type": exc.__class__.__name__,
                "traceback": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
