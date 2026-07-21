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

from .qsar_contracts import TabICLWorkerJob
from .tabicl_toolkit import TabICLToolkit


def _read_job(job_path: Path) -> TabICLWorkerJob:
    return TabICLWorkerJob.model_validate_json(job_path.read_text())


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if len(args) != 1:
        sys.stderr.write(
            "Usage: python -m cs_copilot.tools.prediction.tabicl_train_worker <job.json>\n"
        )
        return 2

    job_path = Path(args[0]).expanduser().resolve()
    job_dir = job_path.parent
    result_path = job_dir / "result.json"
    error_path = job_dir / "error.json"

    try:
        job = _read_job(job_path)
        run_request = job.run_request
        toolkit = TabICLToolkit()
        result = toolkit._run_protocol_training(
            train_csv=run_request.train_csv,
            task_type=run_request.task_type,
            resolved_output_dir=str(Path(run_request.output_dir).expanduser().resolve()),
            target_columns=list(run_request.target_columns),
            feature_columns=run_request.feature_columns,
            split_type=run_request.split_type,
            split_sizes=run_request.split_sizes,
            random_state=run_request.random_state,
            resolved_parameters={
                **run_request.parameter_payload(),
                "seed_policy": job.seed_policy,
                "validation_strategy": job.validation_strategy,
            },
            representation_name=job.representation_name,
            prediction_state=None,
            active_marker_path=Path(run_request.output_dir).expanduser().resolve()
            / ".training_in_progress",
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
