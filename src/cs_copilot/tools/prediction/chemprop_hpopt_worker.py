#!/usr/bin/env python
# coding: utf-8
"""Strict subprocess entry point for customized Chemprop HPO distributions."""

from __future__ import annotations

import sys
from pathlib import Path

from .qsar_contracts import (
    ChempropHpoptWorkerJob,
    FloatSearchRange,
    IntSearchRange,
)


def _ray_distribution(spec):
    from ray import tune

    if isinstance(spec, IntSearchRange):
        upper = spec.high + 1
        if spec.log:
            return tune.lograndint(spec.low, upper)
        if spec.step != 1:
            return tune.qrandint(spec.low, upper, spec.step)
        return tune.randint(spec.low, upper)
    if isinstance(spec, FloatSearchRange):
        if spec.log:
            return tune.loguniform(spec.low, spec.high)
        return tune.uniform(spec.low, spec.high)
    raise TypeError(f"Unsupported Chemprop tuning distribution: {type(spec).__name__}")


def _read_job(path: Path) -> ChempropHpoptWorkerJob:
    return ChempropHpoptWorkerJob.model_validate_json(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if len(args) != 1:
        sys.stderr.write(
            "Usage: python -m cs_copilot.tools.prediction.chemprop_hpopt_worker <job.json>\n"
        )
        return 2

    job = _read_job(Path(args[0]).expanduser().resolve())
    from chemprop.cli import hpopt
    from chemprop.cli.main import main as chemprop_main

    patched_space = dict(hpopt.SEARCH_SPACE)
    for name, _distribution_spec in job.search_space.model_dump(exclude_none=True).items():
        if name == "backend":
            continue
        spec = getattr(job.search_space, name)
        patched_space[name] = _ray_distribution(spec)
    hpopt.SEARCH_SPACE = patched_space

    sys.argv = ["chemprop", *job.argv]
    chemprop_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
