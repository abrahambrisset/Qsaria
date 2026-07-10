#!/usr/bin/env python
# coding: utf-8
"""
Chemprop backend adapter.

This adapter intentionally keeps the rest of the codebase insulated from the
specific Chemprop CLI/API.  The initial implementation uses the Chemprop CLI
because it provides a stable operational path for training and prediction over
CSV files, which fits the project's S3/local file abstraction well.
"""

from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .backend import (
    BackendNotAvailableError,
    InvalidPredictionInputError,
    PredictionBackend,
    PredictionExecutionError,
    PredictionModelRecord,
    PredictionTaskSpec,
)
from .backend_capabilities import enrich_backend_environment
from .chemprop_adapter import normalize_chemprop_classification_predictions
from .training_orchestration import is_classification_task, normalize_task_type

logger = logging.getLogger(__name__)

EPOCH_PROGRESS_RE = re.compile(r"\bepoch\b[^0-9]*(\d+)(?:\s*/\s*(\d+))?", re.IGNORECASE)
DEFAULT_CHEMPROP_FINGERPRINT_FFN_BLOCK_INDEX = -1


class ChempropBackend(PredictionBackend):
    """Prediction backend built around Chemprop v2 CLI commands."""

    backend_name = "chemprop"
    MODEL_EXTENSIONS = (".ckpt", ".pt")

    @staticmethod
    def _classification_targets_from_record(
        model_record: PredictionModelRecord,
    ) -> Dict[str, Dict[str, Any]]:
        targets: Dict[str, Dict[str, Any]] = {}
        for source in (
            model_record.inference_profile or {},
            model_record.training_data_summary or {},
        ):
            stored = source.get("classification_targets")
            if isinstance(stored, dict):
                targets.update(
                    {
                        str(target): dict(metadata)
                        for target, metadata in stored.items()
                        if isinstance(metadata, dict)
                    }
                )
        primary_target = (model_record.task.target_columns or [None])[0]
        if primary_target and primary_target not in targets:
            for source in (
                model_record.inference_profile or {},
                model_record.training_data_summary or {},
            ):
                labels = source.get("class_labels")
                if labels:
                    targets[primary_target] = {
                        "class_labels": list(labels),
                        "class_count": int(source.get("class_count") or len(labels)),
                        "label_mapping": source.get("label_mapping") or {},
                    }
                    break
        return targets

    def _find_cli_path(self) -> Optional[str]:
        cli_path = shutil.which("chemprop")
        if cli_path:
            return cli_path

        candidate_roots = []

        venv_root = os.getenv("VIRTUAL_ENV")
        if venv_root:
            candidate_roots.append(Path(venv_root))

        candidate_roots.append(Path(sys.prefix))
        candidate_roots.append(Path("/app/.venv"))

        seen = set()
        for root in candidate_roots:
            if root in seen:
                continue
            seen.add(root)
            venv_cli = root / "bin" / "chemprop"
            if venv_cli.exists():
                return str(venv_cli)

        return None

    def _package_version(self) -> Optional[str]:
        try:
            return importlib.metadata.version("chemprop")
        except importlib.metadata.PackageNotFoundError:
            return None

    def is_available(self) -> bool:
        cli_path = self._find_cli_path()
        if cli_path:
            return True

        if self._package_version() is None:
            return False

        return importlib.util.find_spec("chemprop") is not None

    def describe_environment(self) -> Dict[str, Any]:
        version = self._package_version()

        return enrich_backend_environment(
            self.backend_name,
            {
                "backend_name": self.backend_name,
                "available": self.is_available(),
                "cli_path": self._find_cli_path(),
                "package_version": version,
            },
        )

    def validate_model_path(self, model_path: str) -> Path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise InvalidPredictionInputError(f"Model path does not exist: {model_path}")

        if path.is_file() and path.suffix not in self.MODEL_EXTENSIONS:
            raise InvalidPredictionInputError(
                f"Chemprop model artifact must end with one of {self.MODEL_EXTENSIONS}: {model_path}"
            )

        return path

    def _ensure_available(self) -> None:
        if not self.is_available():
            env = self.describe_environment()
            raise BackendNotAvailableError(
                "Chemprop backend is not available. "
                "Install the optional dependency and ensure the `chemprop` CLI is on PATH. "
                f"Environment snapshot: {env}"
            )

    def _sanitize_train_extra_args(self, extra_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize or drop legacy/internal Chemprop CLI arguments before training.

        We keep a blacklist here because the agent may legitimately propose
        extra supported Chemprop flags that are unknown to the backend today.
        The blacklist only removes arguments that are known to be invalid for
        the installed CLI or that belong to our orchestration layer rather than
        Chemprop itself.
        """
        sanitized = dict(extra_args or {})

        if "num_folds" in sanitized and "num_replicates" not in sanitized:
            logger.warning(
                "Received deprecated Chemprop train arg `num_folds`; mapping it to `num_replicates`."
            )
            sanitized["num_replicates"] = sanitized.pop("num_folds")
        else:
            sanitized.pop("num_folds", None)

        if "save_dir" in sanitized:
            logger.warning(
                "Dropping deprecated/duplicate Chemprop train arg `save_dir`; `output_dir` is already set."
            )
            sanitized.pop("save_dir", None)

        if sanitized.get("splits_file"):
            sanitized.pop("split_type", None)
            sanitized.pop("split", None)
            sanitized.pop("split_sizes", None)
            sanitized.pop("data_seed", None)

        unsupported_args = {
            "gpus",
            "gpu",
            "use_gpu",
            "seed",
            "extra_metrics",
            "profile",
            "scaffold_split",
            "split_type_mixed",
            "model_type",
            "hidden_size",
            "depth",
            "dropout",
            "init_lr",
            "max_lr",
            "final_lr",
            "warmup_epochs",
            "primary_split",
            "split_strategies",
            "random_seed",
            "checkpoint_dir",
            "metrics_dir",
            "applicability_domain",
            "save_preds",
            "save_checkpoints",
            "dataset_id",
            "validation_protocol",
            "validation_strategy",
            "seed_policy",
            "training_profile",
        }
        dropped_args = sorted(arg for arg in unsupported_args if arg in sanitized)
        for arg in dropped_args:
            sanitized.pop(arg, None)

        if dropped_args:
            logger.warning(
                "Dropping unsupported Chemprop train args for this CLI/runtime: %s",
                ", ".join(dropped_args),
            )
        return sanitized

    def _resolve_artifact_path(
        self,
        model_record: PredictionModelRecord,
        raw_path: Optional[str],
    ) -> Optional[Path]:
        if not raw_path:
            return None

        candidate = Path(raw_path).expanduser()
        candidates = [candidate]
        if not candidate.is_absolute():
            if model_record.metadata_path:
                candidates.append(Path(model_record.metadata_path).expanduser().parent / candidate)
            model_path = Path(model_record.model_path).expanduser()
            candidates.append(model_path.parent / candidate)
            candidates.append(model_path.parent.parent / candidate)

        for item in candidates:
            try:
                if item.exists():
                    return item.resolve()
            except Exception:
                continue
        return None

    def _extract_epoch_progress(self, line: str) -> tuple[Optional[int], Optional[int]]:
        match = EPOCH_PROGRESS_RE.search(line)
        if not match:
            return None, None

        current_epoch = int(match.group(1))
        total_epochs = int(match.group(2)) if match.group(2) else None
        return current_epoch, total_epochs

    def _parse_run_index(self, name: str, prefix: str) -> Optional[int]:
        if not name.startswith(prefix):
            return None
        suffix = name[len(prefix) :]
        if not suffix.isdigit():
            return None
        return int(suffix)

    def _read_last_metrics_row(self, metrics_path: Path) -> Optional[Dict[str, str]]:
        try:
            with metrics_path.open("r", newline="") as fh:
                reader = csv.DictReader(fh)
                last_row: Optional[Dict[str, str]] = None
                for row in reader:
                    if any((value or "").strip() for value in row.values()):
                        last_row = row
                return last_row
        except Exception as exc:
            logger.debug("Could not read metrics CSV %s: %s", metrics_path, exc)
            return None

    def _get_metrics_progress(
        self,
        *,
        output_dir: Path,
        total_epochs: Optional[int],
        total_replicates: Optional[int],
        total_models: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        metrics_candidates = list(output_dir.rglob("metrics.csv"))
        if not metrics_candidates:
            return None

        try:
            latest_metrics = max(metrics_candidates, key=lambda path: path.stat().st_mtime)
        except Exception:
            return None

        last_row = self._read_last_metrics_row(latest_metrics)
        if not last_row:
            return None

        try:
            rel_parts = latest_metrics.relative_to(output_dir).parts
        except Exception:
            rel_parts = latest_metrics.parts

        replicate_idx: Optional[int] = None
        model_idx: Optional[int] = None
        for part in rel_parts:
            if replicate_idx is None:
                replicate_idx = self._parse_run_index(part, "replicate_")
            if model_idx is None:
                model_idx = self._parse_run_index(part, "model_")

        epoch_raw = (last_row.get("epoch") or "").strip()
        epoch = int(float(epoch_raw)) if epoch_raw else None

        return {
            "metrics_path": latest_metrics,
            "replicate_index": replicate_idx,
            "model_index": model_idx,
            "replicate_display": (
                f"{(replicate_idx + 1)}/{total_replicates}"
                if replicate_idx is not None and total_replicates
                else None
            ),
            "model_display": (
                f"{(model_idx + 1)}/{total_models}"
                if model_idx is not None and total_models
                else None
            ),
            "epoch": epoch,
            "epoch_display": (
                f"{epoch}/{total_epochs}" if epoch is not None and total_epochs else None
            ),
            "step": (last_row.get("step") or "").strip() or None,
        }

    def _run_cli(
        self,
        args: list[str],
        *,
        progress_label: Optional[str] = None,
        heartbeat_seconds: float = 120.0,
        output_dir: Optional[Path] = None,
        total_epochs: Optional[int] = None,
        total_replicates: Optional[int] = None,
        total_models: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        self._ensure_available()
        cli_path = self._find_cli_path()
        if cli_path:
            args = [cli_path, *args[1:]]
        logger.info("Running Chemprop command: %s", " ".join(args))

        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stream_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        last_epoch: Optional[int] = None
        observed_total_epochs: Optional[int] = None
        last_progress_line: Optional[str] = None
        started_at = time.monotonic()
        next_heartbeat_at = started_at + heartbeat_seconds

        def _pump_stream(stream, source: str) -> None:
            try:
                if stream is None:
                    return
                for line in iter(stream.readline, ""):
                    stream_queue.put((source, line))
            finally:
                if stream is not None:
                    stream.close()

        stdout_thread = threading.Thread(
            target=_pump_stream, args=(process.stdout, "stdout"), daemon=True
        )
        stderr_thread = threading.Thread(
            target=_pump_stream, args=(process.stderr, "stderr"), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()

        try:
            while True:
                try:
                    source, line = stream_queue.get(timeout=1.0)
                    stripped = line.rstrip()
                    if source == "stdout":
                        stdout_lines.append(line)
                    else:
                        stderr_lines.append(line)

                    if not stripped:
                        continue

                    current_epoch, observed_total = self._extract_epoch_progress(stripped)
                    if current_epoch is not None:
                        last_epoch = current_epoch
                        if observed_total is not None:
                            observed_total_epochs = observed_total
                        last_progress_line = stripped
                except queue.Empty:
                    pass

                if process.poll() is not None:
                    while True:
                        try:
                            source, line = stream_queue.get_nowait()
                            if source == "stdout":
                                stdout_lines.append(line)
                            else:
                                stderr_lines.append(line)
                        except queue.Empty:
                            break
                    break

                now = time.monotonic()
                if now >= next_heartbeat_at:
                    elapsed_seconds = int(now - started_at)
                    minutes, seconds = divmod(elapsed_seconds, 60)
                    metrics_progress = (
                        self._get_metrics_progress(
                            output_dir=output_dir,
                            total_epochs=total_epochs,
                            total_replicates=total_replicates,
                            total_models=total_models,
                        )
                        if output_dir is not None
                        else None
                    )

                    if metrics_progress is not None:
                        parts = [f"split={progress_label or output_dir.name}"]
                        if metrics_progress.get("replicate_display"):
                            parts.append(f"replicate={metrics_progress['replicate_display']}")
                        if metrics_progress.get("model_display"):
                            parts.append(f"model={metrics_progress['model_display']}")
                        if metrics_progress.get("epoch_display"):
                            parts.append(f"epoch={metrics_progress['epoch_display']}")
                        elif metrics_progress.get("epoch") is not None:
                            parts.append(f"epoch={metrics_progress['epoch']}")
                        if metrics_progress.get("step"):
                            parts.append(f"step={metrics_progress['step']}")
                        logger.info(
                            "Training status: %s (elapsed=%dm%02ds)",
                            ", ".join(parts),
                            minutes,
                            seconds,
                        )
                    elif last_epoch is not None:
                        if observed_total_epochs:
                            percent = min(
                                100.0,
                                max(
                                    0.0, (float(last_epoch) / float(observed_total_epochs)) * 100.0
                                ),
                            )
                            logger.info(
                                "Chemprop status [%s]: still running after %dm%02ds, epoch %s/%s (%.1f%%).",
                                progress_label or "training",
                                minutes,
                                seconds,
                                last_epoch,
                                observed_total_epochs,
                                percent,
                            )
                        else:
                            logger.info(
                                "Chemprop status [%s]: still running after %dm%02ds, latest epoch seen: %s.",
                                progress_label or "training",
                                minutes,
                                seconds,
                                last_epoch,
                            )
                    elif output_dir is not None:
                        logger.info(
                            "Training status: split=%s, waiting for first metrics (elapsed=%dm%02ds)",
                            progress_label or output_dir.name,
                            minutes,
                            seconds,
                        )
                    elif last_progress_line:
                        logger.info(
                            "Chemprop status [%s]: still running after %dm%02ds, last progress line: %s",
                            progress_label or "training",
                            minutes,
                            seconds,
                            last_progress_line,
                        )
                    else:
                        logger.info(
                            "Chemprop status [%s]: still running after %dm%02ds.",
                            progress_label or "training",
                            minutes,
                            seconds,
                        )
                    next_heartbeat_at = now + heartbeat_seconds
        finally:
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)

        stdout = "".join(stdout_lines).strip()
        stderr = "".join(stderr_lines).strip()
        if process.returncode != 0:
            details = stderr or stdout or "Chemprop CLI exited with a non-zero status."
            raise PredictionExecutionError(
                "Chemprop execution failed. " f"Command: {' '.join(args)} | Details: {details}"
            )

        return subprocess.CompletedProcess(
            args=args,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def predict_from_csv(
        self,
        input_csv: str,
        model_record: PredictionModelRecord,
        preds_path: str,
        *,
        return_uncertainty: bool = False,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        input_path = Path(input_csv).expanduser()
        if not input_path.exists():
            raise InvalidPredictionInputError(f"Input CSV does not exist: {input_csv}")

        model_path = self.validate_model_path(model_record.model_path)
        output_path = Path(preds_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        args = [
            "chemprop",
            "predict",
            "--test-path",
            str(input_path),
            "--model-paths",
            str(model_path),
            "--preds-path",
            str(output_path),
        ]

        if model_record.task.smiles_columns:
            args.extend(["--smiles-columns", *model_record.task.smiles_columns])

        if model_record.task.reaction_columns:
            args.extend(["--reaction-columns", *model_record.task.reaction_columns])

        if return_uncertainty and model_record.task.uncertainty_method:
            args.extend(["--uncertainty-method", model_record.task.uncertainty_method])
            if model_record.task.calibration_method:
                args.extend(["--calibration-method", model_record.task.calibration_method])

        for key, value in (extra_args or {}).items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            elif isinstance(value, (list, tuple)):
                args.extend([flag, *[str(item) for item in value]])
            elif value is not None:
                args.extend([flag, str(value)])

        completed = self._run_cli(args)
        prediction_format = "chemprop_native"
        if is_classification_task(model_record.task.task_type):
            native_predictions = pd.read_csv(output_path)
            normalized_predictions = normalize_chemprop_classification_predictions(
                native_predictions,
                task=model_record.task,
                classification_targets=self._classification_targets_from_record(model_record),
            )
            normalized_predictions.to_csv(output_path, index=False)
            prediction_format = "qsaria_classification_canonical"
        return {
            "backend": self.backend_name,
            "command": args,
            "preds_path": str(output_path),
            "prediction_format": prediction_format,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }

    def fingerprint_from_csv(
        self,
        *,
        input_csv: str,
        model_path: str,
        output_csv: str,
        smiles_columns: Optional[list[str]] = None,
        ffn_block_index: int = DEFAULT_CHEMPROP_FINGERPRINT_FFN_BLOCK_INDEX,
    ) -> Dict[str, Any]:
        """Extract official Chemprop learned representations for AD scoring."""
        input_path = Path(input_csv).expanduser()
        if not input_path.exists():
            raise InvalidPredictionInputError(f"Input CSV does not exist: {input_csv}")

        resolved_model_path = self.validate_model_path(model_path)
        output_path = Path(output_csv).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        args = [
            "chemprop",
            "fingerprint",
            "--test-path",
            str(input_path),
            "--model-paths",
            str(resolved_model_path),
            "--ffn-block-index",
            str(int(ffn_block_index)),
            "--output",
            str(output_path),
            "--num-workers",
            "0",
            "--accelerator",
            "cpu",
        ]
        if smiles_columns:
            args.extend(["--smiles-columns", *[str(column) for column in smiles_columns]])

        completed = self._run_cli(args, progress_label="chemprop_fingerprint")
        cli_output_path = output_path.with_stem(f"{output_path.stem}_0")
        source_path = cli_output_path if cli_output_path.exists() else output_path
        if not source_path.exists():
            raise PredictionExecutionError(
                "Chemprop fingerprint completed but no fingerprint CSV was produced. "
                f"Expected {cli_output_path} or {output_path}."
            )

        frame = pd.read_csv(source_path)
        rename = {
            column: f"chemprop_{column}"
            for column in frame.columns
            if str(column).startswith("fp_")
        }
        frame = frame.rename(columns=rename)
        feature_columns = [str(column) for column in frame.columns]
        frame.to_csv(output_path, index=False)
        return {
            "fingerprints_path": str(output_path),
            "raw_fingerprints_path": str(source_path),
            "feature_columns": feature_columns,
            "row_count": int(len(frame)),
            "feature_count": int(len(feature_columns)),
            "ffn_block_index": int(ffn_block_index),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }

    def train_model(
        self,
        train_csv: str,
        output_dir: str,
        task: PredictionTaskSpec,
        *,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        input_path = Path(train_csv).expanduser()
        if not input_path.exists():
            raise InvalidPredictionInputError(f"Training CSV does not exist: {train_csv}")

        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now().astimezone()

        normalized_task_type = normalize_task_type(task.task_type)
        chemprop_task_type = (
            "multiclass"
            if normalized_task_type == "multiclass_classification"
            else normalized_task_type
        )

        args = [
            "chemprop",
            "train",
            "--data-path",
            str(input_path),
            "--task-type",
            chemprop_task_type,
            "--output-dir",
            str(output_path),
        ]

        if task.smiles_columns:
            args.extend(["--smiles-columns", *task.smiles_columns])

        if task.target_columns:
            args.extend(["--target-columns", *task.target_columns])

        if task.reaction_columns:
            args.extend(["--reaction-columns", *task.reaction_columns])

        sanitized_extra_args = self._sanitize_train_extra_args(extra_args)

        for key, value in sanitized_extra_args.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            elif isinstance(value, (list, tuple)):
                args.extend([flag, *[str(item) for item in value]])
            elif value is not None:
                args.extend([flag, str(value)])

        completed = self._run_cli(
            args,
            progress_label=output_path.name,
            output_dir=output_path,
            total_epochs=(
                int(sanitized_extra_args.get("epochs"))
                if sanitized_extra_args.get("epochs") is not None
                else None
            ),
            total_replicates=(
                int(sanitized_extra_args.get("num_replicates"))
                if sanitized_extra_args.get("num_replicates") is not None
                else None
            ),
            total_models=(
                int(sanitized_extra_args.get("ensemble_size"))
                if sanitized_extra_args.get("ensemble_size") is not None
                else None
            ),
        )
        completed_at = datetime.now().astimezone()
        return {
            "backend": self.backend_name,
            "command": args,
            "output_dir": str(output_path),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round((completed_at - started_at).total_seconds(), 3),
        }
