#!/usr/bin/env python
# coding: utf-8
"""Classical scikit-learn baselines for QSAR.

This backend supplies the reference points every QSAR result has to beat before
it can be called an improvement: a dummy predictor (the no-skill floor), random
forest on fingerprints (the field's default yardstick), SVM, kNN, PLS and ridge.
It mirrors ``lightgbm_backend`` in structure, artifact layout, pickle
persistence, leakage guards and error types.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from cs_copilot.storage import S3

from .backend import (
    BackendNotAvailableError,
    InvalidPredictionInputError,
    PredictionBackend,
    PredictionExecutionError,
    PredictionModelRecord,
    PredictionTaskSpec,
)
from .backend_capabilities import enrich_backend_environment
from .qsar_contracts import BackendRunBase
from .qsar_splitters import build_full_train_split_payload, build_qsar_split_payload
from .qsar_training_policy import project_now
from .training_orchestration import (
    build_classification_prediction_frame,
    classification_task_kind,
    compute_classification_metrics,
    compute_regression_metrics,
    decode_classification_labels,
    encode_classification_labels,
    is_classification_task,
    json_safe_label,
    resolve_class_labels,
    strip_unnamed_columns,
    write_training_summary,
)

SklearnEstimatorName = Literal["dummy", "random_forest", "svm", "knn", "pls", "ridge"]

SKLEARN_ESTIMATORS: Tuple[str, ...] = ("dummy", "random_forest", "svm", "knn", "pls", "ridge")

#: Estimators that are sensitive to feature scale and are therefore wrapped in a
#: ``StandardScaler`` pipeline.  Fingerprint/descriptor matrices mix 0/1 bits
#: with descriptors spanning several orders of magnitude (molecular weight,
#: TPSA, logP); an RBF kernel, a Euclidean neighbourhood, a PLS component and an
#: L2 penalty are all dominated by the largest-scale column unless the matrix is
#: standardized first.  The scaler lives inside the pipeline so it is fitted on
#: the training split only and travels with the pickled model.
SCALED_ESTIMATORS = frozenset({"svm", "knn", "pls", "ridge"})

#: Tree ensembles and the dummy baseline are invariant to any monotone per-feature
#: rescaling, so scaling them would only add a fitted transform to persist.
UNSCALED_ESTIMATORS = frozenset({"dummy", "random_forest"})

#: Estimators that accept ``class_weight="balanced"`` for imbalanced endpoints.
CLASS_WEIGHT_ESTIMATORS = frozenset({"random_forest", "svm", "ridge"})

#: PLS is a regression-only latent-variable method here; PLS-DA is a different
#: model and is not silently substituted.
REGRESSION_ONLY_ESTIMATORS = frozenset({"pls"})

MODEL_FILENAME = "best.pkl"
TRAINING_SUMMARY_FILENAME = "cs_copilot_training_summary.json"


class SklearnRunRequest(BackendRunBase):
    """Runtime contract for one scikit-learn baseline run.

    Mirrors ``LightGBMRunRequest``.  Hyperparameters are shared across
    estimators where the meaning is identical and ignored where the chosen
    estimator has no such knob.
    """

    backend: Literal["sklearn"] = "sklearn"
    estimator: SklearnEstimatorName = "random_forest"
    feature_columns: Optional[List[str]] = None
    split_sizes: Optional[List[float]] = None
    split_type: str = "random"
    split_payload: Optional[List[Dict[str, Any]]] = None
    validation_protocol: str = "standard_qsar"
    random_state: int = 42
    n_jobs: int = Field(default=1, ge=1)
    class_weight: Optional[Literal["balanced"]] = None
    # random forest
    n_estimators: int = Field(default=500, ge=1)
    max_depth: Optional[int] = Field(default=None, ge=1)
    min_samples_leaf: int = Field(default=1, ge=1)
    max_features: Union[float, Literal["sqrt", "log2"]] = "sqrt"
    # support vector machines
    kernel: Literal["rbf", "linear", "poly", "sigmoid"] = "rbf"
    C: float = Field(default=1.0, gt=0.0)
    gamma: Union[float, Literal["scale", "auto"]] = "scale"
    epsilon: float = Field(default=0.1, ge=0.0)
    # k nearest neighbours
    n_neighbors: int = Field(default=5, ge=1)
    weights: Literal["uniform", "distance"] = "distance"
    metric: str = "euclidean"
    # partial least squares
    n_components: int = Field(default=2, ge=1)
    # ridge regression / L2 logistic regression
    alpha: float = Field(default=1.0, gt=0.0)
    max_iter: int = Field(default=1000, ge=1)
    persist_artifacts: bool = True
    return_prediction_frames: bool = False
    final_refit: bool = False


def _coerce_request(request: Any) -> SklearnRunRequest:
    """Validate an incoming run request into the strict sklearn contract."""
    if isinstance(request, SklearnRunRequest):
        return request
    if isinstance(request, Mapping):
        return SklearnRunRequest.model_validate(dict(request))
    raise InvalidPredictionInputError(
        "The sklearn backend requires a SklearnRunRequest (or a mapping that validates as one); "
        f"received {type(request).__name__}."
    )


def _as_1d_float_array(values: Any) -> np.ndarray:
    """Flatten an estimator output to 1-D floats (PLS predicts a column vector)."""
    return np.asarray(values, dtype=float).reshape(-1)


class SklearnBackend(PredictionBackend):
    """Prediction backend exposing classical scikit-learn QSAR baselines."""

    backend_name = "sklearn"
    MODEL_EXTENSIONS = (".pkl",)

    def _package_version(self) -> Optional[str]:
        try:
            return importlib.metadata.version("scikit-learn")
        except importlib.metadata.PackageNotFoundError:
            return None

    def is_available(self) -> bool:
        return importlib.util.find_spec("sklearn") is not None

    def describe_environment(self) -> Dict[str, Any]:
        """Describe availability, supported estimators and the known limitations."""
        environment: Dict[str, Any] = {
            "backend_name": self.backend_name,
            "available": self.is_available(),
            "package_version": self._package_version(),
            "cpu_available": True,
            "gpu_detected": False,
            "supports_gpu_when_available": False,
            "supported_estimators": list(SKLEARN_ESTIMATORS),
            "scaled_estimators": sorted(SCALED_ESTIMATORS),
            "unscaled_estimators": sorted(UNSCALED_ESTIMATORS),
            "class_weight_estimators": sorted(CLASS_WEIGHT_ESTIMATORS),
            "regression_only_estimators": sorted(REGRESSION_ONLY_ESTIMATORS),
            "supports_multi_target": False,
            "supports_uncertainty": False,
            "limitations": [
                "Single target column per run; multi-target training is not supported.",
                "Requires a prepared numeric feature matrix; SMILES are not featurized here.",
                "Feature columns must be numeric and free of NaN/inf values.",
                "PLS supports regression only; classification is rejected.",
                "No predictive uncertainty export.",
            ],
        }
        try:
            return enrich_backend_environment(self.backend_name, environment)
        except KeyError:
            # The capability registry is owned elsewhere; stay usable until the
            # `sklearn` entry is registered there.
            environment["capabilities"] = None
            environment["capabilities_registered"] = False
            return environment

    def validate_model_path(self, model_path: str) -> Path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise InvalidPredictionInputError(f"Model path does not exist: {model_path}")
        if path.suffix not in self.MODEL_EXTENSIONS:
            raise InvalidPredictionInputError(
                f"sklearn model artifact must end with one of {self.MODEL_EXTENSIONS}: {model_path}"
            )
        return path.resolve()

    def _ensure_available(self) -> None:
        if not self.is_available():
            raise BackendNotAvailableError(
                "scikit-learn backend is not available. Install `scikit-learn` first. "
                f"Environment snapshot: {self.describe_environment()}"
            )

    # ------------------------------------------------------------------
    # feature selection and leakage guards
    # ------------------------------------------------------------------
    def _select_feature_columns(
        self,
        df: pd.DataFrame,
        task: PredictionTaskSpec,
        requested_columns: Optional[Sequence[str]],
    ) -> List[str]:
        """Resolve the feature matrix columns, refusing anything that leaks the target."""
        excluded = {str(column) for column in task.target_columns} | {
            str(column) for column in task.smiles_columns
        }
        if requested_columns:
            feature_columns = [str(column) for column in requested_columns]
            leaked = sorted({column for column in feature_columns if column in excluded})
            if leaked:
                raise InvalidPredictionInputError(
                    "Target and SMILES columns can never be used as features. "
                    f"Leaking columns: {leaked}"
                )
            missing = [column for column in feature_columns if column not in df.columns]
            if missing:
                raise InvalidPredictionInputError(
                    f"Requested feature columns are missing: {missing}"
                )
        else:
            feature_columns = [
                str(column)
                for column in df.columns
                if str(column) not in excluded and pd.api.types.is_numeric_dtype(df[column])
            ]
        if not feature_columns:
            raise InvalidPredictionInputError(
                "No usable sklearn feature columns were found. Provide numeric feature columns."
            )
        duplicates = sorted({c for c in feature_columns if feature_columns.count(c) > 1})
        if duplicates:
            raise InvalidPredictionInputError(f"Duplicate feature columns requested: {duplicates}")
        non_numeric = [
            column for column in feature_columns if not pd.api.types.is_numeric_dtype(df[column])
        ]
        if non_numeric:
            raise InvalidPredictionInputError(
                "sklearn feature columns must be numeric; encode them before training. "
                f"Invalid columns: {non_numeric}"
            )
        self._reject_target_copies(df, feature_columns=feature_columns, task=task)
        return feature_columns

    def _reject_target_copies(
        self,
        df: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        task: PredictionTaskSpec,
    ) -> None:
        """Refuse a feature column that is a value-for-value copy of the target."""
        for target_column in task.target_columns:
            if target_column not in df.columns:
                continue
            target_values = df[target_column]
            copies = [
                column
                for column in feature_columns
                if df[column].reset_index(drop=True).equals(target_values.reset_index(drop=True))
            ]
            if copies:
                raise InvalidPredictionInputError(
                    f"Feature columns {sorted(copies)} are identical to target "
                    f"'{target_column}'; this leaks the label. Drop them from feature_columns."
                )

    def _validate_feature_values(self, features: pd.DataFrame) -> None:
        """Reject non-numeric or NaN/inf features instead of silently imputing them."""
        try:
            values = features.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise InvalidPredictionInputError(
                f"sklearn features must be numeric and convertible to float: {exc}"
            ) from exc
        if np.isfinite(values).all():
            return
        bad = [
            str(column)
            for index, column in enumerate(features.columns)
            if not np.isfinite(values[:, index]).all()
        ]
        raise InvalidPredictionInputError(
            "sklearn features must be finite; impute or drop missing values before training. "
            f"Columns with NaN/inf: {sorted(bad)}"
        )

    # ------------------------------------------------------------------
    # estimator construction
    # ------------------------------------------------------------------
    def _resolve_class_weight(
        self,
        request: SklearnRunRequest,
        *,
        task_is_classification: bool,
    ) -> Optional[str]:
        if request.class_weight is None:
            return None
        if not task_is_classification:
            raise InvalidPredictionInputError(
                "class_weight is a classification-only option; it has no meaning for regression."
            )
        if request.estimator not in CLASS_WEIGHT_ESTIMATORS:
            raise InvalidPredictionInputError(
                f"Estimator '{request.estimator}' does not support class_weight. "
                f"Supported estimators: {sorted(CLASS_WEIGHT_ESTIMATORS)}."
            )
        return request.class_weight

    def _build_estimator(
        self,
        request: SklearnRunRequest,
        *,
        task_is_classification: bool,
        n_train_rows: int,
        n_features: int,
    ) -> Any:
        """Build the configured estimator, scaled inside a pipeline when needed."""
        estimator_name = request.estimator
        if estimator_name in REGRESSION_ONLY_ESTIMATORS and task_is_classification:
            raise InvalidPredictionInputError(
                f"Estimator '{estimator_name}' supports regression only. "
                "Use random_forest, svm, knn, ridge or dummy for classification."
            )
        class_weight = self._resolve_class_weight(
            request, task_is_classification=task_is_classification
        )

        if estimator_name == "dummy":
            estimator: Any = (
                DummyClassifier(strategy="prior")
                if task_is_classification
                else DummyRegressor(strategy="mean")
            )
        elif estimator_name == "random_forest":
            shared = {
                "n_estimators": request.n_estimators,
                "max_depth": request.max_depth,
                "min_samples_leaf": request.min_samples_leaf,
                "max_features": request.max_features,
                "n_jobs": request.n_jobs,
                "random_state": request.random_state,
            }
            estimator = (
                RandomForestClassifier(class_weight=class_weight, **shared)
                if task_is_classification
                else RandomForestRegressor(**shared)
            )
        elif estimator_name == "svm":
            estimator = (
                SVC(
                    kernel=request.kernel,
                    C=request.C,
                    gamma=request.gamma,
                    class_weight=class_weight,
                    probability=True,
                    random_state=request.random_state,
                )
                if task_is_classification
                else SVR(
                    kernel=request.kernel,
                    C=request.C,
                    gamma=request.gamma,
                    epsilon=request.epsilon,
                )
            )
        elif estimator_name == "knn":
            if request.n_neighbors > n_train_rows:
                raise InvalidPredictionInputError(
                    f"n_neighbors={request.n_neighbors} exceeds the {n_train_rows} training rows."
                )
            shared = {
                "n_neighbors": request.n_neighbors,
                "weights": request.weights,
                "metric": request.metric,
                "n_jobs": request.n_jobs,
            }
            estimator = (
                KNeighborsClassifier(**shared)
                if task_is_classification
                else KNeighborsRegressor(**shared)
            )
        elif estimator_name == "pls":
            max_components = max(1, min(n_train_rows - 1, n_features))
            if request.n_components > max_components:
                raise InvalidPredictionInputError(
                    f"n_components={request.n_components} exceeds the maximum of {max_components} "
                    f"for {n_train_rows} training rows and {n_features} features."
                )
            # Scaling is done once by the pipeline's StandardScaler; PLS's own
            # internal scaling would only repeat it with a different ddof.
            estimator = PLSRegression(n_components=request.n_components, scale=False)
        elif estimator_name == "ridge":
            estimator = (
                LogisticRegression(
                    C=request.C,
                    class_weight=class_weight,
                    max_iter=request.max_iter,
                    random_state=request.random_state,
                )
                if task_is_classification
                else Ridge(alpha=request.alpha, random_state=request.random_state)
            )
        else:  # pragma: no cover - the Literal annotation already constrains this
            raise InvalidPredictionInputError(
                f"Unknown sklearn estimator '{estimator_name}'. "
                f"Supported estimators: {list(SKLEARN_ESTIMATORS)}."
            )

        if estimator_name in SCALED_ESTIMATORS:
            return Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])
        return estimator

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def _predict_proba(self, model: Any, features: pd.DataFrame) -> Optional[np.ndarray]:
        if not hasattr(model, "predict_proba"):
            return None
        return np.asarray(model.predict_proba(features), dtype=float)

    def predict_from_csv(
        self,
        input_csv: str,
        model_record: PredictionModelRecord,
        preds_path: str,
        *,
        return_uncertainty: bool = False,
    ) -> Dict[str, Any]:
        """Run batch prediction from a CSV of prepared numeric features."""
        if return_uncertainty:
            raise InvalidPredictionInputError(
                "The sklearn backend does not support predictive uncertainty export."
            )
        model_path = self.validate_model_path(model_record.model_path)
        try:
            with model_path.open("rb") as fh:
                payload = pickle.load(fh)
        except Exception as exc:
            raise PredictionExecutionError(
                f"Could not load sklearn model artifact {model_path}: {exc}"
            ) from exc

        with S3.open(input_csv, "r") as fh:
            df = strip_unnamed_columns(pd.read_csv(fh))

        feature_columns = list((payload or {}).get("feature_columns") or [])
        missing_features = [column for column in feature_columns if column not in df.columns]
        if missing_features:
            raise InvalidPredictionInputError(
                f"Prediction input is missing feature columns: {missing_features}"
            )
        features = df[feature_columns].copy()
        self._validate_feature_values(features)

        target_columns = list(model_record.task.target_columns)
        task_type = str(
            (payload or {}).get("task_type") or model_record.task.task_type or "regression"
        )
        model = (payload or {}).get("model")
        if model is None:
            raise PredictionExecutionError(
                f"sklearn model artifact {model_path} does not contain a fitted model."
            )
        try:
            if is_classification_task(task_type):
                class_labels = list((payload or {}).get("class_labels") or [])
                if not class_labels:
                    raise InvalidPredictionInputError(
                        "sklearn classification artifact is missing class_labels metadata."
                    )
                probabilities = self._predict_proba(model, features)
                output = build_classification_prediction_frame(
                    predicted_codes=model.predict(features),
                    class_labels=class_labels,
                    target_column=target_columns[0] if len(target_columns) == 1 else "prediction",
                    probabilities=(
                        pd.DataFrame(probabilities) if probabilities is not None else None
                    ),
                )
                for column in ("smiles", "Drug_ID"):
                    if column in df.columns:
                        output.insert(0, column, df[column].reset_index(drop=True))
            else:
                predictions = _as_1d_float_array(model.predict(features))
                output = pd.DataFrame({"prediction": pd.Series(predictions, dtype=float)})
                if len(target_columns) == 1:
                    output[target_columns[0]] = output["prediction"]
        except InvalidPredictionInputError:
            raise
        except Exception as exc:
            raise PredictionExecutionError(f"sklearn prediction failed: {exc}") from exc

        with S3.open(preds_path, "w") as fh:
            output.to_csv(fh, index=False)
        return {
            "preds_path": preds_path,
            "rows": int(len(output)),
            "feature_columns": feature_columns,
            "estimator": (payload or {}).get("estimator"),
        }

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def _split_indices(
        self,
        request: SklearnRunRequest,
        *,
        working: pd.DataFrame,
        task: PredictionTaskSpec,
        feature_columns: List[str],
    ) -> Dict[str, Any]:
        split_payload = request.split_payload
        if not split_payload:
            if request.final_refit:
                split_payload = build_full_train_split_payload(df=working)
            else:
                split_payload = build_qsar_split_payload(
                    df=working,
                    split_type=request.split_type,
                    split_sizes=list(request.split_sizes or [0.8, 0.1, 0.1]),
                    random_state=request.random_state,
                    smiles_column=task.smiles_columns[0] if task.smiles_columns else None,
                    feature_columns=feature_columns,
                )
        if not split_payload or not (split_payload[0].get("train") or []):
            raise InvalidPredictionInputError(
                "sklearn training requires a split payload with non-empty train indices."
            )
        return split_payload[0]

    def train_model(self, request: Any) -> Dict[str, Any]:
        """Train one scikit-learn baseline and persist its artifacts."""
        run = _coerce_request(request)
        self._ensure_available()
        task = PredictionTaskSpec(**run.task_payload())
        task_is_classification = is_classification_task(task.task_type)
        if task.task_type != "regression" and not task_is_classification:
            raise InvalidPredictionInputError(
                "The sklearn backend supports regression and classification tasks."
            )
        if len(task.target_columns) != 1:
            raise InvalidPredictionInputError(
                "The sklearn backend requires exactly one target column."
            )
        target_column = task.target_columns[0]
        started_at = project_now()

        with S3.open(run.train_csv, "r") as fh:
            dataset = strip_unnamed_columns(pd.read_csv(fh))
        if target_column not in dataset.columns:
            raise InvalidPredictionInputError(f"Missing target column: {target_column}")
        if dataset[target_column].isna().any():
            raise InvalidPredictionInputError(
                f"sklearn training target '{target_column}' contains missing values."
            )

        feature_columns = self._select_feature_columns(dataset, task, run.feature_columns)
        id_columns = [column for column in ("smiles", "Drug_ID") if column in dataset.columns]
        working = dataset[feature_columns + [target_column] + id_columns].copy()
        if len(working) < 10:
            raise InvalidPredictionInputError(
                "The sklearn backend requires at least 10 rows after target cleanup."
            )
        self._validate_feature_values(working[feature_columns])

        class_labels: List[Any] = []
        class_mapping: Dict[str, int] = {}
        if task_is_classification:
            class_labels = resolve_class_labels(working[target_column])
            if len(class_labels) < 2:
                raise InvalidPredictionInputError(
                    "sklearn classification requires at least two target classes."
                )
            encoded_target, class_mapping = encode_classification_labels(
                working[target_column], class_labels
            )
            if encoded_target.isna().any():
                raise InvalidPredictionInputError(
                    "sklearn classification target contains labels outside class_labels."
                )
            y_all = encoded_target.astype(int)
        else:
            y_all = pd.to_numeric(working[target_column], errors="coerce").astype(float)
            if y_all.isna().any():
                raise InvalidPredictionInputError(
                    f"sklearn regression target '{target_column}' is not numeric."
                )

        split_indices = self._split_indices(
            run, working=working, task=task, feature_columns=feature_columns
        )
        train_idx = [int(index) for index in (split_indices.get("train") or [])]
        val_idx = [int(index) for index in (split_indices.get("val") or [])]
        test_idx = [int(index) for index in (split_indices.get("test") or [])]
        if run.persist_artifacts and not run.final_refit and not (val_idx or test_idx):
            raise InvalidPredictionInputError(
                "sklearn persistent training requires non-empty validation or test indices."
            )
        effective_split: Dict[str, Any] = {"train": train_idx}
        if val_idx:
            effective_split["val"] = val_idx
        if test_idx:
            effective_split["test"] = test_idx
        if split_indices.get("metadata"):
            effective_split["metadata"] = dict(split_indices["metadata"])

        X_train = working.iloc[train_idx][feature_columns].copy()
        y_train = y_all.iloc[train_idx].copy()
        X_val = working.iloc[val_idx][feature_columns].copy() if val_idx else None
        y_val = y_all.iloc[val_idx].copy() if val_idx else None
        X_test = working.iloc[test_idx][feature_columns].copy() if test_idx else None
        y_test = y_all.iloc[test_idx].copy() if test_idx else None
        if task_is_classification:
            missing_classes = sorted(
                set(range(len(class_labels))) - {int(value) for value in y_train.tolist()}
            )
            if missing_classes:
                missing = [json_safe_label(class_labels[index]) for index in missing_classes]
                raise InvalidPredictionInputError(
                    f"sklearn classification training split is missing target classes {missing}."
                )

        model = self._build_estimator(
            run,
            task_is_classification=task_is_classification,
            n_train_rows=len(X_train),
            n_features=len(feature_columns),
        )
        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            raise PredictionExecutionError(f"sklearn training failed: {exc}") from exc

        def _predict_split(
            features: Optional[pd.DataFrame],
        ) -> Tuple[Optional[pd.Series], Optional[np.ndarray]]:
            if features is None:
                return None, None
            try:
                if task_is_classification:
                    codes = pd.Series(model.predict(features), index=features.index)
                    return codes, self._predict_proba(model, features)
                return (
                    pd.Series(
                        _as_1d_float_array(model.predict(features)),
                        index=features.index,
                        dtype=float,
                    ),
                    None,
                )
            except Exception as exc:
                raise PredictionExecutionError(f"sklearn prediction failed: {exc}") from exc

        y_val_pred, y_val_proba = _predict_split(X_val)
        y_test_pred, y_test_proba = _predict_split(X_test)

        def _split_metrics(
            y_true: Optional[pd.Series],
            y_hat: Optional[pd.Series],
            probabilities: Optional[np.ndarray],
        ) -> Dict[str, Any]:
            if y_true is None or y_hat is None:
                return {}
            if task_is_classification:
                positive_scores = (
                    pd.Series(probabilities[:, 1])
                    if probabilities is not None
                    and len(class_labels) == 2
                    and probabilities.shape[1] >= 2
                    else None
                )
                return compute_classification_metrics(
                    decode_classification_labels(y_true.reset_index(drop=True), class_labels),
                    decode_classification_labels(y_hat.reset_index(drop=True), class_labels),
                    class_labels=class_labels,
                    positive_scores=positive_scores,
                    target_column=target_column,
                )
            return compute_regression_metrics(y_true, y_hat, target_column=target_column)

        metrics: Dict[str, Any] = {}
        validation_metrics = _split_metrics(y_val, y_val_pred, y_val_proba)
        if validation_metrics:
            metrics["validation"] = validation_metrics
        test_metrics = _split_metrics(y_test, y_test_pred, y_test_proba)
        if test_metrics:
            metrics["test"] = test_metrics

        def _prediction_frame(
            *,
            split_pred: pd.Series,
            split_proba: Optional[np.ndarray],
            split_y: pd.Series,
            split_source: pd.DataFrame,
        ) -> pd.DataFrame:
            if task_is_classification:
                frame = build_classification_prediction_frame(
                    predicted_codes=split_pred.reset_index(drop=True),
                    class_labels=class_labels,
                    target_column=target_column,
                    probabilities=(pd.DataFrame(split_proba) if split_proba is not None else None),
                )
                for column in id_columns:
                    frame.insert(0, column, split_source[column].reset_index(drop=True))
                frame[f"{target_column}_true"] = decode_classification_labels(
                    split_y.reset_index(drop=True), class_labels
                )
                frame["y_true"] = frame[f"{target_column}_true"]
                frame["y_pred"] = frame["prediction"]
                return frame
            return pd.DataFrame(
                {
                    target_column: split_pred.reset_index(drop=True),
                    "prediction": split_pred.reset_index(drop=True),
                    "y_true": split_y.reset_index(drop=True),
                    "y_pred": split_pred.reset_index(drop=True),
                }
            )

        output_path = Path(run.output_dir).expanduser().resolve()
        model_dir = output_path / "model_0"
        model_path = model_dir / MODEL_FILENAME
        validation_predictions_path: Optional[Path] = model_dir / "validation_predictions.csv"
        test_predictions_path: Optional[Path] = model_dir / "test_predictions.csv"
        splits_path = output_path / "splits.json"
        config_path = output_path / "config.toml"
        summary_path = output_path / TRAINING_SUMMARY_FILENAME
        if run.persist_artifacts:
            model_dir.mkdir(parents=True, exist_ok=True)

        validation_prediction_frame = None
        test_prediction_frame = None
        if y_val_pred is not None and y_val is not None:
            validation_prediction_frame = _prediction_frame(
                split_pred=y_val_pred,
                split_proba=y_val_proba,
                split_y=y_val,
                split_source=working.iloc[val_idx],
            )
            if run.persist_artifacts:
                with S3.open(str(validation_predictions_path), "w") as fh:
                    validation_prediction_frame.to_csv(fh, index=False)
        else:
            validation_predictions_path = None
        if y_test_pred is not None and y_test is not None:
            test_prediction_frame = _prediction_frame(
                split_pred=y_test_pred,
                split_proba=y_test_proba,
                split_y=y_test,
                split_source=working.iloc[test_idx],
            )
            if run.persist_artifacts:
                with S3.open(str(test_predictions_path), "w") as fh:
                    test_prediction_frame.to_csv(fh, index=False)
        else:
            test_predictions_path = None

        model_payload: Dict[str, Any] = {
            "backend_name": self.backend_name,
            "estimator": run.estimator,
            "scaled": run.estimator in SCALED_ESTIMATORS,
            "task_type": task.task_type,
            "target_columns": [target_column],
            "feature_columns": feature_columns,
            "model": model,
            "train_csv": run.train_csv,
            "trained_at": started_at.isoformat(),
        }
        if task_is_classification:
            model_payload.update(
                {
                    "task_kind": classification_task_kind(task.task_type, len(class_labels)),
                    "class_labels": [json_safe_label(label) for label in class_labels],
                    "class_count": len(class_labels),
                    "label_mapping": class_mapping,
                    "positive_class_label": (
                        json_safe_label(class_labels[1]) if len(class_labels) == 2 else None
                    ),
                }
            )

        completed_at = project_now()
        result: Dict[str, Any] = {
            "backend_name": self.backend_name,
            "estimator": run.estimator,
            "feature_scaling": "standard_scaler" if run.estimator in SCALED_ESTIMATORS else "none",
            "model_path": str(model_path) if run.persist_artifacts else None,
            "best_model_path": str(model_path) if run.persist_artifacts else None,
            "validation_predictions_path": (
                str(validation_predictions_path) if validation_predictions_path else None
            ),
            "test_predictions_path": (
                str(test_predictions_path) if test_predictions_path else None
            ),
            "splits_path": str(splits_path) if run.persist_artifacts else None,
            "config_path": str(config_path) if run.persist_artifacts else None,
            "summary_path": str(summary_path) if run.persist_artifacts else None,
            "task_type": task.task_type,
            "task_kind": (
                classification_task_kind(task.task_type, len(class_labels))
                if task_is_classification
                else "regression"
            ),
            "metrics": metrics,
            "metrics_status": "not_evaluated" if not test_idx else "evaluated",
            "evaluation_required": not test_idx,
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "target_column": target_column,
            "class_labels": [json_safe_label(label) for label in class_labels],
            "class_count": len(class_labels) if task_is_classification else None,
            "label_mapping": class_mapping,
            "positive_class_label": (
                json_safe_label(class_labels[1])
                if task_is_classification and len(class_labels) == 2
                else None
            ),
            "split_payload": [effective_split],
            "effective_split_payload": [effective_split],
            "split_metadata": effective_split.get("metadata") or {},
            "has_validation_split": bool(val_idx),
            "effective_train_count": int(len(train_idx)),
            "validation_count": int(len(val_idx)),
            "test_count": int(len(test_idx)),
            "final_refit": run.final_refit,
            "split_type": run.split_type,
            "validation_protocol": run.validation_protocol,
            "random_state": run.random_state,
            "effective_train_args": run.parameter_payload(),
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round((completed_at - started_at).total_seconds(), 3),
        }

        if run.persist_artifacts:
            with model_path.open("wb") as fh:
                pickle.dump(model_payload, fh)
            splits_path.write_text(json.dumps([effective_split], indent=2) + "\n")
            config_path.write_text(
                "\n".join(
                    [
                        'backend_name = "sklearn"',
                        f'estimator = "{run.estimator}"',
                        f'task_type = "{task.task_type}"',
                        f'target_column = "{target_column}"',
                        f'split_type = "{run.split_type}"',
                        f'validation_protocol = "{run.validation_protocol}"',
                    ]
                )
                + "\n"
            )
            write_training_summary(summary_path, result)

        if run.return_prediction_frames:
            result["validation_prediction_frame"] = validation_prediction_frame
            result["test_prediction_frame"] = test_prediction_frame
        return result
