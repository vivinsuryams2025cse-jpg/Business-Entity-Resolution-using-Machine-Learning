"""Training, threshold selection, evaluation, and serialization for match models."""

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import (
    CANDIDATE_PAIR_COLUMN,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    SOURCE1_PAIR_COLUMN,
)


RANDOM_STATE = 42
DEFAULT_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


@dataclass
class ModelBundle:
    """Selected classifier plus its validation-selected operating settings."""

    model_name: str
    estimator: Any
    threshold: float
    feature_columns: list[str]
    feature_config: dict[str, Any]
    feature_engineer: Any
    random_state: int
    validation_summary: dict[str, Any]
    validation_estimator: Any = None


@dataclass
class ModelComparison:
    """Threshold sweeps, best validation metrics, and final model bundle."""

    threshold_results: dict[str, list[dict[str, Any]]]
    best_model_results: dict[str, dict[str, Any]]
    selected_model_name: str
    selected_threshold: float
    bundle: ModelBundle


def f0_5_score(precision: float, recall: float) -> float:
    """Calculate the challenge F0.5 score using its exact formula."""
    denominator = 0.25 * precision + recall
    if denominator == 0:
        return 0.0
    return (1.25 * precision * recall) / denominator


def _metrics_at_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = probabilities >= threshold
    precision = float(precision_score(labels, predictions, zero_division=0))
    recall = float(recall_score(labels, predictions, zero_division=0))
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f0_5": f0_5_score(precision, recall),
        "confusion_matrix": matrix.tolist(),
        "true_negative": int(matrix[0, 0]),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "true_positive": int(matrix[1, 1]),
        "records": int(len(labels)),
        "positive_records": int(labels.sum()),
    }


def tune_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score fixed thresholds and select the highest validation F0.5 only."""
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must have the same number of rows.")
    if not thresholds or any(not 0 < threshold < 1 for threshold in thresholds):
        raise ValueError("Thresholds must be non-empty values strictly between 0 and 1.")
    results = [
        _metrics_at_threshold(labels, probabilities, threshold)
        for threshold in thresholds
    ]
    best_result = max(
        results,
        key=lambda result: (
            result["f0_5"],
            result["precision"],
            result["threshold"],
        ),
    )
    return results, best_result


def split_by_source1_group(
    labeled_pairs: pd.DataFrame,
    *,
    validation_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    group_column: str = SOURCE1_PAIR_COLUMN,
    label_column: str = LABEL_COLUMN,
    attempts: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split candidate rows by Source 1 ID and choose a stratified group split.

    All candidates connected through a Source 1 or target entity stay in the
    same partition. This also prevents a shared Source 2/3 record from crossing
    folds. Among deterministic group-shuffle attempts, choose a split whose
    validation positive rate is closest to the overall candidate positive rate.
    """
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if group_column not in labeled_pairs or label_column not in labeled_pairs:
        raise ValueError(f"Data must contain {group_column!r} and {label_column!r}.")
    labels = labeled_pairs[label_column].to_numpy(dtype=np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Training candidates must contain both positive and negative labels.")
    if CANDIDATE_PAIR_COLUMN in labeled_pairs:
        parents: dict[tuple[str, str], tuple[str, str]] = {}
        ranks: dict[tuple[str, str], int] = {}

        def find(node: tuple[str, str]) -> tuple[str, str]:
            parents.setdefault(node, node)
            ranks.setdefault(node, 0)
            if parents[node] != node:
                parents[node] = find(parents[node])
            return parents[node]

        def union(left: tuple[str, str], right: tuple[str, str]) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                return
            if ranks[left_root] < ranks[right_root]:
                left_root, right_root = right_root, left_root
            parents[right_root] = left_root
            if ranks[left_root] == ranks[right_root]:
                ranks[left_root] += 1

        pair_nodes = []
        for source1_id, candidate_id in zip(
            labeled_pairs[group_column].astype(str),
            labeled_pairs[CANDIDATE_PAIR_COLUMN].astype(str),
        ):
            source1_node = ("source1", source1_id)
            candidate_node = ("candidate", candidate_id)
            union(source1_node, candidate_node)
            pair_nodes.append(source1_node)
        groups = np.empty(len(pair_nodes), dtype=object)
        groups[:] = [find(node) for node in pair_nodes]
    else:
        groups = labeled_pairs[group_column].astype(str).to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two distinct Source 1 entities are required for validation.")

    splitter = GroupShuffleSplit(
        n_splits=attempts,
        test_size=validation_size,
        random_state=random_state,
    )
    overall_positive_rate = float(labels.mean())
    best_split = None
    best_split_error = float("inf")
    for train_indices, validation_indices in splitter.split(
        np.zeros(len(labels)), labels, groups
    ):
        train_labels = labels[train_indices]
        validation_labels = labels[validation_indices]
        if len(np.unique(train_labels)) < 2 or len(np.unique(validation_labels)) < 2:
            continue
        validation_positive_rate = float(validation_labels.mean())
        validation_fraction = len(validation_indices) / len(labels)
        split_error = abs(validation_positive_rate - overall_positive_rate) + abs(
            validation_fraction - validation_size
        )
        if split_error < best_split_error:
            best_split = (train_indices, validation_indices)
            best_split_error = split_error

    if best_split is None:
        raise ValueError(
            "Could not create a group-safe train/validation split containing both "
            "classes in both partitions. More labeled Source 1 entities may be needed."
        )

    train_indices, validation_indices = best_split
    train_data = labeled_pairs.iloc[train_indices].reset_index(drop=True)
    validation_data = labeled_pairs.iloc[validation_indices].reset_index(drop=True)
    return train_data, validation_data


def _feature_matrix(feature_dataset: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    missing_columns = set(feature_columns) - set(feature_dataset.columns)
    if missing_columns:
        raise ValueError(f"Feature dataset is missing columns: {sorted(missing_columns)}")
    matrix = feature_dataset[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")
    return matrix


def build_candidate_models(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Create the two fixed, reproducible model candidates."""
    logistic_regression = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        ),
    )
    random_forest = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    return {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
    }


def compare_models(
    training_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    *,
    feature_engineer: Any,
    feature_config: dict[str, Any],
    feature_columns: list[str] | None = None,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    random_state: int = RANDOM_STATE,
) -> ModelComparison:
    """Train two classifiers and choose model/threshold by validation F0.5."""
    feature_columns = feature_columns or list(FEATURE_COLUMNS)
    train_labels = training_features[LABEL_COLUMN].to_numpy(dtype=np.int8)
    validation_labels = validation_features[LABEL_COLUMN].to_numpy(dtype=np.int8)
    if set(np.unique(train_labels)) != {0, 1}:
        raise ValueError("Training partition must contain both classes.")
    if set(np.unique(validation_labels)) != {0, 1}:
        raise ValueError("Validation partition must contain both classes.")
    training_matrix = _feature_matrix(training_features, feature_columns)
    validation_matrix = _feature_matrix(validation_features, feature_columns)

    threshold_results = {}
    best_model_results = {}
    fitted_models = {}
    for model_name, estimator in build_candidate_models(random_state).items():
        estimator.fit(training_matrix, train_labels)
        probabilities = estimator.predict_proba(validation_matrix)[:, 1]
        model_threshold_results, best_result = tune_threshold(
            validation_labels, probabilities, thresholds
        )
        threshold_results[model_name] = model_threshold_results
        best_model_results[model_name] = best_result
        fitted_models[model_name] = estimator

    selected_model_name = max(
        best_model_results,
        key=lambda name: (
            best_model_results[name]["f0_5"],
            best_model_results[name]["precision"],
            name == "logistic_regression",
        ),
    )
    selected_result = best_model_results[selected_model_name]
    bundle = ModelBundle(
        model_name=selected_model_name,
        estimator=fitted_models[selected_model_name],
        threshold=selected_result["threshold"],
        feature_columns=feature_columns,
        feature_config=feature_config,
        feature_engineer=feature_engineer,
        random_state=random_state,
        validation_summary=selected_result,
        validation_estimator=fitted_models[selected_model_name],
    )
    return ModelComparison(
        threshold_results=threshold_results,
        best_model_results=best_model_results,
        selected_model_name=selected_model_name,
        selected_threshold=selected_result["threshold"],
        bundle=bundle,
    )


def evaluate_model(
    bundle: ModelBundle,
    labeled_features: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate a saved model at its already-selected threshold; never retune."""
    if LABEL_COLUMN not in labeled_features:
        raise ValueError(f"Evaluation data must contain the {LABEL_COLUMN!r} column.")
    labels = labeled_features[LABEL_COLUMN].to_numpy(dtype=np.int8)
    matrix = _feature_matrix(labeled_features, bundle.feature_columns)
    estimator = bundle.validation_estimator or bundle.estimator
    probabilities = estimator.predict_proba(matrix)[:, 1]
    return _metrics_at_threshold(labels, probabilities, bundle.threshold)


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    """Serialize the model, fitted feature engineer, config, and threshold."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as model_file:
        pickle.dump(bundle, model_file, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_model_bundle(path: str | Path) -> ModelBundle:
    """Load a previously trained model bundle."""
    with Path(path).open("rb") as model_file:
        bundle = pickle.load(model_file)
    if not isinstance(bundle, ModelBundle):
        raise ValueError("Model artifact does not contain a ModelBundle.")
    return bundle


def print_evaluation_report(title: str, metrics: dict[str, Any]) -> None:
    """Print the challenge metrics and confusion matrix in a readable format."""
    print(title)
    print(f"  Threshold: {metrics['threshold']:.2f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F0.5:      {metrics['f0_5']:.4f}")
    print("  Confusion matrix [[TN, FP], [FN, TP]]:")
    print(f"    {metrics['confusion_matrix']}")
    print(
        f"  TN={metrics['true_negative']} FP={metrics['false_positive']} "
        f"FN={metrics['false_negative']} TP={metrics['true_positive']}"
    )