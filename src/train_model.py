"""End-to-end supervised training and validation for entity matching."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from src.blocking import BlockingConfig, generate_candidate_pairs
from src.features import (
    CANDIDATE_PAIR_COLUMN,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    SOURCE1_PAIR_COLUMN,
    FeatureConfig,
    PairFeatureEngineer,
    _resolve_id_column,
    _resolve_optional_id_column,
    label_candidate_pairs,
)
from src.model import (
    DEFAULT_THRESHOLDS,
    RANDOM_STATE,
    compare_models,
    print_evaluation_report,
    save_model_bundle,
    split_by_source1_group,
)
from src.preprocessing import preprocess_business_records


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_dir = PROJECT_DIR / "dataset" / "train"
    paths = [
        train_dir / "train_source1.tsv",
        train_dir / "train_source2.tsv",
        train_dir / "train_source3.tsv",
        train_dir / "train_ground_truth.tsv",
    ]
    missing_paths = [path for path in paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing training files: "
            + ", ".join(str(path.relative_to(PROJECT_DIR)) for path in missing_paths)
        )
    loaded_frames = [pd.read_csv(path, sep="\t") for path in paths]
    return tuple(
        preprocess_business_records(frame) if index < 3 else frame
        for index, frame in enumerate(loaded_frames)
    )


def _load_feature_config(config_path: Path | None) -> FeatureConfig:
    if config_path is None:
        return FeatureConfig()
    with config_path.open(encoding="utf-8") as config_file:
        config_values: dict[str, Any] = json.load(config_file)
    return FeatureConfig(**config_values)


def _training_fold_sources(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    ground_truth: pd.DataFrame,
    training_pairs: pd.DataFrame,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select source records for TF-IDF fit without validation-entity text."""
    source1_id_column = _resolve_id_column(source1, 1, config.source1_id_column)
    ground_truth_source1_id = _resolve_id_column(
        ground_truth, 1, config.ground_truth_source1_id_column
    )
    training_source1_ids = set(training_pairs[SOURCE1_PAIR_COLUMN].astype(str))
    source1_fit = source1.loc[
        source1[source1_id_column].astype(str).isin(training_source1_ids)
    ]
    training_ground_truth = ground_truth.loc[
        ground_truth[ground_truth_source1_id].astype(str).isin(training_source1_ids)
    ]

    source2_id_column = _resolve_id_column(source2, 2, config.source2_id_column)
    ground_truth_source2_id = _resolve_optional_id_column(
        ground_truth, 2, config.ground_truth_source2_id_column
    )
    source2_fit_ids = (
        set(training_ground_truth[ground_truth_source2_id].dropna().astype(str))
        if ground_truth_source2_id is not None
        else set()
    )
    source2_fit = source2.loc[source2[source2_id_column].astype(str).isin(source2_fit_ids)]

    source3_id_column = _resolve_id_column(source3, 3, config.source3_id_column)
    ground_truth_source3_id = _resolve_optional_id_column(
        ground_truth, 3, config.ground_truth_source3_id_column
    )
    source3_fit_ids = (
        set(training_ground_truth[ground_truth_source3_id].dropna().astype(str))
        if ground_truth_source3_id is not None
        else set()
    )
    source3_fit = source3.loc[source3[source3_id_column].astype(str).isin(source3_fit_ids)]
    return source1_fit, source2_fit, source3_fit


def _add_pair_labels(
    features: pd.DataFrame,
    labeled_pairs: pd.DataFrame,
) -> pd.DataFrame:
    label_lookup = {
        (str(row[SOURCE1_PAIR_COLUMN]), str(row[CANDIDATE_PAIR_COLUMN])): int(row[LABEL_COLUMN])
        for _, row in labeled_pairs.iterrows()
    }
    result = features.copy()
    result[LABEL_COLUMN] = [
        label_lookup[(str(source1_id), str(candidate_id))]
        for source1_id, candidate_id in zip(
            result[SOURCE1_PAIR_COLUMN], result[CANDIDATE_PAIR_COLUMN]
        )
    ]
    return result


def run_training(
    *,
    output_dir: Path,
    validation_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    feature_config: FeatureConfig | None = None,
) -> dict[str, Any]:
    """Run preprocessing, blocking, labels, leakage-safe split, and model fit."""
    feature_config = feature_config or FeatureConfig()
    source1, source2, source3, ground_truth = _load_training_data()
    blocking_config = BlockingConfig(
        source1_id_column=feature_config.source1_id_column,
        source2_id_column=feature_config.source2_id_column,
        source3_id_column=feature_config.source3_id_column,
        name_column=feature_config.name_column,
        address_column=feature_config.address_column,
        country_column=feature_config.country_column,
    )
    blocking_result = generate_candidate_pairs(
        source1, source2, source3, config=blocking_config
    )
    labeled_pairs = label_candidate_pairs(
        blocking_result.candidate_pairs,
        ground_truth,
        feature_config,
    )
    training_pairs, validation_pairs = split_by_source1_group(
        labeled_pairs,
        validation_size=validation_size,
        random_state=random_state,
    )

    fit_sources = _training_fold_sources(
        source1,
        source2,
        source3,
        ground_truth,
        training_pairs,
        feature_config,
    )
    feature_engineer = PairFeatureEngineer(feature_config).fit(*fit_sources)
    training_features = feature_engineer.transform_candidate_pairs(
        source1, source2, source3, training_pairs
    )
    validation_features = feature_engineer.transform_candidate_pairs(
        source1, source2, source3, validation_pairs
    )
    training_features = _add_pair_labels(training_features, training_pairs)
    validation_features = _add_pair_labels(validation_features, validation_pairs)

    comparison = compare_models(
        training_features,
        validation_features,
        feature_engineer=feature_engineer,
        feature_config=asdict(feature_config),
        feature_columns=list(FEATURE_COLUMNS),
        thresholds=DEFAULT_THRESHOLDS,
        random_state=random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "entity_match_model.pkl"
    feature_config_path = output_dir / "feature_config.json"
    comparison_path = output_dir / "validation_metrics.json"
    training_features_path = output_dir / "train_features.tsv"
    validation_features_path = output_dir / "validation_features.tsv"

    save_model_bundle(comparison.bundle, model_path)
    with feature_config_path.open("w", encoding="utf-8") as config_file:
        json.dump(asdict(feature_config), config_file, indent=2)
    training_features.to_csv(training_features_path, sep="\t", index=False)
    validation_features.to_csv(validation_features_path, sep="\t", index=False)

    comparison_report = {
        "random_state": random_state,
        "validation_size": validation_size,
        "total_possible_pairs": blocking_result.metrics["total_possible_pairs"],
        "candidate_pairs": blocking_result.metrics["candidate_pairs"],
        "training_rows": len(training_features),
        "validation_rows": len(validation_features),
        "training_positive_rows": int(training_features[LABEL_COLUMN].sum()),
        "validation_positive_rows": int(validation_features[LABEL_COLUMN].sum()),
        "thresholds": list(DEFAULT_THRESHOLDS),
        "models": {
            name: {
                "threshold_sweep": comparison.threshold_results[name],
                "best_validation_metrics": comparison.best_model_results[name],
            }
            for name in comparison.threshold_results
        },
        "selected_model": comparison.selected_model_name,
        "selected_threshold": comparison.selected_threshold,
    }
    with comparison_path.open("w", encoding="utf-8") as report_file:
        json.dump(comparison_report, report_file, indent=2)

    print("Candidate generation")
    print(f"  Total possible pairs: {blocking_result.metrics['total_possible_pairs']:,}")
    print(f"  Candidate pairs:      {blocking_result.metrics['candidate_pairs']:,}")
    print(f"  Reduction ratio:      {blocking_result.metrics['reduction_ratio']:.4%}")
    print(f"  Generation time:      {blocking_result.metrics['generation_time_seconds']:.3f}s")
    print("Training/validation split grouped by Source 1 entity ID")
    print(
        f"  Train: {len(training_features):,} rows, "
        f"{int(training_features[LABEL_COLUMN].sum()):,} positives"
    )
    print(
        f"  Validation: {len(validation_features):,} rows, "
        f"{int(validation_features[LABEL_COLUMN].sum()):,} positives"
    )
    for model_name in comparison.threshold_results:
        print(f"\n{model_name} validation threshold sweep")
        print("  threshold  precision  recall  F0.5")
        for result in comparison.threshold_results[model_name]:
            print(
                f"  {result['threshold']:.2f}       {result['precision']:.4f}     "
                f"{result['recall']:.4f}  {result['f0_5']:.4f}"
            )
        print_evaluation_report(
            f"{model_name} best validation threshold", 
            comparison.best_model_results[model_name],
        )

    print("\nSelected model and threshold (validation F0.5 only)")
    print(f"  Model: {comparison.selected_model_name}")
    print(f"  Threshold: {comparison.selected_threshold:.2f}")
    print(f"  Saved model: {model_path}")
    print(f"  Saved feature config: {feature_config_path}")
    print(f"  Saved validation report: {comparison_path}")
    return comparison_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train supervised business entity matchers.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "output")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--feature-config", type=Path)
    args = parser.parse_args()

    try:
        run_training(
            output_dir=args.output_dir,
            validation_size=args.validation_size,
            random_state=args.random_state,
            feature_config=_load_feature_config(args.feature_config),
        )
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())