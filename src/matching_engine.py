"""End-to-end test-time candidate scoring and final entity match output."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.blocking import BlockingConfig, generate_candidate_pairs, write_candidate_pairs
from src.features import (
    CANDIDATE_PAIR_COLUMN,
    SOURCE1_PAIR_COLUMN,
    FeatureConfig,
    _resolve_id_column,
)
from src.model import ModelBundle, _feature_matrix, load_model_bundle
from src.preprocessing import preprocess_business_records


PROJECT_DIR = Path(__file__).resolve().parents[1]
MATCHING_RESULT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


@dataclass
class MatchingResult:
    """Candidate set, final entity matches, and inference measurements."""

    candidate_pairs: pd.DataFrame
    matching_results: pd.DataFrame
    metrics: dict[str, int | float]


def _feature_config_from_bundle(bundle: ModelBundle) -> FeatureConfig:
    try:
        return FeatureConfig(**bundle.feature_config)
    except TypeError as error:
        raise ValueError(f"Saved model has an invalid feature configuration: {error}") from error


def _validated_source_ids(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    config: FeatureConfig,
) -> tuple[list[str], set[str], set[str]]:
    source1_id_column = _resolve_id_column(source1, 1, config.source1_id_column)
    source2_id_column = _resolve_id_column(source2, 2, config.source2_id_column)
    source3_id_column = _resolve_id_column(source3, 3, config.source3_id_column)

    def read_ids(records: pd.DataFrame, column: str, label: str) -> list[str]:
        values = records[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{label} contains missing or blank entity IDs.")
        ids = values.astype(str).tolist()
        if len(set(ids)) != len(ids):
            raise ValueError(f"{label} contains duplicate entity IDs.")
        return ids

    source1_ids = read_ids(source1, source1_id_column, "Source 1")
    source2_ids = read_ids(source2, source2_id_column, "Source 2")
    source3_ids = read_ids(source3, source3_id_column, "Source 3")
    overlap = (set(source1_ids) & set(source2_ids)) | (set(source1_ids) & set(source3_ids))
    if overlap:
        raise ValueError(f"Source 1 IDs overlap target-source IDs: {sorted(overlap)[:5]}")
    target_overlap = set(source2_ids) & set(source3_ids)
    if target_overlap:
        raise ValueError(
            "Source 2 and Source 3 IDs must be distinct for candidate IDs to identify "
            f"their source unambiguously: {sorted(target_overlap)[:5]}"
        )
    return source1_ids, set(source2_ids), set(source3_ids)


def run_matching(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    bundle: ModelBundle,
) -> MatchingResult:
    """Preprocess, block, score, and threshold all test source records.

    The model bundle owns the fitted training feature transformer and selected
    validation threshold. No test labels or threshold tuning are used here.
    """
    started_at = perf_counter()
    if not hasattr(bundle.estimator, "predict_proba"):
        raise ValueError("The saved model must support probability predictions.")
    if not 0 < bundle.threshold < 1:
        raise ValueError("Saved matching threshold must be between 0 and 1.")
    if bundle.feature_engineer is None or not bundle.feature_engineer.is_fitted:
        raise ValueError("The saved model does not contain a fitted feature engineer.")

    feature_config = _feature_config_from_bundle(bundle)
    source1_ids, source2_ids, source3_ids = _validated_source_ids(
        source1, source2, source3, feature_config
    )
    processed_sources = tuple(
        preprocess_business_records(
            records,
            name_column=feature_config.name_column,
            address_column=feature_config.address_column,
            country_column=feature_config.country_column,
        )
        for records in (source1, source2, source3)
    )
    blocking_config = BlockingConfig(
        source1_id_column=feature_config.source1_id_column,
        source2_id_column=feature_config.source2_id_column,
        source3_id_column=feature_config.source3_id_column,
        name_column=feature_config.name_column,
        address_column=feature_config.address_column,
        country_column=feature_config.country_column,
    )
    blocking_result = generate_candidate_pairs(
        *processed_sources,
        config=blocking_config,
    )
    candidates = blocking_result.candidate_pairs
    allowed_candidate_ids = source2_ids | source3_ids
    if not set(candidates[CANDIDATE_PAIR_COLUMN].astype(str)).issubset(allowed_candidate_ids):
        raise ValueError("Blocking returned a candidate ID outside test Source 2/3.")
    if set(candidates[CANDIDATE_PAIR_COLUMN].astype(str)) & set(source1_ids):
        raise ValueError("Blocking returned a Source 1 ID as a candidate.")

    candidate_features = bundle.feature_engineer.transform_candidate_pairs(
        *processed_sources,
        candidates,
    )
    if candidate_features.empty:
        probabilities = np.empty(0, dtype=np.float32)
        predicted_pairs = candidate_features[[SOURCE1_PAIR_COLUMN, CANDIDATE_PAIR_COLUMN]].copy()
    else:
        matrix = _feature_matrix(candidate_features, bundle.feature_columns)
        probabilities = bundle.estimator.predict_proba(matrix)[:, 1]
        predicted_pairs = candidate_features.loc[
            probabilities >= bundle.threshold,
            [SOURCE1_PAIR_COLUMN, CANDIDATE_PAIR_COLUMN],
        ].drop_duplicates(ignore_index=True)

    matches_by_source1: dict[str, list[str]] = {entity_id: [] for entity_id in source1_ids}
    for source1_id, candidate_id in predicted_pairs.itertuples(index=False, name=None):
        source1_id = str(source1_id)
        candidate_id = str(candidate_id)
        if source1_id not in matches_by_source1:
            raise ValueError(f"Prediction contains unknown Source 1 ID {source1_id!r}.")
        if candidate_id not in allowed_candidate_ids or candidate_id in source1_ids:
            raise ValueError(f"Prediction contains invalid target entity ID {candidate_id!r}.")
        if candidate_id not in matches_by_source1[source1_id]:
            matches_by_source1[source1_id].append(candidate_id)

    matching_results = pd.DataFrame(
        {
            SOURCE1_PAIR_COLUMN: source1_ids,
            "matched_entity_ids": [
                ",".join(matches_by_source1[entity_id]) for entity_id in source1_ids
            ],
        },
        columns=MATCHING_RESULT_COLUMNS,
    )
    metrics = {
        "source1_records": len(source1_ids),
        "candidate_pairs": len(candidates),
        "predicted_matches": len(predicted_pairs),
        "matched_source1_records": sum(bool(ids) for ids in matches_by_source1.values()),
        "singleton_predictions": sum(len(ids) == 1 for ids in matches_by_source1.values()),
        "inference_time_seconds": perf_counter() - started_at,
    }
    return MatchingResult(candidates, matching_results, metrics)


def write_matching_outputs(
    result: MatchingResult,
    output_dir: str | Path = PROJECT_DIR / "output",
) -> tuple[Path, Path]:
    """Write candidate pairs and the one-row-per-Source-1 matching results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = write_candidate_pairs(
        result.candidate_pairs,
        output_dir / "candidate_pairs.tsv",
    )
    matching_path = output_dir / "matching_results.tsv"
    result.matching_results[MATCHING_RESULT_COLUMNS].to_csv(
        matching_path,
        sep="\t",
        index=False,
    )
    return candidate_path, matching_path


def _read_test_sources(
    source_paths: tuple[Path, Path, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_paths = [path for path in source_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing test source files: "
            + ", ".join(str(path.relative_to(PROJECT_DIR)) for path in missing_paths)
        )
    return tuple(
        pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        for path in source_paths
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate final business entity matches.")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_DIR / "models" / "entity_match_model.pkl",
    )
    parser.add_argument(
        "--source1",
        type=Path,
        default=PROJECT_DIR / "dataset" / "test" / "test_source1.tsv",
    )
    parser.add_argument(
        "--source2",
        type=Path,
        default=PROJECT_DIR / "dataset" / "test" / "test_source2.tsv",
    )
    parser.add_argument(
        "--source3",
        type=Path,
        default=PROJECT_DIR / "dataset" / "test" / "test_source3.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "output",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Trained model not found: {args.model}", file=sys.stderr)
        return 1

    try:
        source1, source2, source3 = _read_test_sources(
            (args.source1, args.source2, args.source3)
        )
        bundle = load_model_bundle(args.model)
        result = run_matching(source1, source2, source3, bundle)
        candidate_path, matching_path = write_matching_outputs(result, args.output_dir)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(error, file=sys.stderr)
        return 1

    print(f"Source 1 records:       {result.metrics['source1_records']:,}")
    print(f"Candidate pairs:        {result.metrics['candidate_pairs']:,}")
    print(f"Predicted matches:      {result.metrics['predicted_matches']:,}")
    print(f"Matched Source 1 rows:  {result.metrics['matched_source1_records']:,}")
    print(f"Singleton predictions:  {result.metrics['singleton_predictions']:,}")
    print(f"Inference time:         {result.metrics['inference_time_seconds']:.3f}s")
    print(f"Candidate pairs saved:  {candidate_path}")
    print(f"Matching results saved: {matching_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())