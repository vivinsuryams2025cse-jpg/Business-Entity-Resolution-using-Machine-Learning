"""Numeric pair features for business entity matching."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import pickle
import re
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import (
    normalize_business_address,
    normalize_business_name,
    normalize_country,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE1_PAIR_COLUMN = "source1_entity_id"
CANDIDATE_PAIR_COLUMN = "candidate_entity_id"
LABEL_COLUMN = "is_match"
FEATURE_COLUMNS = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort",
    "name_token_set",
    "name_edit_similarity",
    "name_jaccard",
    "name_tfidf",
    "address_ratio",
    "address_partial_ratio",
    "address_token_similarity",
    "address_jaccard",
    "address_tfidf",
    "address_common_tokens",
    "country_match",
    "name_length_difference",
    "address_length_difference",
    "name_common_tokens",
    "common_tokens",
    "name_missing_source1",
    "name_missing_candidate",
    "address_missing_source1",
    "address_missing_candidate",
    "country_missing_source1",
    "country_missing_candidate",
]
_ID_COLUMN_NAMES = {
    1: ("source1entityid", "s1entityid", "entityid", "source1id", "s1id", "id"),
    2: ("source2entityid", "s2entityid", "entityid", "source2id", "s2id", "id"),
    3: ("source3entityid", "s3entityid", "entityid", "source3id", "s3id", "id"),
}


@dataclass(frozen=True)
class FeatureConfig:
    """Column names and memory/performance controls for feature generation."""

    source1_id_column: str | None = None
    source2_id_column: str | None = None
    source3_id_column: str | None = None
    ground_truth_source1_id_column: str | None = None
    ground_truth_source2_id_column: str | None = None
    ground_truth_source3_id_column: str | None = None
    name_column: str = "business_name"
    address_column: str = "business_address"
    country_column: str = "country"
    batch_size: int = 50000
    workers: int = -1


def _normalize_column_name(column: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(column).casefold())


def _resolve_optional_id_column(
    records: pd.DataFrame,
    source_number: int,
    configured_column: str | None = None,
) -> str | None:
    if configured_column is not None:
        if configured_column not in records.columns:
            raise ValueError(f"ID column {configured_column!r} was not found.")
        return configured_column

    columns_by_normalized_name = {
        _normalize_column_name(column): column for column in records.columns
    }
    for possible_name in _ID_COLUMN_NAMES[source_number]:
        if possible_name in columns_by_normalized_name:
            return columns_by_normalized_name[possible_name]
    return None


def _resolve_id_column(
    records: pd.DataFrame,
    source_number: int,
    configured_column: str | None = None,
) -> str:
    resolved_column = _resolve_optional_id_column(records, source_number, configured_column)
    if resolved_column is not None:
        return resolved_column
    raise ValueError(
        f"Could not identify the Source {source_number} entity ID column. "
        "Set the corresponding ID column in FeatureConfig."
    )


def _clean_value(value: object, normalizer) -> str | None:
    normalized = normalizer(value)
    if normalized is None or not normalized.strip():
        return None
    return normalized


def _text_values(records: pd.DataFrame, column: str, normalizer) -> list[str | None]:
    normalized_column = f"{column.removesuffix('_normalized')}_normalized"
    if normalized_column in records.columns:
        source_column = normalized_column
    elif column in records.columns:
        source_column = column
    else:
        return [None] * len(records)
    normalized_values = records[source_column].map(
        lambda value: _clean_value(value, normalizer)
    ).tolist()
    return [
        value if isinstance(value, str) and value.strip() else None
        for value in normalized_values
    ]


def _fit_vectorizer(documents: list[str | None]) -> TfidfVectorizer | None:
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32,
        norm="l2",
    )
    try:
        vectorizer.fit(
            [document if isinstance(document, str) else "" for document in documents]
        )
    except ValueError as error:
        if "empty vocabulary" not in str(error).lower():
            raise
        return None
    return vectorizer


def _transform_vectorizer(
    vectorizer: TfidfVectorizer | None,
    documents: list[str | None],
) -> csr_matrix:
    if vectorizer is None:
        return csr_matrix((len(documents), 0), dtype=np.float32)
    clean_documents = [
        document if isinstance(document, str) else "" for document in documents
    ]
    return vectorizer.transform(clean_documents).tocsr()


def _jaccard_and_common_tokens(
    left: list[str | None], right: list[str | None]
) -> tuple[np.ndarray, np.ndarray]:
    similarities = np.zeros(len(left), dtype=np.float32)
    shared_counts = np.zeros(len(left), dtype=np.float32)
    for row_number, (left_text, right_text) in enumerate(zip(left, right)):
        if left_text is None or right_text is None:
            continue
        left_tokens = set(left_text.split())
        right_tokens = set(right_text.split())
        union = left_tokens | right_tokens
        intersection_count = len(left_tokens & right_tokens)
        if union:
            similarities[row_number] = intersection_count / len(union)
        shared_counts[row_number] = intersection_count
    return similarities, shared_counts


def _length_difference(left: list[str | None], right: list[str | None]) -> np.ndarray:
    return np.fromiter(
        (
            abs(len(left_text) - len(right_text))
            if left_text is not None and right_text is not None
            else 0
            for left_text, right_text in zip(left, right)
        ),
        dtype=np.float32,
        count=len(left),
    )


class PairFeatureEngineer:
    """Fit TF-IDF on training source records and transform candidate pairs."""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        if self.config.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.name_vectorizer: TfidfVectorizer | None = None
        self.address_vectorizer: TfidfVectorizer | None = None
        self.is_fitted = False

    def fit(
        self,
        source1: pd.DataFrame,
        source2: pd.DataFrame,
        source3: pd.DataFrame,
    ) -> "PairFeatureEngineer":
        """Fit name/address TF-IDF vocabularies from training source records only."""
        name_documents = []
        address_documents = []
        for records in (source1, source2, source3):
            name_documents.extend(
                _text_values(records, self.config.name_column, normalize_business_name)
            )
            address_documents.extend(
                _text_values(records, self.config.address_column, normalize_business_address)
            )

        self.name_vectorizer = _fit_vectorizer(name_documents)
        self.address_vectorizer = _fit_vectorizer(address_documents)
        self.is_fitted = True
        return self

    def transform_candidate_pairs(
        self,
        source1: pd.DataFrame,
        source2: pd.DataFrame,
        source3: pd.DataFrame,
        candidate_pairs: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create a numeric feature row for each unique candidate ID pair."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() with training source data before transforming pairs.")
        required_pair_columns = {SOURCE1_PAIR_COLUMN, CANDIDATE_PAIR_COLUMN}
        missing_pair_columns = required_pair_columns - set(candidate_pairs.columns)
        if missing_pair_columns:
            raise ValueError(f"Candidate pairs are missing columns: {sorted(missing_pair_columns)}")

        pairs = candidate_pairs[[SOURCE1_PAIR_COLUMN, CANDIDATE_PAIR_COLUMN]].drop_duplicates(
            ignore_index=True
        )
        if pairs.empty:
            return pairs.assign(**{column: pd.Series(dtype="float32") for column in FEATURE_COLUMNS})

        source1_id_column = _resolve_id_column(
            source1, 1, self.config.source1_id_column
        )
        target_frames = (source2, source3)
        target_id_columns = (
            _resolve_id_column(source2, 2, self.config.source2_id_column),
            _resolve_id_column(source3, 3, self.config.source3_id_column),
        )

        source1_id_values = source1[source1_id_column]
        if source1_id_values.isna().any() or source1_id_values.astype(str).str.strip().eq("").any():
            raise ValueError("Source 1 contains missing or blank entity IDs.")
        source1_ids = source1_id_values.astype(str).tolist()
        source1_id_to_row = {entity_id: row for row, entity_id in enumerate(source1_ids)}
        if len(source1_id_to_row) != len(source1_ids):
            raise ValueError("Source 1 contains duplicate entity IDs.")

        target_id_to_row = {}
        target_names: list[str | None] = []
        target_addresses: list[str | None] = []
        target_countries: list[str | None] = []
        for records, id_column in zip(target_frames, target_id_columns):
            names = _text_values(records, self.config.name_column, normalize_business_name)
            addresses = _text_values(records, self.config.address_column, normalize_business_address)
            countries = _text_values(records, self.config.country_column, normalize_country)
            id_values = records[id_column]
            if id_values.isna().any() or id_values.astype(str).str.strip().eq("").any():
                raise ValueError(f"Target source column {id_column!r} has missing or blank IDs.")
            ids = id_values.astype(str).tolist()
            for row_number, entity_id in enumerate(ids):
                if entity_id in target_id_to_row:
                    raise ValueError(f"Candidate entity ID {entity_id!r} occurs in multiple target rows.")
                target_id_to_row[entity_id] = len(target_names)
                target_names.append(names[row_number])
                target_addresses.append(addresses[row_number])
                target_countries.append(countries[row_number])

        source1_pair_ids = pairs[SOURCE1_PAIR_COLUMN].astype(str).tolist()
        candidate_pair_ids = pairs[CANDIDATE_PAIR_COLUMN].astype(str).tolist()
        unknown_source1 = sorted(set(source1_pair_ids) - set(source1_id_to_row))
        unknown_candidates = sorted(set(candidate_pair_ids) - set(target_id_to_row))
        if unknown_source1 or unknown_candidates:
            raise ValueError(
                f"Candidate IDs not found in source records; Source 1: {unknown_source1[:5]}, "
                f"target: {unknown_candidates[:5]}"
            )

        source1_names = _text_values(source1, self.config.name_column, normalize_business_name)
        source1_addresses = _text_values(source1, self.config.address_column, normalize_business_address)
        source1_countries = _text_values(source1, self.config.country_column, normalize_country)
        source1_name_matrix = _transform_vectorizer(self.name_vectorizer, source1_names)
        target_name_matrix = _transform_vectorizer(self.name_vectorizer, target_names)
        source1_address_matrix = _transform_vectorizer(self.address_vectorizer, source1_addresses)
        target_address_matrix = _transform_vectorizer(self.address_vectorizer, target_addresses)

        source_rows = np.fromiter(
            (source1_id_to_row[entity_id] for entity_id in source1_pair_ids),
            dtype=np.int64,
            count=len(pairs),
        )
        target_rows = np.fromiter(
            (target_id_to_row[entity_id] for entity_id in candidate_pair_ids),
            dtype=np.int64,
            count=len(pairs),
        )

        name_left = [source1_names[row] for row in source_rows]
        name_right = [target_names[row] for row in target_rows]
        address_left = [source1_addresses[row] for row in source_rows]
        address_right = [target_addresses[row] for row in target_rows]
        country_left = [source1_countries[row] for row in source_rows]
        country_right = [target_countries[row] for row in target_rows]

        result_chunks = []
        for start in range(0, len(pairs), self.config.batch_size):
            end = min(start + self.config.batch_size, len(pairs))
            chunk_source_rows = source_rows[start:end]
            chunk_target_rows = target_rows[start:end]
            chunk_name_left = name_left[start:end]
            chunk_name_right = name_right[start:end]
            chunk_address_left = address_left[start:end]
            chunk_address_right = address_right[start:end]
            name_missing_left = np.array([value is None for value in chunk_name_left])
            name_missing_right = np.array([value is None for value in chunk_name_right])
            address_missing_left = np.array([value is None for value in chunk_address_left])
            address_missing_right = np.array([value is None for value in chunk_address_right])
            country_missing_left = np.array([value is None for value in country_left[start:end]])
            country_missing_right = np.array([value is None for value in country_right[start:end]])
            name_present = ~(name_missing_left | name_missing_right)
            address_present = ~(address_missing_left | address_missing_right)

            name_values_left = [value or "" for value in chunk_name_left]
            name_values_right = [value or "" for value in chunk_name_right]
            address_values_left = [value or "" for value in chunk_address_left]
            address_values_right = [value or "" for value in chunk_address_right]

            def score(scorer, left_values, right_values, present_mask):
                values = process.cpdist(
                    left_values,
                    right_values,
                    scorer=scorer,
                    workers=self.config.workers,
                    dtype=np.float32,
                )
                values[~present_mask] = 0
                return values

            name_jaccard, name_common_tokens = _jaccard_and_common_tokens(
                chunk_name_left, chunk_name_right
            )
            address_jaccard, address_common_tokens = _jaccard_and_common_tokens(
                chunk_address_left, chunk_address_right
            )
            name_tfidf = np.asarray(
                source1_name_matrix[chunk_source_rows]
                .multiply(target_name_matrix[chunk_target_rows])
                .sum(axis=1)
            ).ravel().astype(np.float32)
            address_tfidf = np.asarray(
                source1_address_matrix[chunk_source_rows]
                .multiply(target_address_matrix[chunk_target_rows])
                .sum(axis=1)
            ).ravel().astype(np.float32)

            result_chunks.append(
                pd.DataFrame(
                    {
                        "name_ratio": score(fuzz.ratio, name_values_left, name_values_right, name_present),
                        "name_partial_ratio": score(fuzz.partial_ratio, name_values_left, name_values_right, name_present),
                        "name_token_sort": score(fuzz.token_sort_ratio, name_values_left, name_values_right, name_present),
                        "name_token_set": score(fuzz.token_set_ratio, name_values_left, name_values_right, name_present),
                        "name_edit_similarity": score(Levenshtein.normalized_similarity, name_values_left, name_values_right, name_present),
                        "name_jaccard": name_jaccard,
                        "name_tfidf": name_tfidf,
                        "address_ratio": score(fuzz.ratio, address_values_left, address_values_right, address_present),
                        "address_partial_ratio": score(fuzz.partial_ratio, address_values_left, address_values_right, address_present),
                        "address_token_similarity": score(fuzz.token_sort_ratio, address_values_left, address_values_right, address_present),
                        "address_jaccard": address_jaccard,
                        "address_tfidf": address_tfidf,
                        "address_common_tokens": address_common_tokens,
                        "country_match": np.array(
                            [
                                left == right and left is not None
                                for left, right in zip(country_left[start:end], country_right[start:end])
                            ],
                            dtype=np.float32,
                        ),
                        "name_length_difference": _length_difference(chunk_name_left, chunk_name_right),
                        "address_length_difference": _length_difference(chunk_address_left, chunk_address_right),
                        "name_common_tokens": name_common_tokens,
                        "common_tokens": name_common_tokens + address_common_tokens,
                        "name_missing_source1": name_missing_left.astype(np.float32),
                        "name_missing_candidate": name_missing_right.astype(np.float32),
                        "address_missing_source1": address_missing_left.astype(np.float32),
                        "address_missing_candidate": address_missing_right.astype(np.float32),
                        "country_missing_source1": country_missing_left.astype(np.float32),
                        "country_missing_candidate": country_missing_right.astype(np.float32),
                    },
                    columns=FEATURE_COLUMNS,
                )
            )

        features = pd.concat(result_chunks, ignore_index=True)
        features = features.replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)
        return pd.concat([pairs, features], axis=1)

    def save(self, path: str | Path) -> Path:
        """Persist the fitted training vocabularies for repeatable inference."""
        if not self.is_fitted:
            raise RuntimeError("Fit the feature engineer before saving it.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as pipeline_file:
            pickle.dump(self, pipeline_file, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PairFeatureEngineer":
        """Load a saved feature engineer fitted only on training source records."""
        with Path(path).open("rb") as pipeline_file:
            feature_engineer = pickle.load(pipeline_file)
        if not isinstance(feature_engineer, cls) or not feature_engineer.is_fitted:
            raise ValueError("The saved file does not contain a fitted PairFeatureEngineer.")
        return feature_engineer


def _ground_truth_match_pairs(
    ground_truth: pd.DataFrame,
    config: FeatureConfig,
) -> set[tuple[str, str]]:
    source1_id_column = _resolve_id_column(
        ground_truth, 1, config.ground_truth_source1_id_column
    )
    source2_id_column = _resolve_optional_id_column(
        ground_truth, 2, config.ground_truth_source2_id_column
    )
    source3_id_column = _resolve_optional_id_column(
        ground_truth, 3, config.ground_truth_source3_id_column
    )
    if source2_id_column is None and source3_id_column is None:
        raise ValueError("Ground truth must include a Source 2 ID or Source 3 ID column.")

    matches = set()
    target_columns = [
        column for column in (source2_id_column, source3_id_column) if column is not None
    ]
    for _, row in ground_truth.iterrows():
        source1_id = row[source1_id_column]
        if pd.isna(source1_id) or not str(source1_id).strip():
            continue
        for target_column in target_columns:
            target_id = row[target_column]
            if not pd.isna(target_id) and str(target_id).strip():
                matches.add((str(source1_id), str(target_id)))
    return matches


def label_candidate_pairs(
    candidate_pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Add binary training labels to candidate pairs from ground truth only."""
    config = config or FeatureConfig()
    required_columns = {SOURCE1_PAIR_COLUMN, CANDIDATE_PAIR_COLUMN}
    missing_columns = required_columns - set(candidate_pairs.columns)
    if missing_columns:
        raise ValueError(f"Candidate pairs are missing columns: {sorted(missing_columns)}")

    labeled_pairs = candidate_pairs[
        [SOURCE1_PAIR_COLUMN, CANDIDATE_PAIR_COLUMN]
    ].drop_duplicates(ignore_index=True)
    match_pairs = _ground_truth_match_pairs(ground_truth, config)
    labels = [
        int((str(source1_id), str(candidate_id)) in match_pairs)
        for source1_id, candidate_id in zip(
            labeled_pairs[SOURCE1_PAIR_COLUMN],
            labeled_pairs[CANDIDATE_PAIR_COLUMN],
        )
    ]
    labeled_pairs[LABEL_COLUMN] = np.asarray(labels, dtype=np.int8)
    return labeled_pairs


def build_training_feature_dataset(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
    feature_engineer: PairFeatureEngineer,
) -> pd.DataFrame:
    """Add binary labels to candidate features using training ground truth only."""
    feature_dataset = feature_engineer.transform_candidate_pairs(
        source1, source2, source3, candidate_pairs
    )
    labeled_pairs = label_candidate_pairs(
        candidate_pairs, ground_truth, feature_engineer.config
    )
    label_lookup = {
        (str(row[SOURCE1_PAIR_COLUMN]), str(row[CANDIDATE_PAIR_COLUMN])): row[LABEL_COLUMN]
        for _, row in labeled_pairs.iterrows()
    }
    feature_dataset[LABEL_COLUMN] = np.asarray(
        [
            label_lookup[(str(source1_id), str(candidate_id))]
            for source1_id, candidate_id in zip(
                feature_dataset[SOURCE1_PAIR_COLUMN],
                feature_dataset[CANDIDATE_PAIR_COLUMN],
            )
        ],
        dtype=np.int8,
    )
    return feature_dataset


def print_feature_report(feature_dataset: pd.DataFrame, *, include_label: bool = False) -> None:
    """Print numeric feature distributions and a few example candidate rows."""
    columns = FEATURE_COLUMNS + ([LABEL_COLUMN] if include_label and LABEL_COLUMN in feature_dataset else [])
    print("Feature distributions:")
    if feature_dataset.empty:
        print("  No candidate feature rows to summarize.")
    else:
        print(
            feature_dataset[columns]
            .describe()
            .loc[["mean", "std", "min", "max"]]
            .transpose()
            .to_string()
        )
        if include_label and LABEL_COLUMN in feature_dataset:
            print("Label counts:")
            print(feature_dataset[LABEL_COLUMN].value_counts().sort_index().to_string())
        print("Example feature rows:")
        print(feature_dataset.head(5).to_string(index=False))


def _read_sources(split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_paths = [
        PROJECT_DIR / "dataset" / split / f"{split}_source{number}.tsv"
        for number in (1, 2, 3)
    ]
    missing_paths = [path for path in source_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing source files: "
            + ", ".join(str(path.relative_to(PROJECT_DIR)) for path in missing_paths)
        )
    return tuple(pd.read_csv(path, sep="\t") for path in source_paths)


def _run_cli() -> int:
    parser = argparse.ArgumentParser(description="Build numeric candidate-pair features.")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--pipeline", type=Path, default=PROJECT_DIR / "output" / "feature_pipeline.joblib"
    )
    args = parser.parse_args()

    candidate_path = args.candidates or PROJECT_DIR / "output" / (
        "train_candidate_pairs.tsv" if args.split == "train" else "candidate_pairs.tsv"
    )
    output_path = args.output or PROJECT_DIR / "output" / (
        "train_features.tsv" if args.split == "train" else "test_features.tsv"
    )
    if not candidate_path.exists():
        print(f"Candidate file not found: {candidate_path}", file=sys.stderr)
        return 1

    try:
        source1, source2, source3 = _read_sources(args.split)
        candidate_pairs = pd.read_csv(candidate_path, sep="\t")
        if args.split == "train":
            ground_truth_path = PROJECT_DIR / "dataset" / "train" / "train_ground_truth.tsv"
            if not ground_truth_path.exists():
                raise FileNotFoundError(f"Missing ground truth: {ground_truth_path}")
            ground_truth = pd.read_csv(ground_truth_path, sep="\t")
            feature_engineer = PairFeatureEngineer().fit(source1, source2, source3)
            feature_dataset = build_training_feature_dataset(
                source1,
                source2,
                source3,
                candidate_pairs,
                ground_truth,
                feature_engineer,
            )
            feature_engineer.save(args.pipeline)
            include_label = True
        else:
            feature_engineer = PairFeatureEngineer.load(args.pipeline)
            feature_dataset = feature_engineer.transform_candidate_pairs(
                source1, source2, source3, candidate_pairs
            )
            include_label = False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        feature_dataset.to_csv(output_path, sep="\t", index=False)
        print(f"Feature dataset written to: {output_path}")
        if args.split == "train":
            print(f"Fitted feature pipeline saved to: {args.pipeline}")
        print_feature_report(feature_dataset, include_label=include_label)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(_run_cli())