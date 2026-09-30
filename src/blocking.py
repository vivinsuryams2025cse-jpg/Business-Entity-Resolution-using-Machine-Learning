"""Scalable candidate generation for cross-source business entity matching."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from time import perf_counter

import pandas as pd

from src.preprocessing import (
    normalize_business_address,
    normalize_business_name,
    normalize_country,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_COLUMNS = ["source1_entity_id", "candidate_entity_id"]
BlockingKey = tuple[str, ...]

_STRATEGY_PRIORITY = {
    "country_name_token": 0,
    "country_address_token": 0,
    "country_name_prefix": 0,
    "name_token": 1,
    "address_token": 2,
    "name_prefix": 3,
    "country": 4,
}


@dataclass(frozen=True)
class BlockingConfig:
    """Controls the recall/runtime tradeoff of candidate generation."""

    source1_id_column: str | None = None
    source2_id_column: str | None = None
    source3_id_column: str | None = None
    name_column: str = "business_name"
    address_column: str = "business_address"
    country_column: str = "country"
    prefix_length: int = 3
    min_token_length: int = 2
    max_block_pair_count: int = 5000
    max_candidates_per_source1_per_source: int = 100


@dataclass
class BlockingResult:
    """Candidate ID pairs and diagnostics for a single blocking run."""

    candidate_pairs: pd.DataFrame
    metrics: dict[str, int | float]


def _normalized_column_name(column: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(column).casefold())


def _resolve_id_column(
    records: pd.DataFrame,
    source_number: int,
    configured_column: str | None,
) -> str:
    if configured_column is not None:
        if configured_column not in records.columns:
            raise ValueError(f"ID column {configured_column!r} was not found.")
        return configured_column

    column_lookup = {
        _normalized_column_name(column): column for column in records.columns
    }
    candidates = (
        f"source{source_number}entityid",
        f"s{source_number}entityid",
        "entityid",
        f"source{source_number}id",
        f"s{source_number}id",
        "businessid",
        "recordid",
        "id",
    )
    for candidate in candidates:
        if candidate in column_lookup:
            return column_lookup[candidate]

    raise ValueError(
        f"Could not identify Source {source_number} entity IDs. "
        "Pass its column name in BlockingConfig."
    )


def _resolve_feature_column(records: pd.DataFrame, column: str) -> str | None:
    normalized_column = f"{column.removesuffix('_normalized')}_normalized"
    if normalized_column in records.columns:
        return normalized_column
    if column in records.columns:
        return column
    return None


def _validate_ids(records: pd.DataFrame, id_column: str, source_number: int) -> list[object]:
    ids = records[id_column]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError(f"Source {source_number} has missing or blank entity IDs.")
    if ids.duplicated().any():
        raise ValueError(f"Source {source_number} has duplicate IDs in {id_column!r}.")
    return ids.tolist()


def _field_values(
    records: pd.DataFrame,
    column: str,
    normalizer,
) -> list[str | None]:
    resolved_column = _resolve_feature_column(records, column)
    if resolved_column is None:
        return [None] * len(records)
    normalized_values = records[resolved_column].map(normalizer).tolist()
    return [
        value if isinstance(value, str) and value.strip() else None
        for value in normalized_values
    ]


def _record_keys(
    name: str | None,
    address: str | None,
    country: str | None,
    *,
    prefix_length: int,
    min_token_length: int,
) -> set[BlockingKey]:
    keys: set[BlockingKey] = set()
    name_tokens = {
        token for token in (name or "").split() if len(token) >= min_token_length
    }
    address_tokens = {
        token for token in (address or "").split() if len(token) >= min_token_length
    }
    name_prefixes = {
        token[:prefix_length]
        for token in name_tokens
        if len(token) >= prefix_length
    }

    if country:
        keys.add(("country", country))
    keys.update(("name_token", token) for token in name_tokens)
    keys.update(("name_prefix", prefix) for prefix in name_prefixes)
    keys.update(("address_token", token) for token in address_tokens)

    if country:
        keys.update(("country_name_token", country, token) for token in name_tokens)
        keys.update(("country_name_prefix", country, prefix) for prefix in name_prefixes)
        keys.update(("country_address_token", country, token) for token in address_tokens)

    return keys


def _build_keys(
    records: pd.DataFrame,
    config: BlockingConfig,
) -> list[set[BlockingKey]]:
    names = _field_values(records, config.name_column, normalize_business_name)
    addresses = _field_values(records, config.address_column, normalize_business_address)
    countries = _field_values(records, config.country_column, normalize_country)
    return [
        _record_keys(
            name,
            address,
            country,
            prefix_length=config.prefix_length,
            min_token_length=config.min_token_length,
        )
        for name, address, country in zip(names, addresses, countries)
    ]


def _build_index(record_keys: list[set[BlockingKey]]) -> dict[BlockingKey, list[int]]:
    index: dict[BlockingKey, list[int]] = defaultdict(list)
    for row_number, keys in enumerate(record_keys):
        for key in keys:
            index[key].append(row_number)
    return index


def _generate_for_target(
    source1_ids: list[object],
    source1_keys: list[set[BlockingKey]],
    source1_index: dict[BlockingKey, list[int]],
    target_ids: list[object],
    target_index: dict[BlockingKey, list[int]],
    config: BlockingConfig,
) -> tuple[list[tuple[object, object]], set[int], int, int]:
    pairs = []
    covered_source1_rows: set[int] = set()
    truncated_source1_rows = 0
    oversized_blocks: set[BlockingKey] = set()

    for source1_row, keys in enumerate(source1_keys):
        eligible_blocks = []
        for key in keys:
            target_rows = target_index.get(key)
            if not target_rows:
                continue

            block_pair_count = len(source1_index[key]) * len(target_rows)
            if block_pair_count > config.max_block_pair_count:
                oversized_blocks.add(key)
                continue

            eligible_blocks.append(
                (_STRATEGY_PRIORITY[key[0]], block_pair_count, key, target_rows)
            )

        eligible_blocks.sort(key=lambda block: (block[0], block[1], block[2]))
        selected_target_rows: set[int] = set()
        truncated = False

        for _, _, _, target_rows in eligible_blocks:
            for target_row in target_rows:
                if target_row in selected_target_rows:
                    continue
                if len(selected_target_rows) >= config.max_candidates_per_source1_per_source:
                    truncated = True
                    break
                selected_target_rows.add(target_row)
            if truncated:
                break

        if truncated:
            truncated_source1_rows += 1
        if selected_target_rows:
            covered_source1_rows.add(source1_row)
            pairs.extend(
                (source1_ids[source1_row], target_ids[target_row])
                for target_row in sorted(selected_target_rows)
            )

    return pairs, covered_source1_rows, truncated_source1_rows, len(oversized_blocks)


def generate_candidate_pairs(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    *,
    config: BlockingConfig | None = None,
) -> BlockingResult:
    """Generate deduplicated S1-to-S2/S3 candidates without deciding matches.

    Candidate blocks combine exact normalized country, normalized business-name
    tokens, character prefixes, and normalized address tokens. Combined
    country-and-field keys improve precision; standalone field keys preserve
    recall when country values are missing or differ. Blocks exceeding the
    configured pair count are skipped to avoid quadratic expansions. Each S1
    row is capped independently for each target source.
    """
    started_at = perf_counter()
    config = config or BlockingConfig()
    if config.prefix_length < 1 or config.min_token_length < 1:
        raise ValueError("prefix_length and min_token_length must be positive.")
    if config.max_block_pair_count < 1 or config.max_candidates_per_source1_per_source < 1:
        raise ValueError("Blocking pair and candidate limits must be positive.")

    source1_id_column = _resolve_id_column(source1, 1, config.source1_id_column)
    source2_id_column = _resolve_id_column(source2, 2, config.source2_id_column)
    source3_id_column = _resolve_id_column(source3, 3, config.source3_id_column)
    source1_ids = _validate_ids(source1, source1_id_column, 1)
    source2_ids = _validate_ids(source2, source2_id_column, 2)
    source3_ids = _validate_ids(source3, source3_id_column, 3)

    source1_keys = _build_keys(source1, config)
    source2_keys = _build_keys(source2, config)
    source3_keys = _build_keys(source3, config)
    source1_index = _build_index(source1_keys)
    source2_index = _build_index(source2_keys)
    source3_index = _build_index(source3_keys)

    source2_pairs, source2_coverage, source2_truncated, source2_oversized = (
        _generate_for_target(
            source1_ids,
            source1_keys,
            source1_index,
            source2_ids,
            source2_index,
            config,
        )
    )
    source3_pairs, source3_coverage, source3_truncated, source3_oversized = (
        _generate_for_target(
            source1_ids,
            source1_keys,
            source1_index,
            source3_ids,
            source3_index,
            config,
        )
    )

    candidate_pairs = pd.DataFrame(
        source2_pairs + source3_pairs,
        columns=OUTPUT_COLUMNS,
    ).drop_duplicates(ignore_index=True)
    total_possible_pairs = len(source1) * (len(source2) + len(source3))
    candidate_pair_count = len(candidate_pairs)
    reduction_ratio = (
        1 - candidate_pair_count / total_possible_pairs
        if total_possible_pairs
        else 0.0
    )
    metrics = {
        "source1_records": len(source1),
        "source2_records": len(source2),
        "source3_records": len(source3),
        "total_possible_pairs": total_possible_pairs,
        "candidate_pairs": candidate_pair_count,
        "reduction_ratio": reduction_ratio,
        "generation_time_seconds": perf_counter() - started_at,
        "source1_with_source2_candidates": len(source2_coverage),
        "source1_with_source3_candidates": len(source3_coverage),
        "source1_truncated_for_source2": source2_truncated,
        "source1_truncated_for_source3": source3_truncated,
        "oversized_blocks_skipped": source2_oversized + source3_oversized,
    }
    return BlockingResult(candidate_pairs=candidate_pairs, metrics=metrics)


def write_candidate_pairs(candidate_pairs: pd.DataFrame, output_path: str | Path) -> Path:
    """Write the challenge pair schema as a tab-separated file."""
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in candidate_pairs]
    if missing_columns:
        raise ValueError(f"Candidate table is missing required columns: {missing_columns}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_pairs[OUTPUT_COLUMNS].to_csv(output_path, sep="\t", index=False)
    return output_path


def _run_cli() -> int:
    parser = argparse.ArgumentParser(description="Generate business entity candidate pairs.")
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="Which source files to process (default: test).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "output" / "candidate_pairs.tsv",
        help="Candidate TSV output path.",
    )
    args = parser.parse_args()

    input_paths = [
        PROJECT_DIR / "dataset" / args.split / f"{args.split}_source{source_number}.tsv"
        for source_number in (1, 2, 3)
    ]
    missing_paths = [path for path in input_paths if not path.exists()]
    if missing_paths:
        print("Cannot generate candidates; missing input files:", file=sys.stderr)
        for path in missing_paths:
            print(f"  {path.relative_to(PROJECT_DIR)}", file=sys.stderr)
        return 1

    source1, source2, source3 = [pd.read_csv(path, sep="\t") for path in input_paths]
    result = generate_candidate_pairs(source1, source2, source3)
    output_path = write_candidate_pairs(result.candidate_pairs, args.output)

    print(f"Candidate pairs written to: {output_path}")
    for metric, value in result.metrics.items():
        print(f"{metric}: {value:.6f}" if isinstance(value, float) else f"{metric}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(_run_cli())