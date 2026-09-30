"""Load and inspect the supplied business entity resolution TSV files."""

from pathlib import Path
import re
import sys

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILES = {
    "train_source1": PROJECT_DIR / "dataset" / "train" / "train_source1.tsv",
    "train_source2": PROJECT_DIR / "dataset" / "train" / "train_source2.tsv",
    "train_source3": PROJECT_DIR / "dataset" / "train" / "train_source3.tsv",
    "train_ground_truth": PROJECT_DIR / "dataset" / "train" / "train_ground_truth.tsv",
    "test_source1": PROJECT_DIR / "dataset" / "test" / "test_source1.tsv",
    "test_source2": PROJECT_DIR / "dataset" / "test" / "test_source2.tsv",
    "test_source3": PROJECT_DIR / "dataset" / "test" / "test_source3.tsv",
}
SOURCE_PREFIXES = {
    "train_source1": "S1-",
    "train_source2": "S2-",
    "train_source3": "S3-",
    "test_source1": "S1-",
    "test_source2": "S2-",
    "test_source3": "S3-",
}


def load_datasets() -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load available TSV files and collect missing or unreadable file errors."""
    datasets = {}
    problems = []

    for dataset_name, file_path in DATA_FILES.items():
        if not file_path.exists():
            problems.append(f"Missing file: {file_path.relative_to(PROJECT_DIR)}")
            continue

        try:
            datasets[dataset_name] = pd.read_csv(file_path, sep="\t")
        except (OSError, UnicodeError, pd.errors.ParserError) as error:
            problems.append(f"Could not read {file_path.name}: {error}")

    return datasets, problems


def normalize_column_name(column_name: object) -> str:
    """Normalize a column name to simplify ID-column matching."""
    return re.sub(r"[^a-z0-9]", "", str(column_name).lower())


def is_entity_id_column(column_name: object) -> bool:
    """Return whether a column name looks like an entity identifier."""
    normalized_name = normalize_column_name(column_name)
    return (
        normalized_name in {"id", "entityid", "businessid", "recordid"}
        or normalized_name.endswith(("entityid", "businessid", "recordid"))
        or normalized_name in {"source1id", "source2id", "source3id", "s1id", "s2id", "s3id"}
    )


def print_dataset_details(dataset_name: str, data: pd.DataFrame) -> None:
    """Print structure, sample rows, missing values, and ID checks."""
    print(f"\n{'=' * 72}\n{dataset_name} ({len(data):,} records)")
    print("Columns:", list(data.columns))
    print("Sample records:")
    if data.empty:
        print("  No records found.")
    else:
        print(data.head(5).to_string(index=False))

    missing_values = data.isna().sum()
    missing_values = missing_values[missing_values > 0]
    print("Missing values:")
    if missing_values.empty:
        print("  None")
    else:
        print(missing_values.to_string())

    id_columns = [column for column in data.columns if is_entity_id_column(column)]
    if dataset_name in SOURCE_PREFIXES:
        print("Duplicate entity IDs:")
        if not id_columns:
            print("  No recognizable entity ID column found.")
        for column in id_columns:
            duplicate_count = int(data[column].duplicated(keep=False).sum())
            print(f"  {column}: {duplicate_count:,} rows have a duplicated ID")

        expected_prefix = SOURCE_PREFIXES[dataset_name]
        if not id_columns:
            print(f"ID prefix check: could not check for {expected_prefix} (no ID column found).")
        for column in id_columns:
            ids = data[column].dropna().astype(str)
            invalid_ids = ids[~ids.str.startswith(expected_prefix)]
            print(
                f"ID prefix check ({column}, expected {expected_prefix}): "
                f"{len(ids) - len(invalid_ids):,}/{len(ids):,} IDs match"
            )

    if dataset_name == "train_ground_truth":
        print("Ground-truth source ID prefix checks:")
        checked_columns = set()
        for source_number in (1, 2, 3):
            expected_prefix = f"S{source_number}-"
            source_names = {f"source{source_number}id", f"s{source_number}id"}
            source_columns = [
                column
                for column in data.columns
                if normalize_column_name(column) in source_names
            ]
            for column in source_columns:
                checked_columns.add(column)
                ids = data[column].dropna().astype(str)
                matching_count = int(ids.str.startswith(expected_prefix).sum())
                print(
                    f"  {column} (expected {expected_prefix}): "
                    f"{matching_count:,}/{len(ids):,} IDs match"
                )
        if not checked_columns:
            print("  No source-specific ID columns found; check column names in the sample above.")


def print_dataset_summary(datasets: dict[str, pd.DataFrame]) -> None:
    """Print a compact row and column count summary by dataset split."""
    print(f"\n{'=' * 72}\nDataset summary")
    for split in ("train", "test"):
        split_datasets = [
            (name, data) for name, data in datasets.items() if name.startswith(f"{split}_")
        ]
        total_records = sum(len(data) for _, data in split_datasets)
        print(f"{split.title()}: {len(split_datasets)} files loaded, {total_records:,} total records")
        for name, data in split_datasets:
            print(f"  {name}: {len(data):,} records, {len(data.columns)} columns")


def main() -> int:
    """Run the dataset inspection report."""
    print("Business Entity Resolution - Dataset Inspection")
    datasets, problems = load_datasets()

    if not datasets:
        print("\nNo dataset files were loaded.")

    for dataset_name, data in datasets.items():
        print_dataset_details(dataset_name, data)

    print_dataset_summary(datasets)

    if problems:
        print("\nFiles to check:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nPlace the supplied TSV files in the matching dataset/train or dataset/test folder.")
        return 1

    print("\nAll expected TSV files were loaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())