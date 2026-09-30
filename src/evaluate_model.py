"""Evaluate a saved match model on its held-out labeled validation features."""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from src.model import evaluate_model, load_model_bundle, print_evaluation_report


PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report validation metrics for the saved entity matching model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_DIR / "models" / "entity_match_model.pkl",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_DIR / "output" / "validation_features.tsv",
        help="Labeled validation feature TSV saved by train_model.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "output" / "evaluation_report.json",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Trained model not found: {args.model}", file=sys.stderr)
        return 1
    if not args.features.exists():
        print(f"Labeled validation features not found: {args.features}", file=sys.stderr)
        return 1

    try:
        bundle = load_model_bundle(args.model)
        labeled_features = pd.read_csv(args.features, sep="\t")
        metrics = evaluate_model(bundle, labeled_features)
    except (OSError, ValueError, TypeError) as error:
        print(error, file=sys.stderr)
        return 1

    report = {
        "model": bundle.model_name,
        "threshold": bundle.threshold,
        "metrics": metrics,
        "note": "Threshold was selected using validation performance and is not retuned here.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    print_evaluation_report(
        f"Saved model validation evaluation: {bundle.model_name}", metrics
    )
    print(f"Evaluation report saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())