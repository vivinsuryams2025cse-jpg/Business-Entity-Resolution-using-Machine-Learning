import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import FeatureConfig, PairFeatureEngineer
from src.matching_engine import (
    MATCHING_RESULT_COLUMNS,
    run_matching,
    write_matching_outputs,
)
from src.model import ModelBundle


def make_records(rows):
    return pd.DataFrame(
        rows,
        columns=["entity_id", "business_name", "business_address", "country"],
    )


class NameScoreModel:
    """Small test double that returns a probability from a name feature."""

    def __init__(self, threshold=75.0):
        self.threshold = threshold

    def predict_proba(self, feature_matrix):
        name_ratio_column = 0
        scores = np.where(
            feature_matrix[:, name_ratio_column] >= self.threshold,
            0.95,
            0.05,
        )
        return np.column_stack((1 - scores, scores))


class MatchingEngineTests(unittest.TestCase):
    def setUp(self):
        self.source1 = make_records(
            [
                ("S1-1", "Northwind Distribution Inc", "12 Main Street", "France"),
                ("S1-2", "Unknown Company", "88 Hill Road", "Canada"),
                ("S1-3", None, None, None),
            ]
        )
        self.source2 = make_records(
            [
                ("S2-1", "Northwind Distribution Ltd", "12 Main St", "France"),
                ("S2-2", "Unrelated Company", "2 Oak Avenue", "France"),
            ]
        )
        self.source3 = make_records(
            [
                ("S3-1", "Northwind Distributon", "12 Main Street", "France"),
                ("S3-2", "Different Company", "88 Hill Rd", "Canada"),
            ]
        )
        feature_engineer = PairFeatureEngineer(FeatureConfig()).fit(
            self.source1, self.source2, self.source3
        )
        self.bundle = ModelBundle(
            model_name="test-name-score",
            estimator=NameScoreModel(),
            threshold=0.5,
            feature_columns=[
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
            ],
            feature_config={
                "source1_id_column": None,
                "source2_id_column": None,
                "source3_id_column": None,
                "name_column": "business_name",
                "address_column": "business_address",
                "country_column": "country",
                "batch_size": 50000,
                "workers": 1,
            },
            feature_engineer=feature_engineer,
            random_state=42,
            validation_summary={},
        )

    def test_every_source1_has_one_row_and_matches_only_exist_in_candidates(self):
        result = run_matching(self.source1, self.source2, self.source3, self.bundle)

        self.assertEqual(result.matching_results.columns.tolist(), MATCHING_RESULT_COLUMNS)
        self.assertEqual(
            result.matching_results["source1_entity_id"].tolist(),
            self.source1["entity_id"].tolist(),
        )
        self.assertEqual(result.matching_results["source1_entity_id"].nunique(), 3)
        candidate_ids = set(result.candidate_pairs["candidate_entity_id"])
        target_ids = set(self.source2["entity_id"]) | set(self.source3["entity_id"])
        for matched_ids in result.matching_results["matched_entity_ids"]:
            ids = matched_ids.split(",") if matched_ids else []
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(set(ids).issubset(candidate_ids))
            self.assertTrue(set(ids).issubset(target_ids))
            self.assertFalse(any(entity_id.startswith("S1-") for entity_id in ids))

    def test_multiple_matches_empty_matches_and_singleton_metrics(self):
        result = run_matching(self.source1, self.source2, self.source3, self.bundle)
        matches = result.matching_results.set_index("source1_entity_id")

        self.assertEqual(set(matches.loc["S1-1", "matched_entity_ids"].split(",")), {"S2-1", "S3-1"})
        self.assertEqual(matches.loc["S1-2", "matched_entity_ids"], "")
        self.assertEqual(matches.loc["S1-3", "matched_entity_ids"], "")
        self.assertEqual(result.metrics["source1_records"], 3)
        self.assertEqual(result.metrics["predicted_matches"], 2)
        self.assertEqual(result.metrics["singleton_predictions"], 0)
        self.assertGreaterEqual(result.metrics["inference_time_seconds"], 0)

    def test_singleton_prediction_is_counted(self):
        source1 = make_records(
            [("S1-only", "Acme Services Incorporated", "12 Main Street", "France")]
        )
        source2 = make_records(
            [("S2-only", "Acme Services Inc", "12 Main St", "France")]
        )
        source3 = make_records(
            [("S3-other", "Unrelated Enterprise", "88 Hill Road", "Canada")]
        )

        result = run_matching(source1, source2, source3, self.bundle)

        self.assertEqual(
            result.matching_results.loc[0, "matched_entity_ids"], "S2-only"
        )
        self.assertEqual(result.metrics["singleton_predictions"], 1)

    def test_writes_candidate_and_matching_tsvs_with_required_columns(self):
        result = run_matching(self.source1, self.source2, self.source3, self.bundle)
        with tempfile.TemporaryDirectory() as temporary_directory:
            candidate_path, matching_path = write_matching_outputs(result, temporary_directory)
            candidates = pd.read_csv(candidate_path, sep="\t", dtype=str)
            matches = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)

        self.assertEqual(
            candidates.columns.tolist(), ["source1_entity_id", "candidate_entity_id"]
        )
        self.assertEqual(matches.columns.tolist(), MATCHING_RESULT_COLUMNS)
        self.assertEqual(len(matches), len(self.source1))

    def test_all_sources_without_candidates_still_returns_source1_rows(self):
        source1 = make_records([("S1-empty", "No shared tokens", "", "")])
        source2 = make_records([("S2-empty", "Other words", "", "")])
        source3 = make_records([("S3-empty", "Different terms", "", "")])

        result = run_matching(source1, source2, source3, self.bundle)

        self.assertTrue(result.candidate_pairs.empty)
        self.assertEqual(
            result.matching_results.to_dict("records"),
            [{"source1_entity_id": "S1-empty", "matched_entity_ids": ""}],
        )

    def test_full_inference_path_handles_15000_records_per_source(self):
        source1 = make_records(
            [
                (f"S1-{number}", f"Entity Record {number:05d}", f"{number} Main Street", f"Country {number}")
                for number in range(15000)
            ]
        )
        source2 = make_records(
            [
                (f"S2-{number}", f"Entity Record {number:05d}", f"{number} Main St", f"Country {number}")
                for number in range(15000)
            ]
        )
        source3 = make_records(
            [
                (f"S3-{number}", f"Unrelated Account {number:05d}", f"{number} Oak Road", f"Other {number}")
                for number in range(15000)
            ]
        )

        result = run_matching(source1, source2, source3, self.bundle)

        self.assertEqual(result.metrics["source1_records"], 15000)
        self.assertEqual(len(result.matching_results), 15000)
        self.assertEqual(result.metrics["predicted_matches"], 15000)
        self.assertEqual(result.metrics["singleton_predictions"], 15000)
        self.assertLess(result.metrics["candidate_pairs"], 15000 * (15000 + 15000))


if __name__ == "__main__":
    unittest.main()