import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import (
    FEATURE_COLUMNS,
    FeatureConfig,
    LABEL_COLUMN,
    PairFeatureEngineer,
    build_training_feature_dataset,
)


def make_records(rows):
    return pd.DataFrame(
        rows,
        columns=["entity_id", "business_name", "business_address", "country"],
    )


class PairFeatureEngineerTests(unittest.TestCase):
    def setUp(self):
        self.source1 = make_records(
            [
                ("S1-1", "Northwind Manufacturing Ltd", "12 Main Street", "France"),
                ("S1-2", None, None, None),
            ]
        )
        self.source2 = make_records(
            [
                ("S2-1", "Northwind Mfg. Limited", "12 Main St", "France"),
                ("S2-2", "Other Company", "8 Oak Road", "Canada"),
            ]
        )
        self.source3 = make_records(
            [("S3-1", "Unrelated Business", "99 Hill Road", "Germany")]
        )
        self.candidates = pd.DataFrame(
            [
                ("S1-1", "S2-1"),
                ("S1-1", "S2-2"),
                ("S1-1", "S3-1"),
                ("S1-2", "S2-2"),
            ],
            columns=["source1_entity_id", "candidate_entity_id"],
        )
        self.engineer = PairFeatureEngineer().fit(
            self.source1, self.source2, self.source3
        )

    def test_creates_numeric_features_for_every_candidate(self):
        features = self.engineer.transform_candidate_pairs(
            self.source1, self.source2, self.source3, self.candidates
        )

        self.assertEqual(len(features), len(self.candidates))
        self.assertEqual(features[FEATURE_COLUMNS].shape, (4, len(FEATURE_COLUMNS)))
        self.assertTrue(all(pd.api.types.is_numeric_dtype(features[column]) for column in FEATURE_COLUMNS))
        self.assertFalse(features[FEATURE_COLUMNS].isna().any().any())

    def test_similar_pair_has_higher_name_and_address_scores(self):
        features = self.engineer.transform_candidate_pairs(
            self.source1, self.source2, self.source3, self.candidates
        ).set_index(["source1_entity_id", "candidate_entity_id"])

        self.assertGreater(features.loc[("S1-1", "S2-1"), "name_ratio"], features.loc[("S1-1", "S2-2"), "name_ratio"])
        self.assertGreater(features.loc[("S1-1", "S2-1"), "name_jaccard"], features.loc[("S1-1", "S2-2"), "name_jaccard"])
        self.assertGreater(features.loc[("S1-1", "S2-1"), "address_common_tokens"], 0)
        self.assertEqual(features.loc[("S1-1", "S2-1"), "country_match"], 1)

    def test_missing_values_have_indicators_and_zero_similarity(self):
        features = self.engineer.transform_candidate_pairs(
            self.source1,
            self.source2,
            self.source3,
            self.candidates.iloc[[3]],
        ).iloc[0]

        self.assertEqual(features["name_missing_source1"], 1)
        self.assertEqual(features["address_missing_source1"], 1)
        self.assertEqual(features["country_missing_source1"], 1)
        self.assertEqual(features["name_ratio"], 0)
        self.assertEqual(features["address_tfidf"], 0)

    def test_duplicate_candidate_pairs_are_removed(self):
        duplicated_pairs = pd.concat([self.candidates.iloc[[0]]] * 3, ignore_index=True)

        features = self.engineer.transform_candidate_pairs(
            self.source1, self.source2, self.source3, duplicated_pairs
        )

        self.assertEqual(len(features), 1)

    def test_training_labels_come_from_source1_source2_and_source3_ground_truth(self):
        ground_truth = pd.DataFrame(
            {
                "source1_entity_id": ["S1-1", "S1-2"],
                "source2_entity_id": ["S2-1", None],
                "source3_entity_id": ["S3-1", "S3-1"],
            }
        )
        candidate_pairs = pd.concat(
            [
                self.candidates,
                pd.DataFrame(
                    [("S1-2", "S3-1")],
                    columns=["source1_entity_id", "candidate_entity_id"],
                ),
            ],
            ignore_index=True,
        )

        features = build_training_feature_dataset(
            self.source1,
            self.source2,
            self.source3,
            candidate_pairs,
            ground_truth,
            self.engineer,
        ).set_index(["source1_entity_id", "candidate_entity_id"])

        self.assertEqual(features.loc[("S1-1", "S2-1"), LABEL_COLUMN], 1)
        self.assertEqual(features.loc[("S1-1", "S2-2"), LABEL_COLUMN], 0)
        self.assertEqual(features.loc[("S1-1", "S3-1"), LABEL_COLUMN], 1)
        self.assertEqual(features.loc[("S1-2", "S3-1"), LABEL_COLUMN], 1)

    def test_ground_truth_id_columns_can_be_configured_separately(self):
        feature_engineer = PairFeatureEngineer(
            FeatureConfig(
                ground_truth_source1_id_column="left_record",
                ground_truth_source2_id_column="middle_record",
                ground_truth_source3_id_column="right_record",
            )
        ).fit(self.source1, self.source2, self.source3)
        ground_truth = pd.DataFrame(
            {
                "left_record": ["S1-1"],
                "middle_record": ["S2-1"],
                "right_record": ["S3-1"],
            }
        )

        features = build_training_feature_dataset(
            self.source1,
            self.source2,
            self.source3,
            self.candidates,
            ground_truth,
            feature_engineer,
        ).set_index(["source1_entity_id", "candidate_entity_id"])

        self.assertEqual(features.loc[("S1-1", "S2-1"), LABEL_COLUMN], 1)
        self.assertEqual(features.loc[("S1-1", "S3-1"), LABEL_COLUMN], 1)

    def test_test_transform_reuses_training_tfidf_vocabulary(self):
        fitted_vocabulary = set(self.engineer.name_vectorizer.vocabulary_)
        test_source1 = make_records(
            [("S1-T1", "RareNovel Term", "1 Unseen Boulevard", "France")]
        )
        test_source2 = make_records(
            [("S2-T1", "RareNovel Term", "1 Unseen Boulevard", "France")]
        )
        test_source3 = make_records([])
        test_pairs = pd.DataFrame(
            [("S1-T1", "S2-T1")],
            columns=["source1_entity_id", "candidate_entity_id"],
        )

        features = self.engineer.transform_candidate_pairs(
            test_source1, test_source2, test_source3, test_pairs
        )

        self.assertEqual(set(self.engineer.name_vectorizer.vocabulary_), fitted_vocabulary)
        self.assertEqual(features.loc[0, "name_tfidf"], 0)
        self.assertNotIn(LABEL_COLUMN, features.columns)

    def test_pipeline_can_be_saved_and_loaded(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            pipeline_path = Path(temp_directory) / "feature_pipeline.joblib"
            self.engineer.save(pipeline_path)
            loaded_engineer = PairFeatureEngineer.load(pipeline_path)

        original = self.engineer.transform_candidate_pairs(
            self.source1, self.source2, self.source3, self.candidates
        )
        loaded = loaded_engineer.transform_candidate_pairs(
            self.source1, self.source2, self.source3, self.candidates
        )
        pd.testing.assert_frame_equal(original, loaded)

    def test_unfitted_pipeline_cannot_transform(self):
        with self.assertRaises(RuntimeError):
            PairFeatureEngineer().transform_candidate_pairs(
                self.source1, self.source2, self.source3, self.candidates
            )

    def test_zero_candidate_table_has_stable_feature_schema(self):
        empty_candidates = self.candidates.iloc[:0]

        features = self.engineer.transform_candidate_pairs(
            self.source1, self.source2, self.source3, empty_candidates
        )

        self.assertEqual(features.columns.tolist(), [
            "source1_entity_id",
            "candidate_entity_id",
            *FEATURE_COLUMNS,
        ])
        self.assertTrue(features.empty)


if __name__ == "__main__":
    unittest.main()