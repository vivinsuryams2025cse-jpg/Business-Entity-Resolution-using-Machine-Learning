import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, LABEL_COLUMN, SOURCE1_PAIR_COLUMN
from src.model import (
    DEFAULT_THRESHOLDS,
    RANDOM_STATE,
    ModelBundle,
    compare_models,
    evaluate_model,
    f0_5_score,
    load_model_bundle,
    save_model_bundle,
    split_by_source1_group,
    tune_threshold,
)
from src.train_model import _training_fold_sources
from src.features import FeatureConfig


class F05AndThresholdTests(unittest.TestCase):
    def test_f05_uses_the_challenge_formula(self):
        precision = 0.8
        recall = 0.5
        expected = (1.25 * precision * recall) / (0.25 * precision + recall)

        self.assertAlmostEqual(f0_5_score(precision, recall), expected)

    def test_zero_precision_and_recall_returns_zero(self):
        self.assertEqual(f0_5_score(0.0, 0.0), 0.0)

    def test_threshold_tuning_uses_only_provided_validation_scores(self):
        labels = np.array([1, 1, 0, 0], dtype=np.int8)
        probabilities = np.array([0.90, 0.60, 0.55, 0.10])

        results, best_result = tune_threshold(labels, probabilities)

        self.assertEqual([row["threshold"] for row in results], list(DEFAULT_THRESHOLDS))
        self.assertEqual(best_result["threshold"], 0.60)
        self.assertEqual(best_result["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertAlmostEqual(best_result["f0_5"], 1.0)


class GroupSplitTests(unittest.TestCase):
    def test_source1_groups_never_cross_train_and_validation(self):
        rows = []
        for group_number in range(40):
            rows.extend(
                [
                    {SOURCE1_PAIR_COLUMN: f"S1-{group_number}", LABEL_COLUMN: 1},
                    {SOURCE1_PAIR_COLUMN: f"S1-{group_number}", LABEL_COLUMN: 0},
                ]
            )
        labeled_pairs = pd.DataFrame(rows)

        training, validation = split_by_source1_group(
            labeled_pairs, random_state=RANDOM_STATE
        )

        self.assertFalse(
            set(training[SOURCE1_PAIR_COLUMN])
            & set(validation[SOURCE1_PAIR_COLUMN])
        )
        self.assertEqual(set(training[LABEL_COLUMN]), {0, 1})
        self.assertEqual(set(validation[LABEL_COLUMN]), {0, 1})
        repeated_training, repeated_validation = split_by_source1_group(
            labeled_pairs, random_state=RANDOM_STATE
        )
        pd.testing.assert_frame_equal(training, repeated_training)
        pd.testing.assert_frame_equal(validation, repeated_validation)

    def test_validation_entity_text_is_excluded_from_tfidf_training_corpus(self):
        source1 = pd.DataFrame(
            {
                "entity_id": ["S1-1", "S1-2"],
                "business_name": ["training-only", "validation-only"],
            }
        )
        source2 = pd.DataFrame(
            {
                "entity_id": ["S2-1", "S2-2"],
                "business_name": ["training-match", "validation-match"],
            }
        )
        source3 = pd.DataFrame(
            {
                "entity_id": ["S3-1", "S3-2"],
                "business_name": ["training-branch", "validation-branch"],
            }
        )
        ground_truth = pd.DataFrame(
            {
                "source1_entity_id": ["S1-1", "S1-2"],
                "source2_entity_id": ["S2-1", "S2-2"],
                "source3_entity_id": ["S3-1", "S3-2"],
            }
        )
        training_pairs = pd.DataFrame(
            {SOURCE1_PAIR_COLUMN: ["S1-1"], LABEL_COLUMN: [1]}
        )

        fit_source1, fit_source2, fit_source3 = _training_fold_sources(
            source1,
            source2,
            source3,
            ground_truth,
            training_pairs,
            FeatureConfig(),
        )

        self.assertEqual(fit_source1["entity_id"].tolist(), ["S1-1"])
        self.assertEqual(fit_source2["entity_id"].tolist(), ["S2-1"])
        self.assertEqual(fit_source3["entity_id"].tolist(), ["S3-1"])


def make_labeled_features(group_count=40):
    rows = []
    for group_number in range(group_count):
        for label in (1, 0):
            features = {column: 0.0 for column in FEATURE_COLUMNS}
            similarity = 0.9 if label else 0.1
            features.update(
                {
                    "name_ratio": similarity * 100,
                    "name_partial_ratio": similarity * 100,
                    "name_token_sort": similarity * 100,
                    "name_token_set": similarity * 100,
                    "name_edit_similarity": similarity,
                    "name_jaccard": similarity,
                    "name_tfidf": similarity,
                    "address_ratio": similarity * 100,
                    "address_partial_ratio": similarity * 100,
                    "address_token_similarity": similarity * 100,
                    "address_jaccard": similarity,
                    "address_tfidf": similarity,
                    "country_match": float(bool(label)),
                }
            )
            rows.append(
                {
                    SOURCE1_PAIR_COLUMN: f"S1-{group_number}",
                    "candidate_entity_id": f"T-{group_number}-{label}",
                    **features,
                    LABEL_COLUMN: label,
                }
            )
    return pd.DataFrame(rows)


class ModelSelectionTests(unittest.TestCase):
    def test_compares_two_models_and_selects_from_validation_f05(self):
        labeled_features = make_labeled_features()
        training, validation = split_by_source1_group(labeled_features)

        comparison = compare_models(
            training,
            validation,
            feature_engineer=None,
            feature_config={"name_column": "business_name"},
            random_state=RANDOM_STATE,
        )

        self.assertEqual(
            set(comparison.best_model_results), {"logistic_regression", "random_forest"}
        )
        self.assertIn(comparison.selected_model_name, comparison.best_model_results)
        self.assertIn(comparison.selected_threshold, DEFAULT_THRESHOLDS)
        self.assertEqual(
            set(comparison.threshold_results[comparison.selected_model_name][0]),
            {
                "threshold",
                "precision",
                "recall",
                "f0_5",
                "confusion_matrix",
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
                "records",
                "positive_records",
            },
        )

    def test_saved_bundle_retains_threshold_and_evaluates_without_retuning(self):
        labeled_features = make_labeled_features()
        training, validation = split_by_source1_group(labeled_features)
        comparison = compare_models(
            training,
            validation,
            feature_engineer=None,
            feature_config={"name_column": "business_name"},
            random_state=RANDOM_STATE,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.pkl"
            save_model_bundle(comparison.bundle, model_path)
            loaded_bundle = load_model_bundle(model_path)

        self.assertEqual(loaded_bundle.threshold, comparison.selected_threshold)
        metrics = evaluate_model(loaded_bundle, validation)
        self.assertEqual(metrics["threshold"], comparison.selected_threshold)
        self.assertIn("confusion_matrix", metrics)

    def test_test_like_features_without_labels_cannot_be_evaluated(self):
        bundle = ModelBundle(
            model_name="unused",
            estimator=None,
            threshold=0.5,
            feature_columns=list(FEATURE_COLUMNS),
            feature_config={},
            feature_engineer=None,
            random_state=RANDOM_STATE,
            validation_summary={},
        )

        with self.assertRaisesRegex(ValueError, "is_match"):
            evaluate_model(bundle, pd.DataFrame(columns=FEATURE_COLUMNS))


if __name__ == "__main__":
    unittest.main()