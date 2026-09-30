import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.blocking import (
    BlockingConfig,
    OUTPUT_COLUMNS,
    generate_candidate_pairs,
    write_candidate_pairs,
)


def make_records(rows):
    return pd.DataFrame(
        rows,
        columns=["entity_id", "business_name", "business_address", "country"],
    )


class CandidateGenerationTests(unittest.TestCase):
    def test_generates_candidates_for_both_target_sources_and_deduplicates(self):
        source1 = make_records(
            [("S1-1", "Acme Tech Ltd.", "10 Main St, Suite 4", "France")]
        )
        source2 = make_records(
            [("S2-1", "Acme Technology Limited", "10 Main Street Ste 4", "France")]
        )
        source3 = make_records(
            [("S3-1", "Acme Technology", "", "France")]
        )

        result = generate_candidate_pairs(source1, source2, source3)

        self.assertEqual(result.candidate_pairs.columns.tolist(), OUTPUT_COLUMNS)
        self.assertEqual(
            set(map(tuple, result.candidate_pairs.to_numpy())),
            {("S1-1", "S2-1"), ("S1-1", "S3-1")},
        )
        self.assertEqual(len(result.candidate_pairs), 2)
        self.assertEqual(result.metrics["source1_with_source2_candidates"], 1)
        self.assertEqual(result.metrics["source1_with_source3_candidates"], 1)

    def test_name_prefix_and_address_tokens_recover_nonexact_names(self):
        source1 = make_records(
            [("S1-1", "Northwind Manufacturing", "7 Market Road", "")]
        )
        source2 = make_records(
            [("S2-1", "Northwin Manufacturing", "", "")]
        )
        source3 = make_records(
            [("S3-1", "Completely Different", "7 Market Rd", "")]
        )

        result = generate_candidate_pairs(source1, source2, source3)

        self.assertEqual(
            set(result.candidate_pairs["candidate_entity_id"]), {"S2-1", "S3-1"}
        )

    def test_exact_country_block_finds_a_candidate_without_shared_name_or_address(self):
        source1 = make_records([("S1-1", "Alpha Company", "", "  France ")])
        source2 = make_records([("S2-1", "Beta Enterprise", "", "france")])
        source3 = make_records([("S3-1", "Gamma Group", "", "Canada")])

        result = generate_candidate_pairs(source1, source2, source3)

        self.assertEqual(result.candidate_pairs["candidate_entity_id"].tolist(), ["S2-1"])

    def test_oversized_blocks_are_skipped_and_reported(self):
        source1 = make_records([("S1-1", "Alpha Company", "", "France")])
        source2 = make_records(
            [
                ("S2-1", "Beta Enterprise", "", "France"),
                ("S2-2", "Gamma Group", "", "France"),
            ]
        )
        source3 = make_records([])

        result = generate_candidate_pairs(
            source1,
            source2,
            source3,
            config=BlockingConfig(max_block_pair_count=1),
        )

        self.assertTrue(result.candidate_pairs.empty)
        self.assertGreater(result.metrics["oversized_blocks_skipped"], 0)
        self.assertEqual(result.metrics["total_possible_pairs"], 2)

    def test_candidate_cap_is_visible_in_metrics(self):
        source1 = make_records([("S1-1", "shared name", "", "")])
        source2 = make_records(
            [(f"S2-{number}", "shared name", "", "") for number in range(1, 5)]
        )
        source3 = make_records([])

        result = generate_candidate_pairs(
            source1,
            source2,
            source3,
            config=BlockingConfig(max_candidates_per_source1_per_source=2),
        )

        self.assertEqual(len(result.candidate_pairs), 2)
        self.assertEqual(result.metrics["source1_truncated_for_source2"], 1)

    def test_metrics_measure_possible_pairs_candidate_pairs_and_reduction(self):
        source1 = make_records([("S1-1", "Acme Services", "1 Pine Road", "France")])
        source2 = make_records([("S2-1", "Acme Service", "1 Pine Rd", "France")])
        source3 = make_records([("S3-1", "No Match", "", "")])

        result = generate_candidate_pairs(source1, source2, source3)

        self.assertEqual(result.metrics["total_possible_pairs"], 2)
        self.assertEqual(result.metrics["candidate_pairs"], 1)
        self.assertEqual(result.metrics["reduction_ratio"], 0.5)
        self.assertGreaterEqual(result.metrics["generation_time_seconds"], 0)

    def test_writer_outputs_required_tsv_columns(self):
        candidate_pairs = pd.DataFrame(
            [("S1-1", "S2-1")], columns=OUTPUT_COLUMNS
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            output_path = write_candidate_pairs(
                candidate_pairs, Path(temp_directory) / "candidate_pairs.tsv"
            )
            loaded = pd.read_csv(output_path, sep="\t")

        self.assertEqual(loaded.columns.tolist(), OUTPUT_COLUMNS)
        self.assertEqual(loaded.iloc[0].tolist(), ["S1-1", "S2-1"])

    def test_15000_records_generate_a_bounded_candidate_table(self):
        row_numbers = range(15000)
        source1 = make_records(
            [
                (f"S1-{number}", f"Enterprise {number:05d}", f"{number} Oak Road", "France")
                for number in row_numbers
            ]
        )
        source2 = make_records(
            [
                (f"S2-{number}", f"Enterprise {number:05d}", f"{number} Oak Road", "France")
                for number in range(15000)
            ]
        )
        source3 = make_records(
            [
                (f"S3-{number}", f"Enterprise {number:05d}", f"{number} Oak Road", "France")
                for number in range(15000)
            ]
        )

        result = generate_candidate_pairs(source1, source2, source3)

        self.assertEqual(len(result.candidate_pairs), 30000)
        self.assertEqual(result.metrics["total_possible_pairs"], 450000000)
        self.assertGreater(result.metrics["reduction_ratio"], 0.9999)
        self.assertLessEqual(len(result.candidate_pairs), 15000 * 100 * 2)


if __name__ == "__main__":
    unittest.main()