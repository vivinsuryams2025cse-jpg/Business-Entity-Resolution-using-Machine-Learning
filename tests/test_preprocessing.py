import unittest

import pandas as pd

from src.preprocessing import (
    normalize_business_address,
    normalize_business_name,
    normalize_country,
    preprocess_business_records,
)


class BusinessNameTests(unittest.TestCase):
    def test_normalizes_case_punctuation_abbreviations_and_legal_suffix(self):
        self.assertEqual(
            normalize_business_name("  North & West Intl., Inc.  "),
            "north and west international inc",
        )

    def test_normalizes_long_legal_suffix(self):
        self.assertEqual(
            normalize_business_name("Example Limited Liability Company"),
            "example llc",
        )

    def test_missing_business_name(self):
        self.assertIsNone(normalize_business_name(None))
        self.assertIsNone(normalize_business_name(float("nan")))


class BusinessAddressTests(unittest.TestCase):
    def test_normalizes_punctuation_whitespace_and_address_abbreviations(self):
        self.assertEqual(
            normalize_business_address("  24-B, Main St., Suite # 5  "),
            "24 b main street suite 5",
        )

    def test_missing_address(self):
        self.assertIsNone(normalize_business_address(pd.NA))


class CountryTests(unittest.TestCase):
    def test_normalizes_case_and_whitespace_for_france(self):
        self.assertEqual(normalize_country("  fRaNcE  "), "france")

    def test_does_not_restrict_country_values(self):
        self.assertEqual(normalize_country("  Côte d'Ivoire "), "côte d'ivoire")

    def test_missing_country(self):
        self.assertIsNone(normalize_country(None))


class DataFramePreprocessingTests(unittest.TestCase):
    def test_preserves_original_values_and_adds_normalized_columns(self):
        records = pd.DataFrame(
            {
                "business_name": [" Acme, Inc. "],
                "business_address": ["10 Main St."],
                "country": [" France "],
                "entity_id": ["S1-001"],
            }
        )

        processed = preprocess_business_records(records)

        self.assertEqual(processed.loc[0, "business_name"], " Acme, Inc. ")
        self.assertEqual(processed.loc[0, "business_name_normalized"], "acme inc")
        self.assertEqual(processed.loc[0, "business_address_normalized"], "10 main street")
        self.assertEqual(processed.loc[0, "country_normalized"], "france")
        self.assertEqual(processed.loc[0, "entity_id"], "S1-001")
        self.assertEqual(records.columns.tolist(), [
            "business_name",
            "business_address",
            "country",
            "entity_id",
        ])

    def test_supports_custom_column_names_and_keeps_all_rows(self):
        records = pd.DataFrame(
            {
                "name": ["One LLC", "Two Ltd."],
                "address": ["1 Oak Rd", "2 Pine Ave"],
                "nation": ["France", "Canada"],
            }
        )

        processed = preprocess_business_records(
            records,
            name_column="name",
            address_column="address",
            country_column="nation",
        )

        self.assertEqual(len(processed), 2)
        self.assertEqual(processed["name_normalized"].tolist(), ["one llc", "two ltd"])
        self.assertEqual(processed["address_normalized"].tolist(), ["1 oak road", "2 pine avenue"])
        self.assertEqual(processed["nation_normalized"].tolist(), ["france", "canada"])

    def test_processes_15000_rows_without_truncation(self):
        records = pd.DataFrame(
            {
                "business_name": ["Example Ltd."] * 15000,
                "business_address": ["1 Main St."] * 15000,
                "country": ["France"] * 15000,
                "entity_id": [f"S1-{record_number}" for record_number in range(15000)],
            }
        )

        processed = preprocess_business_records(records)

        self.assertEqual(len(processed), 15000)
        self.assertEqual(processed.loc[14999, "entity_id"], "S1-14999")
        self.assertEqual(processed.loc[14999, "business_name_normalized"], "example ltd")


if __name__ == "__main__":
    unittest.main()