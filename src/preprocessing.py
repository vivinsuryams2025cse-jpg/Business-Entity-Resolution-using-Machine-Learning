"""Deterministic text preprocessing for business entity records."""

import re
import unicodedata

import pandas as pd


_BUSINESS_ABBREVIATIONS = {
    "intl": "international",
    "mfg": "manufacturing",
    "svc": "services",
    "svcs": "services",
    "tech": "technology",
    "assn": "association",
    "bros": "brothers",
}

_ADDRESS_ABBREVIATIONS = {
    "st": "street",
    "str": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "hwy": "highway",
    "pkwy": "parkway",
    "apt": "apartment",
    "ste": "suite",
    "bldg": "building",
    "fl": "floor",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}

_LEGAL_SUFFIXES = {
    ("inc",): "inc",
    ("incorporated",): "inc",
    ("corp",): "corp",
    ("corporation",): "corp",
    ("co",): "co",
    ("company",): "co",
    ("ltd",): "ltd",
    ("limited",): "ltd",
    ("llc",): "llc",
    ("l", "l", "c"): "llc",
    ("limited", "liability", "company"): "llc",
    ("llp",): "llp",
    ("l", "l", "p"): "llp",
    ("limited", "liability", "partnership"): "llp",
    ("lp",): "lp",
    ("l", "p"): "lp",
    ("limited", "partnership"): "lp",
    ("plc",): "plc",
    ("p", "l", "c"): "plc",
    ("public", "limited", "company"): "plc",
    ("gmbh",): "gmbh",
    ("g", "m", "b", "h"): "gmbh",
    ("sarl",): "sarl",
    ("s", "a", "r", "l"): "sarl",
    ("sas",): "sas",
    ("s", "a", "s"): "sas",
    ("sa",): "sa",
    ("s", "a"): "sa",
    ("bv",): "bv",
    ("b", "v"): "bv",
    ("nv",): "nv",
    ("n", "v"): "nv",
}
_LEGAL_SUFFIX_ALIASES = sorted(
    _LEGAL_SUFFIXES.items(), key=lambda item: len(item[0]), reverse=True
)


def _is_missing(value: object) -> bool:
    """Return whether a scalar pandas-compatible value is missing."""
    return bool(pd.isna(value))


def _tokenize(value: object) -> list[str]:
    """Lowercase text and replace punctuation with spaces, keeping '&' as 'and'."""
    if _is_missing(value):
        return []

    text = str(value).casefold()
    characters = []
    for character in text:
        if character == "&":
            characters.append(" and ")
        elif unicodedata.category(character).startswith("P"):
            characters.append(" ")
        else:
            characters.append(character)
    return "".join(characters).split()


def _normalize_legal_suffix(tokens: list[str]) -> list[str]:
    """Replace a recognized trailing legal suffix with its canonical token."""
    for suffix, canonical in _LEGAL_SUFFIX_ALIASES:
        suffix_length = len(suffix)
        if tuple(tokens[-suffix_length:]) == suffix:
            return [*tokens[:-suffix_length], canonical]
    return tokens


def normalize_business_name(value: object) -> str | None:
    """Normalize a business name while leaving the source value untouched.

    Common business abbreviations are expanded. Recognized legal suffixes are
    represented by stable short forms such as ``inc``, ``ltd``, and ``llc``.
    Missing scalar values return ``None``.
    """
    if _is_missing(value):
        return None

    tokens = _tokenize(value)
    tokens = [_BUSINESS_ABBREVIATIONS.get(token, token) for token in tokens]
    return " ".join(_normalize_legal_suffix(tokens))


def normalize_business_address(value: object) -> str | None:
    """Normalize address punctuation, whitespace, and common abbreviations.

    Numbers, unit identifiers, and other useful address tokens are retained.
    Missing scalar values return ``None``.
    """
    if _is_missing(value):
        return None

    tokens = _tokenize(value)
    return " ".join(_ADDRESS_ABBREVIATIONS.get(token, token) for token in tokens)


def normalize_country(value: object) -> str | None:
    """Normalize country case and whitespace without restricting country names."""
    if _is_missing(value):
        return None
    return " ".join(str(value).casefold().split())


def preprocess_business_records(
    records: pd.DataFrame,
    *,
    name_column: str = "business_name",
    address_column: str = "business_address",
    country_column: str = "country",
) -> pd.DataFrame:
    """Return a copy with normalized columns, retaining every original column.

    The function processes every input row and does not cap or discard records.
    Column names are configurable to support variations in challenge file schemas.
    Normalized values are written to ``<column>_normalized`` columns.
    """
    processed = records.copy()
    normalizers = {
        name_column: normalize_business_name,
        address_column: normalize_business_address,
        country_column: normalize_country,
    }

    for column, normalizer in normalizers.items():
        if column in processed.columns:
            processed[f"{column}_normalized"] = processed[column].map(normalizer)

    return processed