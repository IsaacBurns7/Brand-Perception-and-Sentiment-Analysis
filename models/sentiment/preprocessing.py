"""Reusable preprocessing helpers for brand sentiment text data."""

from __future__ import annotations

import re

import pandas as pd

from .config import CLEAN_TEXT_COLUMN, EXCLUDED_LABEL


URL_PATTERN = re.compile(r"http\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
LEADING_RT_PATTERN = re.compile(r"^\s*rt\b[:\s-]*", flags=re.IGNORECASE)
LINK_TOKEN_PATTERN = re.compile(r"\{link\}", flags=re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize raw tweet text for classical ML models.

    The cleaning keeps most lexical content intact while removing common
    Twitter artifacts such as URLs, mentions, leading retweet markers,
    and the literal ``{link}`` placeholder.
    """

    if not isinstance(text, str):
        return ""

    cleaned = text.lower()
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = MENTION_PATTERN.sub(" ", cleaned)
    cleaned = LEADING_RT_PATTERN.sub("", cleaned)
    cleaned = LINK_TOKEN_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip()


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str,
    label_col: str | None = None,
) -> pd.DataFrame:
    """Return a cleaned copy of a dataframe with a ``clean_text`` column.

    Args:
        df: Input dataframe.
        text_col: Name of the raw text column.
        label_col: Optional label column. When provided, rows with the
            excluded rare label are removed before training.

    Returns:
        A cleaned dataframe copy with normalized text, empty rows removed,
        duplicate cleaned texts removed, and indices reset.
    """

    if text_col not in df.columns:
        raise KeyError(f"Text column '{text_col}' was not found in dataframe.")

    if label_col is not None and label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' was not found in dataframe.")

    cleaned_df = df.copy()
    cleaned_df[CLEAN_TEXT_COLUMN] = (
        cleaned_df[text_col].fillna("").astype(str).map(clean_text)
    )

    if label_col is not None:
        cleaned_df = cleaned_df[cleaned_df[label_col] != EXCLUDED_LABEL]

    cleaned_df = cleaned_df[cleaned_df[CLEAN_TEXT_COLUMN] != ""]
    cleaned_df = cleaned_df.drop_duplicates(subset=[CLEAN_TEXT_COLUMN])
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df
