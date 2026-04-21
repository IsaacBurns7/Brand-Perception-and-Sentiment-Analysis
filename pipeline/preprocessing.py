from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import CLEAN_DOCUMENT_COLUMNS


_DEFAULT_TEXT_COLUMN_CANDIDATES = [
    "full_content",
    "article",
    "content",
    "text",
]


def _load_existing_cleaner():
    cleaner_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "Topic-Modeling"
        / "preprocessing"
        / "cleaning_utils.py"
    )
    spec = importlib.util.spec_from_file_location("topic_cleaning_utils", cleaner_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load cleaning utilities from {cleaner_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.clean_text


_EXISTING_CLEAN_TEXT = _load_existing_cleaner()


def clean_text(text: str) -> str:
    """Shared text cleaning entrypoint for downstream stages."""
    return _EXISTING_CLEAN_TEXT(text)


def resolve_text_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    candidates = [preferred] if preferred else []
    candidates.extend(_DEFAULT_TEXT_COLUMN_CANDIDATES)

    for column in candidates:
        if column and column in df.columns:
            return column

    if "title" in df.columns:
        return "title"

    raise ValueError("No usable text column found in input dataframe.")


def build_raw_text(row: pd.Series, text_column: str) -> str:
    text = str(row.get(text_column, "") or "")
    if len(text.strip()) < 20:
        title = str(row.get("title", "") or "")
        description = str(row.get("description", "") or "")
        text = " ".join(part for part in [title, description, text] if part).strip()
    return text


def clean_documents(
    documents: pd.DataFrame,
    *,
    text_column: str | None = None,
    doc_id_column: str = "doc_id",
    source_column: str = "source_name",
    created_column: str = "published_at",
) -> pd.DataFrame:
    """
    Normalize raw article-like rows into the canonical clean-document shape.
    """
    if documents.empty:
        return pd.DataFrame(columns=CLEAN_DOCUMENT_COLUMNS)

    df = documents.copy()
    resolved_text_column = resolve_text_column(df, preferred=text_column)

    if doc_id_column not in df.columns:
        if "url" in df.columns:
            doc_id_column = "url"
        elif "id" in df.columns:
            doc_id_column = "id"
        else:
            df[doc_id_column] = df.index.astype(str)

    if source_column not in df.columns:
        source_column = "source" if "source" in df.columns else source_column
    if created_column not in df.columns:
        created_column = "created_utc" if "created_utc" in df.columns else created_column

    clean_df = pd.DataFrame(
        {
            "doc_id": df[doc_id_column].astype(str),
            "text": df.apply(lambda row: clean_text(build_raw_text(row, resolved_text_column)), axis=1),
            "source": df[source_column].fillna("unknown").astype(str) if source_column in df.columns else "unknown",
            "created_utc": pd.to_datetime(
                df[created_column] if created_column in df.columns else pd.NaT,
                utc=True,
                errors="coerce",
            ),
        }
    )

    clean_df["text"] = clean_df["text"].fillna("").astype(str)
    clean_df = clean_df.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
    return clean_df.loc[:, CLEAN_DOCUMENT_COLUMNS]


def clean_document_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    return clean_documents(pd.DataFrame(records))
