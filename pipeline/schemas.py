from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import pandas as pd


RAW_DOCUMENT_COLUMNS = [
    "doc_id",
    "text",
    "source",
    "created_utc",
]

CLEAN_DOCUMENT_COLUMNS = [
    "doc_id",
    "text",
    "source",
    "created_utc",
]

PROCESSED_DOCUMENT_COLUMNS = [
    "doc_id",
    "text",
    "brand",
    "aspect",
    "sentiment",
    "sentiment_label",
    "topic",
    "source",
    "created_utc",
]


@dataclass(slots=True)
class RawDocument:
    doc_id: str
    text: str
    source: str
    created_utc: datetime | str | None


@dataclass(slots=True)
class CleanDocument:
    doc_id: str
    text: str
    source: str
    created_utc: datetime | str | None


@dataclass(slots=True)
class ProcessedDocument:
    doc_id: str
    text: str
    brand: str
    aspect: str
    sentiment: float
    sentiment_label: str
    topic: str
    source: str
    created_utc: datetime | str | None


def _rows_to_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    for column in columns:
        if column not in df.columns:
            df[column] = None
    return df.loc[:, columns]


def raw_documents_to_frame(documents: list[RawDocument]) -> pd.DataFrame:
    return _rows_to_frame([asdict(document) for document in documents], RAW_DOCUMENT_COLUMNS)


def clean_documents_to_frame(documents: list[CleanDocument]) -> pd.DataFrame:
    return _rows_to_frame([asdict(document) for document in documents], CLEAN_DOCUMENT_COLUMNS)


def processed_documents_to_frame(documents: list[ProcessedDocument]) -> pd.DataFrame:
    return _rows_to_frame([asdict(document) for document in documents], PROCESSED_DOCUMENT_COLUMNS)
