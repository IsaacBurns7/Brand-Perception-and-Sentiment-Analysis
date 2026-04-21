from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".mplconfig"))

from pipeline.storage import (
    DEFAULT_PROCESSED_TABLE,
    DEFAULT_STORAGE_BACKEND,
    ensure_processed_documents_available,
    normalize_processed_documents,
    resolve_storage_backend,
)
from temporal_aggregation import run_temporal_aggregation


def _to_iso8601(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def get_summary_metrics(processed_documents: pd.DataFrame) -> dict[str, Any]:
    df = normalize_processed_documents(processed_documents)
    if df.empty:
        return {
            "document_count": 0,
            "source_count": 0,
            "topic_count": 0,
            "brand_count": 0,
            "aspect_count": 0,
            "average_sentiment": 0.0,
            "positive_share": 0.0,
            "negative_share": 0.0,
            "neutral_share": 0.0,
            "first_document_at": None,
            "last_document_at": None,
        }

    label_share = df["sentiment_label"].value_counts(normalize=True)
    return {
        "document_count": int(df["doc_id"].nunique()),
        "row_count": int(len(df)),
        "source_count": int(df["source"].nunique()),
        "topic_count": int(df["topic"].nunique()),
        "brand_count": int(df["brand"].nunique()),
        "aspect_count": int(df["aspect"].nunique()),
        "average_sentiment": round(float(df["sentiment"].mean()), 4),
        "positive_share": round(float(label_share.get("positive", 0.0)), 4),
        "negative_share": round(float(label_share.get("negative", 0.0)), 4),
        "neutral_share": round(float(label_share.get("neutral", 0.0)), 4),
        "first_document_at": _to_iso8601(df["created_utc"].min()),
        "last_document_at": _to_iso8601(df["created_utc"].max()),
    }


def get_sentiment_over_time(
    processed_documents: pd.DataFrame,
    *,
    rolling_window_days: int = 7,
) -> list[dict[str, Any]]:
    df = normalize_processed_documents(processed_documents)
    if df.empty:
        return []

    daily, _ = run_temporal_aggregation(df, rolling_window_days=rolling_window_days)
    rows = daily[["day", "doc_count", "avg_sentiment", "sentiment_7d", "count_7d"]].copy()
    rows["day"] = rows["day"].apply(_to_iso8601)
    rows["avg_sentiment"] = rows["avg_sentiment"].round(4)
    rows["sentiment_7d"] = rows["sentiment_7d"].round(4)
    rows["count_7d"] = rows["count_7d"].round(4)
    rows = rows.where(pd.notnull(rows), None)
    return rows.to_dict(orient="records")


def get_source_breakdown(
    processed_documents: pd.DataFrame,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    df = normalize_processed_documents(processed_documents)
    if df.empty:
        return []

    grouped = (
        df.groupby("source", as_index=False)
        .agg(
            document_count=("doc_id", "nunique"),
            row_count=("doc_id", "count"),
            average_sentiment=("sentiment", "mean"),
            topic_count=("topic", "nunique"),
        )
        .sort_values(["document_count", "source"], ascending=[False, True])
        .head(limit)
        .reset_index(drop=True)
    )
    grouped["average_sentiment"] = grouped["average_sentiment"].round(4)
    return grouped.to_dict(orient="records")


def get_topic_breakdown(
    processed_documents: pd.DataFrame,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    df = normalize_processed_documents(processed_documents)
    if df.empty:
        return []

    grouped = (
        df.groupby("topic", as_index=False)
        .agg(
            document_count=("doc_id", "nunique"),
            row_count=("doc_id", "count"),
            average_sentiment=("sentiment", "mean"),
            source_count=("source", "nunique"),
        )
        .sort_values(["document_count", "topic"], ascending=[False, True])
        .head(limit)
        .reset_index(drop=True)
    )
    grouped["average_sentiment"] = grouped["average_sentiment"].round(4)
    return grouped.to_dict(orient="records")


def get_aspect_breakdown(
    processed_documents: pd.DataFrame,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    df = normalize_processed_documents(processed_documents)
    if df.empty:
        return []

    grouped = (
        df.groupby("aspect", as_index=False)
        .agg(
            document_count=("doc_id", "nunique"),
            row_count=("doc_id", "count"),
            average_sentiment=("sentiment", "mean"),
            topic_count=("topic", "nunique"),
            brand_count=("brand", "nunique"),
        )
        .sort_values(["document_count", "aspect"], ascending=[False, True])
        .head(limit)
        .reset_index(drop=True)
    )
    grouped["average_sentiment"] = grouped["average_sentiment"].round(4)
    return grouped.to_dict(orient="records")


def get_changepoints(
    processed_documents: pd.DataFrame,
    *,
    rolling_window_days: int = 7,
    penalty: int | float = 3,
) -> list[dict[str, Any]]:
    df = normalize_processed_documents(processed_documents)
    if df.empty:
        return []

    daily, changepoints = run_temporal_aggregation(
        df,
        rolling_window_days=rolling_window_days,
        changepoint_penalty=penalty,
    )
    daily_lookup = daily.set_index("day")

    results: list[dict[str, Any]] = []
    for changepoint in changepoints:
        row = daily_lookup.loc[changepoint]
        results.append(
            {
                "day": _to_iso8601(changepoint),
                "avg_sentiment": round(float(row["avg_sentiment_filled"]), 4),
                "doc_count": int(row["doc_count"]),
            }
        )
    return results


@dataclass(slots=True)
class AnalyticsService:
    backend: str = DEFAULT_STORAGE_BACKEND
    path: str | None = None
    table_name: str = DEFAULT_PROCESSED_TABLE
    use_sample_data: bool = True
    persist_sample: bool = False

    def resolved_backend(self) -> str:
        return resolve_storage_backend(self.backend)

    def load_processed_documents(self) -> pd.DataFrame:
        return ensure_processed_documents_available(
            backend=self.backend,
            path=self.path,
            table_name=self.table_name,
            use_sample_data=self.use_sample_data,
            persist_sample=self.persist_sample,
        )

    def summary_metrics(self) -> dict[str, Any]:
        return get_summary_metrics(self.load_processed_documents())

    def sentiment_over_time(self, *, rolling_window_days: int = 7) -> list[dict[str, Any]]:
        return get_sentiment_over_time(
            self.load_processed_documents(),
            rolling_window_days=rolling_window_days,
        )

    def source_breakdown(self, *, limit: int = 10) -> list[dict[str, Any]]:
        return get_source_breakdown(self.load_processed_documents(), limit=limit)

    def topic_breakdown(self, *, limit: int = 10) -> list[dict[str, Any]]:
        return get_topic_breakdown(self.load_processed_documents(), limit=limit)

    def aspect_breakdown(self, *, limit: int = 10) -> list[dict[str, Any]]:
        return get_aspect_breakdown(self.load_processed_documents(), limit=limit)

    def changepoints(
        self,
        *,
        rolling_window_days: int = 7,
        penalty: int | float = 3,
    ) -> list[dict[str, Any]]:
        return get_changepoints(
            self.load_processed_documents(),
            rolling_window_days=rolling_window_days,
            penalty=penalty,
        )
