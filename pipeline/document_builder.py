from __future__ import annotations

import os
from typing import Any

import pandas as pd

from models.absa import ABSAError, run_absa

from .schemas import PROCESSED_DOCUMENT_COLUMNS


DEFAULT_BRAND = "unknown"
DEFAULT_SENTIMENT = 0.0
DEFAULT_SENTIMENT_LABEL = "neutral"
DEFAULT_ASPECT = "general"
DEFAULT_TOPIC = "unclassified"
DEFAULT_SOURCE = "unknown"

_SENTIMENT_MAP = {
    "positive": 1,
    "neutral": 0,
    "negative": -1,
}


def build_stub_sentiment_output(clean_documents: pd.DataFrame) -> pd.DataFrame:
    if clean_documents.empty:
        return pd.DataFrame(columns=["doc_id", "sentiment", "sentiment_label"])

    return pd.DataFrame(
        {
            "doc_id": clean_documents["doc_id"].astype(str),
            "sentiment": DEFAULT_SENTIMENT,
            "sentiment_label": DEFAULT_SENTIMENT_LABEL,
        }
    )
def build_sentiment_output(clean_documents: pd.DataFrame) -> pd.DataFrame:
    if clean_documents.empty:
        return pd.DataFrame(columns=["doc_id", "sentiment", "sentiment_label"])
    

def build_stub_topic_output(clean_documents: pd.DataFrame) -> pd.DataFrame:
    if clean_documents.empty:
        return pd.DataFrame(columns=["doc_id", "topic"])

    return pd.DataFrame(
        {
            "doc_id": clean_documents["doc_id"].astype(str),
            "topic": DEFAULT_TOPIC,
        }
    )


def build_stub_absa_output(clean_documents: pd.DataFrame) -> pd.DataFrame:
    if clean_documents.empty:
        return pd.DataFrame(columns=["doc_id", "aspect", "sentiment", "sentiment_label"])

    return pd.DataFrame(
        {
            "doc_id": clean_documents["doc_id"].astype(str),
            "aspect": DEFAULT_ASPECT,
            "sentiment": DEFAULT_SENTIMENT,
            "sentiment_label": DEFAULT_SENTIMENT_LABEL,
        }
    )


def _coerce_ner_output(ner_output: pd.DataFrame | None, clean_documents: pd.DataFrame) -> pd.DataFrame:
    if ner_output is None or ner_output.empty:
        return pd.DataFrame(
            {
                "doc_id": clean_documents["doc_id"].astype(str),
                "brand": DEFAULT_BRAND,
            }
        )

    df = ner_output.copy()
    if "doc_id" not in df.columns:
        if "article_url" in df.columns:
            df["doc_id"] = df["article_url"].astype(str)
        elif "url" in df.columns:
            df["doc_id"] = df["url"].astype(str)
        else:
            raise ValueError("NER output must include doc_id or a compatible identifier column.")

    if "brand" in df.columns:
        brand_df = df[["doc_id", "brand"]].copy()
    elif "canonical_name" in df.columns:
        brand_df = df[["doc_id", "canonical_name"]].rename(columns={"canonical_name": "brand"})
    elif "ner_brands" in df.columns:
        brand_df = df[["doc_id", "ner_brands"]].explode("ner_brands").rename(columns={"ner_brands": "brand"})
    else:
        brand_df = pd.DataFrame({"doc_id": clean_documents["doc_id"].astype(str), "brand": DEFAULT_BRAND})

    brand_df["doc_id"] = brand_df["doc_id"].astype(str)
    brand_df["brand"] = brand_df["brand"].fillna(DEFAULT_BRAND).astype(str)
    brand_df.loc[brand_df["brand"].str.len() == 0, "brand"] = DEFAULT_BRAND
    return brand_df.drop_duplicates().reset_index(drop=True)


def _coerce_sentiment_output(sentiment_output: pd.DataFrame | None, clean_documents: pd.DataFrame) -> pd.DataFrame:
    df = build_sentiment_output(clean_documents) if sentiment_output is None or sentiment_output.empty else sentiment_output.copy()

    if "doc_id" not in df.columns:
        if "article_url" in df.columns:
            df["doc_id"] = df["article_url"].astype(str)
        else:
            raise ValueError("Sentiment output must include doc_id or article_url.")

    if "sentiment" not in df.columns and "sentiment_score" in df.columns:
        df["sentiment"] = df["sentiment_score"]
    if "sentiment_label" not in df.columns:
        df["sentiment_label"] = DEFAULT_SENTIMENT_LABEL

    out = df[["doc_id", "sentiment", "sentiment_label"]].copy()
    out["doc_id"] = out["doc_id"].astype(str)
    out["sentiment"] = pd.to_numeric(out["sentiment"], errors="coerce").fillna(DEFAULT_SENTIMENT)
    out["sentiment_label"] = out["sentiment_label"].fillna(DEFAULT_SENTIMENT_LABEL).astype(str)
    return out.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)


def _normalize_sentiment_label(value: Any) -> str:
    label = str(value or DEFAULT_SENTIMENT_LABEL).strip().lower()
    if label not in _SENTIMENT_MAP:
        return DEFAULT_SENTIMENT_LABEL
    return label


def _map_sentiment_value(label: str) -> int:
    return _SENTIMENT_MAP.get(_normalize_sentiment_label(label), DEFAULT_SENTIMENT)


def _build_absa_output(
    clean_documents: pd.DataFrame,
    *,
    absa_enabled: bool,
    sentiment_output: pd.DataFrame | None,
) -> pd.DataFrame:
    fallback = build_stub_absa_output(clean_documents)

    if not absa_enabled:
        sentiment_df = _coerce_sentiment_output(sentiment_output, clean_documents)
        sentiment_df["aspect"] = DEFAULT_ASPECT
        return sentiment_df[["doc_id", "aspect", "sentiment", "sentiment_label"]]

    try:
        triplet_batches = run_absa(clean_documents["text"].fillna("").astype(str).tolist())
    except ABSAError:
        return fallback

    rows: list[dict[str, Any]] = []
    doc_ids = clean_documents["doc_id"].astype(str).tolist()
    for index, doc_id in enumerate(doc_ids):
        triplets = triplet_batches[index] if index < len(triplet_batches) else []
        if not triplets:
            rows.append(
                {
                    "doc_id": doc_id,
                    "aspect": DEFAULT_ASPECT,
                    "sentiment": DEFAULT_SENTIMENT,
                    "sentiment_label": DEFAULT_SENTIMENT_LABEL,
                }
            )
            continue

        for aspect, _opinion, sentiment_label in triplets:
            normalized_label = _normalize_sentiment_label(sentiment_label)
            rows.append(
                {
                    "doc_id": doc_id,
                    "aspect": str(aspect or DEFAULT_ASPECT).strip() or DEFAULT_ASPECT,
                    "sentiment": _map_sentiment_value(normalized_label),
                    "sentiment_label": normalized_label,
                }
            )

    if not rows:
        return fallback

    return pd.DataFrame(rows, columns=["doc_id", "aspect", "sentiment", "sentiment_label"])


def _coerce_topic_output(topic_output: pd.DataFrame | None, clean_documents: pd.DataFrame) -> pd.DataFrame:
    df = build_stub_topic_output(clean_documents) if topic_output is None or topic_output.empty else topic_output.copy()

    if "doc_id" not in df.columns:
        if "article_url" in df.columns:
            df["doc_id"] = df["article_url"].astype(str)
        else:
            raise ValueError("Topic output must include doc_id or article_url.")

    if "topic" not in df.columns and "topic_name" in df.columns:
        df["topic"] = df["topic_name"]
    elif "topic" not in df.columns:
        df["topic"] = DEFAULT_TOPIC

    out = df[["doc_id", "topic"]].copy()
    out["doc_id"] = out["doc_id"].astype(str)
    out["topic"] = out["topic"].fillna(DEFAULT_TOPIC).astype(str)
    return out.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)


def build_processed_documents(
    clean_documents: pd.DataFrame,
    *,
    ner_output: pd.DataFrame | None = None,
    sentiment_output: pd.DataFrame | None = None,
    topic_output: pd.DataFrame | None = None,
    absa_enabled: bool | None = None,
) -> pd.DataFrame:
    """
    Join canonical cleaned documents with downstream model outputs.

    Produces one row per extracted aspect, preserving the existing brand join.
    If NER or ABSA are unavailable, fallback rows are emitted so the rest of
    the pipeline stays runnable.
    """
    if clean_documents.empty:
        return pd.DataFrame(columns=PROCESSED_DOCUMENT_COLUMNS)

    docs = clean_documents.copy()
    docs["doc_id"] = docs["doc_id"].astype(str)
    docs["text"] = docs["text"].fillna("").astype(str)
    if "source" not in docs.columns:
        docs["source"] = DEFAULT_SOURCE
    docs["source"] = docs["source"].fillna(DEFAULT_SOURCE).astype(str)
    docs["created_utc"] = pd.to_datetime(
        docs["created_utc"] if "created_utc" in docs.columns else pd.NaT,
        utc=True,
        errors="coerce",
    )
    if absa_enabled is None:
        absa_enabled = os.environ.get("BRAND_PERCEPTION_ENABLE_ABSA", "1") != "0"

    brand_df = _coerce_ner_output(ner_output, docs)
    sentiment_df = _build_absa_output(
        docs,
        absa_enabled=absa_enabled,
        sentiment_output=sentiment_output,
    )
    topic_df = _coerce_topic_output(topic_output, docs)

    processed = docs.merge(brand_df, on="doc_id", how="left")
    processed = processed.merge(sentiment_df, on="doc_id", how="left")
    processed = processed.merge(topic_df, on="doc_id", how="left")

    processed["brand"] = processed["brand"].fillna(DEFAULT_BRAND).astype(str)
    processed["aspect"] = processed["aspect"].fillna(DEFAULT_ASPECT).astype(str)
    processed["sentiment"] = pd.to_numeric(processed["sentiment"], errors="coerce").fillna(DEFAULT_SENTIMENT)
    processed["sentiment_label"] = processed["sentiment_label"].fillna(DEFAULT_SENTIMENT_LABEL).astype(str)
    processed["topic"] = processed["topic"].fillna(DEFAULT_TOPIC).astype(str)
    processed["source"] = processed["source"].fillna(DEFAULT_SOURCE).astype(str)

    return processed.loc[:, PROCESSED_DOCUMENT_COLUMNS].drop_duplicates().reset_index(drop=True)


def build_processed_document_records(
    clean_documents: list[dict[str, Any]],
    *,
    ner_output: list[dict[str, Any]] | None = None,
    sentiment_output: list[dict[str, Any]] | None = None,
    topic_output: list[dict[str, Any]] | None = None,
    absa_enabled: bool | None = None,
) -> pd.DataFrame:
    return build_processed_documents(
        pd.DataFrame(clean_documents),
        ner_output=pd.DataFrame(ner_output or []),
        sentiment_output=pd.DataFrame(sentiment_output or []),
        topic_output=pd.DataFrame(topic_output or []),
        absa_enabled=absa_enabled,
    )
