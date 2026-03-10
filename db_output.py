"""
DB Output — Brand Entity Persistence
======================================
Converts NERResult objects into DB-ready payloads that match the schema
already established by news_dailyworker.py (PostgreSQL + optional Azure Blob).

Existing table (from news_dailyworker.push_data):
    articles / rating  →  source_id, source_name, author, title, description,
                           url, url_to_image, published_at, content, category

New tables added by this module:
    brands                      — one row per unique canonical brand
    document_brand_mentions     — join table (article ↔ brand)

The write helpers mirror the psycopg2 pattern from news_dailyworker.py.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from psycopg2 import sql  # type: ignore  (already a dep via news_dailyworker.py)

from ner_pipeline import NERResult, BrandEntity

logger = logging.getLogger(__name__)


# ── Schema DDL ────────────────────────────────────────────────────────────────

CREATE_BRANDS_TABLE = """
CREATE TABLE IF NOT EXISTS brands (
    id             SERIAL PRIMARY KEY,
    canonical_name TEXT    UNIQUE NOT NULL,
    aliases        JSONB   DEFAULT '[]',
    entity_type    TEXT    DEFAULT 'BRAND',
    created_at     TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_brands_name ON brands (canonical_name);
"""

CREATE_MENTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS document_brand_mentions (
    id            SERIAL PRIMARY KEY,
    article_url   TEXT,                          -- FK-equivalent to articles.url
    brand_id      INTEGER REFERENCES brands(id),
    source_name   TEXT,                          -- from news_dailyworker source_name
    category      TEXT,                          -- NewsAPI query term
    mention_count INTEGER     DEFAULT 1,
    confidence    REAL        DEFAULT 0.0,
    positions     JSONB       DEFAULT '[]',      -- [[start_char, end_char], ...]
    model_sources JSONB       DEFAULT '[]',
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_mentions_url   ON document_brand_mentions (article_url);
CREATE INDEX IF NOT EXISTS idx_mentions_brand ON document_brand_mentions (brand_id);
CREATE INDEX IF NOT EXISTS idx_mentions_cat   ON document_brand_mentions (category);
"""


def create_tables(conn) -> None:
    """Create brands and document_brand_mentions tables if they don't exist."""
    with conn.cursor() as cur:
        cur.execute(CREATE_BRANDS_TABLE)
        cur.execute(CREATE_MENTIONS_TABLE)
    conn.commit()
    logger.info("NER tables ensured.")


# ── Plain-dict payloads (framework-agnostic) ──────────────────────────────────

def brand_upsert_payload(entity: BrandEntity) -> dict:
    return {
        "canonical_name": entity.canonical_name,
        "aliases":        json.dumps(entity.aliases),
        "entity_type":    entity.entity_type.value,
    }


def mention_insert_payload(
    entity:      BrandEntity,
    article_url: Optional[str] = None,
    source_name: Optional[str] = None,
    category:    Optional[str] = None,
) -> dict:
    return {
        "article_url":   article_url or entity.doc_id,
        "brand_id":      None,   # resolved after upsert
        "source_name":   source_name,
        "category":      category,
        "mention_count": entity.mention_count,
        "confidence":    entity.confidence,
        "positions":     json.dumps(entity.positions),
        "model_sources": json.dumps(entity.model_sources),
    }


# ── PostgreSQL writer (matches news_dailyworker.py psycopg2 style) ────────────

def write_result_to_postgres(
    result:      NERResult,
    conn,
    article_url: Optional[str] = None,
    table_prefix: str = "",
) -> int:
    """
    Upsert brand entities and insert mention rows into PostgreSQL.
    Uses the same psycopg2 connection pattern as news_dailyworker.push_data().

    Parameters
    ----------
    result       : NERResult from NERPipeline.run()
    conn         : open psycopg2 connection (caller manages lifecycle)
    article_url  : URL of the article (use articles.url as the join key)
    table_prefix : optional schema prefix, e.g. "public."

    Returns number of brand entities written.
    """
    brands_tbl   = sql.Identifier(f"{table_prefix}brands")        if table_prefix else sql.SQL("brands")
    mentions_tbl = sql.Identifier(f"{table_prefix}document_brand_mentions") if table_prefix else sql.SQL("document_brand_mentions")

    count = 0
    with conn.cursor() as cur:
        for entity in result.brand_entities:
            # Upsert brand
            cur.execute(
                sql.SQL("""
                    INSERT INTO {} (canonical_name, aliases, entity_type)
                    VALUES (%(canonical_name)s, %(aliases)s::jsonb, %(entity_type)s)
                    ON CONFLICT (canonical_name) DO UPDATE
                        SET aliases     = EXCLUDED.aliases,
                            entity_type = EXCLUDED.entity_type
                    RETURNING id
                """).format(sql.SQL("brands")),
                brand_upsert_payload(entity),
            )
            brand_id = cur.fetchone()[0]

            # Insert mention
            payload = mention_insert_payload(
                entity,
                article_url = article_url,
                source_name = result.source_name,
                category    = result.category,
            )
            payload["brand_id"] = brand_id
            cur.execute(
                sql.SQL("""
                    INSERT INTO document_brand_mentions
                        (article_url, brand_id, source_name, category,
                         mention_count, confidence, positions, model_sources)
                    VALUES
                        (%(article_url)s, %(brand_id)s, %(source_name)s, %(category)s,
                         %(mention_count)s, %(confidence)s, %(positions)s::jsonb,
                         %(model_sources)s::jsonb)
                """).format(),
                payload,
            )
            count += 1

    conn.commit()
    return count


# ── DataFrame batch writer ────────────────────────────────────────────────────

def write_dataframe_to_postgres(df, conn) -> int:
    """
    Write NER results for a full daily DataFrame (output of
    NERPipeline.run_on_dataframe) to PostgreSQL.

    Expects columns: ner_raw_json, url, source_name, category
    Returns total brand-entity rows written.
    """
    import pandas as pd

    if "ner_raw_json" not in df.columns:
        raise ValueError("DataFrame must have 'ner_raw_json' column — run NERPipeline.run_on_dataframe() first.")

    total = 0
    for _, row in df.iterrows():
        try:
            payload     = json.loads(row["ner_raw_json"])
            article_url = str(row.get("url", "") or "")
            source_name = str(row.get("source_name", "") or "")
            category    = str(row.get("category", "") or "")

            # Reconstruct a lightweight NERResult-like object for the writer
            from ner_pipeline import NERResult, BrandEntity, EntityType
            brand_entities = []
            for e in payload.get("brand_entities", []):
                brand_entities.append(BrandEntity(
                    canonical_name = e["canonical_name"],
                    aliases        = e.get("aliases", []),
                    entity_type    = EntityType(e.get("entity_type", "BRAND")),
                    confidence     = e.get("confidence", 0.0),
                    mention_count  = e.get("mention_count", 1),
                    positions      = [tuple(p) for p in e.get("positions", [])],
                    model_sources  = e.get("model_sources", []),
                    doc_id         = payload.get("doc_id"),
                ))

            result = NERResult(
                doc_id         = payload.get("doc_id"),
                brand_entities = brand_entities,
                model_used     = payload.get("model_used", ""),
                source_name    = source_name,
                category       = category,
            )

            total += write_result_to_postgres(result, conn, article_url=article_url)

        except Exception as exc:
            logger.warning("Failed to write row to DB: %s", exc)

    logger.info("Wrote %d brand-entity rows to PostgreSQL.", total)
    return total


# ── CSV fallback (useful for local dev without Postgres) ─────────────────────

def write_dataframe_to_csv(df, output_dir: str = "data/dailyworker") -> tuple[str, str]:
    """
    Save flat CSVs for brands and mentions — useful for inspection or
    loading into analysis notebooks (same pattern as DumpSqlData.export_to_csv).

    Returns (brands_csv_path, mentions_csv_path).
    """
    import os, pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    brand_rows:   list[dict] = []
    mention_rows: list[dict] = []

    for _, row in df.iterrows():
        try:
            payload     = json.loads(row.get("ner_raw_json", "{}"))
            article_url = str(row.get("url", ""))
            source_name = str(row.get("source_name", ""))
            category    = str(row.get("category", ""))

            for e in payload.get("brand_entities", []):
                brand_rows.append({
                    "canonical_name": e["canonical_name"],
                    "aliases":        json.dumps(e.get("aliases", [])),
                    "entity_type":    e.get("entity_type", "BRAND"),
                })
                mention_rows.append({
                    "article_url":   article_url,
                    "source_name":   source_name,
                    "category":      category,
                    "canonical_name": e["canonical_name"],
                    "mention_count": e.get("mention_count", 1),
                    "confidence":    e.get("confidence", 0.0),
                    "model_sources": json.dumps(e.get("model_sources", [])),
                })
        except Exception as exc:
            logger.warning("Skipping row in CSV export: %s", exc)

    brands_path   = os.path.join(output_dir, "ner_brands.csv")
    mentions_path = os.path.join(output_dir, "ner_mentions.csv")

    pd.DataFrame(brand_rows).drop_duplicates("canonical_name").to_csv(brands_path,   index=False)
    pd.DataFrame(mention_rows).to_csv(mentions_path, index=False)

    logger.info("Saved brands   → %s (%d rows)", brands_path,   len(brand_rows))
    logger.info("Saved mentions → %s (%d rows)", mentions_path, len(mention_rows))
    return brands_path, mentions_path


# ── SQLAlchemy ORM models (drop into your models.py) ─────────────────────────

SQLALCHEMY_SNIPPET = '''
# --- Add to your SQLAlchemy models file ---

from sqlalchemy import Column, Integer, String, Float, JSON, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

class Brand(Base):
    __tablename__ = "brands"
    id             = Column(Integer, primary_key=True)
    canonical_name = Column(String, unique=True, nullable=False, index=True)
    aliases        = Column(JSON,   default=list)
    entity_type    = Column(String, default="BRAND")
    created_at     = Column(DateTime(timezone=True), server_default=func.now())
    mentions       = relationship("DocumentBrandMention", back_populates="brand")

class DocumentBrandMention(Base):
    __tablename__ = "document_brand_mentions"
    id            = Column(Integer, primary_key=True)
    article_url   = Column(String,  index=True)
    brand_id      = Column(Integer, ForeignKey("brands.id"), index=True)
    source_name   = Column(String)
    category      = Column(String,  index=True)
    mention_count = Column(Integer, default=1)
    confidence    = Column(Float,   default=0.0)
    positions     = Column(JSON,    default=list)
    model_sources = Column(JSON,    default=list)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    brand         = relationship("Brand", back_populates="mentions")
'''
