from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .document_builder import (
    build_processed_documents,
    build_stub_sentiment_output,
    build_stub_topic_output,
)
from .preprocessing import clean_documents
from .schemas import PROCESSED_DOCUMENT_COLUMNS


DEFAULT_STORAGE_BACKEND = os.environ.get("BRAND_PERCEPTION_STORAGE_BACKEND", "duckdb")
DEFAULT_DUCKDB_PATH = Path(
    os.environ.get("BRAND_PERCEPTION_DUCKDB_PATH", "services/duckdb/brand_perception.duckdb")
)
DEFAULT_JSONL_PATH = Path(
    os.environ.get("BRAND_PERCEPTION_JSONL_PATH", "data/processed_documents.jsonl")
)
DEFAULT_PROCESSED_TABLE = os.environ.get("BRAND_PERCEPTION_PROCESSED_TABLE", "processed_documents")


def resolve_storage_backend(backend: str = DEFAULT_STORAGE_BACKEND) -> str:
    if backend != "duckdb":
        return backend

    try:
        import duckdb  # noqa: F401
    except ModuleNotFoundError:
        return "jsonl"

    return "duckdb"


def normalize_processed_documents(processed_documents: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the canonical processed-document shape."""
    if processed_documents.empty:
        return pd.DataFrame(columns=PROCESSED_DOCUMENT_COLUMNS)

    normalized = processed_documents.copy()
    if "aspect" not in normalized.columns:
        normalized["aspect"] = "general"

    missing_columns = [
        column for column in PROCESSED_DOCUMENT_COLUMNS if column not in normalized.columns
    ]
    if missing_columns:
        raise ValueError(
            "Processed documents are missing required columns: "
            + ", ".join(missing_columns)
        )

    normalized = normalized.loc[:, PROCESSED_DOCUMENT_COLUMNS].copy()
    normalized["doc_id"] = normalized["doc_id"].astype(str)
    normalized["text"] = normalized["text"].fillna("").astype(str)
    normalized["brand"] = normalized["brand"].fillna("unknown").astype(str)
    normalized["aspect"] = normalized["aspect"].fillna("general").astype(str)
    normalized["sentiment"] = pd.to_numeric(normalized["sentiment"], errors="coerce").fillna(0.0)
    normalized["sentiment_label"] = normalized["sentiment_label"].fillna("neutral").astype(str)
    normalized["topic"] = normalized["topic"].fillna("unclassified").astype(str)
    normalized["source"] = normalized["source"].fillna("unknown").astype(str)
    normalized["created_utc"] = pd.to_datetime(normalized["created_utc"], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=["created_utc"]).drop_duplicates().reset_index(drop=True)
    return normalized


def _resolve_storage_path(backend: str, path: str | os.PathLike[str] | None = None) -> Path:
    backend = resolve_storage_backend(backend)
    if path is not None:
        return Path(path)
    if backend == "duckdb":
        return DEFAULT_DUCKDB_PATH
    if backend == "jsonl":
        return DEFAULT_JSONL_PATH
    raise ValueError(f"Unsupported storage backend: {backend}")


def _read_duckdb(path: Path, table_name: str) -> pd.DataFrame:
    import duckdb

    if not path.exists():
        return pd.DataFrame(columns=PROCESSED_DOCUMENT_COLUMNS)

    conn = duckdb.connect(str(path))
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        if table_name not in {row[0] for row in tables}:
            return pd.DataFrame(columns=PROCESSED_DOCUMENT_COLUMNS)
        return conn.execute(f"SELECT * FROM {table_name}").df()
    finally:
        conn.close()


def _write_duckdb(
    processed_documents: pd.DataFrame,
    path: Path,
    table_name: str,
    mode: str,
) -> None:
    import duckdb

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    try:
        conn.register("processed_documents_df", processed_documents)
        if mode == "replace":
            conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM processed_documents_df")
        elif mode == "append":
            tables = conn.execute("SHOW TABLES").fetchall()
            if table_name in {row[0] for row in tables}:
                existing_columns = {
                    row[1]
                    for row in conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
                }
                if "aspect" not in existing_columns:
                    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN aspect VARCHAR")
                    conn.execute(
                        f"UPDATE {table_name} SET aspect = 'general' WHERE aspect IS NULL"
                    )
            else:
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM processed_documents_df WHERE 1=0")
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM processed_documents_df")
        else:
            raise ValueError("mode must be 'replace' or 'append'")
    finally:
        conn.close()


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=PROCESSED_DOCUMENT_COLUMNS)

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows, columns=PROCESSED_DOCUMENT_COLUMNS)


def _write_jsonl(processed_documents: pd.DataFrame, path: Path, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = processed_documents.copy()
    records["created_utc"] = records["created_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    open_mode = "a" if mode == "append" and path.exists() else "w"
    with path.open(open_mode, encoding="utf-8") as handle:
        for row in records.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_processed_documents(
    processed_documents: pd.DataFrame,
    *,
    backend: str = DEFAULT_STORAGE_BACKEND,
    path: str | os.PathLike[str] | None = None,
    table_name: str = DEFAULT_PROCESSED_TABLE,
    mode: str = "replace",
) -> dict[str, Any]:
    resolved_backend = resolve_storage_backend(backend)
    normalized = normalize_processed_documents(processed_documents)
    target_path = _resolve_storage_path(resolved_backend, path)

    if resolved_backend == "duckdb":
        _write_duckdb(normalized, target_path, table_name, mode)
    elif resolved_backend == "jsonl":
        _write_jsonl(normalized, target_path, mode)
    else:
        raise ValueError(f"Unsupported storage backend: {resolved_backend}")

    return {
        "backend": resolved_backend,
        "path": str(target_path),
        "table_name": table_name if resolved_backend == "duckdb" else None,
        "rows_written": int(len(normalized)),
        "mode": mode,
    }


def read_processed_documents(
    *,
    backend: str = DEFAULT_STORAGE_BACKEND,
    path: str | os.PathLike[str] | None = None,
    table_name: str = DEFAULT_PROCESSED_TABLE,
) -> pd.DataFrame:
    resolved_backend = resolve_storage_backend(backend)
    target_path = _resolve_storage_path(resolved_backend, path)

    if resolved_backend == "duckdb":
        df = _read_duckdb(target_path, table_name)
    elif resolved_backend == "jsonl":
        df = _read_jsonl(target_path)
    else:
        raise ValueError(f"Unsupported storage backend: {resolved_backend}")

    return normalize_processed_documents(df)


def build_sample_processed_documents() -> pd.DataFrame:
    """Immediate local sample for API/dev use when storage is empty."""
    raw_rows = pd.DataFrame(
        [
            {
                "url": "https://news.example.com/acme-earnings-2026-04-18",
                "title": "Acme posts stronger Q1 revenue as product demand improves",
                "description": "Executives said enterprise renewals remained stable.",
                "full_content": (
                    "Acme reported better-than-expected first-quarter revenue on April 18, 2026. "
                    "Management said demand improved across software subscriptions and support services, "
                    "while analysts noted steadier retention among large enterprise customers."
                ),
                "source_name": "Reuters",
                "published_at": "2026-04-18T13:15:00Z",
            },
            {
                "url": "https://news.example.com/acme-shipping-delays-2026-04-18",
                "title": "Customers flag Acme shipping delays after warehouse outage",
                "description": "Some orders were delayed for two days.",
                "full_content": (
                    "Customers on social platforms described delayed deliveries after an outage at a regional "
                    "warehouse. Acme said it restored operations and was working through a backlog of orders."
                ),
                "source_name": "CNBC",
                "published_at": "2026-04-18T20:40:00Z",
            },
            {
                "url": "https://news.example.com/acme-ai-partnership-2026-04-19",
                "title": "Acme announces AI partnership with major cloud provider",
                "description": "The deal focuses on analytics and customer support tools.",
                "full_content": (
                    "Acme announced a multiyear partnership to expand AI-based analytics and automate customer "
                    "support workflows. Executives said the agreement should reduce response times and improve "
                    "internal productivity."
                ),
                "source_name": "The Verge",
                "published_at": "2026-04-19T15:05:00Z",
            },
            {
                "url": "https://news.example.com/acme-pricing-2026-04-20",
                "title": "Analysts watch Acme pricing changes ahead of summer launch",
                "description": "Investors are looking for margin impact.",
                "full_content": (
                    "Analysts said Acme's upcoming pricing changes could support margins if demand remains solid. "
                    "Several firms kept a neutral stance while waiting for more guidance on the summer product launch."
                ),
                "source_name": "Bloomberg",
                "published_at": "2026-04-20T11:00:00Z",
            },
        ]
    )

    clean_df = clean_documents(raw_rows)
    topic_df = build_stub_topic_output(clean_df)
    return build_processed_documents(
        clean_df,
        sentiment_output=build_stub_sentiment_output(clean_df),
        topic_output=topic_df,
        absa_enabled=False,
    )


def ensure_processed_documents_available(
    *,
    backend: str = DEFAULT_STORAGE_BACKEND,
    path: str | os.PathLike[str] | None = None,
    table_name: str = DEFAULT_PROCESSED_TABLE,
    use_sample_data: bool = True,
    persist_sample: bool = False,
) -> pd.DataFrame:
    df = read_processed_documents(backend=backend, path=path, table_name=table_name)
    if not df.empty or not use_sample_data:
        return df

    sample_df = build_sample_processed_documents()
    if persist_sample:
        write_processed_documents(
            sample_df,
            backend=backend,
            path=path,
            table_name=table_name,
            mode="replace",
        )
    return sample_df
