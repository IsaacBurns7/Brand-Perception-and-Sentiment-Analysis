# Ingestion Service Plan (Apify -> Parquet -> DuckDB)

## 1) Goal

Build a local data service that:

1. Pulls documents from an Apify endpoint on a schedule or on demand.
2. Stores immutable raw snapshots in Parquet files.
3. Uses DuckDB to query those Parquet files directly (zero-copy reads).
4. Exposes a stable, deduplicated corpus view for downstream model integrations.

Success criteria:

1. Ingestion is idempotent (no duplicate docs across repeated runs).
2. Every run is traceable (run metadata, counts, errors).
3. Corpus snapshots are reproducible and queryable without copy-loading into DuckDB tables.

## 2) Phase-First Execution Breakdown

Phase 1: Ingestion foundation

1. Implement config, Apify client, retries, and run metadata.
2. Normalize endpoint payloads into canonical schema.
3. Write partitioned raw Parquet + ingestion run manifest.
4. Exit criteria:
	1. At least one run writes valid raw partitions.
	2. Run metrics are persisted and queryable.

Phase 2: Storage and query serving

1. Define zero-copy DuckDB views over raw Parquet.
2. Add deduped current-corpus view (`v_latest_docs`).
3. Validate scan performance and partition pruning.
4. Exit criteria:
	1. `v_raw_docs` and `v_latest_docs` produce expected row counts.
	2. No DB copy/load step is required for reads.

Phase 3: Data quality and observability

1. Add quality checks (nulls, duplicates, timestamp validity, language distribution).
2. Add structured run logging and failure reporting.
3. Add lightweight benchmark timings for common queries.
4. Exit criteria:
	1. Quality report produced for each run.
	2. Errors are actionable from logs + manifest metrics.

Phase 4: Integration-ready handoff

1. Publish stable output contracts for downstream NLP consumers.
2. Add documentation for model-team integration points.
3. Lock schema versioning and backward-compatibility rules.
4. Exit criteria:
	1. Downstream teams can consume `v_latest_docs` without code changes.
	2. Contract changes are versioned and traceable.

---

## 3) Proposed Service Layout

Suggested folder layout under services/reddit_data:

1. service.py: orchestration entry point (run once or daemon/scheduler).
2. config.py: environment config (Apify token, endpoint, paths).
3. apify_client.py: API calls + pagination/retries.
4. normalize.py: schema normalization from raw Apify payload.
5. storage.py: Parquet writing and local manifest updates.
6. duckdb_views.sql: external table/view definitions.
7. quality_checks.py: ingestion quality checks and diagnostics.
8. contracts.md: downstream schema contracts and versioning notes.
9. plan.md: this implementation plan.

---

## 4) Data Flow

1. Extract:
	1. Call Apify endpoint with pagination and time filters (if available).
	2. Capture payload + HTTP metadata.
2. Normalize:
	1. Map raw fields into canonical schema.
	2. Derive stable document key and timestamps.
3. Load (Parquet):
	1. Write raw normalized records into partitioned Parquet.
	2. Write run manifest row (counts, timing, source endpoint).
4. Serve in DuckDB:
	1. Query Parquet directly using read_parquet/glob paths.
	2. Build SQL views for raw and deduplicated corpus access.
5. Validate:
	1. Execute quality checks and record summary metrics.
	2. Persist quality report for each run.

---

## 5) Storage + Zero-Copy Strategy

Root paths:

1. data/raw/apify/year=YYYY/month=MM/day=DD/run_id=.../*.parquet
2. data/meta/ingestion_runs.parquet
3. data/meta/quality_reports.parquet

Why this works:

1. Parquet gives efficient columnar scans.
2. DuckDB can query Parquet files in place, avoiding row-by-row database inserts.
3. Partitioning by date/run supports incremental loads and selective reads.

DuckDB external view pattern:

```sql
CREATE OR REPLACE VIEW v_raw_docs AS
SELECT *
FROM read_parquet('data/raw/apify/year=*/month=*/day=*/run_id=*/*.parquet');
```

---

## 6) Canonical Raw Schema

Required columns for normalized raw documents:

1. doc_id: deterministic unique key (hash of source_id + source + text + created_at).
2. source: platform identifier (reddit, x, news, etc.).
3. source_record_id: upstream id from Apify payload.
4. source_url: canonical URL/permalink.
5. title: document headline/title (nullable).
6. body_text: main text content.
7. author: author/user id (nullable/anonymized if needed).
8. language: ISO code (detected or provided).
9. created_at_utc: upstream content timestamp.
10. ingested_at_utc: service ingestion timestamp.
11. run_id: ingestion run id.
12. endpoint: endpoint path used for retrieval.
13. raw_json: original record JSON string for audit/debug.

Recommended additional fields:

1. engagement metrics (score/upvotes/comments/shares if available).
2. subreddit/forum/channel tags.
3. extraction_status and error_reason.

---

## 7) Idempotency + Incremental Ingestion

Use a two-layer strategy:

1. Run-level append-only raw files:
	1. Every ingestion run writes a new run_id partition.
2. Query-level deduped view:
	1. Define v_latest_docs as most recent row per doc_id.

Example dedupe view in DuckDB:

```sql
CREATE OR REPLACE VIEW v_latest_docs AS
SELECT * EXCLUDE (rn)
FROM (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY doc_id
           ORDER BY ingested_at_utc DESC
         ) AS rn
  FROM v_raw_docs
)
WHERE rn = 1;
```

---

## 8) Operational Plan

Runtime modes:

1. Batch mode: run every N minutes/hours via cron.
2. One-shot mode: fetch now and exit.

Reliability requirements:

1. Retry on transient HTTP errors with exponential backoff.
2. Enforce request timeouts.
3. Log structured events (run_id, stage, elapsed_ms, counts, errors).
4. Persist ingestion run metrics:
	1. records_fetched
	2. records_normalized
	3. records_written
	4. records_deduped
	5. failures

---

## 9) DuckDB Views for Downstream Consumers

Core views:

1. v_raw_docs: union of all raw partitions.
2. v_latest_docs: deduped canonical corpus.
3. v_recent_docs: optional rolling-window view for recent ingestion slices.

Example stable consumer view:

```sql
CREATE OR REPLACE VIEW v_recent_docs AS
SELECT doc_id, source, title, body_text, language, created_at_utc, ingested_at_utc
FROM v_latest_docs
WHERE body_text IS NOT NULL
  AND ingested_at_utc >= NOW() - INTERVAL '7 days';
```

---

## 10) Detailed Task Checklist by Phase

Phase 1: Ingestion foundation

1. Implement config + Apify client + run_id metadata.
2. Normalize payload to canonical schema.
3. Write partitioned raw Parquet + run metrics.

Phase 2: DuckDB integration

1. Add SQL views for v_raw_docs and v_latest_docs.
2. Validate zero-copy scans on Parquet paths.

Phase 3: Validation and monitoring

1. Add data-quality checks (nulls, duplicates, language distribution).
2. Add quality report outputs to metadata storage.
3. Add benchmark query timings (reuse benchmark style from duckdb service).

Phase 4: Consumer handoff

1. Document query contracts and schema versioning.
2. Add integration notes for downstream NLP teams.
3. Freeze v1 contract after first successful integration.

---

## 11) Data Quality Checks (Required)

1. Raw ingestion:
	1. non-null doc_id rate = 100%
	2. duplicate doc_id rate near zero in deduped view
	3. valid created_at_utc parse rate within target threshold
2. Serving layer:
	1. v_raw_docs row count equals raw partition count aggregation
	2. v_latest_docs row count <= v_raw_docs row count
3. Operational:
	1. failed API page rate below threshold
	2. p95 run duration tracked and stable over time

---

## 12) Config Contract (Environment Variables)

1. APIFY_TOKEN
2. APIFY_ENDPOINT
3. APIFY_DATASET_ID (optional depending on endpoint style)
4. INGEST_PAGE_SIZE
5. INGEST_LOOKBACK_HOURS
6. DATA_ROOT (default: ./data)
7. DUCKDB_PATH (default: ./services/duckdb/my_local_db.duckdb)
8. SCHEMA_VERSION

---

## 13) Out of Scope (Deferred)

1. Topic modeling preprocessing.
2. ABSA preprocessing.
3. NER preprocessing.
4. Any model training or inference pipelines.

These will be integrated in a later plan revision after ingestion contracts are stable.

---

## 14) Deliverables

1. Working ingestion service with retry/idempotency and run metrics.
2. Partitioned raw Parquet lake layout.
3. DuckDB views over Parquet (zero-copy query path).
4. Data quality and benchmark reporting for ingestion and serving layers.
5. Schema contract documentation for downstream integrations.
