# Ingestion + Preprocessing Service Plan (Apify -> Parquet -> DuckDB)

## 1) Goal

Build a local data service that:

1. Pulls documents from an Apify endpoint on a schedule or on demand.
2. Stores immutable raw snapshots in Parquet files.
3. Uses DuckDB to query those Parquet files directly (zero-copy reads).
4. Produces task-specific preprocessed views for:
	1. Topic Modeling
	2. ABSA (Aspect-Based Sentiment Analysis)
	3. NER (Named Entity Recognition)

Success criteria:

1. Ingestion is idempotent (no duplicate docs across repeated runs).
2. Every run is traceable (run metadata, counts, errors).
3. NLP-ready datasets are reproducible and versioned.

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

Phase 3: Topic preprocessing

1. Implement text normalization and token pipeline.
2. Output topic-ready Parquet (`tokens`, `tokens_str`, diagnostics).
3. Add quality checks (empty token rows, token count distribution).
4. Exit criteria:
	1. Topic dataset is reproducible with `preprocess_version`.
	2. Downstream topic modeling script can consume output directly.

Phase 4: ABSA preprocessing

1. Segment sentences and extract aspect candidates.
2. Build one row per (doc, sentence, aspect) with context windows.
3. Preserve sentiment cues (negation/intensity markers).
4. Exit criteria:
	1. ABSA schema is complete and stable.
	2. Coverage diagnostics are reported.

Phase 5: NER preprocessing + validation

1. Build NER-ready segments with token-offset integrity.
2. Run end-to-end data quality checks across raw/topic/absa/ner.
3. Add benchmark timings and operational runbook.
4. Exit criteria:
	1. NER output has valid offsets and chunk-size compliance.
	2. Service is production-ready for scheduled runs.

---

## 3) Proposed Service Layout

Suggested folder layout under services/reddit_data:

1. service.py: orchestration entry point (run once or daemon/scheduler).
2. config.py: environment config (Apify token, endpoint, paths).
3. apify_client.py: API calls + pagination/retries.
4. normalize.py: schema normalization from raw Apify payload.
5. storage.py: Parquet writing and local manifest updates.
6. duckdb_views.sql: external table/view definitions.
7. preprocess_topic.py: topic-modeling preprocessing output.
8. preprocess_absa.py: ABSA preprocessing output.
9. preprocess_ner.py: NER preprocessing output.
10. plan.md: this implementation plan.

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
	2. Build SQL views for deduped current corpus and each NLP task.
5. Preprocess outputs:
	1. Create task-specific Parquet artifacts.
	2. Optionally create CSV exports for model scripts.

---

## 5) Storage + Zero-Copy Strategy

Root paths:

1. data/raw/apify/year=YYYY/month=MM/day=DD/run_id=.../*.parquet
2. data/processed/topic/year=YYYY/month=MM/day=DD/*.parquet
3. data/processed/absa/year=YYYY/month=MM/day=DD/*.parquet
4. data/processed/ner/year=YYYY/month=MM/day=DD/*.parquet
5. data/meta/ingestion_runs.parquet

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

## 9) Preprocessing Plan: Topic Modeling

Objective:

1. Produce a clean token stream that preserves topical signals and removes noise.

Input:

1. v_latest_docs.body_text

Transformations:

1. Normalize:
	1. lowercase
	2. unicode normalization
	3. URL, handle, and markup cleanup
2. Linguistic filtering:
	1. language filter (for now, keep en)
	2. remove stopwords
	3. lemmatize
3. Token policy:
	1. keep nouns, adjectives, domain terms
	2. remove short tokens, pure digits, punctuation-only tokens
4. Phrase modeling:
	1. optional bigram/trigram formation for stable collocations
5. Vocabulary controls:
	1. min_df/no_below threshold
	2. max_df/no_above threshold
6. Output fields:
	1. doc_id
	2. cleaned_text
	3. tokens (array<string>)
	4. tokens_str (space-joined)
	5. token_count
	6. preprocess_version

Output location:

1. data/processed/topic/.../*.parquet

---

## 10) Preprocessing Plan: ABSA

Objective:

1. Prepare sentence-level and aspect-candidate-level records for aspect and sentiment models.

Input:

1. v_latest_docs.title
2. v_latest_docs.body_text

Transformations:

1. Sentence segmentation:
	1. split into sentences with sentence_id per doc
2. Aspect candidate extraction:
	1. noun/noun-phrase chunks
	2. domain lexicon matching (brand/product/feature terms)
3. Context windows:
	1. capture left/right token windows around each aspect mention
4. Label-ready schema:
	1. one row per (doc_id, sentence_id, aspect_term)
5. Sentiment cues:
	1. keep negation markers
	2. preserve intensifiers/diminishers
	3. normalize emojis/slang if social text heavy
6. Output fields:
	1. doc_id
	2. sentence_id
	3. sentence_text
	4. aspect_term
	5. aspect_start
	6. aspect_end
	7. context_left
	8. context_right
	9. preprocess_version

Output location:

1. data/processed/absa/.../*.parquet

---

## 11) Preprocessing Plan: NER

Objective:

1. Produce model-ready text segments with offsets and stable tokenization for entity extraction.

Input:

1. v_latest_docs.title
2. v_latest_docs.body_text

Transformations:

1. Text preparation:
	1. preserve case for entity quality
	2. keep punctuation that influences entity boundaries
2. Segmentation:
	1. sentence/document chunking to model max length
3. Token-offset mapping:
	1. keep char_start/char_end offsets for each token
4. Optional weak supervision:
	1. gazetteer hits (ORG/PRODUCT/PERSON/LOC)
5. Output fields:
	1. doc_id
	2. segment_id
	3. segment_text
	4. token_texts (array<string>)
	5. token_offsets (array<struct<start:int,end:int>> or JSON)
	6. preprocess_version

Output location:

1. data/processed/ner/.../*.parquet

---

## 12) DuckDB Views for Downstream Tasks

Core views:

1. v_raw_docs: union of all raw partitions.
2. v_latest_docs: deduped canonical corpus.
3. v_topic_input: selects topic preprocessing fields.
4. v_absa_input: selects ABSA preprocessing fields.
5. v_ner_input: selects NER preprocessing fields.

Example topic input view:

```sql
CREATE OR REPLACE VIEW v_topic_input AS
SELECT doc_id, body_text
FROM v_latest_docs
WHERE language = 'en' AND body_text IS NOT NULL;
```

---

## 13) Detailed Task Checklist by Phase

Phase 1: Ingestion foundation

1. Implement config + Apify client + run_id metadata.
2. Normalize payload to canonical schema.
3. Write partitioned raw Parquet + run metrics.

Phase 2: DuckDB integration

1. Add SQL views for v_raw_docs and v_latest_docs.
2. Validate zero-copy scans on Parquet paths.

Phase 3: NLP preprocessors

1. Implement preprocess_topic.py and write Parquet output.
2. Implement preprocess_absa.py and write Parquet output.
3. Implement preprocess_ner.py and write Parquet output.

Phase 4: Validation and monitoring

1. Add data-quality checks (nulls, duplicates, language distribution).
2. Add preprocessing diagnostics (token counts, sentence counts, truncation rate).
3. Add benchmark query timings (reuse benchmark style from duckdb service).

---

## 14) Data Quality Checks (Required)

1. Raw ingestion:
	1. non-null doc_id rate = 100%
	2. duplicate doc_id rate near zero in deduped view
2. Topic preprocessing:
	1. token_count within expected band
	2. empty token rows below threshold
3. ABSA preprocessing:
	1. sentence split success rate
	2. aspect extraction coverage
4. NER preprocessing:
	1. segment length distribution
	2. offset integrity checks

---

## 15) Config Contract (Environment Variables)

1. APIFY_TOKEN
2. APIFY_ENDPOINT
3. APIFY_DATASET_ID (optional depending on endpoint style)
4. INGEST_PAGE_SIZE
5. INGEST_LOOKBACK_HOURS
6. DATA_ROOT (default: ./data)
7. DUCKDB_PATH (default: ./services/duckdb/my_local_db.duckdb)
8. PREPROCESS_VERSION

---

## 16) Deliverables

1. Working ingestion service with retry/idempotency and run metrics.
2. Partitioned raw Parquet lake layout.
3. DuckDB views over Parquet (zero-copy query path).
4. Three preprocessing outputs (topic, ABSA, NER) written as Parquet.
5. Basic profiling/benchmark script + quality report.

