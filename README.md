# NER — Brand Entity Extraction

Part of the **Brand Perception & Sentiment Analysis** pipeline.  
Sits between the data-cleaning step (`news_dailyworker.py`) and the modelling steps (LDA, sentiment classifiers).

```
news_dailyworker.py        →   NER/ner_pipeline.py   →   LDA / Sentiment models
(CollectData + Preprocessing)   (brand extraction)        (downstream analysis)
```

---

## Files

| File | Purpose |
|------|---------|
| `ner_pipeline.py` | Core pipeline — `NERPipeline`, model backends, normaliser |
| `db_output.py` | PostgreSQL writers, CSV export, SQLAlchemy ORM snippet |
| `evaluate_ner.py` | Benchmark backends, P/R/F1 output with `[timing]` logs |

---

## Where it fits in the project

### Input — daily article CSV
`news_dailyworker.Preprocessing().runner()` produces:

```
data/dailyworker/YYYY-MM-DD.csv
  source_id, source_name, author, title, description,
  url, url_to_image, published_at, content, category, full_content
```

NER runs on `full_content` (scraped body text) and falls back to `content → article → title+description` automatically.

### Output — two new columns

| Column | Type | Notes |
|--------|------|-------|
| `ner_brands` | `list[str]` | Canonical brand names — ready for DB insert |
| `ner_raw_json` | `str (JSON)` | Full payload: confidence, positions, model used |

---

## Quick start

```python
from NER.ner_pipeline import NERPipeline
import pandas as pd

# Load a daily output from news_dailyworker
df = pd.read_csv("data/dailyworker/2025-01-15.csv")

# Run NER (uses en_core_web_md — already downloaded for LDA)
pipeline = NERPipeline()
df = pipeline.run_on_dataframe(df)

# Inspect
print(df[["title", "source_name", "category", "ner_brands"]].head(10))

# Save enriched CSV
df.to_csv("data/dailyworker/2025-01-15_ner.csv", index=False)
```

---

## Model backends

### spaCy (default, recommended)
The project **already uses spaCy** in `LDA.py` and `LDA_normalize_corpus.py`.  
NER reuses the same installed model — no extra download needed.

| Model | Already in project? | Notes |
|-------|--------------------|-|
| `en_core_web_md` | ✅ (LDA_normalize_corpus.py) | Recommended — better NER accuracy |
| `en_core_web_sm` | ✅ (LDA.py, spacy_test.py) | Fallback |

The LDA pipeline loads spaCy with `disable=["ner", "parser"]` to save memory.  
The NER pipeline loads it with `disable=["parser", "lemmatizer"]` — NER **enabled**, parser disabled.  
Both pipelines can coexist without conflict.

### Rules (always-on fallback)
Zero dependencies. Runs alongside spaCy by default (`combine_rules=True`) to recover brands that spaCy lemmatises away or misses in short title/description fragments.

Covers:
- Legal-suffix orgs: `Apple Inc.`, `Rivian Automotive Inc.`, `AstraZeneca PLC`
- Dictionary of ~100 known brands relevant to the scraped news domains
- CamelCase tokens: `ExxonMobil`, `McKinsey`, `BlackRock`
- Ticker symbols: `$AAPL`, `(NASDAQ: NVDA)`

---

## Database integration

### Write to PostgreSQL (same conn as `news_dailyworker.py`)

```python
import psycopg2
from NER.db_output import create_tables, write_dataframe_to_postgres

conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
create_tables(conn)   # run once

df = pipeline.run_on_dataframe(df)
write_dataframe_to_postgres(df, conn)
conn.close()
```

### New tables

```sql
-- One row per unique brand across all articles
CREATE TABLE brands (
    id             SERIAL PRIMARY KEY,
    canonical_name TEXT    UNIQUE NOT NULL,
    aliases        JSONB,           -- ["Apple", "AAPL"]
    entity_type    TEXT,            -- BRAND | ORG | PRODUCT
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Join table: which brands appear in which articles
CREATE TABLE document_brand_mentions (
    id            SERIAL PRIMARY KEY,
    article_url   TEXT,             -- matches articles.url
    brand_id      INTEGER REFERENCES brands(id),
    source_name   TEXT,             -- e.g. "Forbes", "BBC News"
    category      TEXT,             -- NewsAPI query term (e.g. "technology")
    mention_count INTEGER,
    confidence    REAL,
    positions     JSONB,            -- [[start_char, end_char], ...]
    model_sources JSONB,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

### Export to CSV (no Postgres needed)

```python
from NER.db_output import write_dataframe_to_csv
brands_path, mentions_path = write_dataframe_to_csv(df, output_dir="data/dailyworker")
```

---

## Evaluation

```bash
python NER/evaluate_ner.py                 # compare spaCy+Rules vs Rules-only
python NER/evaluate_ner.py --verbose       # show per-article detail
python NER/evaluate_ner.py --backend rules # Rules only
```

Expected output (Rules-only baseline, no spaCy):
```
spaCy+Rules  →  P=0.810  R=0.829  F1=0.819  avg=~45ms/article
Rules-only   →  P=0.645  R=0.769  F1=0.702  avg=0.2ms/article
```

---

## Integration with `news_dailyworker.py` runner

Drop this into `news_dailyworker.runner()` after `Preprocessing().runner()`:

```python
# In news_dailyworker.py — after line:
#   Preprocessing().runner()

if f'{str(previous_date)}.csv' in list_files_in_directory():
    from NER.ner_pipeline import NERPipeline
    from NER.db_output    import create_tables, write_dataframe_to_postgres

    df       = pd.read_csv(f"{data_directory}/{previous_date}.csv")
    pipeline = NERPipeline()
    df       = pipeline.run_on_dataframe(df)
    df.to_csv(f"{data_directory}/{previous_date}.csv", index=False)

    conn = CollectData().connect_to_database(dbname, user, password, host, port)
    create_tables(conn)
    write_dataframe_to_postgres(df, conn)
    conn.close()
    append_word_to_file("NER complete — brands written to DB")
```

---

## Configuration

```python
NERPipeline(
    spacy_model    = "en_core_web_md",  # or en_core_web_sm
    min_confidence = 0.45,              # drop entities below this
    text_column    = "full_content",    # or "content", "article"
    combine_rules  = True,              # merge Rules output with spaCy
)
```
