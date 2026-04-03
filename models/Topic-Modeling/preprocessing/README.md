# Preprocessing Notes

This directory contains the LDA preprocessing pipeline and supporting modules.

## Quick Run

```bash
python LDA_normalize_corpus.py --input <input.csv> --text-col content --output <output.csv>
```

## Optional Diagnostics (Phase 5)

Diagnostics are disabled unless you pass `--diagnostics-output`.

```bash
python LDA_normalize_corpus.py \
  --input <input.csv> \
  --text-col content \
  --output <output.csv> \
  --diagnostics-output baseline_phase1/diagnostics.json \
  --diagnostics-top-n 25
```

Example below 
```bash
python LDA_normalize_corpus.py --input ../data/rating.csv --text-col article --output ../data/phase5_experiment_rating_normalized_for_LDA.csv --diagnostics-output diagnostics/preprocessing_phase5.json --diagnostics-top-n 25 
```

## Optional N-grams (Phase 6)

N-gram detection is disabled by default.
Enable it explicitly with `--enable-ngrams`.

```bash
python LDA_normalize_corpus.py \
  --input <input.csv> \
  --text-col content \
  --output <output.csv> \
  --enable-ngrams \
  --ngram-min-count 15 \
  --ngram-threshold 10.0
```

Example below 
```bash
python LDA_normalize_corpus.py --input ../data/rating.csv --text-col article --output ../data/phase6_experiment_rating_normalized_for_LDA.csv --diagnostics-output diagnostics/preprocessing_phase6.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 
```

## Add max_document_count to encompass larger corpus 

Example below
```bash
python LDA_normalize_corpus.py --input ../data/rating.csv --text-col article --output ../data/phasefinal_experiment_rating_normalized_for_LDA.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 10000
```

Example background process with 4 processes, 100,000 max documents, and batch size of 100 
```bash
python LDA_normalize_corpus.py --input ../data/rating.csv --text-col article --output ../data/rating_normalized_for_LDA.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000 --n-process 4 --batch-size 100 > LDA_normalize_corpus.out 2>&1 & 
```

## BERTopic Mode

Use BERTopic mode to keep cleaned document text instead of aggressively pruning to token lists.

```bash
python LDA_normalize_corpus.py \
  --input ../data/rating.csv \
  --text-col article \
  --output ../data/rating_normalized_for_BERTopic.csv \
  --topic-model-target bertopic \
  --bertopic-min-words 3 \
  --bertopic-drop-empty-text \
  --max-doc-count 1000
```

BERTopic mode output columns:
- `cleaned_text`
- `token_count_light`
- optional original source text when `--bertopic-keep-original-text` is enabled