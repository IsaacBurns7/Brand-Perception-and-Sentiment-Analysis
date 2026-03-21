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
