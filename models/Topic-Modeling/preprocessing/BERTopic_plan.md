# BERTopic Plan

## Goal
Adapt the current preprocessing flow so it supports two targets safely:
- LDA (current behavior)
- BERTopic (text-preserving behavior)

## Current Pipeline Reality
The current pipeline in lda_pipeline.py is strongly LDA-oriented:
- strict token filtering
- optional Gensim n-grams
- rare-token pruning
- drop-empty-doc logic
- output focused on tokens_str

This can over-prune BERTopic input and has already caused empty corpus outcomes during debugging.

## Design Direction
Add a mode switch and keep one shared pipeline:
- topic_model_target=lda keeps existing behavior.
- topic_model_target=bertopic uses a lighter branch that prioritizes cleaned text over filtered tokens.

## Required Changes

### 1) pipeline_config.py
Add fields:
- topic_model_target: str = "lda"  # allowed: lda, bertopic
- bertopic_min_words: int = 3
- bertopic_drop_empty_text: bool = True
- bertopic_keep_original_text: bool = False

Validation:
- enforce topic_model_target in {"lda", "bertopic"}
- enforce bertopic_min_words >= 1

### 2) LDA_normalize_corpus.py
Add CLI args:
- --topic-model-target {lda,bertopic}
- --bertopic-min-words
- --bertopic-drop-empty-text / --no-bertopic-drop-empty-text
- --bertopic-keep-original-text / --no-bertopic-keep-original-text

Pass these fields into PipelineConfig.

### 3) lda_pipeline.py
Keep shared stages:
- validate config
- load data
- drop nulls
- clean text

Add branch after clean text:

If topic_model_target == bertopic:
- create cleaned_text from text_col
- optional short-doc filter using bertopic_min_words
- skip build_vocab and filter_rare
- skip drop-empty-by-token_count
- write BERTopic-ready output columns:
  - cleaned_text (required)
  - optional source text and doc id
- run BERTopic-specific diagnostics (text length stats, docs kept/dropped)

If topic_model_target == lda:
- run existing tokenization, n-gram, vocab, rare-filter path

Also harden empty-corpus behavior:
- guard stats logging when df is empty
- avoid int conversion on NaN in logs and diagnostics

### 4) token_vocab_utils.py
Keep this module LDA-first.
For BERTopic path, avoid dependence on OOV filtering.

Key rule:
- never remove BERTopic documents due to OOV-heavy tokenization.

### 5) README.md
Add BERTopic section:
- why preprocessing is lighter
- BERTopic-mode command
- output schema for BERTopic mode

## Recommended BERTopic Preprocessing Defaults
- keep regex cleaning and lowercasing
- preserve full cleaned text
- avoid corpus-wide rare-token pruning at preprocess stage
- avoid Gensim phrase training in BERTopic mode

Note:
For BERTopic, ngrams are usually better applied in the BERTopic vectorizer (CountVectorizer ngram_range), not in preprocessing.

## Output Contract
In BERTopic mode, output should contain:
- cleaned_text
- source_row_id (recommended)
- token_count_light (optional)

## Implementation Sequence
1. Add config + CLI mode flags.
2. Add BERTopic branch in lda_pipeline.py.
3. Add empty-data-safe stats and diagnostics.
4. Update README examples.
5. Run smoke tests for both modes.

## Smoke Tests

BERTopic mode smoke:
python LDA_normalize_corpus.py --input ../data/rating.csv --text-col article --output ../data/bertopic_preprocessed_smoke.csv --topic-model-target bertopic --max-doc-count 100 --batch-size 100

LDA regression smoke:
python LDA_normalize_corpus.py --input ../data/rating.csv --text-col article --output ../data/lda_preprocessed_smoke.csv --topic-model-target lda --enable-ngrams --max-doc-count 100 --batch-size 100

Expected results:
- BERTopic mode keeps non-empty cleaned_text for most docs.
- LDA mode behavior remains unchanged.
- no NaN/min-max crashes in either mode.

## Definition Of Done
- one command supports lda and bertopic modes
- BERTopic mode does not collapse corpus to empty docs
- diagnostics are stable for empty and non-empty outputs
- docs include examples for both targets
