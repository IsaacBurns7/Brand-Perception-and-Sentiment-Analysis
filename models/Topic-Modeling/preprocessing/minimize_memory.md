# Minimize Memory Plan: DataFrame In-Place Refactor

## Goal
Refactor these functions so they operate on pandas DataFrame slices/columns instead of returning 2D Python lists:
- process_batch
- apply_ngrams
- build_vocab
- filter_rare

Primary objective: reduce peak memory by avoiding full-corpus list materialization and minimizing copied intermediates.

## Current Pain Points
- process_batch returns list[list[str]], which keeps per-batch token lists as Python objects.
- apply_ngrams returns a newly transformed list[list[str]].
- build_vocab expects list[list[str]], forcing re-materialization when called after batch processing.
- filter_rare returns a new corpus copy.
- run_pipeline still references tokenized after the batch loop, but tokenized is not defined in current flow.

## Target DataFrame Contract
Use DataFrame columns as the source of truth through the pipeline.
- Required columns after tokenization stage:
  - article_clean: cleaned text for the current batch scope only (optional transient)
  - tokens: object dtype (list[str]) while filtering/diagnostics is still needed
  - tokens_str: normalized whitespace-joined tokens
  - token_count: int32 token length per row

Design rule: functions mutate provided DataFrame rows/columns in place and return only lightweight metadata when needed.

## Proposed API Changes

### 1) process_batch
Current:
- process_batch(texts, nlp, stopwords, cfg) -> list[list[str]]

Proposed:
- process_batch(df, text_col, row_indexer, nlp, stopwords, cfg, tokens_col="tokens") -> None

Behavior:
- Read text from df.loc[row_indexer, text_col] lazily.
- Run nlp.pipe on that batch.
- Write list[str] results directly into df.loc[row_indexer, tokens_col].
- No corpus-level return object.

Memory impact:
- Removes one returned list allocation per batch call.
- Keeps only current batch structures live.

### 2) apply_ngrams
Current:
- apply_ngrams(tokenized_docs, cfg) -> list[list[str]]

Proposed:
- apply_ngrams(df, row_indexer, cfg, tokens_col="tokens") -> None

Behavior:
- Build/train Phrases only from df.loc[row_indexer, tokens_col].
- Replace tokens in the same rows in place.
- Avoid creating a second full-batch Python container where possible.

Memory impact:
- Keeps transformation scoped to batch rows, avoids global copied corpus.

### 3) build_vocab
Current:
- build_vocab(tokenized_docs, min_freq, min_doc_freq) -> (vocab, term_count)

Proposed:
- build_vocab(df, tokens_col="tokens", min_freq=..., min_doc_freq=...) -> (vocab, term_count, total_tokens_before)

Behavior:
- Iterate once over df[tokens_col] (stream-like iteration over series values).
- Maintain Counter objects only; no additional tokenized copy.
- Return metadata required for diagnostics.

Memory impact:
- Avoids constructing side structures beyond counters.

### 4) filter_rare
Current:
- filter_rare(tokenized_docs, vocab) -> list[list[str]]

Proposed:
- filter_rare(df, vocab, tokens_col="tokens", token_count_col="token_count") -> (total_before, total_after)

Behavior:
- Mutate each row in df[tokens_col] in place by filtering tokens against vocab.
- Recompute token_count in place (prefer vectorized length map to int32).
- Return token count stats only.

Memory impact:
- Removes returned full-corpus filtered copy.

## run_pipeline Integration Plan

1. Establish columns early:
- Initialize df["tokens"] as empty object column (or assign during first batch only).
- Keep token_count as int32.

2. Batch loop:
- For each slice:
  - clean text for slice only
  - process_batch writes tokens into df for slice
  - apply_ngrams mutates same slice if enabled
  - derive tokens_str and token_count for same slice
- Do not accumulate batch token output in Python variables beyond temporary local references.

3. Rare filtering + vocab:
- build_vocab consumes df["tokens"].
- filter_rare mutates df["tokens"] and df["token_count"] in place.
- refresh df["tokens_str"] from filtered tokens if output must reflect post-filter vocabulary.

4. Diagnostics:
- top_terms and token stats read directly from df["tokens"] and df["token_count"].
- avoid constructing a full flattened list; feed generator directly to Counter.

5. Final output:
- write only required columns.
- if tokens column is not needed in returned df, optionally drop it in place right before return.

## Migration Steps (Ordered)

1. Update token_vocab_utils function signatures and docstrings.
2. Implement process_batch DataFrame write-in-place behavior.
3. Implement apply_ngrams DataFrame slice mutation behavior.
4. Implement build_vocab DataFrame-based iteration.
5. Implement filter_rare DataFrame in-place mutation and stats return.
6. Update run_pipeline call sites for all four functions.
7. Remove stale tokenized references in run_pipeline.
8. Recompute diagnostics from DataFrame-backed columns only.
9. Add lightweight assertions (optional) for expected columns/types during development.
10. Run pipeline smoke tests and compare outputs with baseline for correctness.

## Validation Checklist

Functional parity:
- Same row count after null drop and empty-doc filtering.
- Similar vocabulary size and token removal percentages (allowing small ngram-order differences if training scope changes).
- Same output schema: tokens_str, token_count.
- Diagnostics JSON fields preserved.

Performance checks:
- Peak RSS lower than list-based version on same input.
- No major regression in total runtime.

## Risk Notes
- pandas object columns still store Python lists, so gains come from lifecycle control and avoiding duplicate corpus copies, not from eliminating Python object overhead entirely.
- In-place DataFrame writes can trigger hidden copies if chained indexing is used; always use .loc with explicit row indexers.
- If ngram fitting is done per batch, results may differ from corpus-wide fitting. Decide explicitly between:
  - batch-local fit (lower memory, possibly different phrases), or
  - two-pass global fit with streamed input (higher consistency, moderate memory).

## Suggested Implementation Order for Minimal Breakage
- First refactor build_vocab and filter_rare to consume/mutate df["tokens"].
- Then refactor process_batch and apply_ngrams call path.
- Finally update diagnostics and remove old tokenized assumptions.
