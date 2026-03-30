import logging
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
from tqdm import tqdm

from cleaning_utils import EXTRA_STOPWORDS, clean_text
from pipeline_config import PipelineConfig
from token_vocab_utils import apply_ngrams, build_vocab, filter_rare, process_batch

log = logging.getLogger(__name__)


def run_pipeline(
    input_path: str,
    text_col: str,
    output_path: str,
    cfg: PipelineConfig,
    sep: str = ",",
) -> pd.DataFrame:
    cfg.validate()

    log.info("Loading data from %s ...", input_path)
    # I/O hotspot: large CSV scan/parsing from disk (420 MB input scale).
    # High memory overhead: reads selected CSV column into a full in-memory DataFrame.
    df = pd.read_csv(input_path, sep=sep, usecols=[text_col], low_memory=False)
    df = df.head(cfg.max_doc_count)
    log.info("  Loaded %d rows.", len(df))

    # High memory overhead: dropna+reset_index creates a new DataFrame copy.
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    log.info("  %d rows after dropping nulls.", len(df))

    log.info("Loading spaCy model ...")
    try:
        nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
    except OSError as e:
        raise RuntimeError(
            "Failed to load spaCy model 'en_core_web_md'. "
            "Ensure it is installed by running "
            "'python -m spacy download en_core_web_md', or configure the "
            "pipeline to use a different spaCy model."
        ) from e
    nlp.max_length = 2_000_000

    stopwords = EXTRA_STOPWORDS | {word.lower() for word in nlp.Defaults.stop_words}

    log.info("Running spaCy pipeline (batch_size=%d) ...", cfg.batch_size)
    batch_size = cfg.batch_size

    df["tokens"] = None
    df["tokens_str"] = ""
    df["token_count"] = 0

    log.info("Cleaning + tokenizing in batches (in-place DataFrame updates) ...")

    for i in tqdm(range(0, len(df), batch_size), desc="Batches"):
        row_indexer = df.index[i : i + batch_size]
        df.loc[row_indexer, text_col] = df.loc[row_indexer, text_col].map(clean_text)

        # CPU hotspot: token-level spaCy processing and filtering for each item in this batch.
        process_batch(
            df,
            text_col,
            row_indexer,
            nlp,
            stopwords,
            cfg,
            tokens_col="tokens",
        )

    if cfg.enable_ngrams:
        log.info(
            "Applying n-gram detection (min_count=%d, threshold=%.2f) ...",
            cfg.ngram_min_count,
            cfg.ngram_threshold,
        )
        apply_ngrams(df, cfg, tokens_col="tokens")

    df["tokens_str"] = df["tokens"].map(" ".join)
    df["token_count"] = df["tokens"].map(len).astype("int32", copy=False)

    log.info("Building vocabulary and filtering rare tokens ...")
    # High memory overhead: vocabulary building traverses full corpus and keeps frequency maps.
    vocab, term_count, token_count_before_filter = build_vocab(
        df,
        cfg.min_freq,
        cfg.min_doc_freq,
        tokens_col="tokens",
    )
    log.info(
        "Vocabulary: %d unique tokens before filtering -> %d after "
        "(min_freq=%d, min_doc_freq=%d)",
        term_count,
        len(vocab),
        cfg.min_freq,
        cfg.min_doc_freq,
    )
    token_count_before_filter, token_count_after_filter = filter_rare(
        df,
        vocab,
        tokens_col="tokens",
        token_count_col="token_count",
    )
    df["tokens_str"] = df["tokens"].map(" ".join)
    removed_tokens = token_count_before_filter - token_count_after_filter

    if token_count_before_filter > 0:
        token_removed_pct = (removed_tokens / token_count_before_filter) * 100.0
    else:
        token_removed_pct = 0.0

    before = len(df)
    df = df[df["token_count"] > 0].reset_index(drop=True) #oh wow
    dropped_docs = before - len(df)
    log.info("Dropped %d empty documents after filtering.", dropped_docs)
    log.info("Final corpus: %d documents.", len(df))

    # out = df[[text_col, "tokens_str", "token_count"]] #prints original article, adds lowkey too much data 
    out = df[["tokens_str", "token_count"]]
    # I/O hotspot: writes full normalized corpus to disk; throughput depends on filesystem speed.
    out.to_csv(output_path, index=False)
    log.info("Saved normalized output to %s", output_path)

    log.info(
        "Token stats -> mean: %.1f | median: %.1f | min: %d | max: %d",
        df["token_count"].mean(),
        df["token_count"].median(),
        df["token_count"].min(),
        df["token_count"].max(),
    )

    if cfg.diagnostics_output:
        # CPU hotspot: full-corpus flatten + counting touches every token again.
        # High memory overhead: computes global term counts across all tokens.
        top_terms = Counter(token for doc in df["tokens"] for token in doc).most_common(cfg.diagnostics_top_n)
        diagnostics_path = Path(cfg.diagnostics_output)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

        if before > 0:
            empty_docs_removed_pct = (dropped_docs / before) * 100.0
        else:
            empty_docs_removed_pct = 0.0

        diagnostics_payload = {
            "input_path": input_path,
            "output_path": output_path,
            "text_col": text_col,
            "loaded_docs": before,
            "final_docs": len(df),
            "dropped_empty_docs": dropped_docs,
            "empty_docs_removed_pct": round(empty_docs_removed_pct, 4),
            "tokens_before_rare_filter": token_count_before_filter,
            "tokens_after_rare_filter": token_count_after_filter,
            "tokens_removed_by_rare_filter": removed_tokens,
            "tokens_removed_by_rare_filter_pct": round(token_removed_pct, 4),
            "vocab_size_before_filtering": term_count,
            "vocab_size_after_filtering": len(vocab),
            "ngrams": {
                "enabled": cfg.enable_ngrams,
                "min_count": cfg.ngram_min_count,
                "threshold": cfg.ngram_threshold,
            },
            "token_stats": {
                "mean": float(df["token_count"].mean()),
                "median": float(df["token_count"].median()),
                "min": int(df["token_count"].min()),
                "max": int(df["token_count"].max()),
            },
            "top_terms": [{"term": term, "count": count} for term, count in top_terms],
        }

        diagnostics_path.write_text(
            # I/O hotspot: serializes and writes diagnostics payload to disk in one operation.
            json.dumps(diagnostics_payload, indent=2) + "\n",
            encoding="ascii",
        )
        log.info("Saved diagnostics output to %s", diagnostics_path)

    return df
