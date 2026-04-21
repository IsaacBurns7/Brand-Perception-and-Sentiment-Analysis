import logging
import json
from collections import Counter
from pathlib import Path
import os

import pandas as pd
import spacy
from tqdm import tqdm

from benchmark import benchmark
from cleaning_utils import EXTRA_STOPWORDS, clean_text
from pipeline_config import PipelineConfig
from token_vocab_utils import apply_ngrams, build_vocab, filter_rare, process_batch

log = logging.getLogger(__name__)


def _safe_count_stats(counts: pd.Series) -> dict[str, float | int] | None:
    if counts.empty:
        return None
    return {
        "mean": float(counts.mean()),
        "median": float(counts.median()),
        "min": int(counts.min()),
        "max": int(counts.max()),
    }


def run_pipeline(
    input_path: str,
    df: pd.DataFrame,
    text_col: str,
    output_path: str,
    cfg: PipelineConfig,
    sep: str = ",",
) -> pd.DataFrame:
    with benchmark("Validate configuration"):
        cfg.validate()

    with benchmark("Load CSV and cap document count"):
        # log.info("Loading data from %s ...", input_path)
        # I/O hotspot: large CSV scan/parsing from disk (420 MB input scale).
        # High memory overhead: reads selected CSV column into a full in-memory DataFrame.
        # df = pd.read_csv(input_path, sep=sep, usecols=[text_col], low_memory=False)
        # df = df.head(cfg.max_doc_count)
        log.info("  Loaded %d rows.", len(df))

    with benchmark("Drop null rows"):
        # High memory overhead: dropna+reset_index creates a new DataFrame copy.
        df = df.dropna(subset=[text_col]).reset_index(drop=True)
        log.info("  %d rows after dropping nulls.", len(df))

    nlp = None
    stopwords: set[str] = set()
    if cfg.topic_model_target == "lda":
        with benchmark("Load spaCy model and stopwords"):
            log.info("Loading spaCy model '%s' ...", cfg.spacy_model)
            try:
                nlp = spacy.load(cfg.spacy_model, disable=["parser", "ner"])
            except OSError as e:
                raise RuntimeError(
                    f"Failed to load spaCy model '{cfg.spacy_model}'. "
                    "Ensure it is installed by running "
                    f"'python -m spacy download {cfg.spacy_model}', or configure the "
                    "pipeline to use a different spaCy model."
                ) from e
            nlp.max_length = 2_000_000

            stopwords = EXTRA_STOPWORDS | {word.lower() for word in nlp.Defaults.stop_words}

        with benchmark("Initialize token columns"):
            log.info("Running spaCy pipeline (batch_size=%d) ...", cfg.batch_size)
            df["tokens"] = None
            df["tokens_str"] = ""
            df["token_count"] = 0

    with benchmark("Clean text batches"):
        log.info("Cleaning text in batches (in-place DataFrame updates) ...")
        batch_size = cfg.batch_size
        for i in tqdm(range(0, len(df), batch_size), desc="Clean Batches"):
            row_indexer = df.index[i : i + batch_size]
            df.loc[row_indexer, text_col] = df.loc[row_indexer, text_col].map(clean_text)

    if cfg.topic_model_target == "bertopic":
        with benchmark("Prepare BERTopic text output"):
            before = len(df)
            df["cleaned_text"] = df[text_col].astype(str)
            df["token_count_light"] = df["cleaned_text"].str.split().map(len).astype("int32", copy=False)

            dropped_docs = 0
            if cfg.bertopic_drop_empty_text:
                df = df[df["token_count_light"] >= cfg.bertopic_min_words].reset_index(drop=True)
                dropped_docs = before - len(df)

            log.info("Dropped %d short/empty documents for BERTopic mode.", dropped_docs)
            log.info("Final corpus: %d documents.", len(df))

        with benchmark("Write normalized output CSV"):
            out_cols = ["cleaned_text", "token_count_light"]
            if cfg.bertopic_keep_original_text:
                out_cols = [text_col] + out_cols
            out = df[out_cols]
            file_exists = os.path.isfile(output_path)
            out.to_csv(output_path, index=False, mode='a', header=not file_exists) #append to output csv if it already exists 
            log.info("Saved normalized output to %s", output_path)

        stats = _safe_count_stats(df["token_count_light"])
        if stats is None:
            log.info("Token stats -> corpus is empty after BERTopic preprocessing.")
        else:
            log.info(
                "Token stats -> mean: %.1f | median: %.1f | min: %d | max: %d",
                stats["mean"],
                stats["median"],
                stats["min"],
                stats["max"],
            )

        with benchmark("Optional diagnostics generation"):
            if cfg.diagnostics_output:
                top_terms = Counter(
                    token
                    for doc in df["cleaned_text"].str.split()
                    for token in doc
                ).most_common(cfg.diagnostics_top_n)

                diagnostics_path = Path(cfg.diagnostics_output)
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

                if before > 0:
                    empty_docs_removed_pct = (dropped_docs / before) * 100.0
                else:
                    empty_docs_removed_pct = 0.0

                diagnostics_payload = {
                    "mode": "bertopic",
                    "input_path": input_path,
                    "output_path": output_path,
                    "text_col": text_col,
                    "pre_empty_filter_docs": before,
                    "final_docs": len(df),
                    "dropped_empty_docs": dropped_docs,
                    "empty_docs_removed_pct": round(empty_docs_removed_pct, 4),
                    "token_stats": (
                        {
                            "mean": float(stats["mean"]),
                            "median": float(stats["median"]),
                            "min": int(stats["min"]),
                            "max": int(stats["max"]),
                        }
                        if stats is not None
                        else {
                            "mean": None,
                            "median": None,
                            "min": None,
                            "max": None,
                        }
                    ),
                    "top_terms": [{"term": term, "count": count} for term, count in top_terms],
                }

                diagnostics_path.write_text(
                    json.dumps(diagnostics_payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                log.info("Saved diagnostics output to %s", diagnostics_path)

        return df

    with benchmark("Tokenize corpus with spaCy"):
        log.info("Tokenizing full corpus in one spaCy pipe pass (n_process=%d) ...", cfg.n_process)
        # Keep one long-lived nlp.pipe invocation to avoid repeated worker pool teardown/startup.
        process_batch(
            df,
            text_col,
            nlp,
            stopwords,
            cfg,
            tokens_col="tokens",
            # logger = log,
        )

    with benchmark("Optional n-gram detection"):
        if cfg.enable_ngrams:
            log.info(
                "Applying n-gram detection (min_count=%d, threshold=%.2f) ...",
                cfg.ngram_min_count,
                cfg.ngram_threshold,
            )
            apply_ngrams(df, cfg, tokens_col="tokens")

    with benchmark("Compute token strings and counts"):
        df["tokens_str"] = df["tokens"].map(" ".join)
        df["token_count"] = df["tokens"].map(len).astype("int32", copy=False)

    with benchmark("Build vocabulary and filter rare tokens"):
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

    with benchmark("Drop empty documents"):
        before = len(df)
        df = df[df["token_count"] > 0].reset_index(drop=True)
        dropped_docs = before - len(df)
        log.info("Dropped %d empty documents after filtering.", dropped_docs)
        log.info("Final corpus: %d documents.", len(df))

    with benchmark("Write normalized output CSV"):
        # out = df[[text_col, "tokens_str", "token_count"]] #prints original article, adds lowkey too much data
        out = df[["tokens_str", "token_count"]]
        # I/O hotspot: writes full normalized corpus to disk; throughput depends on filesystem speed.
        file_exists = os.path.isfile(output_path)
        out.to_csv(
            output_path, 
            mode='a', 
            index=False, 
            header=not file_exists
        )
        log.info("Saved normalized output to %s", output_path)

    stats = _safe_count_stats(df["token_count"])
    if stats is None:
        log.info("Token stats -> corpus is empty after LDA preprocessing.")
    else:
        log.info(
            "Token stats -> mean: %.1f | median: %.1f | min: %d | max: %d",
            stats["mean"],
            stats["median"],
            stats["min"],
            stats["max"],
        )

    with benchmark("Optional diagnostics generation"):
        if cfg.diagnostics_output:
            # CPU hotspot: full-corpus flatten + counting touches every token again.
            # High memory overhead: computes global term counts across all tokens.
            top_terms = Counter(
                token
                for doc in df["tokens"]
                if isinstance(doc, list)
                for token in doc
            ).most_common(cfg.diagnostics_top_n)
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
                "pre_empty_filter_docs": before,
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
                    "mean": float(stats["mean"]) if stats is not None else None,
                    "median": float(stats["median"]) if stats is not None else None,
                    "min": int(stats["min"]) if stats is not None else None,
                    "max": int(stats["max"]) if stats is not None else None,
                },
                "top_terms": [{"term": term, "count": count} for term, count in top_terms],
            }

            diagnostics_path.write_text(
                # I/O hotspot: serializes and writes diagnostics payload to disk in one operation.
                json.dumps(diagnostics_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            log.info("Saved diagnostics output to %s", diagnostics_path)

    return df
