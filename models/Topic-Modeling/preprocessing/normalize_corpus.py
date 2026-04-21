"""
spaCy-based Text Normalization Pipeline for LDA
================================================
Handles: Tokenization, Noise Removal, Text Normalization,
         Stop Word Removal, Lemmatization, Rare/OOV Word Handling
         NOISE REMOVAL BASED ON LDA, BEWARE IT IS A BAG OF WORDS MODEL AND MAY NOT REQUIRE THE SAME PREPROCESSING AS OTHER MODELS

Requirements:
    pip install spacy pandas tqdm
    python -m spacy download en_core_web_sm

Usage:
    look in README.md
"""

import argparse
import logging
import pandas as pd

from pipeline import run_pipeline
from pipeline_config import DEFAULT_CONFIG, PipelineConfig

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="spaCy LDA pre-processing pipeline")
    p.add_argument("--input",      required=True,  help="Path to input CSV")
    p.add_argument("--text-col",   required=True,  help="Name of the text column")
    p.add_argument("--output",     required=True,  help="Path for output CSV")
    p.add_argument("--sep",        default=",",    help="CSV separator (default: ,)")
    p.add_argument("--min-freq",   type=int, default=DEFAULT_CONFIG.min_freq,
                   help="Minimum corpus frequency to keep a token")
    p.add_argument("--min-doc-freq", type=int, default=DEFAULT_CONFIG.min_doc_freq,
                   help="Minimum document frequency to keep a token")
    p.add_argument("--min-token-len", type=int, default=DEFAULT_CONFIG.min_token_len,
                   help="Minimum token character length")
    p.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size,
                   help="spaCy pipe batch size")
    p.add_argument("--n-process",  type=int, default=DEFAULT_CONFIG.n_process,
                   help="Number of parallel spaCy workers (use 1 on Windows)")
    p.add_argument("--spacy-model", default=DEFAULT_CONFIG.spacy_model,
                   help="spaCy model name (default: en_core_web_md)")
    p.add_argument("--oov-placeholder", default=None,
                   help="Replace OOV tokens with this string instead of dropping them")
    p.add_argument("--diagnostics-output", default=None,
                   help="Optional path to write diagnostics JSON")
    p.add_argument("--diagnostics-top-n", type=int, default=DEFAULT_CONFIG.diagnostics_top_n,
                   help="Number of top terms to include in diagnostics output")
    p.add_argument("--enable-ngrams", action="store_true",
                   help="Enable n-gram detection stage (disabled by default)")
    p.add_argument("--ngram-min-count", type=int, default=DEFAULT_CONFIG.ngram_min_count,
                   help="Minimum count for n-gram phrase detection")
    p.add_argument("--ngram-threshold", type=float, default=DEFAULT_CONFIG.ngram_threshold,
                   help="Threshold for n-gram phrase detection")
    p.add_argument("--max-doc-count", type=int, default=DEFAULT_CONFIG.max_doc_count)
    p.add_argument(
        "--topic-model-target",
        choices=["lda", "bertopic"],
        default=DEFAULT_CONFIG.topic_model_target,
        help="Select preprocessing target behavior (default: lda)",
    )
    p.add_argument(
        "--bertopic-min-words",
        type=int,
        default=DEFAULT_CONFIG.bertopic_min_words,
        help="Minimum word count for a cleaned document in BERTopic mode",
    )
    p.add_argument(
        "--bertopic-drop-empty-text",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG.bertopic_drop_empty_text,
        help="Drop short/empty cleaned documents in BERTopic mode",
    )
    p.add_argument(
        "--bertopic-keep-original-text",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG.bertopic_keep_original_text,
        help="Include original source text column in BERTopic output",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = PipelineConfig(
        topic_model_target=args.topic_model_target,
        spacy_model=args.spacy_model,
        min_token_len=args.min_token_len,
        max_token_len=DEFAULT_CONFIG.max_token_len,
        min_freq=args.min_freq,
        min_doc_freq=args.min_doc_freq,
        batch_size=args.batch_size,
        n_process=args.n_process,
        oov_placeholder=args.oov_placeholder,
        diagnostics_output=args.diagnostics_output,
        diagnostics_top_n=args.diagnostics_top_n,
        enable_ngrams=args.enable_ngrams,
        ngram_min_count=args.ngram_min_count,
        ngram_threshold=args.ngram_threshold,
        max_doc_count=args.max_doc_count,
        bertopic_min_words=args.bertopic_min_words,
        bertopic_drop_empty_text=args.bertopic_drop_empty_text,
        bertopic_keep_original_text=args.bertopic_keep_original_text,
    )

    # Stream the input CSV in chunks and call run_pipeline for each chunk.
    # This respects the configured maximum document count (`cfg.max_doc_count`).
    chunksize = 5000
    processed = 0
    for chunk in pd.read_csv(args.input, sep=args.sep, usecols=[args.text_col], chunksize=chunksize, low_memory=False):
        # enforce global max doc count if set
        remaining = cfg.max_doc_count - processed if cfg.max_doc_count is not None else None
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None and len(chunk) > remaining:
            chunk = chunk.iloc[:remaining]

        processed += len(chunk)

        run_pipeline(
            input_path = args.input,
            df = chunk,
            text_col    = args.text_col,
            output_path = args.output,
            cfg         = cfg,
            sep         = args.sep,
        )
