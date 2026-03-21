"""
spaCy-based Text Normalization Pipeline for LDA
================================================
Handles: Tokenization, Noise Removal, Text Normalization,
         Stop Word Removal, Lemmatization, Rare/OOV Word Handling
         NOISE REMOVAL BASED ON LDA, BECAUSE IT IS BAG OF WORDS MODEL

Requirements:
    pip install spacy pandas tqdm
    python -m spacy download en_core_web_md

Usage:
    python normalize_for_lda.py --input articles.csv --text-col content --output normalized.csv
    python normalize_for_lda.py --input articles.csv --text-col content --output normalized.csv --min-freq 5 --batch-size 500
"""

import argparse
import logging

import pandas as pd
import spacy
from tqdm import tqdm

from cleaning_utils import EXTRA_STOPWORDS, clean_text
from pipeline_config import DEFAULT_CONFIG, PipelineConfig
from token_vocab_utils import build_vocab, filter_rare, process_batch

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Pipeline Orchestration ────────────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    text_col: str,
    output_path: str,
    cfg: PipelineConfig,
    sep: str = ",",
) -> pd.DataFrame:

    cfg.validate()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log.info("Loading data from %s …", input_path)
    df = pd.read_csv(input_path, sep=sep, usecols=[text_col], low_memory=False)
    df = df.head(1000)
    log.info("  Loaded %d rows.", len(df))

    # Drop empty rows
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    log.info("  %d rows after dropping nulls.", len(df))

    # ── 2. Noise removal + text normalisation (regex, fast) ───────────────────
    log.info("Cleaning text (regex pass) …")
    cleaned = df[text_col].map(clean_text).tolist()

    # ── 3. Load spaCy ─────────────────────────────────────────────────────────
    log.info("Loading spaCy model …")
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])  # keep tagger/morphologizer
    nlp.max_length = 2_000_000

    # Merge extra stopwords into spaCy's list
    stopwords = EXTRA_STOPWORDS | {w.lower() for w in nlp.Defaults.stop_words}

    # ── 4. Tokenisation, stop-word removal, lemmatisation, OOV handling ───────
    log.info("Running spaCy pipeline (batch_size=%d) …", cfg.batch_size)
    tokenized: list[list[str]] = []
    batch_size = cfg.batch_size

    for i in tqdm(range(0, len(cleaned), batch_size), desc="Batches"):
        batch = cleaned[i : i + batch_size]
        tokenized.extend(process_batch(batch, nlp, stopwords, cfg))

    # -- FOR SURE WANT TO ADD DETECTING BIGRAMS HERE -- # 
    # bigram = Phrases(tokenized, min_count=15, threshold=10)
    # bigram_model = Phraser(bigram)
    # tokenized = [bigram_model[doc] for doc in tokenized]


    #maybe for each step I output a small random portion(sample like 1000 docs, and output to inspect?)
    #maybe I should have each step be in its own file if this gets to be too big? 

    # ── 5. Rare-word removal ──────────────────────────────────────────────────
    log.info("Building vocabulary and filtering rare tokens …")
    vocab, term_count = build_vocab(tokenized, cfg.min_freq, cfg.min_doc_freq)
    log.info(
        "Vocabulary: %d unique tokens before filtering → %d after "
        "(min_freq=%d, min_doc_freq=%d)",
        term_count,
        len(vocab),
        cfg.min_freq,
        cfg.min_doc_freq,
    )
    tokenized = filter_rare(tokenized, vocab)

    # ── 6. Assemble output ────────────────────────────────────────────────────
    df["tokens"]          = tokenized
    df["tokens_str"]      = df["tokens"].map(" ".join)   # space-joined string (for gensim)
    df["token_count"]     = df["tokens"].map(len)

    # Drop documents that ended up empty
    before = len(df)
    df = df[df["token_count"] > 0].reset_index(drop=True)
    log.info("Dropped %d empty documents after filtering.", before - len(df))
    log.info("Final corpus: %d documents.", len(df))

    # ── 7. Save ───────────────────────────────────────────────────────────────
    out = df[[text_col, "tokens_str", "token_count"]]
    out.to_csv(output_path, index=False)
    log.info("Saved normalized output to %s", output_path)

    # Quick stats
    log.info(
        "Token stats → mean: %.1f | median: %.1f | min: %d | max: %d",
        df["token_count"].mean(),
        df["token_count"].median(),
        df["token_count"].min(),
        df["token_count"].max(),
    )
    return df


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
    p.add_argument("--oov-placeholder", default=None,
                   help="Replace OOV tokens with this string instead of dropping them")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = PipelineConfig(
        min_token_len=args.min_token_len,
        max_token_len=DEFAULT_CONFIG.max_token_len,
        min_freq=args.min_freq,
        min_doc_freq=args.min_doc_freq,
        batch_size=args.batch_size,
        n_process=args.n_process,
        oov_placeholder=args.oov_placeholder,
    )

    run_pipeline(
        input_path  = args.input,
        text_col    = args.text_col,
        output_path = args.output,
        cfg         = cfg,
        sep         = args.sep,
    )