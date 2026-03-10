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

import re
import argparse
import logging
from pathlib import Path
from collections import Counter

import pandas as pd
import spacy
from tqdm import tqdm

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "min_token_len": 3,        # drop tokens shorter than this
    "max_token_len": 40,       # drop tokens longer than this (garbled text)
    "min_freq": 5,             # rare-word threshold (absolute count)
    "min_doc_freq": 2,         # drop tokens that appear in fewer than N docs
    "batch_size": 500,         # spaCy pipe batch size
    "n_process": 1,            # parallel workers (set >1 on Linux; use 1 on Windows)
    "oov_placeholder": None,   # replace OOV tokens with this string, or None to drop
}

# Extended stop words on top of spaCy's built-in list
EXTRA_STOPWORDS = {
    "said", "say", "says", "would", "could", "also", "one",
    "two", "three", "new", "like", "get", "make", "know",
    "use", "just", "year", "time", "way", "day", "man",
    "woman", "people", "thing",
}


# ── Noise Removal ─────────────────────────────────────────────────────────────
_URL_RE      = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE     = re.compile(r"<[^>]+>")
_EMAIL_RE    = re.compile(r"\S+@\S+\.\S+")
_MENTION_RE  = re.compile(r"@\w+")
_HASHTAG_RE  = re.compile(r"#\w+")
_NUM_RE      = re.compile(r"\b\d+\b")               # standalone numbers
_PUNCT_RE    = re.compile(r"[^\w\s]")               # all remaining punctuation
_WHITESPACE  = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Step 1 & 3 – Noise removal + basic text normalisation."""
    # print("clean text IN: ", text)
    if not isinstance(text, str):
        return ""
    text = text.lower()                              # lowercase (normalisation)
    text = _URL_RE.sub(" ", text)                    # remove URLs
    text = _HTML_RE.sub(" ", text)                   # strip HTML tags
    text = _EMAIL_RE.sub(" ", text)                  # remove e-mail addresses
    text = _MENTION_RE.sub(" ", text)                # remove @mentions
    text = _HASHTAG_RE.sub(" ", text)                # remove #hashtags
    text = text.replace("\n", " ").replace("\r", " ")
    text = _NUM_RE.sub(" ", text)                    # drop bare numbers
    text = _PUNCT_RE.sub(" ", text)                  # remove punctuation
    text = _WHITESPACE.sub(" ", text).strip()        # collapse whitespace
    # print("clean text OUT: ", text)
    return text


# ── spaCy Token Filter ────────────────────────────────────────────────────────

def is_valid_token(token, stopwords: set, cfg: dict) -> bool:
    """
    Returns True if the token should be kept.
    Combines: stop word removal, POS filtering, length filtering.
    """
    lemma = token.lemma_

    # Step 4 – Stop word removal (spaCy built-in + extras)
    if token.is_stop or lemma in stopwords:
        return False

    # Step 2 – Noise removal at token level
    if token.is_punct or token.is_space or token.like_num:
        return False

    # POS filter: keep only nouns, verbs, adjectives, adverbs
    if token.pos_ not in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}:
        return False

    # Length guard
    if not (cfg["min_token_len"] <= len(lemma) <= cfg["max_token_len"]):
        return False

    # Alphabetic only (catches leftover symbols)
    if not lemma.isalpha():
        return False

    return True


# ── OOV Handling ──────────────────────────────────────────────────────────────

def handle_oov(token, cfg: dict) -> str | None:
    """
    Step 6 – Handle out-of-vocabulary / unknown tokens.
    If the token has no word vector AND is flagged OOV, either
    replace it with a placeholder or drop it (return None).
    """
    if token.is_oov:
        # print("token determined to be out of vocabulary: ", token)
        return cfg.get("oov_placeholder")   # None → will be filtered downstream
    return token.lemma_


# ── Core Processing ───────────────────────────────────────────────────────────

def process_batch(texts: list[str], nlp, stopwords: set, cfg: dict) -> list[list[str]]:
    """
    Run a batch of pre-cleaned texts through the spaCy pipeline.
    Returns a list of token lists (one per document).
    """
    print("BATCH PROCESSING TEXTS IN: ", texts)
    results = []
    docs = nlp.pipe(texts, batch_size=cfg["batch_size"], n_process=cfg["n_process"])
    # print("DOCS AFTER NLP PIPE: ", docs)
    for doc in docs:
        tokens = []
        for token in doc:
            # print("token trying to pass through: ", token)
            if not is_valid_token(token, stopwords, cfg):
                continue
            # Step 5 – Lemmatisation + Step 6 – OOV handling
            lemma = handle_oov(token, cfg)
            if not lemma:
                continue
            tokens.append(lemma)
            # print("token that passed through lemma: ", lemma)

        results.append(tokens)
    print("BATCH PROCESSING RESULTS OUT: ",results)
    return results


# ── Rare-word Removal ─────────────────────────────────────────────────────────

def build_vocab(tokenized_docs: list[list[str]], min_freq: int, min_doc_freq: int):
    """
    Build a vocabulary from a fully tokenised corpus.
    Removes tokens that are too rare (corpus-level and document-level).
    """
    term_freq   = Counter()
    doc_freq    = Counter()

    for doc in tokenized_docs:
        term_freq.update(doc)
        doc_freq.update(set(doc))   # count each term once per document

    vocab = {
        term for term, freq in term_freq.items()
        if freq >= min_freq and doc_freq[term] >= min_doc_freq
    }
    log.info(
        "Vocabulary: %d unique tokens before filtering → %d after "
        "(min_freq=%d, min_doc_freq=%d)",
        len(term_freq), len(vocab), min_freq, min_doc_freq,
    )
    return vocab


def filter_rare(tokenized_docs: list[list[str]], vocab: set) -> list[list[str]]:
    """Remove rare / OOV tokens from all documents using the final vocab."""
    return [[t for t in doc if t in vocab] for doc in tokenized_docs]


# ── Pipeline Orchestration ────────────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    text_col: str,
    output_path: str,
    cfg: dict,
    sep: str = ",",
) -> pd.DataFrame:

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log.info("Loading data from %s …", input_path)
    df = pd.read_csv(input_path, sep=sep, usecols=[text_col], low_memory=False)
    # df = df.head(100)
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
    log.info("Running spaCy pipeline (batch_size=%d) …", cfg["batch_size"])
    tokenized: list[list[str]] = []
    batch_size = cfg["batch_size"]

    for i in tqdm(range(0, len(cleaned), batch_size), desc="Batches"):
        batch = cleaned[i : i + batch_size]
        tokenized.extend(process_batch(batch, nlp, stopwords, cfg))

    # ── 5. Rare-word removal ──────────────────────────────────────────────────
    log.info("Building vocabulary and filtering rare tokens …")
    vocab = build_vocab(tokenized, cfg["min_freq"], cfg["min_doc_freq"])
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
    p.add_argument("--min-freq",   type=int, default=DEFAULT_CONFIG["min_freq"],
                   help="Minimum corpus frequency to keep a token")
    p.add_argument("--min-doc-freq", type=int, default=DEFAULT_CONFIG["min_doc_freq"],
                   help="Minimum document frequency to keep a token")
    p.add_argument("--min-token-len", type=int, default=DEFAULT_CONFIG["min_token_len"],
                   help="Minimum token character length")
    p.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
                   help="spaCy pipe batch size")
    p.add_argument("--n-process",  type=int, default=DEFAULT_CONFIG["n_process"],
                   help="Number of parallel spaCy workers (use 1 on Windows)")
    p.add_argument("--oov-placeholder", default=None,
                   help="Replace OOV tokens with this string instead of dropping them")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = {**DEFAULT_CONFIG}
    cfg["min_freq"]         = args.min_freq
    cfg["min_doc_freq"]     = args.min_doc_freq
    cfg["min_token_len"]    = args.min_token_len
    cfg["batch_size"]       = args.batch_size
    cfg["n_process"]        = args.n_process
    cfg["oov_placeholder"]  = args.oov_placeholder

    run_pipeline(
        input_path  = args.input,
        text_col    = args.text_col,
        output_path = args.output,
        cfg         = cfg,
        sep         = args.sep,
    )