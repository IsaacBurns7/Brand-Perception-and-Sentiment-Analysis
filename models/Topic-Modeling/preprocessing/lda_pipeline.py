import logging

import pandas as pd
import spacy
from tqdm import tqdm

from cleaning_utils import EXTRA_STOPWORDS, clean_text
from pipeline_config import PipelineConfig
from token_vocab_utils import build_vocab, filter_rare, process_batch

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
    df = pd.read_csv(input_path, sep=sep, usecols=[text_col], low_memory=False)
    df = df.head(1000)
    log.info("  Loaded %d rows.", len(df))

    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    log.info("  %d rows after dropping nulls.", len(df))

    log.info("Cleaning text (regex pass) ...")
    cleaned = df[text_col].map(clean_text).tolist()

    log.info("Loading spaCy model ...")
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
    nlp.max_length = 2_000_000

    stopwords = EXTRA_STOPWORDS | {word.lower() for word in nlp.Defaults.stop_words}

    log.info("Running spaCy pipeline (batch_size=%d) ...", cfg.batch_size)
    tokenized: list[list[str]] = []
    batch_size = cfg.batch_size

    for i in tqdm(range(0, len(cleaned), batch_size), desc="Batches"):
        batch = cleaned[i : i + batch_size]
        tokenized.extend(process_batch(batch, nlp, stopwords, cfg))

    log.info("Building vocabulary and filtering rare tokens ...")
    vocab, term_count = build_vocab(tokenized, cfg.min_freq, cfg.min_doc_freq)
    log.info(
        "Vocabulary: %d unique tokens before filtering -> %d after "
        "(min_freq=%d, min_doc_freq=%d)",
        term_count,
        len(vocab),
        cfg.min_freq,
        cfg.min_doc_freq,
    )
    tokenized = filter_rare(tokenized, vocab)

    df["tokens"] = tokenized
    df["tokens_str"] = df["tokens"].map(" ".join)
    df["token_count"] = df["tokens"].map(len)

    before = len(df)
    df = df[df["token_count"] > 0].reset_index(drop=True)
    log.info("Dropped %d empty documents after filtering.", before - len(df))
    log.info("Final corpus: %d documents.", len(df))

    out = df[[text_col, "tokens_str", "token_count"]]
    out.to_csv(output_path, index=False)
    log.info("Saved normalized output to %s", output_path)

    log.info(
        "Token stats -> mean: %.1f | median: %.1f | min: %d | max: %d",
        df["token_count"].mean(),
        df["token_count"].median(),
        df["token_count"].min(),
        df["token_count"].max(),
    )
    return df
