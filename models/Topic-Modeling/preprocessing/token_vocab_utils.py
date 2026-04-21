from collections import Counter

try:
    from gensim.models import Phrases
    from gensim.models.phrases import Phraser
except ImportError:  # Allow pipeline to run without gensim when n-grams are disabled
    Phrases = None
    Phraser = None
import pandas as pd
import numpy as np
from spacy.attrs import IS_STOP, IS_PUNCT, IS_SPACE, LIKE_NUM, POS
from tqdm import tqdm

from pipeline_config import PipelineConfig


def _preview(value, cutoff: int = 200) -> str:
    text = str(value)
    if len(text) <= cutoff:
        return text
    return text[:cutoff] + "..."


# def is_valid_token(token, stopwords: set[str], cfg: PipelineConfig) -> bool:
#     # CPU hotspot: this predicate executes multiple attribute checks for every token.
#     lemma = token.lemma_
#     if token.is_stop or lemma in stopwords:
#         return False
#     if token.is_punct or token.is_space or token.like_num:
#         return False
#     if token.pos_ not in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}: #possibly try deleting later in evaluation phase, may improve performance on more complex documents
#         return False
#     if not (cfg.min_token_len <= len(lemma) <= cfg.max_token_len):
#         return False
#     if not lemma.isalpha():
#         return False

#     return True


# def handle_oov(token, cfg: PipelineConfig) -> str | None:
#     if token.is_oov:
#         return cfg.oov_placeholder
#     return token.lemma_

# def _extract_tokens(doc, stopwords, cfg):
#     valid_pos = {"NOUN","VERB","ADJ","ADV","PROPN"}
#     min_len, max_len = cfg.min_token_len, cfg.max_token_len
#     placeholder = cfg.oov_placeholder
#     tokens = []
#     for token in doc:
#         if token.is_stop or token.is_punct or token.is_space or token.like_num:
#             continue
#         if token.pos_ not in valid_pos:
#             continue
#         lemma = token.lemma_          # single lookup
#         if lemma in stopwords:
#             continue
#         if not (min_len <= len(lemma) <= max_len) or not lemma.isalpha():
#             continue
#         tokens.append(placeholder if token.is_oov else lemma)
#     return tokens


def _extract_tokens_vectorized(doc, stopwords, cfg, VALID_POS_IDS, logger):
    if logger is not None:
        logger.info(
            "Extracting tokens: type=%s len=%d preview=%s",
            type(doc).__name__,
            len(doc) if doc is not None else 0,
            _preview(doc.text if doc is not None else "", cutoff=200),
        )
    if not doc:
        return []
    arr = doc.to_array([IS_STOP, IS_PUNCT, IS_SPACE, LIKE_NUM, POS])
    mask = (
        (arr[:, 0] == 0) &   # not stop
        (arr[:, 1] == 0) &   # not punct
        (arr[:, 2] == 0) &   # not space
        (arr[:, 3] == 0) &   # not num
        (np.isin(arr[:, 4], list(VALID_POS_IDS)))
    )
    # Stage 1: tokens passing numeric mask
    masked_tokens = [doc[i] for i in np.where(mask)[0]]
    if logger is not None:
        logger.info(
            "After numeric mask: total=%d masked=%d preview=%s",
            len(doc),
            len(masked_tokens),
            [t.text for t in masked_tokens[:5]],
        )

    # Stage 2: drop oov tokens with no placeholder
    oov_filtered = [
        t for t in masked_tokens
        if not (t.is_oov and cfg.oov_placeholder is None)
    ]
    if logger is not None:
        logger.info(
            "After OOV filter: before=%d after=%d dropped=%d",
            len(masked_tokens),
            len(oov_filtered),
            len(masked_tokens) - len(oov_filtered),
        )

    # Stage 3: resolve to lemma or placeholder
    lemmatized = [
        (cfg.oov_placeholder if t.is_oov else t.lemma_)
        for t in oov_filtered
    ]
    if logger is not None:
        logger.info(
            "After lemmatization: count=%d preview=%s",
            len(lemmatized),
            lemmatized[:5],
        )

    # Stage 4: stopword, length, and alpha filters
    ans = [
        l for l in lemmatized
        if l not in stopwords
        and cfg.min_token_len <= len(l) <= cfg.max_token_len
        and l.isalpha()
    ]
    if logger is not None:
        logger.info(
            "After stopword/length/alpha filter: before=%d after=%d preview=%s",
            len(lemmatized),
            len(ans),
            ans[:5],
        )

    if logger is not None:
        logger.info(
            "Extracted tokens: count=%d preview=%s",
            len(ans),
            _preview(ans, cutoff=200),
        )
    return ans

def process_batch(
    df: pd.DataFrame,
    text_col: str,
    nlp,
    stopwords: set[str],
    cfg: PipelineConfig,
    *,
    tokens_col: str = "tokens",
    logger = None
) -> None:
    """Tokenize full DataFrame text column and write tokens in place."""
    texts = df[text_col].tolist()
    if logger is not None:
        logger.info(
            "Tokenization input: docs=%d first_text_preview=%s",
            len(texts),
            _preview(texts[0], cutoff=200) if texts else "",
        )
    docs = nlp.pipe(texts, batch_size=cfg.batch_size)#, n_process=cfg.n_process)
    docs_with_progress = tqdm(
        docs,
        total=len(texts),
        desc="Tokenize Docs",
        unit="doc",
        mininterval=100,
    )
    VALID_POS_IDS = {nlp.vocab.strings[p] for p in {"NOUN","VERB","ADJ","ADV","PROPN"}}

    try:
        df[tokens_col] = pd.Series(
            (_extract_tokens_vectorized(doc, stopwords, cfg, VALID_POS_IDS, logger=logger) for doc in docs_with_progress),
            index=df.index,
        )
        #logger.info("TOKENS COL: ", df[tokens_col])
    finally:
        close_docs = getattr(docs, "close", None)
        if callable(close_docs):
            close_docs()


def build_vocab(
    df: pd.DataFrame,
    min_freq: int,
    min_doc_freq: int,
    *,
    tokens_col: str = "tokens",
) -> tuple[set[str], int, int]:
    """Build vocabulary from ``df[tokens_col]`` without corpus copies."""
    # High memory overhead: keeps full term-frequency and doc-frequency counters in memory.
    term_freq = Counter()
    doc_freq = Counter()
    total_tokens = 0

    # CPU hotspot: full corpus scan with both term and per-document set updates.
    for doc in df[tokens_col]:
        if not isinstance(doc, list):
            continue
        term_freq.update(doc)
        doc_freq.update(set(doc))
        total_tokens += len(doc)

    vocab = {
        term for term, freq in term_freq.items()
        if freq >= min_freq and doc_freq[term] >= min_doc_freq
    }
    return vocab, len(term_freq), total_tokens


def filter_rare(
    df: pd.DataFrame,
    vocab: set[str],
    *,
    tokens_col: str = "tokens",
    token_count_col: str = "token_count",
) -> tuple[int, int]:
    """Filter rare tokens in place and update token counts."""
    token_count_before_filter = 0
    for doc in df[tokens_col]:
        if isinstance(doc, list):
            token_count_before_filter += len(doc)

    # Use set membership for faster filtering over large vocabularies.
    vocab_lookup = set(vocab)

    # CPU hotspot: list-comprehension membership checks run across every token.
    # High memory overhead: this still materializes a new list per row, but avoids full-corpus copies.
    df[tokens_col] = df[tokens_col].map(
        lambda doc: [token for token in doc if token in vocab_lookup] if isinstance(doc, list) else []
    )
    df[token_count_col] = df[tokens_col].map(len).astype("int32", copy=False)
    token_count_after_filter = int(df[token_count_col].sum())
    return token_count_before_filter, token_count_after_filter


def apply_ngrams(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    tokens_col: str = "tokens",
) -> None:
    """Optionally apply corpus-wide bigram detection in place."""
    if not cfg.enable_ngrams:
        return

    tokenized_docs = df[tokens_col].tolist()
    if not tokenized_docs:
        return

    # CPU hotspot: phrase scoring/training across the corpus can be expensive on large inputs.
    # High memory overhead: phrase model training and transformed output can both be large.
    bigram = Phrases(tokenized_docs, min_count=cfg.ngram_min_count, threshold=cfg.ngram_threshold)
    bigram_model = Phraser(bigram)
    #Adding trigrams may improve performance, try at evaluation 
    transformed = [list(bigram_model[doc]) for doc in tokenized_docs]
    df[tokens_col] = pd.Series(transformed, index=df.index)
