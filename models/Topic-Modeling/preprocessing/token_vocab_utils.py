from collections import Counter

from gensim.models import Phrases
from gensim.models.phrases import Phraser
import pandas as pd

from pipeline_config import PipelineConfig


def is_valid_token(token, stopwords: set[str], cfg: PipelineConfig) -> bool:
    # CPU hotspot: this predicate executes multiple attribute checks for every token.
    lemma = token.lemma_
    if token.is_stop or lemma in stopwords:
        return False
    if token.is_punct or token.is_space or token.like_num:
        return False
    if token.pos_ not in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}: #possibly try deleting later in evaluation phase, may improve performance on more complex documents
        return False
    if not (cfg.min_token_len <= len(lemma) <= cfg.max_token_len):
        return False
    if not lemma.isalpha():
        return False

    return True


def handle_oov(token, cfg: PipelineConfig) -> str | None:
    if token.is_oov:
        return cfg.oov_placeholder
    return token.lemma_


def process_batch(
    df: pd.DataFrame,
    text_col: str,
    row_indexer,
    nlp,
    stopwords: set[str],
    cfg: PipelineConfig,
    *,
    tokens_col: str = "tokens",
) -> None:
    """Tokenize a DataFrame row slice and write tokens in place."""
    # High memory overhead at corpus scale: this accumulates token lists for every doc in the batch.
    results: list[list[str]] = []
    texts = df.loc[row_indexer, text_col].tolist()
    # CPU hotspot: spaCy pipeline inference is typically the most expensive stage.
    docs = nlp.pipe(texts, batch_size=cfg.batch_size, n_process=cfg.n_process)
    for doc in docs:
        tokens: list[str] = []
        for token in doc:
            if not is_valid_token(token, stopwords, cfg):
                continue
            lemma = handle_oov(token, cfg)
            if not lemma:
                continue
            tokens.append(lemma)
        results.append(tokens)
    df.loc[row_indexer, tokens_col] = pd.Series(results, index=row_indexer)


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
