from collections import Counter

from gensim.models import Phrases
from gensim.models.phrases import Phraser

from pipeline_config import PipelineConfig


def is_valid_token(token, stopwords: set[str], cfg: PipelineConfig) -> bool:
    lemma = token.lemma_

    if token.is_stop or lemma in stopwords:
        return False

    if token.is_punct or token.is_space or token.like_num:
        return False

    if token.pos_ not in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}:
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


def process_batch(texts: list[str], nlp, stopwords: set[str], cfg: PipelineConfig) -> list[list[str]]:
    results: list[list[str]] = []
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
    return results


def build_vocab(tokenized_docs: list[list[str]], min_freq: int, min_doc_freq: int) -> tuple[set[str], int]:
    term_freq = Counter()
    doc_freq = Counter()

    for doc in tokenized_docs:
        term_freq.update(doc)
        doc_freq.update(set(doc))

    vocab = {
        term for term, freq in term_freq.items()
        if freq >= min_freq and doc_freq[term] >= min_doc_freq
    }
    return vocab, len(term_freq)


def filter_rare(tokenized_docs: list[list[str]], vocab: set[str]) -> list[list[str]]:
    return [[token for token in doc if token in vocab] for doc in tokenized_docs]


def apply_ngrams(tokenized_docs: list[list[str]], cfg: PipelineConfig) -> list[list[str]]:
    """Optionally apply bigram detection to tokenized documents."""
    if not cfg.enable_ngrams:
        return tokenized_docs

    bigram = Phrases(tokenized_docs, min_count=cfg.ngram_min_count, threshold=cfg.ngram_threshold)
    bigram_model = Phraser(bigram)
    return [list(bigram_model[doc]) for doc in tokenized_docs]
