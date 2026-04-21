import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
import time

import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from rbo import RankingSimilarity
import json

@contextmanager
def stage(name: str):
    """Print wall-clock timings for long-running pipeline stages."""
    start = time.perf_counter()
    print(f"[timing] start: {name}", flush=True)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[timing] end:   {name} ({elapsed:.2f}s)", flush=True)

data_dir = Path("./data")

# Default LDA training hyperparameters.
DEFAULT_LDA_CONFIG: dict[str, int | str | float] = {
    "num_topics": 15,
    "chunksize": 500,
    "passes": 20,
    "iterations": 400,
    "eval_every": 10,
    "alpha": "auto",
    "eta": "auto",
    "random_state": 42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LDA model from tokenized CSV")
    parser.add_argument(
        "--input",
        default=str(data_dir / "lda" / "rating.csv"),
        help="Path to input CSV containing tokenized text",
    )
    parser.add_argument(
        "--cache-path",
        default="./cache/lda",
        help="Directory to save trained LDA artifacts",
    )
    parser.add_argument(
        "--token-column",
        default="tokens_str",
        help="CSV column containing whitespace-tokenized documents",
    )
    parser.add_argument(
        "--out-path",
        help="Path to output directory for model on dataset during final iter"
    )   
    parser.add_argument(
        "--eval-path",
        help="Path to evaluation directory for model on dataset during final iter"
    )

    parser.add_argument("--num-topics", type=int, default=int(DEFAULT_LDA_CONFIG["num_topics"]))
    parser.add_argument("--chunksize", type=int, default=int(DEFAULT_LDA_CONFIG["chunksize"]))
    parser.add_argument("--passes", type=int, default=int(DEFAULT_LDA_CONFIG["passes"]))
    parser.add_argument("--iterations", type=int, default=int(DEFAULT_LDA_CONFIG["iterations"]))
    parser.add_argument("--eval-every", type=int, default=int(DEFAULT_LDA_CONFIG["eval_every"]))
    parser.add_argument("--alpha", default=str(DEFAULT_LDA_CONFIG["alpha"]))
    parser.add_argument("--eta", default=str(DEFAULT_LDA_CONFIG["eta"]))
    parser.add_argument("--random-state", type=int, default=int(DEFAULT_LDA_CONFIG["random_state"]))

    return parser.parse_args()


def _parse_prior(value: str) -> str | float:
    lowered = value.strip().lower()
    if lowered in {"auto", "symmetric", "asymmetric"}:
        return lowered
    try:
        return float(value)
    except ValueError:
        raise ValueError(
            f"Invalid prior value '{value}'. Use one of auto/symmetric/asymmetric or a float."
        )

def _resolve_lda_config(config: Mapping[str, object] | None = None) -> dict[str, object]:
    """Merge user config into defaults for LdaModel."""
    resolved = dict(DEFAULT_LDA_CONFIG)
    if config:
        resolved.update(dict(config))
    return resolved

def evaluate_lda(
    lda_model,
    corpus,
    dictionary,
    tokenized_docs,
    topk: int = 10,
) -> dict:
    """
    Evaluate a trained LDA model using all coherence metrics, perplexity,
    topic diversity, and inverted RBO.

    Args:
        lda_model:      Trained Gensim LdaModel instance.
        corpus:         BoW corpus (list of list of (int, int)) used to train the model.
        dictionary:     Gensim Dictionary used to build the corpus.
        tokenized_docs: Raw tokenized documents (list of list of str) — required for
                        corpus-based coherence metrics (c_v, c_npmi).
        topk:           Number of top words per topic used in all calculations.

    Returns:
        Dictionary with keys:
            coherence_cv        – float, c_v coherence (best proxy for human judgment)
            coherence_umass     – float, u_mass coherence (less negative = better)
            coherence_cnpmi     – float, c_npmi coherence
            coherence_cuci      – float, c_uci coherence
            perplexity          – float, per-word perplexity (lower = better fit)
            topic_diversity     – float in [0, 1] (higher = less redundancy)
            inverted_rbo        – float in [0, 1] (higher = more diverse topic rankings)
            num_topics          – int
            top_words           – list of lists of str, top-k words per topic
    """

    # ------------------------------------------------------------------ #
    # 1. Extract top-k words for every topic as plain strings.
    #    show_topic() returns [(word, prob), ...] sorted by descending prob.
    # ------------------------------------------------------------------ #
    top_words: list[list[str]] = [
        [word for word, _ in lda_model.show_topic(topic_id, topn=topk)]
        for topic_id in range(lda_model.num_topics)
    ]

    # ------------------------------------------------------------------ #
    # 2. All four coherence metrics.
    #
    #    c_v    – sliding window + NPMI + cosine sim; best correlation with
    #             human judgments; range roughly [0, 1], higher is better.
    #    u_mass – document co-occurrence counts; fast, no external corpus
    #             needed; range (-inf, 0), less negative is better.
    #    c_npmi – direct NPMI on word pairs with sliding window; range
    #             [-1, 1], higher is better; good for short texts.
    #    c_uci  – PMI (not normalised) with sliding window; range (-inf, inf),
    #             higher is better; older metric, c_npmi generally preferred.
    # ------------------------------------------------------------------ #
    def _coherence(metric: str) -> float:
        kwargs = dict(model=lda_model, dictionary=dictionary, coherence=metric)
        # u_mass only needs the BoW corpus; the rest need raw token lists
        if metric == "u_mass":
            kwargs["corpus"] = corpus
        else:
            kwargs["texts"] = tokenized_docs
        return CoherenceModel(**kwargs).get_coherence()

    coherence_cv    = _coherence("c_v")
    coherence_umass = _coherence("u_mass")
    coherence_cnpmi = _coherence("c_npmi")
    coherence_cuci  = _coherence("c_uci")

    # ------------------------------------------------------------------ #
    # 3. Per-word perplexity.
    #    log_perplexity() returns the ELBO bound (negative, higher = better).
    #    We negate and exponentiate to get conventional perplexity (lower = better).
    #    Perplexity and coherence are often inversely correlated — treat
    #    coherence as the primary signal when they disagree.
    # ------------------------------------------------------------------ #
    log_perplexity: float = lda_model.log_perplexity(corpus)
    perplexity: float = float(np.exp2(-log_perplexity))

    # ------------------------------------------------------------------ #
    # 4. Topic diversity.
    #    Fraction of unique words across the union of all topics' top-k lists.
    #    1.0 = every word appears in exactly one topic (maximally diverse).
    #    ~0  = topics are near-duplicates sharing almost all top words.
    # ------------------------------------------------------------------ #
    all_top_words = [word for topic in top_words for word in topic]
    topic_diversity: float = len(set(all_top_words)) / len(all_top_words)

    # ------------------------------------------------------------------ #
    # 5. Inverted RBO (Rank-Biased Overlap).
    #    Measures pairwise similarity between every pair of topic word rankings.
    #    rbo.RBO().score() ∈ [0, 1]: 1 = identical ranking, 0 = no overlap.
    #    We compute the AVERAGE inverted RBO across all pairs:
    #        inverted_rbo = 1 - mean(pairwise RBO scores)
    #    So 1.0 = all topics are completely distinct (best diversity),
    #       0.0 = all topics are identical (worst diversity).
    #
    #    Install: pip install rbo
    # ------------------------------------------------------------------ #
    num_topics = lda_model.num_topics
    rbo_scores: list[float] = []

    for i in range(num_topics):
        for j in range(i + 1, num_topics):
            # rbo.RBO().score() expects two ranked lists
            score = RankingSimilarity(top_words[i], top_words[j]).rbo()
            rbo_scores.append(score)

    # If only one topic exists there are no pairs; default diversity to 1.0
    inverted_rbo: float = 1.0 - float(np.mean(rbo_scores)) if rbo_scores else 1.0

    return {
        "coherence_cv":     coherence_cv,
        "coherence_umass":  coherence_umass,
        "coherence_cnpmi":  coherence_cnpmi,
        "coherence_cuci":   coherence_cuci,
        "perplexity":       perplexity,
        "topic_diversity":  topic_diversity,
        "inverted_rbo":     inverted_rbo,
        "num_topics":       num_topics,
        "top_words":        top_words,
    }


def train_lda(
    csv_path: Path,
    cache_path: Path,
    evaluation_path: Path,
    config: Mapping[str, object] | None = None,
    token_column: str = "tokens_str",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train an LDA model from a CSV and return topic/doc vectors.

    Args:
        csv_path: Input CSV path with tokenized text column.
        cache_path: Directory to store saved LDA artifacts.
        config: Optional hyperparameter overrides for LdaModel.
        token_column: Name of tokenized text column in the CSV.

    Returns:
        topic_vec: DataFrame with topic_id + topic_terms.
        doc_vec: DataFrame with doc_id + topic distribution.
    """
    with stage("load CSV"):
        df = pd.read_csv(csv_path)

    if token_column not in df.columns:
        raise ValueError(
            f"Column '{token_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    with stage("tokenize documents"):
        documents = df[token_column].fillna("").astype(str).apply(lambda x: x.split()).to_list()

    with stage("build gensim dictionary"):
        dictionary = corpora.Dictionary(documents)

    with stage("build bag-of-words corpus"):
        corpus = [dictionary.doc2bow(text) for text in documents]

    lda_config = _resolve_lda_config(config)

    with stage("train LDA model"):
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            **lda_config,
        )
    
    with stage("evaluate LDA model"):
        evaluation = evaluate_lda(lda_model, corpus, dictionary, documents, 100)

    cache_path.mkdir(parents=True, exist_ok=True)
    model_path = cache_path / "lda_model.gensim"
    dictionary_path = cache_path / "lda_dictionary.gensim"

    with stage("save LDA artifacts"):
        lda_model.save(str(model_path))
        dictionary.save(str(dictionary_path))
        with open(evaluation_path, 'w') as f:
            json.dump(evaluation, f, indent=2)

    with stage("build topic vectors"):
        topic_vec = pd.DataFrame(
            [
                {"topic_id": topic_id, "topic_terms": topic_terms}
                for topic_id, topic_terms in lda_model.print_topics(num_topics=-1)
            ]
        )

    with stage("build document vectors"):
        doc_vec = pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "topic_distribution": lda_model.get_document_topics(bow),
                }
                for doc_id, bow in enumerate(corpus)
            ]
        )

    return topic_vec, doc_vec


if __name__ == "__main__":
    args = parse_args()
    cli_config: dict[str, object] = {
        "num_topics": args.num_topics,
        "chunksize": args.chunksize,
        "passes": args.passes,
        "iterations": args.iterations,
        "eval_every": args.eval_every,
        "alpha": _parse_prior(args.alpha),
        "eta": _parse_prior(args.eta),
        "random_state": args.random_state,
    }

    # Train model on dataset specified by CLI args and save artifacts under cache path.
    topic_vec_df, doc_vec_df = train_lda(
        csv_path=Path(args.input),
        cache_path=Path(args.cache_path),
        evaluation_path=Path(args.eval_path),
        config=cli_config,
        token_column=args.token_column,
    )
    # print(topic_vec_df.head(), flush=True)
    # print(doc_vec_df.head(), flush=True)
    with stage("Saving topic and doc vectors"):
        out_path = Path(args.out_path)
        out_path.mkdir(parents=True, exist_ok=True)
        topic_vec_df.to_csv(out_path / "topic_vec.csv", index=False)
        doc_vec_df.to_csv(out_path / "doc_vec.csv", index=False)

### 2) Train LDA on preprocessed output
#python LDA_train.py --input ./data/lda/rating.csv --token-column tokens_str --cache-path ./cache/lda

### 3) LDA preset cookbook (5 configs)

# 1) Balanced baseline for medium/large corpora
# Goal: Good interpretability and stable topic separation for general use
# Performs well on: clean tokenized reviews/news with recurring themes
# Performs medium on: mixed-quality text with moderate noise
# Performs poorly on: very short texts where bag-of-words is sparse
#python LDA_train.py --input ./data/lda/rating.csv --token-column tokens_str --cache-path ./cache/rating/lda_preset01_balanced --num-topics 15 --chunksize 500 --passes 20 --iterations 400 --eval-every 10 --alpha auto --eta auto --random-state 42 > lda_preset01_balanced_out.txt 2>&1 &

# 2) Fine-grained discovery for diverse corpora
# Goal: Increase topical detail by using more topics and stronger fitting
# Performs well on: large and diverse corpora with many sub-themes
# Performs medium on: medium corpora with moderate diversity
# Performs poorly on: small corpora where topics become fragmented/noisy
#python LDA_train.py --input ./data/lda/rating.csv --token-column tokens_str --cache-path ./cache/rating/lda_preset02_fine_grained --num-topics 35 --chunksize 600 --passes 25 --iterations 500 --eval-every 5 --alpha asymmetric --eta auto --random-state 42 > lda_preset02_fine_grained_out.txt 2>&1 &

# 3) Small dataset stabilization
# Goal: Reduce over-fragmentation and keep broader, cleaner topics
# Performs well on: small pilot datasets and quick experiments
# Performs medium on: medium corpora that only need broad themes
# Performs poorly on: large/high-diversity corpora that need more granularity
#python LDA_train.py --input ./data/lda/rating.csv --token-column tokens_str --cache-path ./cache/rating/lda_preset03_small_stable --num-topics 8 --chunksize 200 --passes 30 --iterations 300 --eval-every 5 --alpha symmetric --eta symmetric --random-state 42 > lda_preset03_small_stable_out.txt 2>&1 &

# 4) Coarse themes for very large corpora
# Goal: Build high-level topic buckets for dashboard/reporting use
# Performs well on: very large corpora where broad trends are enough
# Performs medium on: medium corpora with repetitive language
# Performs poorly on: deep discovery tasks needing nuanced subtopics
#python LDA_train.py --input ./data/lda/rating.csv --token-column tokens_str --cache-path ./cache/rating/lda_preset04_coarse_large --num-topics 12 --chunksize 2000 --passes 12 --iterations 250 --eval-every 10 --alpha auto --eta auto --random-state 42 > lda_preset04_large_coarse_out.txt 2>&1 &

# 5) Noisy short-text tolerant setup
# Goal: Improve robustness on short/noisy docs via stronger smoothing
# Performs well on: short comments, snippets, and noisy social text
# Performs medium on: standard review datasets with mixed doc lengths
# Performs poorly on: long technical text where this may oversmooth distinctions
#python LDA_train.py --input ./data/lda/rating.csv --token-column tokens_str --cache-path ./cache/rating/lda_preset05_short_noisy --num-topics 20 --chunksize 300 --passes 35 --iterations 450 --eval-every 5 --alpha 0.1 --eta 0.1 --random-state 42 > lda_preset05_short_noisy_out.txt 2>&1 &