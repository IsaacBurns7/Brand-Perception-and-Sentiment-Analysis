import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
import time

import pandas as pd
from gensim import corpora
from gensim.models.ldamodel import LdaModel

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


def train_lda(
    csv_path: Path,
    cache_path: Path,
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

    cache_path.mkdir(parents=True, exist_ok=True)
    model_path = cache_path / "lda_model.gensim"
    dictionary_path = cache_path / "lda_dictionary.gensim"

    with stage("save LDA artifacts"):
        lda_model.save(str(model_path))
        dictionary.save(str(dictionary_path))

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
        config=cli_config,
        token_column=args.token_column,
    )
    print(topic_vec_df.head(), flush=True)
    print(doc_vec_df.head(), flush=True)

### 2) Train LDA on preprocessed output
#python LDA_train.py --input ./data/lda/rating.csv --token-column tokens_str --cache-path ./cache/lda

### 3) LDA preset cookbook (5 configs)

# 1) Balanced baseline for medium/large corpora
# Goal: Good interpretability and stable topic separation for general use
# Performs well on: clean tokenized reviews/news with recurring themes
# Performs medium on: mixed-quality text with moderate noise
# Performs poorly on: very short texts where bag-of-words is sparse
#python LDA_train.py --input ./data/lda/rating.csv --token-column tokens_str --cache-path ./cache/lda_preset01_balanced --num-topics 15 --chunksize 500 --passes 20 --iterations 400 --eval-every 10 --alpha auto --eta auto --random-state 42

# 2) Fine-grained discovery for diverse corpora
# Goal: Increase topical detail by using more topics and stronger fitting
# Performs well on: large and diverse corpora with many sub-themes
# Performs medium on: medium corpora with moderate diversity
# Performs poorly on: small corpora where topics become fragmented/noisy
#python LDA_train.py --input ./data/lda/rating.csv --token-column tokens_str --cache-path ./cache/lda_preset02_fine_grained --num-topics 35 --chunksize 600 --passes 25 --iterations 500 --eval-every 5 --alpha asymmetric --eta auto --random-state 42

# 3) Small dataset stabilization
# Goal: Reduce over-fragmentation and keep broader, cleaner topics
# Performs well on: small pilot datasets and quick experiments
# Performs medium on: medium corpora that only need broad themes
# Performs poorly on: large/high-diversity corpora that need more granularity
#python LDA_train.py --input ./data/lda/rating.csv --token-column tokens_str --cache-path ./cache/lda_preset03_small_stable --num-topics 8 --chunksize 200 --passes 30 --iterations 300 --eval-every 5 --alpha symmetric --eta symmetric --random-state 42

# 4) Coarse themes for very large corpora
# Goal: Build high-level topic buckets for dashboard/reporting use
# Performs well on: very large corpora where broad trends are enough
# Performs medium on: medium corpora with repetitive language
# Performs poorly on: deep discovery tasks needing nuanced subtopics
#python LDA_train.py --input ./data/lda/rating.csv --token-column tokens_str --cache-path ./cache/lda_preset04_coarse_large --num-topics 12 --chunksize 2000 --passes 12 --iterations 250 --eval-every 10 --alpha auto --eta auto --random-state 42

# 5) Noisy short-text tolerant setup
# Goal: Improve robustness on short/noisy docs via stronger smoothing
# Performs well on: short comments, snippets, and noisy social text
# Performs medium on: standard review datasets with mixed doc lengths
# Performs poorly on: long technical text where this may oversmooth distinctions
#python LDA_train.py --input ./data/lda/rating.csv --token-column tokens_str --cache-path ./cache/lda_preset05_short_noisy --num-topics 20 --chunksize 300 --passes 35 --iterations 450 --eval-every 5 --alpha 0.1 --eta 0.1 --random-state 42