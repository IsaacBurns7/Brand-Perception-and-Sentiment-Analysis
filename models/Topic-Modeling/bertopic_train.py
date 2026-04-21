import argparse
import importlib
import logging
import os
from pathlib import Path
import json
from contextlib import contextmanager
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.coherencemodel import CoherenceModel
from rbo import RankingSimilarity

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERTopic from preprocessed CSV text")
    parser.add_argument("--input", required=True, help="Path to preprocessed CSV")
    parser.add_argument("--text-col", default="cleaned_text", help="Text column to train BERTopic on")
    parser.add_argument("--output-dir", default="./cache/bertopic", help="Directory for BERTopic outputs")
    parser.add_argument("--eval-path", help = "Path for eval outputs")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name passed to BERTopic",
    )
    parser.add_argument(
        "--embedding-device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="SentenceTransformer device selection (default: auto)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU embedding (off by default)",
    )
    parser.add_argument("--language", default="english", help="Language passed to BERTopic")
    parser.add_argument("--min-topic-size", type=int, default=10, help="Minimum cluster/topic size")
    parser.add_argument(
        "--nr-topics",
        default=None,
        help="Target number of topics (int or 'auto'). Omit for no reduction.",
    )
    parser.add_argument("--top-n-words", type=int, default=10, help="Top words per topic")
    parser.add_argument("--min-df", type=int, default=2, help="Min document frequency for vectorizer")
    parser.add_argument("--ngram-min", type=int, default=1, help="Min ngram size for vectorizer")
    parser.add_argument("--ngram-max", type=int, default=2, help="Max ngram size for vectorizer")
    parser.add_argument(
        "--calculate-probabilities",
        action="store_true",
        help="Compute topic probabilities (slower, more memory)",
    )
    parser.add_argument("--max-doc-count", type=int, default=None, help="Optional cap for training docs")
    parser.add_argument("--umap-n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--umap-n-components", type=int, default=5, help="UMAP output dimensions")
    parser.add_argument("--umap-min-dist", type=float, default=0.0, help="UMAP min_dist")
    parser.add_argument("--umap-metric", default="cosine", help="UMAP metric")
    parser.add_argument("--umap-random-state", type=int, default=42, help="UMAP random_state")
    parser.add_argument("--umap-n-jobs", type=int, default=1, help="UMAP n_jobs (safer default: 1)")
    parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples (default: None, lets library infer)",
    )
    parser.add_argument(
        "--hdbscan-core-dist-n-jobs",
        type=int,
        default=1,
        help="HDBSCAN core distance workers (safer default: 1)",
    )
    parser.add_argument(
        "--omp-num-threads",
        type=int,
        default=1,
        help="Set OMP_NUM_THREADS for native kernels (safer default: 1)",
    )
    return parser.parse_args()


def _parse_nr_topics(value: str | None):
    if value is None:
        return None
    if value == "auto":
        return "auto"
    return int(value)

def evaluate_bertopic(
    topic_model,
    # docs,
    tokenized_docs: list[list[str]],
    dictionary,
    topics: list[int],
    # probabilities,
    topk: int = 10,
) -> dict:
    """
    Evaluate a trained BERTopic model using coherence, outlier rate,
    topic diversity, and inverted RBO.

    Args:
        topic_model:    Trained BERTopic instance.
        docs:           Raw document strings (used for outlier rate).
        tokenized_docs: Raw tokenized documents (list of list of str) for coherence.
        dictionary:     Gensim Dictionary built from tokenized_docs.
        topics:         Topic assignments returned by fit_transform().
        probabilities:  Probability matrix returned by fit_transform() — may be None
                        if calculate_probabilities=False.
        topk:           Number of top words per topic used in all calculations.

    Returns:
        Dictionary with keys:
            coherence_cv        – float, c_v coherence (higher = better)
            coherence_umass     – float, u_mass coherence (less negative = better)
            coherence_cnpmi     – float, c_npmi coherence (higher = better)
            coherence_cuci      – float, c_uci coherence (higher = better)
            outlier_rate        – float in [0, 1], fraction of docs assigned to topic -1
            topic_diversity     – float in [0, 1] (higher = less redundancy)
            inverted_rbo        – float in [0, 1] (higher = more distinct topic rankings)
            mean_topic_size     – float, average number of docs per non-outlier topic
            std_topic_size      – float, std dev of topic sizes (high = imbalanced)
            num_topics          – int, number of topics excluding outlier topic -1
            top_words           – list of lists of str, top-k words per topic
    """

    # ------------------------------------------------------------------ #
    # 1. Extract top-k words for every non-outlier topic.
    #    get_topic() returns [(word, score), ...]; topic -1 is the outlier
    #    bucket and has no meaningful word distribution so we skip it.
    # ------------------------------------------------------------------ #
    topic_ids = [tid for tid in topic_model.get_topics().keys() if tid != -1]
    top_words: list[list[str]] = []
    for tid in topic_ids:
        words = topic_model.get_topic(tid)
        if words:  # get_topic() returns False for empty/missing topics
            top_words.append([word for word, _ in words[:topk]])

    num_topics = len(top_words)

    # ------------------------------------------------------------------ #
    # 2. Coherence — identical approach to evaluate_lda().
    #    We reuse the Gensim CoherenceModel by passing top_words directly
    #    instead of a Gensim model object, which both APIs support.
    # ------------------------------------------------------------------ #
    valid_vocab = set(dictionary.token2id.keys())

    # CoherenceModel requires topic words to exist in dictionary vocab.
    # BERTopic phrase presets can emit n-grams that are absent from this unigram dictionary.
    topics_for_coherence = []
    for topic in top_words:
        cleaned_topic = []
        for tok in topic:
            tok_str = str(tok).strip()
            if tok_str and tok_str in valid_vocab:
                cleaned_topic.append(tok_str)
        if len(cleaned_topic) >= 2:
            topics_for_coherence.append(cleaned_topic)

    def _coherence(metric: str) -> float:
        if not topics_for_coherence:
            return float(np.nan)

        kwargs = dict(topics=topics_for_coherence, dictionary=dictionary, coherence=metric)
        if metric == "u_mass":
            kwargs["corpus"] = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        else:
            kwargs["texts"] = tokenized_docs

        try:
            return CoherenceModel(**kwargs).get_coherence()
        except Exception:
            return float(np.nan)

    coherence_cv = _coherence("c_v")
    coherence_umass = _coherence("u_mass")
    coherence_cnpmi = _coherence("c_npmi")
    coherence_cuci = _coherence("c_uci")

    # ------------------------------------------------------------------ #
    # 3. Outlier rate.
    #    HDBSCAN assigns topic -1 to documents it cannot cluster.
    #    High outlier rates (>20–30%) suggest min_cluster_size is too large
    #    or the corpus is too heterogeneous for the chosen embeddings.
    # ------------------------------------------------------------------ #
    outlier_rate: float = sum(1 for t in topics if t == -1) / len(topics)

    # ------------------------------------------------------------------ #
    # 4. Topic size distribution (excluding outliers).
    #    High std_topic_size relative to mean indicates a few mega-topics
    #    dominating and many tiny ones — a sign of poor granularity.
    # ------------------------------------------------------------------ #
    topic_info = topic_model.get_topic_info()
    non_outlier_sizes = topic_info.loc[topic_info["Topic"] != -1, "Count"].values
    mean_topic_size = float(np.mean(non_outlier_sizes)) if len(non_outlier_sizes) else 0.0
    std_topic_size  = float(np.std(non_outlier_sizes))  if len(non_outlier_sizes) else 0.0

    # ------------------------------------------------------------------ #
    # 5. Topic diversity — same formula as evaluate_lda().
    # ------------------------------------------------------------------ #
    all_top_words = [word for topic in top_words for word in topic]
    topic_diversity: float = len(set(all_top_words)) / len(all_top_words) if all_top_words else 0.0

    # ------------------------------------------------------------------ #
    # 6. Inverted RBO — same formula as evaluate_lda().
    # ------------------------------------------------------------------ #
    rbo_scores: list[float] = []
    for i in range(num_topics):
        for j in range(i + 1, num_topics):
            list_i = list(dict.fromkeys(top_words[i]))
            list_j = list(dict.fromkeys(top_words[j]))

            score = RankingSimilarity(list_i, list_j).rbo()
            rbo_scores.append(score)

    inverted_rbo: float = 1.0 - float(np.mean(rbo_scores)) if rbo_scores else 1.0

    return {
        "coherence_cv":     coherence_cv,
        "coherence_umass":  coherence_umass,
        "coherence_cnpmi":  coherence_cnpmi,
        "coherence_cuci":   coherence_cuci,
        "outlier_rate":     outlier_rate,
        "topic_diversity":  topic_diversity,
        "inverted_rbo":     inverted_rbo,
        "mean_topic_size":  mean_topic_size,
        "std_topic_size":   std_topic_size,
        "num_topics":       num_topics,
        "top_words":        top_words,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    args = parse_args()

    if args.omp_num_threads < 1:
        raise ValueError("--omp-num-threads must be >= 1")
    os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)

    try:
        bertopic_module = importlib.import_module("bertopic")
        BERTopic = bertopic_module.BERTopic
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "BERTopic is not installed. Install with: pip install bertopic sentence-transformers"
        ) from exc

    try:
        sentence_transformers_module = importlib.import_module("sentence_transformers")
        SentenceTransformer = sentence_transformers_module.SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install with: pip install sentence-transformers"
        ) from exc

    try:
        umap_module = importlib.import_module("umap")
        UMAP = umap_module.UMAP
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "umap-learn is not installed. Install with: pip install umap-learn"
        ) from exc

    try:
        hdbscan_module = importlib.import_module("hdbscan")
        HDBSCAN = hdbscan_module.HDBSCAN
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "hdbscan is not installed. Install with: pip install hdbscan"
        ) from exc

    if args.ngram_min < 1:
        raise ValueError("--ngram-min must be >= 1")
    if args.ngram_max < args.ngram_min:
        raise ValueError("--ngram-max must be >= --ngram-min")
    if args.min_topic_size < 2:
        raise ValueError("--min-topic-size must be >= 2")
    if args.umap_n_neighbors < 2:
        raise ValueError("--umap-n-neighbors must be >= 2")
    if args.umap_n_components < 2:
        raise ValueError("--umap-n-components must be >= 2")
    if args.umap_n_jobs < 1:
        raise ValueError("--umap-n-jobs must be >= 1")
    if args.hdbscan_min_samples is not None and args.hdbscan_min_samples < 1:
        raise ValueError("--hdbscan-min-samples must be >= 1")
    if args.hdbscan_core_dist_n_jobs < 1:
        raise ValueError("--hdbscan-core-dist-n-jobs must be >= 1")

    nr_topics = _parse_nr_topics(args.nr_topics)

    log.info("Loading input CSV: %s", args.input)
    df = pd.read_csv(args.input, low_memory=False)
    # df = df.head(10000)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in input CSV")

    docs_df = df[[args.text_col]].copy()
    docs_df[args.text_col] = docs_df[args.text_col].fillna("").astype(str).str.strip()
    docs_df = docs_df[docs_df[args.text_col] != ""].reset_index(drop=True)

    if args.max_doc_count is not None:
        docs_df = docs_df.head(args.max_doc_count)

    if docs_df.empty:
        raise ValueError("No non-empty documents available for BERTopic training")

    docs = docs_df[args.text_col].tolist()
    log.info("Training docs: %d", len(docs))

    if args.cpu_only:
        device = "cpu"
    elif args.embedding_device == "auto":
        device = None
    else:
        device = args.embedding_device

    if device is None:
        embedding_model = SentenceTransformer(args.embedding_model)
    else:
        embedding_model = SentenceTransformer(args.embedding_model, device=device)

    umap_model = UMAP(
        n_neighbors=args.umap_n_neighbors,
        n_components=args.umap_n_components,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.umap_random_state,
        low_memory=True,
        n_jobs=args.umap_n_jobs,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_topic_size,
        min_samples=args.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        core_dist_n_jobs=args.hdbscan_core_dist_n_jobs,
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language=args.language,
        min_topic_size=args.min_topic_size,
        nr_topics=nr_topics,
        top_n_words=args.top_n_words,
        calculate_probabilities=args.calculate_probabilities,
        verbose=True,
    )

    log.info("Fitting BERTopic ...")
    topics, probabilities = topic_model.fit_transform(docs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_topics = docs_df.copy()
    doc_topics.insert(0, "doc_id", range(len(doc_topics)))
    doc_topics["topic"] = topics

    if probabilities is not None:
        try:
            doc_topics["topic_probability_max"] = probabilities.max(axis=1)
        except Exception:
            log.warning("Could not compute max topic probability per document")

    doc_topics_path = output_dir / "document_topics.csv"
    doc_topics.to_csv(doc_topics_path, index=False)
    log.info("Saved document topics: %s", doc_topics_path)

    topic_info = topic_model.get_topic_info()
    topic_info_path = output_dir / "topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False)
    log.info("Saved topic info: %s", topic_info_path)

    model_path = output_dir / "bertopic_model"
    topic_model.save(str(model_path), save_ctfidf=True)
    log.info("Saved model: %s", model_path)

    # ================================================================= #
    # Prepare tokenized documents and Gensim dictionary for evaluation
    # ================================================================= #
    log.info("Preparing evaluation data ...")
    try:
        nltk_module = importlib.import_module("nltk")
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "nltk is not installed. Install with: pip install nltk"
        ) from exc
 
    try:
        from gensim.corpora import Dictionary
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "gensim is not installed. Install with: pip install gensim"
        ) from exc
 
    # Download required NLTK data if not already present
    try:
        stopwords.words('english')
        word_tokenize("test")
    except LookupError:
        nltk_module.download('stopwords', quiet=True)
        nltk_module.download('punkt', quiet=True)
        nltk_module.download('punkt_tab', quiet=True)
 
    stop_words = set(stopwords.words('english'))
    tokenized_docs = [
        [word.lower() for word in word_tokenize(doc) 
         if word.isalpha() and word.lower() not in stop_words]
        for doc in docs
    ]
 
    dictionary = Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=args.min_df, no_above=0.95)
 
    eval_path = Path(args.eval_path)
    evaluation_json = evaluate_bertopic(
        topic_model=topic_model,
        tokenized_docs=tokenized_docs,
        dictionary=dictionary,
        topics=topics,
        topk=args.top_n_words,
    )

    # cache_path.mkdir(parents=True, exist_ok=True)
    # model_path = cache_path / "lda_model.gensim"
    # dictionary_path = cache_path / "lda_dictionary.gensim"

    # with stage("save LDA artifacts"):
    #     lda_model.save(str(model_path))
    #     dictionary.save(str(dictionary_path))
    #     with open(evaluation_path, 'w') as f:
    #         json.dump(evaluation, f, indent=2)
    with stage("Evaluation"):
        with open(eval_path, 'w') as f:
            json.dump(evaluation_json, f, indent=2)
        log.info("Saved eval: %s", eval_path)

if __name__ == "__main__":
    main()

### 2) Train BERTopic on preprocessed output
#python bertopic_train.py --input ./data/bertopic/rating.csv --text-col cleaned_text

### 3) BERTopic preset cookbook (10 configs)

# 1) Balanced baseline for medium/large clean corpora
# Goal: Stable, interpretable topics with moderate granularity
# Performs well on: clean news/blog/review text with clear themes
# Performs medium on: mixed quality text with some slang/noise
# Performs poorly on: very short posts and strongly multilingual corpora
#python bertopic_train.py --input ./data/bertopic/rating.csv --text-col cleaned_text --output-dir ./cache/bertopic/preset01_balanced --embedding-model sentence-transformers/all-MiniLM-L6-v2 --min-topic-size 25 --nr-topics auto --top-n-words 12 --min-df 8 --ngram-min 1 --ngram-max 2 --umap-n-neighbors 30 --umap-n-components 5 --umap-min-dist 0.05 --hdbscan-min-samples 10

# 2) High sensitivity for short social posts
# Goal: Capture many small/niche topics from short documents
# Performs well on: short comments, posts, titles, snippets
# Performs medium on: mixed short+long datasets
# Performs poorly on: tiny datasets where micro-topics become unstable
#python bertopic_train.py --input ./data/bertopic/rating.csv --text-col cleaned_text --output-dir ./cache/bertopic/preset02_social_granular --min-topic-size 5 --nr-topics auto --top-n-words 15 --min-df 1 --ngram-min 1 --ngram-max 1 --umap-n-neighbors 10 --umap-n-components 5 --umap-min-dist 0.0 --hdbscan-min-samples 2

# 3) Small dataset stabilization
# Goal: Keep topics stable on small corpora by limiting topic count
# Performs well on: pilot studies and small experiments
# Performs medium on: medium corpora with limited diversity
# Performs poorly on: large/high-diversity corpora that need more topics
#python bertopic_train.py --input ./data/bertopic/rating.csv --text-col cleaned_text --output-dir ./cache/bertopic/preset03_small_stable --min-topic-size 3 --nr-topics 12 --top-n-words 10 --min-df 1 --ngram-min 1 --ngram-max 2 --umap-n-neighbors 5 --umap-n-components 3 --umap-min-dist 0.1 --hdbscan-min-samples 1

# 4) Large-scale coarse clustering
# Goal: Reduce noise and avoid many tiny topics on very large corpora
# Performs well on: very large review/support corpora with repeated language
# Performs medium on: medium corpora where some detail is still needed
# Performs poorly on: small corpora and rare-topic discovery tasks
#python bertopic_train.py --input ./data/bertopic/rating.csv --text-col cleaned_text --output-dir ./cache/bertopic/preset04_large_coarse --min-topic-size 60 --nr-topics auto --top-n-words 10 --min-df 20 --ngram-min 1 --ngram-max 2 --umap-n-neighbors 50 --umap-n-components 5 --umap-min-dist 0.15 --hdbscan-min-samples 20

# 5) Phrase-aware (bigrams/trigrams)
# Goal: Surface multi-word themes like "customer service" or "battery life"
# Performs well on: review and feedback data where phrases carry meaning
# Performs medium on: general article text with weaker phrase structure
# Performs poorly on: very short text where trigrams are sparse
#python bertopic_train.py --input ./data/bertopic/rating.csv --text-col cleaned_text --output-dir ./cache/bertopic/preset05_phrase_aware --min-topic-size 12 --nr-topics auto --top-n-words 15 --min-df 2 --ngram-min 1 --ngram-max 3 --umap-n-neighbors 20 --umap-n-components 5 --umap-min-dist 0.0 --hdbscan-min-samples 5

# 6) Executive-level theme buckets
# Goal: Produce broad, high-level themes for dashboards/reports
# Performs well on: large enterprise corpora needing summary-level insights
# Performs medium on: mixed corpora with medium granularity goals
# Performs poorly on: deep exploration requiring fine subtopics
#python bertopic_train.py --input ./data/bertopic/rating.csv --text-col cleaned_text --output-dir ./cache/bertopic/preset06_exec_coarse --min-topic-size 100 --nr-topics 20 --top-n-words 8 --min-df 25 --ngram-min 1 --ngram-max 2 --umap-n-neighbors 70 --umap-n-components 5 --umap-min-dist 0.2 --hdbscan-min-samples 30

# 7) Fine-grained discovery for long technical text
# Goal: Discover nuanced subtopics in dense/long documents
# Performs well on: research, technical docs, long-form posts
# Performs medium on: standard consumer reviews
# Performs poorly on: noisy short text with weak semantic signal
#python bertopic_train.py --input ./data/bertopic/rating.csv --text-col cleaned_text --output-dir ./cache/bertopic/preset07_research_fine --min-topic-size 8 --nr-topics auto --top-n-words 20 --min-df 3 --ngram-min 1 --ngram-max 3 --umap-n-neighbors 12 --umap-n-components 8 --umap-min-dist 0.0 --hdbscan-min-samples 3

# 8) Confidence-first with probability outputs
# Goal: Keep conservative clusters and support confidence thresholding
# Performs well on: QA workflows and downstream filtering by confidence
# Performs medium on: smaller datasets where probabilities are noisier
# Performs poorly on: memory-constrained large runs
#python bertopic_train.py --input ./data/bertopic/rating.csv --text-col cleaned_text --output-dir ./cache/bertopic/preset08_confidence --min-topic-size 20 --nr-topics auto --top-n-words 10 --min-df 5 --ngram-min 1 --ngram-max 2 --umap-n-neighbors 25 --umap-n-components 5 --umap-min-dist 0.1 --hdbscan-min-samples 15 --calculate-probabilities

# 9) Multilingual embedding setup
# Goal: Group semantically similar content across multiple languages
# Performs well on: mixed-language corpora with overlapping topics
# Performs medium on: mostly English corpora
# Performs poorly on: language-specific analyses needing custom stopwords/tokenization
#python bertopic_train.py --input ./data/bertopic/rating.csv --text-col cleaned_text --output-dir ./cache/bertopic/preset09_multilingual --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --language multilingual --min-topic-size 15 --nr-topics auto --top-n-words 12 --min-df 3 --ngram-min 1 --ngram-max 2 --umap-n-neighbors 20 --umap-n-components 5 --umap-min-dist 0.05 --hdbscan-min-samples 6

# 10) Reproducible CPU benchmark baseline
# Goal: Maximize repeatability for fair config comparisons
# Performs well on: benchmarking and experiment tracking
# Performs medium on: regular training where speed is not critical
# Performs poorly on: very large corpora that benefit from GPU acceleration
#python bertopic_train.py --input ./data/bertopic/rating.csv --text-col cleaned_text --output-dir ./cache/bertopic/preset10_repro_cpu --cpu-only --embedding-model sentence-transformers/all-MiniLM-L6-v2 --min-topic-size 15 --nr-topics auto --top-n-words 10 --min-df 4 --ngram-min 1 --ngram-max 2 --umap-n-neighbors 15 --umap-n-components 5 --umap-min-dist 0.0 --umap-random-state 42 --umap-n-jobs 1 --hdbscan-min-samples 5 --hdbscan-core-dist-n-jobs 1 --omp-num-threads 1
