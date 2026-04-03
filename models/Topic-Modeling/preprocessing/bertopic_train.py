import argparse
import importlib
import logging
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERTopic from preprocessed CSV text")
    parser.add_argument("--input", required=True, help="Path to preprocessed CSV")
    parser.add_argument("--text-col", default="cleaned_text", help="Text column to train BERTopic on")
    parser.add_argument("--output-dir", default="../data/bertopic_outputs", help="Directory for BERTopic outputs")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name passed to BERTopic",
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
    return parser.parse_args()


def _parse_nr_topics(value: str | None):
    if value is None:
        return None
    if value == "auto":
        return "auto"
    return int(value)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    args = parse_args()

    try:
        bertopic_module = importlib.import_module("bertopic")
        BERTopic = bertopic_module.BERTopic
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "BERTopic is not installed. Install with: pip install bertopic sentence-transformers"
        ) from exc

    if args.ngram_min < 1:
        raise ValueError("--ngram-min must be >= 1")
    if args.ngram_max < args.ngram_min:
        raise ValueError("--ngram-max must be >= --ngram-min")
    if args.min_topic_size < 2:
        raise ValueError("--min-topic-size must be >= 2")

    nr_topics = _parse_nr_topics(args.nr_topics)

    log.info("Loading input CSV: %s", args.input)
    df = pd.read_csv(args.input, low_memory=False)
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

    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
    )

    topic_model = BERTopic(
        embedding_model=args.embedding_model,
        vectorizer_model=vectorizer_model,
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


if __name__ == "__main__":
    main()
