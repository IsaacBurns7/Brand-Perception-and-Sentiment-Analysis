import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bertopic_train import evaluate_bertopic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained BERTopic model on a test CSV")
    parser.add_argument("--input", required=True, help="Path to test CSV")
    parser.add_argument("--text-col", default="cleaned_text", help="Text column in test CSV")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to BERTopic model artifact (legacy bertopic_model or bertopic_bundle directory)",
    )
    parser.add_argument("--output-path", required=True, help="Path to output evaluation JSON")
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency used for evaluation dictionary filtering",
    )
    parser.add_argument(
        "--top-n-words",
        type=int,
        default=10,
        help="Top words per topic used in evaluation metrics",
    )
    parser.add_argument(
        "--max-doc-count",
        type=int,
        default=None,
        help="Optional cap for number of test docs",
    )
    parser.add_argument(
        "--assignment-method",
        choices=["approximate", "transform"],
        default="approximate",
        help="How to assign topics to test docs (default: approximate; safer on macOS)",
    )
    return parser.parse_args()


def _infer_topics(topic_model, docs: list[str], assignment_method: str) -> list[int]:
    if assignment_method == "transform":
        topics, _ = topic_model.transform(docs)
        return [int(t) for t in topics]

    if hasattr(topic_model, "approximate_distribution"):
        distributions, _ = topic_model.approximate_distribution(docs, calculate_tokens=False)
        if distributions is None or len(distributions) == 0:
            raise RuntimeError("BERTopic approximate_distribution returned no assignments")

        topics: list[int] = []
        for row in distributions:
            # Keep parity with BERTopic outlier semantics when no topic mass is present.
            if np.sum(row) <= 0:
                topics.append(-1)
            else:
                topics.append(int(np.argmax(row)))
        return topics

    topics, _ = topic_model.transform(docs)
    return [int(t) for t in topics]


def _prepare_docs(input_path: Path, text_col: str, max_doc_count: int | None) -> list[str]:
    df = pd.read_csv(input_path, low_memory=False)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in input CSV")

    docs_df = df[[text_col]].copy()
    docs_df[text_col] = docs_df[text_col].fillna("").astype(str).str.strip()
    docs_df = docs_df[docs_df[text_col] != ""].reset_index(drop=True)

    if max_doc_count is not None:
        docs_df = docs_df.head(max_doc_count)

    docs = docs_df[text_col].tolist()
    if not docs:
        raise ValueError("No non-empty documents available for BERTopic evaluation")
    return docs


def _prepare_eval_tokens(docs: list[str], min_df: int):
    try:
        nltk_module = importlib.import_module("nltk")
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
    except ModuleNotFoundError as exc:
        raise RuntimeError("nltk is not installed. Install with: pip install nltk") from exc

    try:
        from gensim.corpora import Dictionary
    except ModuleNotFoundError as exc:
        raise RuntimeError("gensim is not installed. Install with: pip install gensim") from exc

    try:
        stopwords.words("english")
        word_tokenize("test")
    except LookupError:
        nltk_module.download("stopwords", quiet=True)
        nltk_module.download("punkt", quiet=True)
        nltk_module.download("punkt_tab", quiet=True)

    stop_words = set(stopwords.words("english"))
    tokenized_docs = [
        [
            word.lower()
            for word in word_tokenize(doc)
            if word.isalpha() and word.lower() not in stop_words
        ]
        for doc in docs
    ]

    dictionary = Dictionary(tokenized_docs)
    if len(dictionary) == 0:
        raise ValueError("Dictionary is empty after tokenization; cannot evaluate BERTopic model")

    dictionary.filter_extremes(no_below=max(1, min_df), no_above=0.95)
    if len(dictionary) == 0:
        dictionary = Dictionary(tokenized_docs)
    return tokenized_docs, dictionary


def _resolve_model_load_path(model_path: Path) -> Path:
    """Resolve legacy/portable model paths to a BERTopic.load-compatible target."""
    if model_path.is_file():
        return model_path

    if model_path.is_dir():
        state_dir = model_path / "artifacts" / "bertopic_state"
        if state_dir.is_dir():
            return state_dir
        # Support passing the state dir directly.
        if (model_path / "config.json").exists() and (model_path / "topics.json").exists():
            return model_path

    raise FileNotFoundError(
        "Could not resolve BERTopic model path. Expected a legacy model file or "
        "a bertopic_bundle directory containing artifacts/bertopic_state."
    )


def main() -> None:
    args = parse_args()

    if args.min_df < 1:
        raise ValueError("--min-df must be >= 1")
    if args.top_n_words < 1:
        raise ValueError("--top-n-words must be >= 1")

    input_path = Path(args.input)
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"BERTopic model not found: {model_path}")

    try:
        bertopic_module = importlib.import_module("bertopic")
        BERTopic = bertopic_module.BERTopic
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "BERTopic is not installed. Install with: pip install bertopic sentence-transformers"
        ) from exc

    docs = _prepare_docs(input_path=input_path, text_col=args.text_col, max_doc_count=args.max_doc_count)
    tokenized_docs, dictionary = _prepare_eval_tokens(docs=docs, min_df=args.min_df)

    resolved_model_path = _resolve_model_load_path(model_path)
    topic_model = BERTopic.load(str(resolved_model_path))
    topics = _infer_topics(topic_model=topic_model, docs=docs, assignment_method=args.assignment_method)

    evaluation_json = evaluate_bertopic(
        topic_model=topic_model,
        tokenized_docs=tokenized_docs,
        dictionary=dictionary,
        topics=topics,
        topk=args.top_n_words,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(evaluation_json, f, indent=2)


if __name__ == "__main__":
    main()
