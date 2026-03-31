"""Factory helpers for building sentiment model backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import (
    BERTWEET_MODEL_NAME,
    DISTILBERT_MODEL_NAME,
    LINEAR_SVC_CONFIG,
    LOGISTIC_REGRESSION_CONFIG,
    PROJECT_LABELS,
    SUPPORTED_MODEL_NAMES,
    TFIDF_CONFIG,
    TWITTER_ROBERTA_MODEL_NAME,
)


@dataclass
class ModelSpec:
    """Container describing a supported sentiment model backend."""

    name: str
    family: Literal["sklearn", "transformer"]
    estimator: Any
    tokenizer: Any | None = None
    pretrained_name: str | None = None
    supports_training: bool = True
    supports_predict_proba: bool = False
    label_mapping: dict[str, str] | None = None
    implementation_status: Literal["ready", "pretrained_inference_only", "not_implemented"] = "ready"
    notes: str | None = None


def _build_tfidf_vectorizer() -> TfidfVectorizer:
    """Create the shared TF-IDF vectorizer configuration."""

    return TfidfVectorizer(**TFIDF_CONFIG)


def _build_logreg_model() -> ModelSpec:
    """Return the TF-IDF + Logistic Regression baseline."""

    pipeline = Pipeline(
        steps=[
            ("tfidf", _build_tfidf_vectorizer()),
            ("classifier", LogisticRegression(**LOGISTIC_REGRESSION_CONFIG)),
        ]
    )
    return ModelSpec(
        name="logreg",
        family="sklearn",
        estimator=pipeline,
        supports_training=True,
        supports_predict_proba=True,
        implementation_status="ready",
    )


def _build_svm_model() -> ModelSpec:
    """Return the TF-IDF + LinearSVC baseline."""

    pipeline = Pipeline(
        steps=[
            ("tfidf", _build_tfidf_vectorizer()),
            ("classifier", LinearSVC(**LINEAR_SVC_CONFIG)),
        ]
    )
    return ModelSpec(
        name="svm",
        family="sklearn",
        estimator=pipeline,
        supports_training=True,
        supports_predict_proba=False,
        implementation_status="ready",
    )


def _import_transformers() -> tuple[Any, Any, Any]:
    """Import transformer classes only when a transformer backend is requested."""

    try:
        from transformers import (  # pyright: ignore[reportMissingImports]
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise ImportError(
            "Transformer models require the 'transformers' package and a backend such as "
            "'torch'. Install them before requesting transformer backends such as "
            "'twitter_roberta', 'bertweet', or 'distilbert'."
        ) from exc

    return AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def _build_transformer_classifier(
    *,
    name: str,
    pretrained_name: str,
    label_names: list[str],
    tokenizer_kwargs: dict[str, Any] | None = None,
    notes: str | None = None,
) -> ModelSpec:
    """Create a sequence-classification transformer configured for project labels."""

    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer = _import_transformers()
    label2id = {label: index for index, label in enumerate(label_names)}
    id2label = {index: label for label, index in label2id.items()}

    config = AutoConfig.from_pretrained(
        pretrained_name,
        num_labels=len(label_names),
        label2id=label2id,
        id2label=id2label,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_name,
        **(tokenizer_kwargs or {}),
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_name,
        config=config,
        ignore_mismatched_sizes=True,
    )

    return ModelSpec(
        name=name,
        family="transformer",
        estimator=model,
        tokenizer=tokenizer,
        pretrained_name=pretrained_name,
        supports_training=True,
        supports_predict_proba=True,
        implementation_status="ready",
        notes=notes,
    )


def _build_twitter_roberta_model() -> ModelSpec:
    """Return the pretrained CardiffNLP Twitter-RoBERTa sentiment model."""

    _, AutoModelForSequenceClassification, AutoTokenizer = _import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(TWITTER_ROBERTA_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(TWITTER_ROBERTA_MODEL_NAME)

    label_mapping = {
        "negative": PROJECT_LABELS[0],
        "neutral": PROJECT_LABELS[1],
        "positive": PROJECT_LABELS[2],
        "LABEL_0": PROJECT_LABELS[0],
        "LABEL_1": PROJECT_LABELS[1],
        "LABEL_2": PROJECT_LABELS[2],
    }

    return ModelSpec(
        name="twitter_roberta",
        family="transformer",
        estimator=model,
        tokenizer=tokenizer,
        pretrained_name=TWITTER_ROBERTA_MODEL_NAME,
        supports_training=False,
        supports_predict_proba=True,
        label_mapping=label_mapping,
        implementation_status="pretrained_inference_only",
        notes="Runs as a pretrained inference baseline without fine-tuning.",
    )


def _build_bertweet_model(label_names: list[str]) -> ModelSpec:
    """Return a fine-tunable BERTweet classifier."""

    return _build_transformer_classifier(
        name="bertweet",
        pretrained_name=BERTWEET_MODEL_NAME,
        label_names=label_names,
        tokenizer_kwargs={"use_fast": False},
        notes="Fine-tunes BERTweet on the shared train split before validation scoring.",
    )


def _build_distilbert_model(label_names: list[str]) -> ModelSpec:
    """Return a fine-tunable DistilBERT classifier."""

    return _build_transformer_classifier(
        name="distilbert",
        pretrained_name=DISTILBERT_MODEL_NAME,
        label_names=label_names,
        notes="Fine-tunes DistilBERT on the shared train split before validation scoring.",
    )


def get_model(model_name: str, label_names: list[str] | None = None) -> ModelSpec:
    """Build and return a supported model backend by name."""

    normalized_name = model_name.strip().lower()
    resolved_labels = label_names or list(PROJECT_LABELS)

    if normalized_name == "logreg":
        return _build_logreg_model()
    if normalized_name == "svm":
        return _build_svm_model()
    if normalized_name == "twitter_roberta":
        return _build_twitter_roberta_model()
    if normalized_name == "bertweet":
        return _build_bertweet_model(resolved_labels)
    if normalized_name == "distilbert":
        return _build_distilbert_model(resolved_labels)

    raise ValueError(
        f"Unsupported model '{model_name}'. Supported models: {', '.join(SUPPORTED_MODEL_NAMES)}"
    )


def list_supported_models() -> tuple[str, ...]:
    """Return the names of supported sentiment models."""

    return SUPPORTED_MODEL_NAMES
