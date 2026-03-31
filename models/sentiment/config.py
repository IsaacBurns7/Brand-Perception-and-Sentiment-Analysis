"""Central configuration for the sentiment pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
TWITTER_SENTIMENT_DIR = DATASETS_DIR / "twitter-sentiment"
GOEMOTIONS_PROCESSED_DIR = DATASETS_DIR / "gomotions_processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_ARTIFACTS_DIR = ARTIFACTS_DIR / "models"
REPORT_ARTIFACTS_DIR = ARTIFACTS_DIR / "reports"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
COMPARISON_OUTPUT_DIR = PROCESSED_DATA_DIR / "model_comparison"

TRAIN_CSV_PATH = TWITTER_SENTIMENT_DIR / "Dataset - Train.csv"
TEST_CSV_PATH = TWITTER_SENTIMENT_DIR / "Dataset - Test.csv"
GOEMOTIONS_TRAIN_CSV_PATH = GOEMOTIONS_PROCESSED_DIR / "goemotions_train.csv"
GOEMOTIONS_TEST_CSV_PATH = GOEMOTIONS_PROCESSED_DIR / "goemotions_test.csv"
PROCESSED_TRAIN_SPLIT_PATH = PROCESSED_DATA_DIR / "tweet_sentiment_train_processed.csv"
PROCESSED_VAL_SPLIT_PATH = PROCESSED_DATA_DIR / "tweet_sentiment_val_processed.csv"

DEFAULT_MODEL_PATH = MODEL_ARTIFACTS_DIR / "sentiment_model.pkl"
DEFAULT_METADATA_PATH = REPORT_ARTIFACTS_DIR / "sentiment_model_metadata.json"

TEXT_COLUMN = "tweet_text"
ENTITY_COLUMN = "emotion_in_tweet_is_directed_at"
LABEL_COLUMN = "is_there_an_emotion_directed_at_a_brand_or_product"
EXCLUDED_LABEL = "I can't tell"
CLEAN_TEXT_COLUMN = "clean_text"

RANDOM_SEED = 42
TEST_SIZE = 0.2

TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_STOP_WORDS = "english"

LOGISTIC_REGRESSION_MAX_ITER = 1000
LOGISTIC_REGRESSION_CLASS_WEIGHT = "balanced"

TFIDF_CONFIG = {
    "max_features": TFIDF_MAX_FEATURES,
    "ngram_range": TFIDF_NGRAM_RANGE,
    "stop_words": TFIDF_STOP_WORDS,
}

LOGISTIC_REGRESSION_CONFIG = {
    "max_iter": LOGISTIC_REGRESSION_MAX_ITER,
    "class_weight": LOGISTIC_REGRESSION_CLASS_WEIGHT,
}

LINEAR_SVC_CONFIG = {
    "class_weight": "balanced",
}

MODEL_NAME = "tfidf_logistic_regression"
COMPARISON_SUMMARY_CSV_PATH = REPORT_ARTIFACTS_DIR / "model_comparison_summary.csv"
COMPARISON_SUMMARY_JSON_PATH = REPORT_ARTIFACTS_DIR / "model_comparison_summary.json"

PROJECT_LABELS = [
    "Negative emotion",
    "No emotion toward brand or product",
    "Positive emotion",
]

SUPPORTED_MODEL_NAMES = (
    "distilbert",
    "bertweet",
)

TRANSFORMER_MODEL_NAMES = (
    "distilbert",
    "bertweet",
)

HF_FINETUNE_MODEL_NAMES = (
    "distilbert",
    "bertweet",
)

TWITTER_ROBERTA_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BERTWEET_MODEL_NAME = "vinai/bertweet-base"
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"

TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_EPOCHS = 3
TRANSFORMER_LEARNING_RATE = 2e-5
TRANSFORMER_MAX_LENGTH = 64
TRANSFORMER_WEIGHT_DECAY = 0.01
TRANSFORMER_WARMUP_RATIO = 0.1
