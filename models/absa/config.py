"""Central configuration for the ABSA pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
MAMS_DIR = DATASETS_DIR / "mams"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_ARTIFACTS_DIR = ARTIFACTS_DIR / "models"
REPORT_ARTIFACTS_DIR = ARTIFACTS_DIR / "reports"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MAMS_TRAIN_XML = MAMS_DIR / "train.xml"
MAMS_VAL_XML = MAMS_DIR / "val.xml"
MAMS_TEST_XML = MAMS_DIR / "test.xml"

MAMS_ASTE_DIR = PROCESSED_DATA_DIR / "mams_aste"
MAMS_ASTE_TRAIN = MAMS_ASTE_DIR / "train.dat.aste"
MAMS_ASTE_VAL = MAMS_ASTE_DIR / "val.dat.aste"
MAMS_ASTE_TEST = MAMS_ASTE_DIR / "test.dat.aste"

ABSA_MODEL_DIR = MODEL_ARTIFACTS_DIR / "absa_aste"
ABSA_REPORT_DIR = REPORT_ARTIFACTS_DIR / "absa"

# RoBERTa stage-2 sentiment model (artifacts/models/roberta_stage2/)
ROBERTA_STAGE2_DIR = MODEL_ARTIFACTS_DIR / "roberta_stage2"

# Sentiment label map matching roberta_stage2 training convention (integer string keys)
# Stage-2 was trained with labels 0=negative, 1=neutral, 2=positive
ROBERTA_ID_TO_SENTIMENT = {"0": "negative", "1": "neutral", "2": "positive"}
ROBERTA_SENTIMENT_TO_POLARITY = {"negative": "NEG", "neutral": "NEU", "positive": "POS"}

# ASTE polarity tokens used in .dat.aste files
POLARITY_MAP = {"positive": "POS", "negative": "NEG", "neutral": "NEU"}
POLARITY_REVERSE = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}

# PyABSA SemEval built-in dataset keys to merge with MAMS
SEMEVAL_BUILTIN_DATASETS = ["Laptop14", "Restaurant14", "Restaurant15", "Restaurant16"]

RANDOM_SEED = 42

# PyABSA training defaults
ATEPC_EPOCHS = 50
ATEPC_BATCH_SIZE = 16
ATEPC_LEARNING_RATE = 1e-5
ATEPC_MAX_LENGTH = 128
ATEPC_L2_LAMBDA = 1e-5
ATEPC_PATIENCE = 10

# Eval CSVs shared with the sentiment pipeline (text + label columns)
EVAL_SA_TEST_CSV = PROCESSED_DATA_DIR / "eval_sentiment_analysis_test.csv"
EVAL_MANUAL_CSV = PROCESSED_DATA_DIR / "eval_testdata_manual_2009.csv"

# Column name candidates for auto-detection (same priority order as sentiment/predict.py)
TEXT_COLUMN_CANDIDATES = ("tweet_text", "clean_comment", "text", "comment")

# RoBERTa inference defaults (mirrors roberta_stage2 training settings)
ROBERTA_BATCH_SIZE = 32
ROBERTA_MAX_LENGTH = 128

# Input template for aspect-aware RoBERTa scoring
# "{aspect} : {sentence}" surfaces the aspect to the sentence-level classifier
ROBERTA_INPUT_TEMPLATE = "{aspect} : {sentence}"
