# Sentiment Module Integration Notes (Mockup)

This document explains how the files in `models/sentiment` work together so integration into the larger project is straightforward.

## What This Module Does

At a high level, this module supports four jobs:

1. Prepare normalized training/evaluation datasets from multiple sources.
2. Train sentiment models (classic sklearn and Hugging Face models).
3. Run inference on single text or batch CSV files.
4. Evaluate prediction quality with standard metrics.

## End-to-End Flow

Typical pipeline flow is:

1. Data prep: `prepare_datasets.py`
2. Training: `train.py`
3. Inference: `predict.py`
4. Metrics on prediction files: `hf_predictions_metrics.py`

Artifacts are mostly written under:

- `artifacts/models/`
- `artifacts/reports/`
- `data/processed/`

## File-By-File Walkthrough

### __init__.py

Purpose: package-level export surface.

- Re-exports the most common constants and functions from config/preprocessing/predict/model_factory.
- Lets callers import from `models.sentiment` directly instead of deep module paths.

### config.py

Purpose: central configuration and paths.

Key responsibilities:

- Defines repo-relative folders (`PROJECT_ROOT`, dataset paths, artifacts paths).
- Stores label/text column names used throughout training and preprocessing.
- Stores model hyperparameters (TF-IDF, logistic regression, transformer defaults).
- Lists supported model names and base pretrained model IDs.

Integration note:

- If your system has different storage locations, this is the first place to adapt.

### preprocessing.py

Purpose: text normalization and training-frame cleanup.

Main functions:

- `clean_text(text)`: lowercases and removes URLs, mentions, leading RT markers, `{link}` token, and extra spaces.
- `preprocess_dataframe(df, text_col, label_col=None)`: adds `clean_text`, drops empty rows, drops duplicates, and optionally removes excluded label values.

Integration note:

- This is the canonical cleaning logic for the sklearn path and should be reused in upstream ETL to keep train/infer behavior aligned.

### utils.py

Purpose: general utility helpers.

Main helpers:

- `ensure_directories(...)`: creates directory trees safely.
- `current_timestamp()`: UTC timestamp for metadata.
- `to_jsonable(...)` and `save_json(...)`: convert numpy/path types to JSON-safe values and write formatted JSON.
- `format_label_distribution(...)`: computes count + proportion summaries.

Integration note:

- Metadata/report writing in `train.py` depends on these functions.

### evaluate.py

Purpose: metric functions for classification outputs.

Main functions:

- `get_accuracy`, `get_macro_f1`, `get_weighted_f1`
- `get_classification_report`
- `get_confusion_matrix`
- `serialize_evaluation_results`: returns a JSON-friendly summary including report and confusion matrix.

Integration note:

- Use this module when you need consistent metric definitions between model experiments.

### model_factory.py

Purpose: construct model backends in one place.

Main pieces:

- `ModelSpec` dataclass: normalized descriptor of a model backend.
- `get_model(model_name, label_names=None)`: returns sklearn or transformer model specs.
- Lazy transformer import: only imports `transformers` when a transformer backend is requested.

Backends included:

- Classic: `logreg`, `svm`
- Transformer: `twitter_roberta`, `bertweet`, `distilbert`, `roberta`, `deberta`

Integration note:

- This is the abstraction boundary if you want to plug in another model family.

### prepare_datasets.py

Purpose: build standardized training/eval CSVs from raw sources.

Main functions:

- Source loaders (`load_sentiment140`, `load_reddit`, `load_sentiment_analysis_train`, `load_twitter_train`, etc.)
- `build_stage_processed_csvs()`: writes:
  - `data/processed/stage1_pretrain.csv` (binary labels)
  - `data/processed/stage2_finetune.csv` (3-class labels)
  - eval CSVs for downstream scoring
- `print_pipeline_commands()`: prints example train/eval command sequences.

Integration note:

- If your orchestration layer already prepares a unified dataset, this script can be bypassed.

### train.py

Purpose: training entry point for multiple model paths.

Main paths:

1. `--model sklearn`
   - Uses `train_model(...)`.
   - Trains TF-IDF + Logistic Regression on configured Twitter dataset.
   - Saves model pickle + metadata JSON.

2. `--model deberta --dataset goemotions`
   - Uses `train_deberta_goemotions(...)`.
   - Fine-tunes DeBERTa on GoEmotions-style labels (`negative`, `neutral`, `positive`).

3. `--model distilbert|roberta|bertweet|deberta` with stage CSVs
   - Uses `train_hf_stage_csv(...)`.
   - Fine-tunes transformer models on stage-1/stage-2 CSVs.

Integration note:

- This file is the primary place to integrate experiment tracking, checkpoint policies, and infra-specific device settings.

### predict.py

Purpose: inference utilities + CLI for sklearn and HF models.

Main functions:

- `load_model(...)`: cached joblib loading.
- `predict_sentiment(text)`: one-text sklearn inference.
- `predict_batch(texts)`: list-based sklearn inference.
- `run_sklearn_batch_inference(...)`: reads CSV, detects text column, writes predictions CSV with confidence/probabilities.
- `run_hf_batch_inference(...)`: batched transformer inference on CSV and writes predictions.

CLI subcommands:

- `sklearn-batch`
- `hf-batch`

Integration note:

- Real-time service integration should generally load once and call `predict_batch` for throughput.

### hf_predictions_metrics.py

Purpose: quick scoring utility for HF output CSV files.

Behavior:

- Reads a predictions CSV that must contain `true_label` and `predicted_label`.
- Prints rows, accuracy, macro F1, weighted F1.

Integration note:

- Useful as a lightweight post-processing step in CI or experiment scripts.

## Suggested Integration Contracts

To integrate cleanly with the broader system, keep these contracts stable:

1. Input text contract:
   - For sklearn CSV batch inference, include one of: `tweet_text`, `clean_comment`, `text`, `comment`.
   - Or pass `--text-column` explicitly.

2. HF evaluation contract:
   - Prediction CSV for `hf_predictions_metrics.py` must include `true_label` and `predicted_label`.

3. Label conventions:
   - Stage/HF training often uses integer labels (0/1/2 or 0/2).
   - GoEmotions path uses string labels (`negative`, `neutral`, `positive`).

4. Artifact locations:
   - Trained models and reports are expected under configured artifacts folders in `config.py`.

## Practical Commands (From Project Root)

Prepare stage datasets:

```bash
python -m models.sentiment.prepare_datasets --print-commands
```

Train sklearn baseline:

```bash
python -m models.sentiment.train --model sklearn
```

Run sklearn batch inference:

```bash
python -m models.sentiment.predict sklearn-batch \
  --input-path data/datasets/twitter-sentiment/Dataset\ -\ Test.csv \
  --output-path data/processed/twitter_sentiment_test_with_predictions.csv
```

Run HF batch inference:

```bash
python -m models.sentiment.predict hf-batch \
  --model-path artifacts/models/deberta_stage2 \
  --input-path data/processed/eval_sentiment_analysis_test.csv \
  --output-path artifacts/reports/deberta_stage2_eval_predictions.csv \
  --text-column text \
  --label-column label
```

Score HF predictions file:

```bash
python -m models.sentiment.hf_predictions_metrics \
  artifacts/reports/deberta_stage2_eval_predictions.csv
```

## Integration Risks To Watch

1. `predict_sentiment` and `predict_batch` assume `predict_proba` is available.
   - This is true for logistic regression, but not true for all sklearn models (for example plain LinearSVC).
2. Dataset label format differs across paths (string labels vs integer labels).
3. `config.py` contains the source-of-truth paths; path drift between environments will break pipelines.

## Recommended Next Refactor (Optional)

If this module is going into production serving, consider adding:

1. A single service-level adapter that normalizes label outputs to one schema.
2. Explicit pydantic/dataclass request-response models for inference I/O.
3. A smoke-test script that trains on a tiny sample and runs one end-to-end predict + evaluate step.
