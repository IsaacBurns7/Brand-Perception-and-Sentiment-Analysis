# Brand perception and sentiment analysis

Baseline **TF-IDF + logistic regression** sentiment models for brand-directed tweets, with optional **Hugging Face** backends (DistilBERT, BERTweet) for experiments defined in `models/sentiment/model_factory.py`.

## Layout

| Path | Purpose |
|------|---------|
| `data/datasets/` | CSV datasets (e.g. Twitter sentiment, processed GoEmotions splits) |
| `data/processed/` | Train/val splits and derived tables from preprocessing |
| `notebooks/` | Exploratory analysis (e.g. GoEmotions EDA) |
| `models/sentiment/` | Training, evaluation, prediction, config, preprocessing |
| `artifacts/` | Trained weights, reports (gitignored; regenerate locally) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train the sklearn baseline

```bash
python -m models.sentiment.train
```

## Predict

Single-call API (from Python): `from models.sentiment.predict import predict_sentiment`.

Batch CSV (sklearn joblib model):

```bash
python -m models.sentiment.predict sklearn-batch --input-path data/datasets/twitter-sentiment/Dataset\ -\ Test.csv --output-path data/processed/twitter_sentiment_test_with_predictions.csv
```

Batch CSV (saved Hugging Face sequence classifier directory):

```bash
python -m models.sentiment.predict hf-batch --model-path /path/to/saved_hf_model --input-path data/datasets/gomotions_processed/goemotions_test.csv --output-path predictions.csv
```

## Sample entry point

```bash
python main.py
# Brand Perception & Sentiment Analysis

News article pipeline for brand extraction, topic modelling, and sentiment analysis.

## Structure

```
models/
  NER/            — brand entity extraction (spaCy + rules)
  Topic-Modeling/ — LDA topic modelling
```

## NER

Runs after `news_dailyworker.Preprocessing().runner()`. Adds `ner_brands` and `ner_raw_json` columns to the daily article CSV.

```python
from models.NER.ner_pipeline import NERPipeline
import pandas as pd

df = pd.read_csv("data/dailyworker/2025-01-15.csv")
df = NERPipeline().run_on_dataframe(df)
```

Evaluate:

```bash
python -m models.NER.evaluate_ner
python -m models.NER.evaluate_ner --csv path/to/rating.csv
```
