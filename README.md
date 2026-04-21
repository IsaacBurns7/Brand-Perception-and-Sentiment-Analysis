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

News article pipeline for brand extraction, topic modelling, sentiment analysis, and aspect-based sentiment extraction.

## Structure

```
models/
  absa/           — PyABSA EMCGCN triplet extraction wrapper
  NER/            — brand entity extraction (spaCy + rules)
  Topic-Modeling/ — LDA topic modelling
```

## ABSA

The pipeline supports multi-aspect extraction with PyABSA using the Hugging Face checkpoint `deepakm10/brand-absa-emcgcn`.

Install dependencies:

```bash
pip install pyabsa
```

Control ABSA with:

```bash
export BRAND_PERCEPTION_ENABLE_ABSA=1
export BRAND_PERCEPTION_ABSA_MODEL=deepakm10/brand-absa-emcgcn
```

If ABSA is disabled or the model fails, the pipeline falls back to the existing stub sentiment behavior and emits a default `aspect` of `general`.

Run the local demo again:

```bash
python demo_integration.py
```

Run the API:

```bash
uvicorn api.app:app --reload
```

Run the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

If the API is not running, the dashboard will fall back to built-in sample data so the layout remains usable for demos and presentation prep.

Test API endpoints:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/analytics/summary
curl http://127.0.0.1:8000/analytics/topics
curl http://127.0.0.1:8000/analytics/aspects
curl "http://127.0.0.1:8000/analytics/timeseries?rolling_window_days=7"
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
