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

*/5 * * * * /Users/kingisaac/Github/Brand-Perception-and-Sentiment-Analysis/.venv/bin/python3 /Users/kingisaac/Github/Brand-Perception-and-Sentiment-Analysis/services/reddit_data/apify_client2.py > /Users/kingisaac/Github/Brand-Perception-and-Sentiment-Analysis/services/reddit_data/apify_client2.out 2>&1