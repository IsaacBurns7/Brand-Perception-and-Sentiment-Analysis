# Brand Perception and Sentiment

This project contains reusable NLP EDA utilities plus notebook workflows for multiple sentiment datasets.

## Setup

Create virtual environment:

```bash
python3 -m venv .venv
```

Activate environment:

Mac/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download NLTK stopwords:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

Start Jupyter:

```bash
jupyter notebook
```

## Running The EDA Notebooks

Available notebooks:

- `EDA/notebooks/brand_sentiment_eda.ipynb`
- `EDA/notebooks/binary_sentiment_eda.ipynb`

Datasets analyzed:

- `EDA/datasets/twitter-sentiment/Dataset - Train.csv`
- `EDA/datasets/binary-classification/sentiment_analysis.csv`

Generated outputs are saved automatically to:

- `EDA/output/brand_sentiment/`
- `EDA/output/binary_sentiment/`
- `EDA/output/text/`

For the smoothest experience, launch Jupyter from the project root so the notebooks can import `EDA.eda_utils` and resolve dataset paths consistently.
