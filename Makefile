SHELL := /bin/zsh

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
VENV_DIR ?= .venv

INPUT ?=
TEXT_COL ?=
OUTPUT ?=
SEP ?=,
MIN_FREQ ?=5
BATCH_SIZE ?=500

.PHONY: help venv install install-spacy-model run-aggregation run-news-eda run-lda normalize clean

help:
	@echo "Available targets:"
	@echo "  make venv                 Create a local virtual environment (.venv)"
	@echo "  make install              Install dependencies from requirements.txt"
	@echo "  make install-spacy-model  Download spaCy model en_core_web_md"
	@echo "  make run-aggregation      Run temporal aggregation pipeline"
	@echo "  make run-news-eda         Run EDA/news.py"
	@echo "  make run-lda              Run models/Topic-Modeling/LDA.py"
	@echo "  make normalize INPUT=<csv> TEXT_COL=<col> OUTPUT=<csv> [MIN_FREQ=5] [BATCH_SIZE=500]"
	@echo "  make clean                Remove common Python cache artifacts"

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Activate with: source $(VENV_DIR)/bin/activate"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-spacy-model:
	$(PYTHON) -m spacy download en_core_web_md

run-aggregation:
	$(PYTHON) temporal_aggregation.py

run-news-eda:
	$(PYTHON) EDA/news.py

run-lda:
	cd models/Topic-Modeling && $(PYTHON) LDA.py

normalize:
	@if [ -z "$(INPUT)" ] || [ -z "$(TEXT_COL)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make normalize INPUT=<input.csv> TEXT_COL=<column_name> OUTPUT=<output.csv>"; \
		exit 1; \
	fi
	$(PYTHON) models/Topic-Modeling/preprocessing/LDA_normalize_corpus.py \
		--input "$(INPUT)" \
		--text-col "$(TEXT_COL)" \
		--output "$(OUTPUT)" \
		--sep "$(SEP)" \
		--min-freq $(MIN_FREQ) \
		--batch-size $(BATCH_SIZE)

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
