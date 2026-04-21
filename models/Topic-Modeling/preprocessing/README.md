## Add max_document_count to encompass larger corpus 

Example below
```bash
python normalize_corpus.py --input ../data/rating.csv --text-col article --output ../data/lda/rating.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000
```

```bash
python normalize_corpus.py --input ../data/rating.csv --text-col article --output ../data/lda/rating.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --max-doc-count 100000 --batch-size 100 > normalize_corpus_out_lda.txt 2>&1 & 
```

Example background processes for different datasets
```bash
python normalize_corpus.py --input ../data/raw-data.csv --text-col content --output ../data/lda/raw-data.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000 --batch-size 100 --n-process 4 > normalize_corpus_out_lda_news_raw.txt 2>&1 & 
```
```bash
python normalize_corpus.py --input ../data/rating.csv --text-col article --output ../data/lda/rating.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000 --batch-size 100 --n-process 4 > normalize_corpus_out_lda_rating.txt 2>&1 & 
```
```bash
python normalize_corpus.py --input ../data/arxiv_data.csv --text-col summaries --output ../data/lda/arxiv_data.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000 --batch-size 100 --n-process 4 > normalize_corpus_out_lda_arxiv1.txt 2>&1 & 
```
```bash
python normalize_corpus.py --input ../data/arxiv_data_210930-054931.csv --text-col abstracts --output ../data/lda/arxiv_data_210930-054931.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000 --batch-size 100 --n-process 4 > normalize_corpus_out_lda_arxiv2.txt 2>&1 & 
```
```bash
python normalize_corpus.py --input ../data/bbc_news.csv --text-col description --output ../data/lda/bbc_news.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000 --batch-size 100 --n-process 4 > normalize_corpus_out_lda.txt 2>&1 & 
```
```bash
python normalize_corpus.py --input ../data/reuters.csv --text-col text --output ../data/lda/reuters.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000 --batch-size 100 --n-process 4 > normalize_corpus_out_lda.txt 2>&1 & 
```
```bash
python normalize_corpus.py --input ../data/ModLewis_train.csv --text-col text --output ../data/lda/ModLewis_train.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000 --batch-size 100 --n-process 4 > normalize_corpus_out_lda_train.txt 2>&1 & 
```
```bash
python normalize_corpus.py --input ../data/ModLewis_test.csv --text-col text --output ../data/lda/ModLewis_test.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000 --batch-size 100 --n-process 4 > normalize_corpus_out_lda_test.txt 2>&1 & 
```


## BERTopic Mode

Use BERTopic mode to keep cleaned document text instead of aggressively pruning to token lists.
```bash
python normalize_corpus.py --input ../data/raw-data.csv --text-col content --output ../data/bertopic/raw-data.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000 --batch-size 100 --n-process 4 --topic-model-target bertopic --bertopic-min-words 3 --bertopic-drop-empty-text > normalize_corpus_out_bertopic_news_raw.txt 2>&1 & 
```
```bash
python normalize_corpus.py --input ../data/rating.csv --text-col article --output ../data/bertopic/rating.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000 --batch-size 100 --topic-model-target bertopic --bertopic-min-words 3 --bertopic-drop-empty-text > normalize_corpus_out_bertopic.txt 2>&1 & 
```
```bash
python normalize_corpus.py --input ../data/arxiv_data.csv --text-col summaries --output ../data/bertopic/arxiv_data.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000 --batch-size 100 --n-process 4 --topic-model-target bertopic --bertopic-min-words 3 --bertopic-drop-empty-text > normalize_corpus_out_bertopic.txt 2>&1 & 
```
```bash
python normalize_corpus.py --input ../data/arxiv_data_210930-054931.csv --text-col abstracts --output ../data/bertopic/arxiv_data_210930-054931.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000 --batch-size 100 --n-process 4 --topic-model-target bertopic --bertopic-min-words 3 --bertopic-drop-empty-text > normalize_corpus_out_bertopic.txt 2>&1 & 
```
```bash
python normalize_corpus.py --input ../data/bbc_news.csv --text-col description --output ../data/bertopic/bbc_news.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000 --batch-size 100 --n-process 4 --topic-model-target bertopic --bertopic-min-words 3 --bertopic-drop-empty-text > normalize_corpus_out_bertopic.txt 2>&1 & 
```
```bash
python normalize_corpus.py --input ../data/reuters.csv --text-col text --output ../data/bertopic/reuters.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000 --batch-size 100 --n-process 4 --topic-model-target bertopic --bertopic-min-words 3 --bertopic-drop-empty-text > normalize_corpus_out_bertopic_all.txt 2>&1 &
```
```bash
python normalize_corpus.py --input ../data/ModLewis_train.csv --text-col text --output ../data/bertopic/ModLewis_train.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000 --batch-size 100 --n-process 4 --topic-model-target bertopic --bertopic-min-words 3 --bertopic-drop-empty-text > normalize_corpus_out_bertopic_train.txt 2>&1 &
```
```bash
python normalize_corpus.py --input ../data/ModLewis_test.csv --text-col text --output ../data/bertopic/ModLewis_test.csv --diagnostics-output diagnostics/preprocessing_phasefinal.json --diagnostics-top-n 25 --enable-ngrams --ngram-min-count 15 --ngram-threshold 10.0 --max-doc-count 100000 --batch-size 100 --n-process 4 --topic-model-target bertopic --bertopic-min-words 3 --bertopic-drop-empty-text > normalize_corpus_out_bertopic_test.txt 2>&1 &
```

BERTopic mode output columns:
- `cleaned_text`
- `token_count_light`
- optional original source text when `--bertopic-keep-original-text` is enabled