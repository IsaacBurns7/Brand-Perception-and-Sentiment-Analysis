import pandas as pd
from pathlib import Path
import re
import spacy
import time
from contextlib import contextmanager
from gensim import corpora
from gensim.models.ldamodel import LdaModel


@contextmanager
def stage(name: str):
    """Print wall-clock timings for long-running pipeline stages."""
    start = time.perf_counter()
    print(f"[timing] start: {name}", flush=True)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[timing] end:   {name} ({elapsed:.2f}s)", flush=True)

data_dir = Path("./data")

with stage("load CSV"):
    df = pd.read_csv(data_dir / "rating.csv")

with stage("trim rows"):
    df = df.head(1000)  # cut that shit down
text_column = "article"


with stage("load spaCy model"):
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # what type is this ? is this a function ?

with stage("extract texts"):
    texts = df[text_column].astype(str).tolist()

print(f"[info] documents: {len(texts)}", flush=True)

def preprocess_doc(doc):
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_digit and len(token) > 2
    ]

with stage("preprocess (spaCy nlp.pipe + lemmatize/stopword filter)"):
    processed_texts = [preprocess_doc(doc) for doc in nlp.pipe(texts, batch_size=500)]

# Create a dictionary and corpus for LDA
with stage("build gensim dictionary"):
    dictionary = corpora.Dictionary(processed_texts)

with stage("build bag-of-words corpus"):
    corpus = [dictionary.doc2bow(text) for text in processed_texts]


num_topics = 15  # choose number of topics
with stage("train LDA model"):
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=5,  # number of passes over the corpus
        alpha="auto",  # helps with topic sparsity
        per_word_topics=True,
    )

with stage("print topics"):
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic {idx}: {topic}\n", flush=True)

with stage("infer example document"):
    example_doc = nlp("Example new document text")
    doc_bow = dictionary.doc2bow(preprocess_doc(example_doc))
    doc_topics = lda_model.get_document_topics(doc_bow)

print(doc_topics, flush=True)  # list of (topic_id, probability)