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
    df = pd.read_csv(data_dir / "rating_normalized_for_LDA.csv")

# with stage("trim rows"):
#     df = df.head(1000)  # cut that shit down
text_column = "article"
token_column = "tokens_str"

with stage("Tokenizing documents"):
    documents = df[token_column].apply(lambda x:x.split())
    documents = documents.to_list()
    for doc in documents:
        for token in doc:
            assert isinstance(token, str)
# Create a dictionary and corpus for LDA
with stage("build gensim dictionary"):
    dictionary = corpora.Dictionary(documents)

with stage("build bag-of-words corpus"):
    corpus = [dictionary.doc2bow(text) for text in documents]

# print("BEFORE FILTERING: ", len(dictionary))
# dictionary.filter_extremes(no_below=20, no_above=0.5)
# print("AFTER FILTERING: ", len(dictionary))
# print(len(corpus))
# empty_docs = sum(1 for doc in corpus if len(doc) == 0)
# print("empty docs:", empty_docs)

num_topics = 15  # choose number of topics
chunksize= 500
passes = 20
iterations = 400
eval_every = True
# id2word = dictionary.id2token

with stage("train LDA model"):
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

with stage("print topics"):
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic {idx}: {topic}\n", flush=True)

with stage("infer example document"):
    example_doc = ["Example","text", "of", "ukraine"]
    doc_bow = dictionary.doc2bow(example_doc)
    doc_topics = lda_model.get_document_topics(doc_bow)

print(doc_topics, flush=True)  