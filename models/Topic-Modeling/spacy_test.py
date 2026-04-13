import spacy
import pandas as pd

df = pd.read_csv("./data/rating.csv")
df = df.head(1)
text_column = "article"
texts = df[text_column].astype(str).tolist()

def run_nlp_test(model):
    nlp = spacy.load(model, disable=["ner", "parser"]) 

    docs = nlp.pipe(texts)

    for doc in docs:
        for token in doc:
            print(token.text, token.has_vector, token.vector_norm, token.is_oov)

models = ["en_core_web_sm", "en_core_web_md"]#, "en_core_web_lg"]
for model in models:
    print("=" * 50)
    print(f"RUNNING NLP TEST WITH MODEL {model}")
    print("=" * 50)
    run_nlp_test(model)