import typer 
from pathlib import Path 
from typing import Optional
from dataclasses import dataclass 

from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

import numpy as np
import pandas as pd
from rbo import RankingSimilarity
import json 
from itertools import combinations

@dataclass 
class EvalConfig:
    input_path: Path
    output_path: Path
    cache_dir: Path
    text_column:str

#parse args 
def evaluate(
        input_path:Path = typer.Argument(..., help="Path to eval dataset csv", exists=True, file_okay=True, dir_okay=False, readable=True),
        # output_path:Path = typer.Argument(..., help="Path to evaluation json", exists=True, file_okay=True, dir_okay=False, readable=True),
        output_path:Path = typer.Argument(..., help="Path to evaluation json"),
        cache_dir:Path = typer.Argument(..., help="Path to cache directory of LDA gensim model", exists=True, file_okay=False, dir_okay=True),
        text_column:str = typer.Argument(..., help="Text column in eval dataset csv")
    ):
    # if not input_path.exists():
    #     typer.echo(f"Error: Input path {input_path} does not exist", err = True)
    #     raise typer.Exit(code=1)
    # if not output_path.exists():
    #     typer.echo(f"Error: Output path {output_path} does not exist", err = True)
    #     raise typer.Exit(code=1)
    # if not cache_dir.exists():
    #     typer.echo(f"Error: Cache path {cache_dir} does not exist", err = True)
    #     raise typer.Exit(code=1)
    if input_path.suffix.lower() != ".csv":
        raise typer.BadParameter("Only .csv files are allowed for the input dataset.")
    config = EvalConfig(input_path = input_path, output_path = output_path, cache_dir = cache_dir, text_column = text_column)
    run_pipeline(config)    

def run_pipeline(config):
    model, dictionary = load_model(cache_dir=config.cache_dir)
    df = load_csv_defensively(file_path=config.input_path, text_column=config.text_column)
    tokenized_docs, corpus = prepare_test_data(df=df, text_column=config.text_column, dictionary=dictionary)
    eval_json = evaluate_lda(lda_model=model, dictionary=dictionary, tokenized_docs=tokenized_docs, corpus=corpus, topk=10)
    with open(config.output_path, "w") as f:
        json.dump(eval_json, f, indent=4)

#load model from cache_path
def load_model(cache_dir:Path):
    model_path = cache_dir / "lda_model.gensim"
    dict_path = cache_dir / "lda_dictionary.gensim"

    if not model_path.exists() or not dict_path.exists():
        typer.secho(f"Error: Required files not found in {cache_dir}", fg=typer.colors.RED)
        raise typer.Exit(1)
    dictionary = Dictionary.load(str(dict_path))
    model = LdaModel.load(str(model_path))
    return model, dictionary

#load csv from input_path
def load_csv_defensively(file_path, text_column):
    try:
        df = pd.read_csv(
            file_path,
            # 1. Encoding: 'utf-8-sig' handles the Excel BOM, 'latin1' is a good fallback
            encoding='utf-8-sig', 
            
            # 2. Cleaning: Automatically strips whitespace from headers
            skipinitialspace=True,
            
            # 3. Missing Data: Define what counts as "NaN" explicitly
            na_values=['NA', 'N/A', 'null', '', '?', '-'],
            
            # 4. Error Handling: Skip rows with too many commas instead of crashing
            on_bad_lines='warn', 
            
            # 5. Memory/Safety: Force specific types if you know them
            dtype={'user_id': str}, 
            
            # 6. Limits: Don't load 10GB into RAM by accident if you just need a peek
            low_memory=False 
        )
        
        # 7. Post-load Header Cleanup
        # df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        if text_column not in set(df.columns):
            raise ValueError(f"Required column not found: {text_column}")

        return df

    except FileNotFoundError:
        print(f"Error: The file at {file_path} does not exist.")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def prepare_test_data(df, text_column, dictionary):
    tokenized_docs = [
        str(doc).lower().split()
        for doc in df[text_column]
        if str(doc).strip() != ""
    ]
    valid_vocab = set(dictionary.token2id.keys())
    filtered_docs = [
        [word for word in doc if word in valid_vocab]
        for doc in tokenized_docs
    ]
    final_docs = [doc for doc in filtered_docs if len(doc) > 0]
    corpus = [dictionary.doc2bow(text) for text in final_docs]

    # 1. Flatten the tokenized_docs and get all unique words
    unique_doc_words = set(word for doc in final_docs for word in doc)

    # 2. Get all valid words from the Gensim dictionary
    valid_dict_words = set(dictionary.token2id.keys())

    # 3. Find the difference (words in docs but NOT in the dictionary)
    missing_words = unique_doc_words - valid_dict_words

    # Print the results
    print(f"Total unique words in test docs: {len(unique_doc_words)}")
    print(f"Total words missing from training dict: {len(missing_words)}")

    # View a sample to see exactly what is being excluded
    if missing_words:
        print(f"Sample missing words: {list(missing_words)[:20]}")

    return final_docs, corpus

#evaluate model on CSV 
def evaluate_lda(
    lda_model,
    dictionary,
    tokenized_docs, 
    corpus,
    topk: int = 10,
) -> dict:
    """
    Evaluates LDA model performance. Optimized for speed and robustness.
    """

    num_topics = lda_model.num_topics
    if num_topics <= 0:
        return {"error": "Model has no topics."}

    # 1. Extract top-k words once
    top_words = [
        [word for word, _ in lda_model.show_topic(t, topn=topk)]
        for t in range(num_topics)
    ]

    # 2. Build coherence diagnostics and keep only stable topic words.
    # A word is supported if it appears in at least one eval document.
    # A topic is stable if, after pruning unsupported/isolated words, >=2 words remain.
    doc_sets = [set(doc) for doc in tokenized_docs]
    top_word_set = {word for topic in top_words for word in topic}
    word_doc_ids = {word: set() for word in top_word_set}

    for idx, doc_set in enumerate(doc_sets):
        for word in doc_set.intersection(top_word_set):
            word_doc_ids[word].add(idx)

    zero_docfreq_words = sorted([w for w, ids in word_doc_ids.items() if len(ids) == 0])

    stable_topics = []
    unstable_topics = []
    per_topic_support = []

    for topic_idx, topic_words in enumerate(top_words):
        supported_words = [w for w in topic_words if len(word_doc_ids[w]) > 0]
        supported_set = set(supported_words)

        # Keep words that co-occur with at least one other word in this topic.
        # This avoids zero-magnitude vectors in c_v internals.
        cooccurring_words = set()
        pair_count = 0
        for w1, w2 in combinations(supported_words, 2):
            if word_doc_ids[w1].intersection(word_doc_ids[w2]):
                pair_count += 1
                cooccurring_words.add(w1)
                cooccurring_words.add(w2)

        stable_topic_words = [w for w in topic_words if w in cooccurring_words]

        if len(stable_topic_words) >= 2:
            stable_topics.append(stable_topic_words)
        else:
            unstable_topics.append(topic_idx)

        per_topic_support.append(
            {
                "topic_index": topic_idx,
                "supported_words": len(supported_set),
                "cooccurring_words": len(cooccurring_words),
                "cooccurring_pairs": pair_count,
                "dropped_words": [w for w in topic_words if w not in stable_topic_words],
            }
        )

    coherence_diagnostics = {
        "total_topics": num_topics,
        "stable_topics_used_for_coherence": len(stable_topics),
        "unstable_topic_indices": unstable_topics,
        "unique_top_topic_words": len(top_word_set),
        "top_topic_words_with_zero_doc_frequency": len(zero_docfreq_words),
        "zero_doc_frequency_words_sample": zero_docfreq_words[:25],
        "per_topic_support": per_topic_support,
        "coherence_computation_mode": "topics_filtered_by_eval_support",
    }

    print(
        "Coherence diagnostics: "
        f"stable_topics={len(stable_topics)}/{num_topics}, "
        f"zero_docfreq_top_words={len(zero_docfreq_words)}"
    )

    # 3. Coherence Metrics (Optimized Loop)
    coherence_results = {}
    metrics = ["c_v", "u_mass", "c_npmi", "c_uci"]

    if not stable_topics:
        print("No stable topics available for coherence computation. Returning NaN coherences.")
        for m in metrics:
            coherence_results[f"coherence_{m.replace('c_', '')}"] = np.nan
    else:
        for m in metrics:
            try:
                kwargs = {
                    "topics": stable_topics,
                    "dictionary": dictionary,
                    "coherence": m,
                    "topn": topk,
                }
                if m == "u_mass":
                    kwargs["corpus"] = corpus
                else:
                    kwargs["texts"] = tokenized_docs

                coherence_results[f"coherence_{m.replace('c_', '')}"] = CoherenceModel(**kwargs).get_coherence()
            except Exception:
                print(f"Failed to get coherence results for coherence_{m}. Defaulting to np.nan")
                coherence_results[f"coherence_{m.replace('c_', '')}"] = np.nan

    # 4. Perplexity (Standard natural log exponentiation)
    log_perplexity = lda_model.log_perplexity(corpus)
    perplexity = np.exp(-log_perplexity) 

    # 5. Topic Diversity (Added safety check)
    all_top_words = [word for topic in top_words for word in topic]
    if len(all_top_words) > 0:
        topic_diversity = len(set(all_top_words)) / len(all_top_words)
    else:
        topic_diversity = 0.0

    # 6. Inverted RBO
    rbo_scores = []
    for i in range(num_topics):
        for j in range(i + 1, num_topics):
            # Using .get() or checking existence of RankingSimilarity helps robustness
            score = RankingSimilarity(top_words[i], top_words[j]).rbo()
            rbo_scores.append(score)

    inverted_rbo = 1.0 - np.mean(rbo_scores) if rbo_scores else 1.0

    # 7. Final assembly
    return {
        **coherence_results,
        "perplexity": float(perplexity),
        "topic_diversity": float(topic_diversity),
        "inverted_rbo": float(inverted_rbo),
        "num_topics": num_topics,
        "top_words": top_words,
        "coherence_diagnostics": coherence_diagnostics,
    }

#output results to output_path 



def main():
    typer.run(evaluate)
if __name__ == "__main__":
    main()

#python LDA_evaluate.py ./data/lda/arxiv_data.csv ./eval.json ./cache/arxiv_data_210930-054931/lda_preset01_balanced tokens_str 