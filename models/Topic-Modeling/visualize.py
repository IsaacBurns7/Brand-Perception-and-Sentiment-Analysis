import typer 
from pathlib import Path 
from typing import Optional
from dataclasses import dataclass 
import pprint

from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

import numpy as np
import pandas as pd
from rbo import RankingSimilarity
import json 
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt

@dataclass 
class VisualizeConfig:
    input_path: Path
    output_path: Path

#parse args 
def visualize(
        input_path:Path = typer.Argument(..., help="Path to evaluation json directory", exists=True, file_okay=False, dir_okay=True),
        output_path:Path = typer.Argument(..., help="Path to visualizations", exists=True, file_okay=False, dir_okay=True),
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
    # if input_path.suffix.lower() != ".csv":
    #     raise typer.BadParameter("Only .csv files are allowed for the input dataset.")
    config = VisualizeConfig(input_path = input_path, output_path = output_path)
    run_pipeline(config)    

def create_json_map(input_path: Path, target_keys: list[str]):
    """
    Maps JSON files from a root directory structure into a nested dictionary.
    
    Structure: {json_file_name: {child_dir: {key: value}}}
    """
    result_map = {}

    # Iterate through every directory in the root
    for child_dir in input_path.iterdir():
        if child_dir.is_dir():
            
            # Look for all .json files within that child directory
            for json_file in child_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Create a sub-dictionary containing only the requested keys
                    # If a key isn't in the JSON, it defaults to None
                    extracted_data = {key: data.get(key) for key in target_keys}
                    
                    file_name = json_file.name
                    dir_name = child_dir.name

                    if file_name not in result_map:
                        result_map[file_name] = {}

                    result_map[file_name][dir_name] = extracted_data
                
                except (json.JSONDecodeError, OSError) as e:
                    print(f"Skipping {json_file.name} due to error: {e}")

    return result_map

def create_transfer_heatmap(results_map, target_metric:str, save_path:Path):
    all_records = []

    dataset_name_map = {
        "arxiv_data": "arxiv1 abstracts",
        "arxiv_data_210930-054931": "arxiv2 abstracts",
        "bbc_news": "bbc news headlines",
        "rating": "global news articles",
        "raw-data": "global news headlines",
        "reuters": "global news articles2"
    }

    # 1. Collect all data from all models
    for model_name, data_dict in results_map.items(): #model_name="lda_preset01_balanced.json"
        parts = model_name.replace(".json", "").split("_") #[lda, preset01, balanced]
        model_name = f"{parts[1].replace('preset0', 'v')}-{parts[2]}" #v1-balanced
        for key, inner_dict in data_dict.items():
            train, test = key.split("__on__")
            train = dataset_name_map[train]
            test = dataset_name_map[test]
            val = inner_dict.get(target_metric)
            all_records.append({
                "Model": model_name,
                "Train Dataset": train,
                "Display Label": f"({model_name}, {train})", # Combined label
                "Test Dataset": test,
                target_metric: val
            })

    # 2. Create DataFrame
    df = pd.DataFrame(all_records)

    # Pivot using the combined Display Label
    pivot_df = df.pivot(
        index="Display Label", 
        columns="Test Dataset", 
        values=target_metric
    )

    # Sort the index so models stay grouped together
    pivot_df = pivot_df.sort_index()

    # 4. Plotting
    # Dynamic height based on number of rows to keep it readable
    # plt.figure(figsize=(14, len(pivot_df) * 0.5 + 2)) # Adjust height based on rows
    # sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="mako")

    # Aggregate to see how Datasets transfer to each other across ALL models
    df[target_metric] = pd.to_numeric(df[target_metric], errors='coerce')
    compact_df = df.groupby(["Train Dataset", "Test Dataset"])[target_metric].mean().unstack()

    plt.figure(figsize=(10, 8))
    sns.heatmap(compact_df, annot=True, fmt=".3f", cmap="mako", mask=compact_df.isnull())
    plt.title("Aggregate Dataset Transferability")
    
    plt.ylabel("Model Configuration (Version, Source Domain)")

    # Aesthetics
    plt.title(f"Cross-Model Transfer Learning: {target_metric.upper()}", fontsize=16, pad=25)
    plt.xlabel("Test Dataset (Target Domain)", fontsize=12, labelpad=10)
    # plt.ylabel("(Model Version, Training Source)", fontsize=12, labelpad=10)
    
    # 5. Save with Path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Heatmap successfully saved to: {save_path.absolute()}")
    # plt.show()

def plot_faceted_transfer(results_map, target_metric: str, save_path: Path):
    all_records = []

    dataset_name_map = {
        "arxiv_data": "arxiv1 abstracts",
        "arxiv_data_210930-054931": "arxiv2 abstracts",
        "bbc_news": "bbc news headlines",
        "rating": "global news articles",
        "raw-data": "global news headlines",
        "reuters": "global news articles2"
    }

    # 1. Collect and Clean Data
    for model_name, data_dict in results_map.items():
        parts = model_name.replace(".json", "").split("_")
        clean_model_name = f"{parts[1].replace('preset0', 'v')}-{parts[2]}"
        
        for key, inner_dict in data_dict.items():
            train, test = key.split("__on__")
            val = inner_dict.get(target_metric)
            
            all_records.append({
                "Model": clean_model_name,
                "Train Dataset": dataset_name_map[train],
                "Test Dataset": dataset_name_map[test],
                target_metric: val
            })

    df = pd.DataFrame(all_records)
    
    # 2. Robust numeric conversion (handles "None" strings and Python None)
    df[target_metric] = pd.to_numeric(df[target_metric], errors='coerce')
    df = df.sort_values(by="Model")

    # 3. Setup FacetGrid
    # We use sharex=False and sharey=False so each "mini-heatmap" 
    # only shows the datasets relevant to that model version.
    g = sns.FacetGrid(
        df, 
        col="Model", 
        col_wrap=2, 
        sharex=False, 
        sharey=False, 
        height=5, 
        aspect=1.2
    )

    def draw_heatmap(data, **kwargs):
        # Pivot the subset for this specific facet
        pivot_data = data.pivot(index="Train Dataset", columns="Test Dataset", values=target_metric)
        
        # Plot the heatmap
        # cbar=True inside the facet can be crowded, but helpful for scale
        # We increase formatting size slightly
        sns.heatmap(
            pivot_data, 
            annot=True, 
            fmt=".3f", 
            cmap="mako", 
            mask=pivot_data.isnull(), 
            cbar=True,
            annot_kws={"size": 10}, # slightly larger numbers
            **kwargs
        )

    g.map_dataframe(draw_heatmap)

    # Increase from previous 0.4 to 0.7/0.8 for significantly more space.
    g.figure.subplots_adjust(top=0.9, hspace=0.7, wspace=0.6)
    
    g.figure.suptitle(f"Sequential Transfer Grid: {target_metric.upper()}", fontsize=18, y=0.97)
    
    # Iterate through axes to fix messy labels
    for ax in g.axes.flat:
        # Get the current title (which will be "Model = v1-balanced")
        current_title = ax.get_title()
        # Simplify it to just "v1-balanced"
        new_title = current_title.replace("Model = ", "")
        ax.set_title(new_title, fontsize=14, pad=10)
        
        # Rotate X labels more aggressively and fix horizontal alignment
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        # Flatten Y labels
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        
        # We remove redundant axis labels for the inner plots to keep it clean
        ax.set_xlabel("Test (Target Domain)", fontsize=10, labelpad=5)
        ax.set_ylabel("Train (Source Domain)", fontsize=10, labelpad=5)

    # 5. Save with Path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Faceted heatmap successfully saved to: {save_path.absolute()}")
    # plt.show()


def run_pipeline(config):
    #python visualize.py ./eval/cross_bertopic ./visualizations/bertopic > test_out.txt 2>&1
    metrics = ["coherence_cv", "topic_diversity", "coherence_cnpmi", "inverted_rbo", "std_topic_size"]
    #python visualize.py ./eval/cross ./visualizations/lda > test_out.txt 2>&1
    # metrics = ["coherence_v", "topic_diversity", "coherence_npmi", "inverted_rbo", "perplexity"]
    #gather all eval dirs
    #gather all models
    #build map [model: {metric: {dataset: value}}]
    final_map = create_json_map(config.input_path, metrics)
    # Increase 'indent' or 'width' to control the look
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(final_map)
    #Make transfer heatmap for topic coherence("coherence_v") and topic diversity("topic_diversity")
    for metric in metrics:
        create_transfer_heatmap(final_map, metric, config.output_path / "transfer_heatmap" / metric) 
        plot_faceted_transfer(final_map, metric, config.output_path / "faceted_transfer_heatmap" / metric)
def main():
    typer.run(visualize)

if __name__ == "__main__":
    main()