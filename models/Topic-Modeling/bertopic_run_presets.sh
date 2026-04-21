#!/usr/bin/env bash
set -euo pipefail

# Run the BERTopic presets from bertopic_train.py for a given dataset.
# Usage:
#   ./bertopic_run_presets.sh rating
#   ./bertopic_run_presets.sh ./data/bertopic/my_dataset.csv

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <dataset-name|path/to/dataset.csv>"
  exit 2
fi

INPUT_ARG="$1"
if [[ "$INPUT_ARG" == *.csv ]]; then
  INPUT="$INPUT_ARG"
  DATASET="$(basename "$INPUT" .csv)"
else
  DATASET="$INPUT_ARG"
  INPUT="./data/bertopic/${DATASET}.csv"
fi

if [ ! -f "$INPUT" ]; then
  echo "Error: input file not found: $INPUT"
  exit 1
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Preset definitions (copied from bertopic_train.py comments)
names=(
#   "preset01_balanced"
#   "preset02_social_granular"
  "preset03_small_stable"
  "preset04_large_coarse"
  "preset05_phrase_aware"
  "preset06_exec_coarse"
  "preset07_research_fine"
  "preset08_confidence"

#   "preset09_multilingual"
#   "preset10_repro_cpu"
)

extra_args=(
#   "--embedding-model sentence-transformers/all-MiniLM-L6-v2 --min-topic-size 25 --nr-topics auto --top-n-words 12 --min-df 8 --ngram-min 1 --ngram-max 2 --umap-n-neighbors 30 --umap-n-components 5 --umap-min-dist 0.05 --hdbscan-min-samples 10"
#   "--min-topic-size 5 --nr-topics auto --top-n-words 15 --min-df 1 --ngram-min 1 --ngram-max 1 --umap-n-neighbors 10 --umap-n-components 5 --umap-min-dist 0.0 --hdbscan-min-samples 2"
  "--min-topic-size 3 --nr-topics 12 --top-n-words 10 --min-df 1 --ngram-min 1 --ngram-max 2 --umap-n-neighbors 5 --umap-n-components 3 --umap-min-dist 0.1 --hdbscan-min-samples 1"
  "--min-topic-size 60 --nr-topics auto --top-n-words 10 --min-df 20 --ngram-min 1 --ngram-max 2 --umap-n-neighbors 50 --umap-n-components 5 --umap-min-dist 0.15 --hdbscan-min-samples 20"
  "--min-topic-size 12 --nr-topics auto --top-n-words 15 --min-df 2 --ngram-min 1 --ngram-max 3 --umap-n-neighbors 20 --umap-n-components 5 --umap-min-dist 0.0 --hdbscan-min-samples 5"
  "--min-topic-size 100 --nr-topics 20 --top-n-words 8 --min-df 25 --ngram-min 1 --ngram-max 2 --umap-n-neighbors 70 --umap-n-components 5 --umap-min-dist 0.2 --hdbscan-min-samples 30"
  "--min-topic-size 8 --nr-topics auto --top-n-words 20 --min-df 3 --ngram-min 1 --ngram-max 3 --umap-n-neighbors 12 --umap-n-components 8 --umap-min-dist 0.0 --hdbscan-min-samples 3"
  "--min-topic-size 20 --nr-topics auto --top-n-words 10 --min-df 5 --ngram-min 1 --ngram-max 2 --umap-n-neighbors 25 --umap-n-components 5 --umap-min-dist 0.1 --hdbscan-min-samples 15 --calculate-probabilities"

#   "--embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --language multilingual --min-topic-size 15 --nr-topics auto --top-n-words 12 --min-df 3 --ngram-min 1 --ngram-max 2 --umap-n-neighbors 20 --umap-n-components 5 --umap-min-dist 0.05 --hdbscan-min-samples 6"
#   "--cpu-only --embedding-model sentence-transformers/all-MiniLM-L6-v2 --min-topic-size 15 --nr-topics auto --top-n-words 10 --min-df 4 --ngram-min 1 --ngram-max 2 --umap-n-neighbors 15 --umap-n-components 5 --umap-min-dist 0.0 --umap-random-state 42 --umap-n-jobs 1 --hdbscan-min-samples 5 --hdbscan-core-dist-n-jobs 1 --omp-num-threads 1"
)

echo "Launching ${#names[@]} BERTopic jobs for dataset '${DATASET}' (input: $INPUT)"

MAX_JOBS=1

# Track launched PIDs explicitly (portable across macOS bash variants).
pids=()

running_jobs() {
  local alive=()
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    fi
  done
  pids=("${alive[@]}")
  echo "${#pids[@]}"
}

for i in "${!names[@]}"; do
  preset="${names[$i]}"
#   cache="$DIR/cache/${DATASET}/${preset}"
  log="$DIR/log/${DATASET}/bertopic_${preset}.txt"
  output="$DIR/output/${DATASET}/${preset}"
  eval_path="$DIR/eval/${DATASET}/bertopic_${preset}.json"

  # Ensure log directory exists (use $log variable)
  mkdir -p "$(dirname "$log")"
#   mkdir -p "$cache"
  mkdir -p "$output"
  mkdir -p "$(dirname "$eval_path")"

  # Safely split extra args and build command array
  extra="${extra_args[$i]}"
  IFS=' ' read -r -a extra_parts <<< "$extra"

  cmd=(python "$DIR/bertopic_train.py" --input "$INPUT" --text-col cleaned_text --output-dir "$output" --eval-path "$eval_path")
  cmd+=("${extra_parts[@]}")

  # Respect concurrency cap before launching next preset.
  while [ "$(running_jobs)" -ge "$MAX_JOBS" ]; do
    sleep 1
  done

  echo "Starting $preset -> output: $output, log: $log"
  "${cmd[@]}" > "$log" 2>&1 &
  pids+=("$!")

  sleep 1
done

echo "All BERTopic jobs launched. Tail logs with: tail -f $DIR/log/${DATASET}/*.txt"
