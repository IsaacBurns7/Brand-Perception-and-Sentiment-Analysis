#!/usr/bin/env bash
set -euo pipefail

# Run the 5 LDA presets from LDA_train.py for a given dataset.
# Usage:
#   ./run_presets.sh rating
#   ./run_presets.sh ./data/lda/my_dataset.csv

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
  INPUT="./data/lda/${DATASET}.csv"
fi

if [ ! -f "$INPUT" ]; then
  echo "Error: input file not found: $INPUT"
  exit 1
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Preset definitions (matched to the cookbook in LDA_train.py)
names=(
  "lda_preset01_balanced"
  "lda_preset02_fine_grained"
  "lda_preset03_small_stable"
  "lda_preset04_coarse_large"
  "lda_preset05_short_noisy"
)
num_topics=(15 35 8 12 20)
chunks=(500 600 200 2000 300)
passes=(20 25 30 12 35)
iters=(400 500 300 250 450)
evals=(10 5 5 10 5)
alphas=(auto asymmetric symmetric auto 0.1)
etas=(auto auto symmetric auto 0.1)

echo "Launching ${#names[@]} LDA jobs for dataset '${DATASET}' (input: $INPUT)"

for i in "${!names[@]}"; do
  preset="${names[$i]}"
  cache="$DIR/cache/${DATASET}/${preset}"
  log="$DIR/log/${DATASET}/${preset}.txt"
  output="$DIR/output/${DATASET}/${preset}"
  eval="$DIR/eval/${DATASET}/${preset}.json"

  cmd=(python "$DIR/LDA_train.py" \
    --input "$INPUT" \
    --token-column tokens_str \
    --cache-path "$cache" \
    --out-path "$output" \
    --eval-path "$eval" \
    --num-topics "${num_topics[$i]}" \
    --chunksize "${chunks[$i]}" \
    --passes "${passes[$i]}" \
    --iterations "${iters[$i]}" \
    --eval-every "${evals[$i]}" \
    --alpha "${alphas[$i]}" \
    --eta "${etas[$i]}" \
    --random-state 42)

  mkdir -p "$(dirname "$log")" #ensure log dir exists
  mkdir -p "$(dirname "$eval")" #ensure eval dir exists
  echo "Starting $preset -> cache: $cache, log: $log"
  "${cmd[@]}" > "$log" 2>&1 &
  sleep 1
done

