#!/usr/bin/env bash
set -euo pipefail

# Cross-dataset LDA evaluation runner.
#
# Plan:
# 1) Resolve friendly dataset aliases (arxiv1, arxiv2, raw_data, etc.) to real
#    file/cache dataset IDs used in this repo.
# 2) Read an editable train->test map (EVAL_MAP) where each train dataset can
#    target one or more test datasets.
# 3) For each mapping, run all 5 LDA presets with LDA_evaluate.py:
#      python LDA_evaluate.py <test_csv> <output_json> <train_cache_dir> <text_column>
# 4) Write outputs under eval/cross/ and logs under log/cross_eval/, then print
#    a final success/skip/failure summary.
#
# Usage:
#   ./lda_cross_eval_presets.sh
#   TEXT_COLUMN=tokens_str ./lda_cross_eval_presets.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEXT_COLUMN="${TEXT_COLUMN:-tokens_str}"

# Current LDA preset cache names in this project.
PRESETS=(
  "lda_preset01_balanced"
  "lda_preset02_fine_grained"
  "lda_preset03_small_stable"
  "lda_preset04_coarse_large"
  "lda_preset05_short_noisy"
)

# Map your logical labels -> actual dataset IDs used by ./data/lda and ./cache.
resolve_dataset() {
  local key
  key="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  case "$key" in
    arxiv1|arxiv_data)
      echo "arxiv_data"
      ;;
    arxiv2|arxiv_data_210930-054931)
      echo "arxiv_data_210930-054931"
      ;;
    bbc_news)
      echo "bbc_news"
      ;;
    raw_data|raw-data)
      echo "raw-data"
      ;;
    rating)
      echo "rating"
      ;;
    reuters)
      echo "reuters"
      ;;
    *)
      echo "$1"
      ;;
  esac
}

# -----------------------------------------------------------------------------
# Edit this map to control what runs.
# Key   = training dataset (where model cache lives).
# Value = space-separated testing datasets (which CSVs to evaluate on).
# -----------------------------------------------------------------------------
declare -A EVAL_MAP=(
#   [arxiv1]="arxiv2"
#   [arxiv2]="arxiv1"
#   [bbc_news]="raw_data"
#   [raw_data]="bbc_news"
#   [rating]="reuters"
  [reuters]="rating"
)

success_count=0
skip_count=0
fail_count=0

echo "Starting cross-dataset LDA evaluation"
echo "Text column: ${TEXT_COLUMN}"
echo "Total train datasets configured: ${#EVAL_MAP[@]}"

for train_alias in "${!EVAL_MAP[@]}"; do
  train_ds="$(resolve_dataset "$train_alias")"

  for test_alias in ${EVAL_MAP[$train_alias]}; do
    test_ds="$(resolve_dataset "$test_alias")"

    if [[ "$train_ds" == "$test_ds" ]]; then
      echo "Skipping self-eval mapping: ${train_alias} -> ${test_alias}"
      skip_count=$((skip_count + 1))
      continue
    fi

    test_csv="$DIR/data/lda/${test_ds}.csv"
    if [[ ! -f "$test_csv" ]]; then
      echo "Skipping: missing test csv for ${train_alias} -> ${test_alias}: $test_csv"
      skip_count=$((skip_count + 1))
      continue
    fi

    echo "\n=== ${train_alias} (${train_ds}) -> ${test_alias} (${test_ds}) ==="

    for preset in "${PRESETS[@]}"; do
      cache_dir="$DIR/cache/${train_ds}/${preset}"
      out_dir="$DIR/eval/cross/${train_ds}__on__${test_ds}"
      out_json="$out_dir/${preset}.json"
      log_dir="$DIR/log/cross_eval/${train_ds}__on__${test_ds}"
      log_file="$log_dir/${preset}.txt"

      if [[ ! -d "$cache_dir" ]]; then
        echo "Skipping preset (missing cache): $cache_dir"
        skip_count=$((skip_count + 1))
        continue
      fi

      mkdir -p "$out_dir" "$log_dir"

      echo "Running ${preset}"
      if python "$DIR/LDA_evaluate.py" "$test_csv" "$out_json" "$cache_dir" "$TEXT_COLUMN" >"$log_file" 2>&1; then
        success_count=$((success_count + 1))
      else
        echo "FAILED ${train_alias} -> ${test_alias} :: ${preset} (see $log_file)"
        fail_count=$((fail_count + 1))
      fi
    done
  done
done

echo "\nDone."
echo "Successful runs: $success_count"
echo "Skipped runs:    $skip_count"
echo "Failed runs:     $fail_count"

if [[ "$fail_count" -gt 0 ]]; then
  exit 1
fi
