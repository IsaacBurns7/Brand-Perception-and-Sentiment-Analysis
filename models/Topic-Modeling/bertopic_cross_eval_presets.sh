#!/usr/bin/env bash
set -euo pipefail

# Cross-dataset BERTopic evaluation runner (10 presets).
#
# Plan:
# 1) Resolve friendly dataset aliases (arxiv1, arxiv2, raw_data, etc.) to real
#    dataset IDs used by ./data/bertopic and ./cache.
# 2) Read an editable train->test map (EVAL_MAP).
# 3) For each mapping, evaluate all 10 BERTopic presets by loading the trained
#    model from cache and transforming test documents.
# 4) Write JSON outputs and logs per train/test/preset, then print a summary.
#
# Usage:
#   ./bertopic_cross_eval_presets.sh
#   TEXT_COLUMN=cleaned_text ./bertopic_cross_eval_presets.sh
#   MAX_DOC_COUNT=5000 ./bertopic_cross_eval_presets.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$DIR/../../.venv/bin/python" ]]; then
    PYTHON_BIN="$DIR/../../.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
TEXT_COLUMN="${TEXT_COLUMN:-cleaned_text}"
MAX_DOC_COUNT="${MAX_DOC_COUNT:-}"
MIN_DF="${MIN_DF:-2}"
TOP_N_WORDS="${TOP_N_WORDS:-100}"
ASSIGNMENT_METHOD="${ASSIGNMENT_METHOD:-approximate}"

# Conservative defaults that reduce BERTopic runtime instability on some systems.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PRESETS=(
  "preset01_balanced"
  "preset02_social_granular"
  "preset03_small_stable"
  "preset04_large_coarse"
  "preset05_phrase_aware"
  "preset06_exec_coarse"
  "preset07_research_fine"
  "preset08_confidence"
  "preset09_multilingual"
  "preset10_repro_cpu"
)

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

resolve_preset_cache_dir() {
  local train_ds="$1"
  local preset="$2"
  local candidate1="$DIR/output/${train_ds}/${preset}"
  local candidate2="$DIR/output/${train_ds}/bertopic_${preset}"

  if [[ -d "$candidate1" ]]; then
    echo "$candidate1"
    return 0
  fi
  if [[ -d "$candidate2" ]]; then
    echo "$candidate2"
    return 0
  fi
  return 1
}

# -----------------------------------------------------------------------------
# Edit this map to control what runs.
# Key   = training dataset (where preset model cache lives).
# Value = space-separated testing datasets.
# -----------------------------------------------------------------------------
declare -A EVAL_MAP=(
  [arxiv1]="arxiv2"
  # [arxiv2]="arxiv1"
#   [bbc_news]="raw_data"
#   [raw_data]="bbc_news"
#   [rating]="reuters"
#   [reuters]="rating"
)

success_count=0
skip_count=0
fail_count=0

echo "Starting cross-dataset BERTopic evaluation"
echo "Text column: ${TEXT_COLUMN}"
echo "Min DF: ${MIN_DF}, Top N words: ${TOP_N_WORDS}"
echo "Assignment method: ${ASSIGNMENT_METHOD}"
if [[ -n "$MAX_DOC_COUNT" ]]; then
  echo "Max docs per run: ${MAX_DOC_COUNT}"
fi
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

    test_csv="$DIR/data/bertopic/${test_ds}.csv"
    if [[ ! -f "$test_csv" ]]; then
      echo "Skipping: missing test csv for ${train_alias} -> ${test_alias}: $test_csv"
      skip_count=$((skip_count + 1))
      continue
    fi

    echo "\n=== ${train_alias} (${train_ds}) -> ${test_alias} (${test_ds}) ==="

    for preset in "${PRESETS[@]}"; do
      if ! cache_dir="$(resolve_preset_cache_dir "$train_ds" "$preset")"; then
        echo "Skipping preset (missing cache dir): ${train_ds}/${preset}"
        skip_count=$((skip_count + 1))
        continue
      fi

      model_path="$cache_dir/bertopic_model"
      if [[ ! -f "$model_path" ]]; then
        echo "Skipping preset (missing model file): $model_path"
        skip_count=$((skip_count + 1))
        continue
      fi

      out_dir="$DIR/eval/cross_bertopic/${train_ds}__on__${test_ds}"
      out_json="$out_dir/bertopic_${preset}.json"
      log_dir="$DIR/log/cross_eval_bertopic/${train_ds}__on__${test_ds}"
      log_file="$log_dir/bertopic_${preset}.txt"

      mkdir -p "$out_dir" "$log_dir"

      cmd=("$PYTHON_BIN" "$DIR/bertopic_evaluate.py"
        --input "$test_csv"
        --text-col "$TEXT_COLUMN"
        --model-path "$model_path"
        --output-path "$out_json"
        --min-df "$MIN_DF"
        --top-n-words "$TOP_N_WORDS"
        --assignment-method "$ASSIGNMENT_METHOD")

      if [[ -n "$MAX_DOC_COUNT" ]]; then
        cmd+=(--max-doc-count "$MAX_DOC_COUNT")
      fi

      echo "Running ${preset}"
      if "${cmd[@]}" >"$log_file" 2>&1; then
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
