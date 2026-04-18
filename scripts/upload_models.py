"""Upload trained model artifacts and datasets to Hugging Face Hub.

Models  → repo_type="model"   e.g. deepakm10/brand-perception-models
Datasets → repo_type="dataset" e.g. deepakm10/brand-perception-datasets

Only inference-ready root files are uploaded for models (model.safetensors,
config.json, tokenizer.*, training_args.bin, *.pkl). Checkpoint folders and
optimizer/scheduler state are skipped. Empty model dirs are skipped.

Usage
-----
    # Dry run — see everything that would be uploaded
    python scripts/upload_models.py --dry-run

    # Upload models + datasets
    python scripts/upload_models.py

    # Upload a single model only
    python scripts/upload_models.py --only-models roberta_stage2

    # Make repos private
    python scripts/upload_models.py --private
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"

DEFAULT_MODEL_REPO = "deepakm10/brand-perception-models"
DEFAULT_DATASET_REPO = "deepakm10/brand-perception-datasets"

# Files to skip at any level
_SKIP_NAMES = {".DS_Store", "optimizer.pt", "rng_state.pth", "scheduler.pt"}
# Checkpoint folders — only needed to resume training
_SKIP_DIR_PREFIX = "checkpoint-"

# Completed models only — bertweet_stage1 is empty so excluded
UPLOAD_MODELS: dict[str, str] = {
    "roberta_stage1":    "roberta_stage1",
    "roberta_stage2":    "roberta_stage2",
    "distilbert_stage1": "distilbert_stage1",
    "distilbert_stage2": "distilbert_stage2",
    "sentiment_model.pkl": "sklearn",
}

# Datasets: local folder → subfolder in HF dataset repo
# sentiment-analysis/training.1600000.processed.noemoticon.csv is a duplicate
# of sentiment140 — excluded to avoid uploading 228 MB twice
UPLOAD_DATASETS: dict[str, str] = {
    "mams":               "mams",
    "reddit_comments":    "reddit_comments",
    "sentiment-analysis": "sentiment_analysis",
    "sentiment140":       "sentiment140",
    "twitter-sentiment":  "twitter_sentiment",
}

# Files inside dataset folders to skip (duplicates / noise)
_SKIP_DATASET_FILES = {
    "training.1600000.processed.noemoticon.csv",  # already in sentiment140/
}
_SKIP_DATASET_DIRS = {"sentiment-analysis"}  # dir that contains the duplicate


def _collect_model_files(model_path: Path) -> list[tuple[Path, str]]:
    """Return [(local_path, repo_relative_path)] for inference-ready files only."""
    pairs: list[tuple[Path, str]] = []

    if model_path.is_file():
        pairs.append((model_path, model_path.name))
        return pairs

    for item in sorted(model_path.iterdir()):
        if item.name in _SKIP_NAMES:
            continue
        if item.is_dir():
            if item.name.startswith(_SKIP_DIR_PREFIX):
                continue
            for sub in sorted(item.iterdir()):
                if sub.is_file() and sub.name not in _SKIP_NAMES:
                    pairs.append((sub, f"{item.name}/{sub.name}"))
        elif item.is_file():
            pairs.append((item, item.name))

    return pairs


def _collect_dataset_files(dataset_path: Path, local_name: str) -> list[tuple[Path, str]]:
    """Return [(local_path, repo_relative_path)] for a dataset folder."""
    pairs: list[tuple[Path, str]] = []

    for item in sorted(dataset_path.rglob("*")):
        if not item.is_file():
            continue
        if item.name in _SKIP_NAMES:
            continue
        # Skip the duplicate sentiment140 file inside sentiment-analysis/
        if local_name == "sentiment-analysis" and item.name in _SKIP_DATASET_FILES:
            print(f"    skip duplicate: {item.name}")
            continue
        rel = item.relative_to(dataset_path).as_posix()
        pairs.append((item, rel))

    return pairs


def _upload_batch(
    api: object,
    files: list[tuple[Path, str]],
    repo_id: str,
    repo_subfolder: str,
    repo_type: str,
    dry_run: bool,
    local_name: str,
) -> None:
    total_bytes = sum(f.stat().st_size for f, _ in files)
    print(f"  Files : {len(files)}   Size: {total_bytes / 1e6:.1f} MB")

    for local_file, rel_path in files:
        dest = f"{repo_subfolder}/{rel_path}"
        size_mb = local_file.stat().st_size / 1e6
        if dry_run:
            print(f"    would upload: {local_file.relative_to(PROJECT_ROOT)}  →  {dest}  ({size_mb:.1f} MB)")
        else:
            print(f"    {dest}  ({size_mb:.1f} MB) … ", end="", flush=True)
            api.upload_file(  # type: ignore[attr-defined]
                path_or_fileobj=str(local_file),
                path_in_repo=dest,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            print("done")


def upload_models(
    repo_id: str,
    *,
    only: list[str] | None = None,
    dry_run: bool = False,
    private: bool = False,
    api: object,
) -> None:
    if not dry_run:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)  # type: ignore[attr-defined]
        print(f"Model repo : https://huggingface.co/{repo_id}")

    targets = {k: v for k, v in UPLOAD_MODELS.items() if only is None or k in only}

    for local_name, repo_subfolder in targets.items():
        local_path = MODELS_DIR / local_name

        if not local_path.exists():
            print(f"\n  SKIP  {local_name}  (not found locally)")
            continue

        # Skip empty directories (e.g. bertweet_stage1)
        if local_path.is_dir() and not any(local_path.iterdir()):
            print(f"\n  SKIP  {local_name}  (directory is empty)")
            continue

        files = _collect_model_files(local_path)
        if not files:
            print(f"\n  SKIP  {local_name}  (no inference files found)")
            continue

        print(f"\n{'[DRY RUN] ' if dry_run else ''}Uploading model: {local_name} → {repo_id}/{repo_subfolder}/")
        _upload_batch(api, files, repo_id, repo_subfolder, "model", dry_run, local_name)

    if not dry_run:
        print(f"\nModels done → https://huggingface.co/{repo_id}")


def upload_datasets(
    repo_id: str,
    *,
    only: list[str] | None = None,
    dry_run: bool = False,
    private: bool = False,
    api: object,
) -> None:
    if not dry_run:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)  # type: ignore[attr-defined]
        print(f"Dataset repo: https://huggingface.co/datasets/{repo_id}")

    targets = {k: v for k, v in UPLOAD_DATASETS.items() if only is None or k in only}

    for local_name, repo_subfolder in targets.items():
        local_path = DATASETS_DIR / local_name

        if not local_path.exists():
            print(f"\n  SKIP  {local_name}  (not found locally)")
            continue

        files = _collect_dataset_files(local_path, local_name)
        if not files:
            print(f"\n  SKIP  {local_name}  (no files found)")
            continue

        print(f"\n{'[DRY RUN] ' if dry_run else ''}Uploading dataset: {local_name} → {repo_id}/{repo_subfolder}/")
        _upload_batch(api, files, repo_id, repo_subfolder, "dataset", dry_run, local_name)

    if not dry_run:
        print(f"\nDatasets done → https://huggingface.co/datasets/{repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload model artifacts and datasets to Hugging Face Hub."
    )
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_MODEL_REPO,
        metavar="USERNAME/REPO",
        help=f"HF model repo id (default: {DEFAULT_MODEL_REPO}).",
    )
    parser.add_argument(
        "--dataset-repo",
        default=DEFAULT_DATASET_REPO,
        metavar="USERNAME/REPO",
        help=f"HF dataset repo id (default: {DEFAULT_DATASET_REPO}).",
    )
    parser.add_argument(
        "--only-models",
        nargs="+",
        default=None,
        metavar="MODEL",
        choices=list(UPLOAD_MODELS),
        help=f"Upload only these models: {list(UPLOAD_MODELS)}",
    )
    parser.add_argument(
        "--only-datasets",
        nargs="+",
        default=None,
        metavar="DATASET",
        choices=list(UPLOAD_DATASETS),
        help=f"Upload only these datasets: {list(UPLOAD_DATASETS)}",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model uploads, only upload datasets.",
    )
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Skip dataset uploads, only upload models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without sending anything.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create both HF repos as private.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from huggingface_hub import HfApi  # type: ignore[import]
    except ImportError:
        print("Error: huggingface_hub is not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    api = HfApi()

    if not args.skip_models:
        upload_models(
            args.model_repo,
            only=args.only_models,
            dry_run=args.dry_run,
            private=args.private,
            api=api,
        )

    if not args.skip_datasets:
        upload_datasets(
            args.dataset_repo,
            only=args.only_datasets,
            dry_run=args.dry_run,
            private=args.private,
            api=api,
        )

    if not args.dry_run:
        print("\nAll uploads complete.")


if __name__ == "__main__":
    main()
