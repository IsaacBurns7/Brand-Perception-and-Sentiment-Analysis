"""Train a PyABSA ATEPC model on SemEval built-ins + MAMS, then save the checkpoint.

The training loop:
  1. (Optional) Convert MAMS XML → .dat.aste if not already done.
  2. Merge SemEval built-in datasets with the MAMS .dat.aste directory.
  3. Run PyABSA ATEPCTrainer; checkpoint lands in artifacts/models/absa_aste/.

Usage:
    python -m models.absa.train
    python -m models.absa.train --batch-size 32 --epochs 20
    python -m models.absa.train --no-semeval --epochs 5
    python -m models.absa.train --max-mams 1000 --semeval-datasets Restaurant14
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from .config import (
    ABSA_MODEL_DIR,
    ATEPC_BATCH_SIZE,
    ATEPC_EPOCHS,
    ATEPC_L2_LAMBDA,
    ATEPC_LEARNING_RATE,
    ATEPC_MAX_LENGTH,
    ATEPC_PATIENCE,
    MAMS_ASTE_DIR,
    MAMS_ASTE_TEST,
    MAMS_ASTE_TRAIN,
    MAMS_ASTE_VAL,
    PROJECT_ROOT,
    RANDOM_SEED,
    SEMEVAL_BUILTIN_DATASETS,
)

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))


def _resolve_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on hardware availability."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _ensure_mams_aste(max_sentences: int | None = None) -> None:
    """Convert MAMS XML → .dat.aste (always re-runs to apply any sentence cap)."""
    from .convert_mams import convert_all_splits

    print(
        f"Converting MAMS XML → .dat.aste"
        + (f" (cap: {max_sentences} per split)" if max_sentences else "")
        + " …"
    )
    convert_all_splits(max_sentences=max_sentences)


def train_atepc(
    *,
    epochs: int = ATEPC_EPOCHS,
    batch_size: int = ATEPC_BATCH_SIZE,
    learning_rate: float = ATEPC_LEARNING_RATE,
    max_length: int = ATEPC_MAX_LENGTH,
    l2_lambda: float = ATEPC_L2_LAMBDA,
    patience: int = ATEPC_PATIENCE,
    output_dir: Path = ABSA_MODEL_DIR,
    include_semeval: bool = True,
    include_mams: bool = True,
    semeval_datasets: list[str] | None = None,
    max_mams_sentences: int | None = None,
    from_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Configure and launch a PyABSA v2 ASTE training run (EMCGCN model).

    MAMS is a built-in PyABSA dataset — no XML conversion needed unless
    ``max_mams_sentences`` is set to cap training data per split.
    SemEval built-in datasets are merged via ``ABSADatasetList``.
    The best checkpoint is saved under output_dir.
    """
    try:
        from pyabsa import AspectSentimentTripletExtraction as ASTE  # type: ignore[import]
        from pyabsa import DatasetItem  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PyABSA is required for ABSA training. "
            "Install it with: pip install pyabsa"
        ) from exc

    # ── 1. Build dataset list ───────────────────────────────────────────────
    # Use ASTEDatasetList (IDs 401–404) not ABSADatasetList (APC IDs 113–116)
    _semeval_map: dict[str, list[str]] = {
        "Laptop14":     ASTE.ASTEDatasetList.Laptop14,
        "Restaurant14": ASTE.ASTEDatasetList.Restaurant14,
        "Restaurant15": ASTE.ASTEDatasetList.Restaurant15,
        "Restaurant16": ASTE.ASTEDatasetList.Restaurant16,
    }
    semeval_to_use = semeval_datasets if semeval_datasets is not None else SEMEVAL_BUILTIN_DATASETS

    dataset: list[Any] = []
    if include_semeval:
        for name in semeval_to_use:
            dataset.extend(_semeval_map[name])

    # MAMS has no built-in ASTE dataset — always use locally converted .dat.aste files
    if include_mams:
        _ensure_mams_aste(max_sentences=max_mams_sentences)
        dataset.append(str(MAMS_ASTE_DIR))

    # ── 2. Configure the ASTE model ────────────────────────────────────────
    config = ASTE.ASTEConfigManager.get_aste_config_english()
    config.model = ASTE.ASTEModelList.EMCGCN
    config.num_epoch = epochs
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.max_seq_len = max_length
    config.l2reg = l2_lambda
    config.patience = patience
    config.seed = [RANDOM_SEED]
    config.log_step = 50

    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device()
    print(f"Device         : {device}")
    print(f"Epochs         : {epochs}  Batch: {batch_size}  LR: {learning_rate}")
    print(f"Datasets       : {dataset}")
    if include_mams and max_mams_sentences is not None:
        print(f"MAMS cap       : {max_mams_sentences} sentences per split")
    print(f"Output dir     : {output_dir}")

    # ── 3. Train ────────────────────────────────────────────────────────────
    # Wrap the list in DatasetItem so PyABSA can resolve dataset_name internally
    dataset_item = DatasetItem("brand_perception", dataset)

    trainer = ASTE.ASTETrainer(
        config=config,
        dataset=dataset_item,
        from_checkpoint=from_checkpoint,
        checkpoint_save_mode=1,
        auto_device=device,
        path_to_save=str(output_dir),
    )
    result = trainer.load_trained_model()

    print(f"\nTraining complete. Checkpoint saved to: {output_dir}")
    return {
        "output_dir": str(output_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "datasets": dataset,
        "device": device,
        "result": str(result),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PyABSA ATEPC (ASTE) model on SemEval + MAMS."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=ATEPC_EPOCHS,
        help=f"Number of training epochs (default: {ATEPC_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=ATEPC_BATCH_SIZE,
        help=f"Training batch size (default: {ATEPC_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--learning-rate",
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=ATEPC_LEARNING_RATE,
        help=f"Learning rate (default: {ATEPC_LEARNING_RATE}).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=ATEPC_MAX_LENGTH,
        help=f"Max token length (default: {ATEPC_MAX_LENGTH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ABSA_MODEL_DIR,
        help=f"Where to save the ATEPC checkpoint (default: {ABSA_MODEL_DIR.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--no-semeval",
        action="store_true",
        help="Train on MAMS only; omit SemEval built-in datasets.",
    )
    parser.add_argument(
        "--no-mams",
        action="store_true",
        help="Train on SemEval datasets only; omit MAMS data.",
    )
    parser.add_argument(
        "--semeval-datasets",
        nargs="+",
        default=None,
        metavar="DATASET",
        choices=SEMEVAL_BUILTIN_DATASETS,
        help=(
            f"Subset of SemEval datasets to include (default: all {SEMEVAL_BUILTIN_DATASETS}). "
            "Ignored when --no-semeval is set."
        ),
    )
    parser.add_argument(
        "--max-mams",
        type=int,
        default=None,
        metavar="N",
        help="Cap MAMS training data at N sentences per split (default: use all ~4 300).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=ATEPC_PATIENCE,
        help=f"Early-stopping patience in epochs (default: {ATEPC_PATIENCE}).",
    )
    parser.add_argument(
        "--from-checkpoint",
        "--from_checkpoint",
        dest="from_checkpoint",
        default=None,
        metavar="PATH",
        help="Resume training from a saved checkpoint directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_atepc(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        patience=args.patience,
        output_dir=args.output_dir,
        include_semeval=not args.no_semeval,
        include_mams=not args.no_mams,
        semeval_datasets=args.semeval_datasets,
        max_mams_sentences=args.max_mams,
        from_checkpoint=args.from_checkpoint,
    )


if __name__ == "__main__":
    main()
