"""Convert MAMS XML annotation files to PyABSA ASTE .dat.aste format.

Output format per line:
    sentence#### #### ####[(word_from, word_to, POLARITY), ...]

where word_from/word_to are 0-based whitespace-token indices (inclusive) and
POLARITY is one of: POS | NEG | NEU.

Usage:
    python -m models.absa.convert_mams
    python -m models.absa.convert_mams --splits train val test --output-dir data/processed/mams_aste
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from .config import (
    MAMS_ASTE_DIR,
    MAMS_ASTE_TEST,
    MAMS_ASTE_TRAIN,
    MAMS_ASTE_VAL,
    MAMS_DIR,
    POLARITY_MAP,
    PROJECT_ROOT,
)

# Matches runs of non-whitespace characters (tokens as PyABSA sees them)
_TOKEN_RE = re.compile(r"\S+")

SPLIT_PATHS: dict[str, tuple[Path, Path]] = {
    "train": (MAMS_DIR / "train.xml", MAMS_ASTE_TRAIN),
    "val": (MAMS_DIR / "val.xml", MAMS_ASTE_VAL),
    "test": (MAMS_DIR / "test.xml", MAMS_ASTE_TEST),
}


def char_to_word_span(text: str, char_from: int, char_to: int) -> tuple[int, int] | tuple[None, None]:
    """Map character offsets [char_from, char_to) to word indices [word_from, word_to] (inclusive).

    MAMS uses exclusive char_to (i.e. text[char_from:char_to] == term).
    Returns (None, None) when the span cannot be resolved against tokenised text.
    """
    tokens = list(_TOKEN_RE.finditer(text))
    word_from: int | None = None
    word_to: int | None = None

    for word_idx, match in enumerate(tokens):
        tok_start, tok_end = match.start(), match.end()

        if word_from is None and tok_start <= char_from < tok_end:
            word_from = word_idx

        # char_to is exclusive; the last character of the span is at char_to - 1
        if tok_start < char_to <= tok_end:
            word_to = word_idx

    return word_from, word_to


def iter_aste_lines(xml_path: Path) -> Iterator[str]:
    """Yield formatted .dat.aste lines parsed from a MAMS XML file.

    Sentences with no resolvable aspect spans are skipped with a warning.
    Sentences whose aspect term list is empty (rare) are also skipped.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for sentence_el in root.iter("sentence"):
        text_el = sentence_el.find("text")
        if text_el is None or not (text_el.text or "").strip():
            continue

        text = (text_el.text or "").strip()
        terms_el = sentence_el.find("aspectTerms")
        if terms_el is None:
            continue

        # Skip the entire sentence if ANY aspect has polarity="conflict"
        aspect_term_els = terms_el.findall("aspectTerm")
        if any(
            (el.get("polarity") or "").strip().lower() == "conflict"
            for el in aspect_term_els
        ):
            continue

        triplets: list[str] = []
        for term_el in aspect_term_els:
            raw_polarity = (term_el.get("polarity") or "").strip().lower()
            polarity_code = POLARITY_MAP.get(raw_polarity)
            if polarity_code is None:
                continue

            try:
                char_from = int(term_el.get("from", -1))
                char_to = int(term_el.get("to", -1))
            except ValueError:
                continue

            if char_from < 0 or char_to <= char_from:
                continue

            word_from, word_to = char_to_word_span(text, char_from, char_to)
            if word_from is None or word_to is None:
                print(
                    f"Warning: could not resolve span ({char_from},{char_to}) in: {text!r}",
                    file=sys.stderr,
                )
                continue

            triplets.append(f"({word_from}, {word_to}, {polarity_code})")

        if not triplets:
            continue

        yield f"{text}#### #### ####[{', '.join(triplets)}]"


def convert_split(xml_path: Path, aste_path: Path, max_sentences: int | None = None) -> int:
    """Convert one MAMS XML split to .dat.aste and return the number of written lines.

    Args:
        max_sentences: If set, only the first N sentences are written (useful for
                       lightweight training runs).
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"MAMS XML not found: {xml_path}")

    aste_path.parent.mkdir(parents=True, exist_ok=True)
    lines = list(iter_aste_lines(xml_path))
    if max_sentences is not None:
        lines = lines[:max_sentences]

    with aste_path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")

    return len(lines)


def convert_all_splits(
    splits: list[str] | None = None,
    output_dir: Path | None = None,
    max_sentences: int | None = None,
) -> dict[str, int]:
    """Convert specified MAMS splits (default: all) and return {split: line_count}.

    Args:
        max_sentences: Cap each split at this many sentences.
    """
    requested = splits or list(SPLIT_PATHS)
    out_dir = output_dir or MAMS_ASTE_DIR
    counts: dict[str, int] = {}

    for split in requested:
        if split not in SPLIT_PATHS:
            raise ValueError(f"Unknown split {split!r}. Valid: {list(SPLIT_PATHS)}")

        xml_path, default_aste = SPLIT_PATHS[split]
        aste_path = out_dir / default_aste.name

        count = convert_split(xml_path, aste_path, max_sentences=max_sentences)
        counts[split] = count
        suffix = f" (capped at {max_sentences})" if max_sentences else ""
        print(f"Converted {split:5s}: {count:5d} sentences → {aste_path}{suffix}")

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MAMS XML annotation files to PyABSA ASTE .dat.aste format."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=list(SPLIT_PATHS),
        metavar="SPLIT",
        help="Which splits to convert (default: train val test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MAMS_ASTE_DIR,
        help=f"Output directory for .dat.aste files (default: {MAMS_ASTE_DIR.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        metavar="N",
        help="Cap each split at N sentences (default: use all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = convert_all_splits(
        splits=args.splits,
        output_dir=args.output_dir,
        max_sentences=args.max_sentences,
    )
    total = sum(counts.values())
    print(f"\nTotal sentences written: {total}")


if __name__ == "__main__":
    main()
