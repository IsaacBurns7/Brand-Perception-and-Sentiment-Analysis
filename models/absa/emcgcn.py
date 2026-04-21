from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


ABSA_MODEL_NAME = os.environ.get(
    "BRAND_PERCEPTION_ABSA_MODEL",
    "./brand-absa-emcgcn",
)


class ABSAError(RuntimeError):
    """Raised when the ABSA model cannot be loaded or executed."""


def _normalize_triplets(result: Any) -> list[tuple[str, str, str]]:
    if not isinstance(result, dict):
        return []

    raw_triplets = result.get("Triplets") or result.get("triplets") or []
    if not isinstance(raw_triplets, list):
        return []

    normalized: list[tuple[str, str, str]] = []
    for triplet in raw_triplets:
        if not isinstance(triplet, dict):
            continue

        aspect = str(triplet.get("Aspect") or triplet.get("aspect") or "").strip()
        opinion = str(triplet.get("Opinion") or triplet.get("opinion") or "").strip()
        sentiment = str(
            triplet.get("Polarity")
            or triplet.get("Sentiment")
            or triplet.get("sentiment")
            or ""
        ).strip()

        if not aspect:
            continue

        normalized.append((aspect, opinion, sentiment))

    return normalized


@lru_cache(maxsize=1)
def _load_extractor() -> Any:
    try:
        from pyabsa import AspectSentimentTripletExtraction as ASTE
    except ModuleNotFoundError as exc:
        raise ABSAError(
            "PyABSA is not installed. Install it with: pip install pyabsa"
        ) from exc

    try:
        return ASTE.AspectSentimentTripletExtractor(
            checkpoint=ABSA_MODEL_NAME,
            auto_device=True,
        )
    except Exception as exc:  # pragma: no cover - depends on external model/runtime
        raise ABSAError(
            f"Failed to load ABSA checkpoint '{ABSA_MODEL_NAME}'."
        ) from exc


def run_absa(texts: list[str]) -> list[list[tuple[str, str, str]]]:
    """
    Run ASTE inference and return normalized (aspect, opinion, sentiment) triplets.
    """
    if not texts:
        return []

    extractor = _load_extractor()

    try:
        results = extractor.predict(
            list(texts),
            print_result=False,
            ignore_error=True,
        )
    except Exception as exc:  # pragma: no cover - depends on external model/runtime
        raise ABSAError("ABSA inference failed.") from exc

    if isinstance(results, dict):
        results = [results]
    if not isinstance(results, list):
        raise ABSAError("Unexpected ABSA prediction payload.")

    return [_normalize_triplets(result) for result in results]
