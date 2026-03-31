"""
NER Pipeline — Brand Entity Extraction
=======================================
Slot in the Brand Perception & Sentiment Analysis pipeline
between the data-cleaning step (news_dailyworker.py / Preprocessing)
and downstream modelling (LDA, sentiment).

Data contract
-------------
Input  : pandas DataFrame with columns produced by news_dailyworker.py
           - source_name  (str)   — e.g. "Forbes", "BBC News"
           - title        (str)
           - description  (str)
           - content      (str)   — raw scraped full text
           - full_content (str)   — cleaned full text (after Preprocessing.runner)
           - category     (str)   — NewsAPI query term used for collection
           - published_at (str)

Output : same DataFrame with two new columns
           - ner_brands   (list[str]) — canonical brand names found in article
           - ner_raw_json (str)       — full JSON payload for DB insertion

The NERPipeline.run_on_dataframe() method is the single public entry point.

Model backends (in priority order)
------------------------------------
  spaCy  — already in the project (LDA.py / LDA_normalize_corpus.py use it).
            Reuses en_core_web_sm / en_core_web_md that are already downloaded.
            NER is disabled in the LDA pipeline — we re-enable it here.
  Rules  — zero-dependency fallback using regex + brand dictionary.
            Always runs alongside spaCy to recover brands spaCy misses.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import time

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


# ── Timing helper (matches existing style in LDA.py) ─────────────────────────

@contextmanager
def stage(name: str):
    start = time.perf_counter()
    logger.info("[timing] start: %s", name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("[timing] end:   %s (%.2fs)", name, elapsed)


# ── Data models ───────────────────────────────────────────────────────────────

class EntityType(str, Enum):
    BRAND    = "BRAND"
    ORG      = "ORG"
    PRODUCT  = "PRODUCT"
    PERSON   = "PERSON"
    LOCATION = "LOCATION"
    OTHER    = "OTHER"


@dataclass
class RawEntity:
    text:        str
    entity_type: EntityType
    start:       int
    end:         int
    confidence:  float
    model:       str


@dataclass
class BrandEntity:
    """One de-duplicated brand mention — maps directly to the `brands` DB table."""
    canonical_name: str
    aliases:        list[str]       = field(default_factory=list)
    entity_type:    EntityType      = EntityType.BRAND
    confidence:     float           = 0.0
    mention_count:  int             = 1
    positions:      list[tuple]     = field(default_factory=list)
    model_sources:  list[str]       = field(default_factory=list)
    doc_id:         Optional[str]   = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entity_type"] = self.entity_type.value
        return d


@dataclass
class NERResult:
    doc_id:         Optional[str]
    brand_entities: list[BrandEntity]
    model_used:     str
    source_name:    Optional[str] = None   # maps to news_dailyworker source_name
    category:       Optional[str] = None   # maps to news_dailyworker category

    def brand_names(self) -> list[str]:
        return [e.canonical_name for e in self.brand_entities]

    def to_dict(self) -> dict:
        return {
            "doc_id":         self.doc_id,
            "source_name":    self.source_name,
            "category":       self.category,
            "model_used":     self.model_used,
            "brand_entities": [e.to_dict() for e in self.brand_entities],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ── Rules backend ─────────────────────────────────────────────────────────────
# The project already uses regex cleaning in LDA_normalize_corpus.py and
# eda_utils.py — these patterns complement that work rather than duplicating it.

_ORG_SUFFIXES = (
    r"Inc\.?", r"LLC\.?", r"Ltd\.?", r"Corp\.?", r"Co\.?",
    r"GmbH", r"S\.A\.", r"N\.V\.", r"PLC", r"AG", r"SE",
    r"Holdings?", r"Group", r"Partners?", r"Associates?",
    r"Enterprises?", r"Solutions?", r"Technologies?", r"Tech",
    r"Systems?", r"Services?", r"Labs?", r"Studios?",
    r"Industries", r"International", r"Global",
)

_SUFFIX_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&'\-\.]{1,40}(?:\s+[A-Z][A-Za-z0-9&'\-\.]{0,30}){0,4})"
    r"\s*,?\s*(?:" + "|".join(_ORG_SUFFIXES) + r")\b"
)

# Based on top entities seen in the EDA outputs (brand_sentiment, binary_sentiment)
# and the news source domains already scraped by news_dailyworker.py
_KNOWN_BRANDS: set[str] = {
    # Big tech (dominant in news corpus)
    "Google", "Apple", "Microsoft", "Amazon", "Meta", "Facebook",
    "Tesla", "Netflix", "Spotify", "Twitter", "X", "LinkedIn",
    "Adobe", "Salesforce", "Oracle", "SAP", "IBM", "Intel",
    "Nvidia", "AMD", "Qualcomm", "Samsung", "Sony", "LG",
    "OpenAI", "Anthropic", "DeepMind", "Hugging Face",
    "GitHub", "GitLab", "Slack", "Zoom", "Dropbox",
    "Shopify", "Stripe", "PayPal", "Visa", "Mastercard",
    "Uber", "Lyft", "Airbnb", "DoorDash",
    "Cloudflare", "Twilio", "Snowflake", "Databricks",
    # Media / publishing (sources already in news_dailyworker.py)
    "Forbes", "CNN", "BBC", "Reuters", "Bloomberg", "Wired",
    "TechCrunch", "Gizmodo", "The Verge", "Deadline", "NPR",
    "Al Jazeera", "ABC News", "Time", "Euronews",
    # Finance
    "JPMorgan", "Goldman Sachs", "BlackRock", "Berkshire Hathaway",
    "Citigroup", "Wells Fargo", "Morgan Stanley",
    # Consumer / retail
    "Nike", "Adidas", "Puma", "Zara", "IKEA", "Walmart", "Target",
    "Costco", "Amazon", "Starbucks", "McDonald's", "Chipotle",
    "Coca-Cola", "Pepsi", "Nestlé", "Unilever",
    # Pharma / health
    "Pfizer", "Moderna", "AstraZeneca", "Johnson & Johnson",
    # Auto
    "Tesla", "Ford", "GM", "BMW", "Mercedes", "Toyota",
    "Volkswagen", "Hyundai", "Kia", "Rivian", "Lucid",
    # Aerospace / defence
    "Boeing", "Airbus", "SpaceX", "Lockheed Martin",
}

_KNOWN_BRANDS_RE = re.compile(
    r"\b(" + "|".join(re.escape(b) for b in sorted(_KNOWN_BRANDS, key=len, reverse=True)) + r")\b"
)

_CAMEL_RE = re.compile(r'\b([A-Z][a-z]{1,15}[A-Z][A-Za-z]{2,20})\b')
_TICKER_RE = re.compile(r'(?:\$([A-Z]{1,5})\b|\((?:NYSE|NASDAQ|LSE):\s*([A-Z]{1,5})\))')

_STOPWORDS = {
    "The", "This", "That", "These", "Those", "We", "Our", "Their",
    "His", "Her", "Its", "For", "And", "But", "Or", "Not", "All",
    "Any", "Some", "More", "Most", "Many", "How", "Why", "What",
    "When", "Where", "Who", "Also", "Still", "Just", "Even",
    "New", "Old", "First", "Last", "High", "Low", "Good", "Bad",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
}


class RulesNERModel:
    """
    Regex + dictionary NER.  Always available — no extra installs needed.
    Complements spaCy: catches known brand names that spaCy lemmatises away,
    and legal-suffix organisations that spaCy may miss in short fragments.
    """
    name = "rules-v1"

    def predict(self, text: str) -> list[RawEntity]:
        entities: list[RawEntity] = []

        for m in _SUFFIX_RE.finditer(text):
            full = m.group(0).strip().rstrip(",")
            if self._valid(full):
                entities.append(RawEntity(full, EntityType.ORG, m.start(), m.end(), 0.88, self.name))

        for m in _KNOWN_BRANDS_RE.finditer(text):
            entities.append(RawEntity(m.group(1), EntityType.BRAND, m.start(), m.end(), 0.92, self.name))

        for m in _CAMEL_RE.finditer(text):
            tok = m.group(1)
            if self._valid(tok) and not self._covered(m.start(), m.end(), entities):
                entities.append(RawEntity(tok, EntityType.ORG, m.start(), m.end(), 0.60, self.name))

        for m in _TICKER_RE.finditer(text):
            ticker = m.group(1) or m.group(2)
            entities.append(RawEntity(ticker, EntityType.BRAND, m.start(), m.end(), 0.80, self.name))

        return entities

    def _valid(self, tok: str) -> bool:
        if not tok or len(tok) < 2 or len(tok) > 80:
            return False
        if tok.split()[0] in _STOPWORDS:
            return False
        return any(c.isalpha() for c in tok)

    def _covered(self, start: int, end: int, existing: list[RawEntity]) -> bool:
        return any(e.start <= start and e.end >= end for e in existing)


# ── spaCy backend ─────────────────────────────────────────────────────────────
# The project already loads en_core_web_sm / en_core_web_md for LDA.
# We load with NER *enabled* and parser disabled (same pattern as LDA.py).

_SPACY_LABEL_MAP = {
    "ORG":        EntityType.ORG,
    "PRODUCT":    EntityType.PRODUCT,
    "PERSON":     EntityType.PERSON,
    "GPE":        EntityType.LOCATION,
    "LOC":        EntityType.LOCATION,
    "FAC":        EntityType.ORG,
    "WORK_OF_ART": EntityType.PRODUCT,
}


class SpacyNERModel:
    """
    Wraps spaCy NER — re-uses the model already present in the project.
    Loads with parser disabled (mirrors LDA.py) but NER enabled.

    Preferred model : en_core_web_md  (already downloaded for LDA_normalize_corpus.py)
    Fallback        : en_core_web_sm  (also used in LDA.py)
    """

    def __init__(self, model_name: str = "en_core_web_md"):
        self._available = False
        for candidate in [model_name, "en_core_web_sm"]:
            try:
                import spacy  # type: ignore
                # disable parser + lemmatizer to save memory (same pattern as LDA.py)
                # but keep NER — that's what we need here
                self._nlp = spacy.load(candidate, disable=["parser", "lemmatizer"])
                self._available = True
                self.name = f"spacy/{candidate}"
                logger.info("spaCy NER model loaded: %s", candidate)
                break
            except (ImportError, OSError):
                continue
        if not self._available:
            logger.warning("spaCy not available. Falling back to Rules only.")
            self.name = "spacy/unavailable"

    @property
    def available(self) -> bool:
        return self._available

    def predict(self, text: str) -> list[RawEntity]:
        if not self._available or not text.strip():
            return []
        # spaCy has a max_length guard — chunk if needed (articles can be long,
        # matching the up-to-1.2M char articles seen in news.txt EDA output)
        results: list[RawEntity] = []
        chunk_size  = 100_000
        overlap_str = 200
        offset = 0
        while offset < len(text):
            chunk = text[offset: offset + chunk_size]
            doc   = self._nlp(chunk)
            for ent in doc.ents:
                etype = _SPACY_LABEL_MAP.get(ent.label_, EntityType.OTHER)
                results.append(RawEntity(
                    text        = ent.text,
                    entity_type = etype,
                    start       = offset + ent.start_char,
                    end         = offset + ent.end_char,
                    confidence  = 0.85,
                    model       = self.name,
                ))
            offset += chunk_size - overlap_str
        return results


# ── Normaliser ────────────────────────────────────────────────────────────────
# Mirrors the text-cleaning philosophy of eda_utils._normalize_text and
# LDA_normalize_corpus.clean_text — strip noise, collapse whitespace.

_STRIP_CHARS   = " \t\n.,;:!?()[]{}\"'"
_NOISE_SUFFIX  = re.compile(
    r'\s+(said|says|has|have|had|is|are|was|were|will|would|could|should|'
    r'announced|reported|confirmed|stated|noted|added|revealed|claims|told)$',
    re.IGNORECASE,
)

BRAND_TYPES = {EntityType.BRAND, EntityType.ORG, EntityType.PRODUCT}


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text).strip(_STRIP_CHARS)
    text = _NOISE_SUFFIX.sub("", text).strip(_STRIP_CHARS)
    return re.sub(r'\s{2,}', ' ', text)


def _key(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


class EntityNormaliser:
    def normalise(
        self,
        raw: list[RawEntity],
        doc_id: Optional[str] = None,
        min_confidence: float = 0.45,
    ) -> list[BrandEntity]:

        cleaned = [
            RawEntity(_normalise(e.text), e.entity_type, e.start, e.end, e.confidence, e.model)
            for e in raw
            if e.confidence >= min_confidence
            and e.entity_type in BRAND_TYPES
            and len(_normalise(e.text)) >= 2
            and _normalise(e.text).split()[0] not in _STOPWORDS
        ]

        groups: dict[str, list[RawEntity]] = {}
        for e in cleaned:
            groups.setdefault(_key(e.text), []).append(e)

        results: list[BrandEntity] = []
        for group in groups.values():
            canonical = max(group, key=lambda x: len(x.text)).text
            aliases   = list({e.text for e in group if e.text != canonical})
            results.append(BrandEntity(
                canonical_name = canonical,
                aliases        = aliases,
                entity_type    = group[0].entity_type,
                confidence     = round(sum(e.confidence for e in group) / len(group), 4),
                mention_count  = len(group),
                positions      = sorted({(e.start, e.end) for e in group}),
                model_sources  = list({e.model for e in group}),
                doc_id         = doc_id,
            ))

        results.sort(key=lambda x: (-x.confidence, x.canonical_name))
        return results


# ── Pipeline ──────────────────────────────────────────────────────────────────

class NERPipeline:
    """
    Main entry point.

    Usage in the project
    --------------------
    # After news_dailyworker.Preprocessing().runner() produces the daily CSV:

    from NER.ner_pipeline import NERPipeline
    import pandas as pd

    df = pd.read_csv("data/dailyworker/2025-01-15.csv")
    pipeline = NERPipeline()
    df = pipeline.run_on_dataframe(df)
    df.to_csv("data/dailyworker/2025-01-15_ner.csv", index=False)

    Parameters
    ----------
    spacy_model     : spaCy model name — prefer en_core_web_md (already in project)
    min_confidence  : drop entities below this threshold (default 0.45)
    text_column     : which DataFrame column to run NER on (default: full_content,
                      falls back to content then article)
    combine_rules   : also run the Rules model alongside spaCy (improves recall)
    """

    def __init__(
        self,
        spacy_model:    str   = "en_core_web_md",
        min_confidence: float = 0.45,
        text_column:    str   = "full_content",
        combine_rules:  bool  = True,
    ):
        self.min_confidence = min_confidence
        self.text_column    = text_column
        self.combine_rules  = combine_rules
        self._normaliser    = EntityNormaliser()
        self._rules         = RulesNERModel()

        with stage("load spaCy NER model"):
            self._spacy = SpacyNERModel(spacy_model)

        if self._spacy.available:
            self.model_used = self._spacy.name + ("+rules" if combine_rules else "")
        else:
            self.model_used = self._rules.name
            logger.warning("Running in Rules-only mode.")

    # ── Single document ───────────────────────────────────────────────────────

    def run(
        self,
        text:        str,
        doc_id:      Optional[str] = None,
        source_name: Optional[str] = None,
        category:    Optional[str] = None,
    ) -> NERResult:
        if not text or not text.strip():
            return NERResult(doc_id, [], self.model_used, source_name, category)

        raw: list[RawEntity] = []

        if self._spacy.available:
            raw.extend(self._spacy.predict(text))
            if self.combine_rules:
                raw.extend(self._rules.predict(text))
        else:
            raw.extend(self._rules.predict(text))

        brands = self._normaliser.normalise(raw, doc_id=doc_id, min_confidence=self.min_confidence)
        return NERResult(doc_id, brands, self.model_used, source_name, category)

    # ── DataFrame batch (main integration point) ──────────────────────────────

    def run_on_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run NER on every row of the DataFrame produced by news_dailyworker.py.

        Picks the best available text column:
          full_content  (after Preprocessing.runner scrapes & cleans)
          → content     (raw NewsAPI truncated content)
          → article     (used in news.py / LDA.py exploratory work)
          → title + ' ' + description  (minimum viable fallback)

        Adds columns
        ------------
        ner_brands   : list[str]  — canonical brand names (for direct DB use)
        ner_raw_json : str        — full JSON with confidence, positions etc.
        """
        text_col = self._resolve_text_column(df)
        logger.info("Running NER on column '%s' (%d rows)", text_col, len(df))

        ner_brands:   list[list[str]] = []
        ner_raw_json: list[str]       = []

        with stage(f"NER batch ({len(df)} articles)"):
            for idx, row in df.iterrows():
                text        = self._get_text(row, text_col)
                doc_id      = str(row.get("id", idx))
                source_name = str(row.get("source_name", ""))
                category    = str(row.get("category",    ""))

                result = self.run(text, doc_id=doc_id, source_name=source_name, category=category)

                ner_brands.append(result.brand_names())
                ner_raw_json.append(result.to_json())

                if (idx + 1) % 500 == 0:
                    logger.info("  processed %d / %d rows", idx + 1, len(df))

        df = df.copy()
        df["ner_brands"]   = ner_brands
        df["ner_raw_json"] = ner_raw_json
        logger.info("NER complete. Columns added: ner_brands, ner_raw_json")
        return df

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_text_column(self, df: pd.DataFrame) -> str:
        """Return the first available text column in priority order.

        Priority: constructor text_column arg → article → full_content → content
        This order reflects typical column richness in the project's CSVs:
        'article' (scraped full text in rating.csv / news.py) tends to be the
        longest, followed by 'full_content' (Preprocessing output), then the
        truncated NewsAPI 'content' field.  Mirrors column names used across
        news_dailyworker.py, news.py, and LDA.py.
        """
        for col in [self.text_column, "article", "full_content", "content"]:
            if col in df.columns:
                return col
        # Fallback: concatenate title + description (both present in NewsAPI output)
        if "title" in df.columns:
            return "title"
        return df.columns[0]

    def _get_text(self, row: pd.Series, text_col: str) -> str:
        text = str(row.get(text_col, "") or "")
        # Supplement with title + description for short / empty full_content rows
        # (matches how news.py treats short articles)
        if len(text) < 200:
            title = str(row.get("title", "") or "")
            desc  = str(row.get("description", "") or "")
            text  = " ".join(filter(None, [title, desc, text]))
        return text
