import re


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_NUM_RE = re.compile(r"\b\d+\b")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


EXTRA_STOPWORDS = {
    "said", "say", "says", "would", "could", "also", "one",
    "two", "three", "new", "like", "get", "make", "know",
    "use", "just", "year", "time", "way", "day", "man",
    "woman", "people", "thing",
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(" ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = _NUM_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text
