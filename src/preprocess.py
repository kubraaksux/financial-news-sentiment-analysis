"""Text preprocessing pipeline shared across project tasks."""

from __future__ import annotations

import re
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def _ensure_nltk() -> None:
    for pkg in ("punkt", "stopwords"):
        try:
            if pkg == "punkt":
                nltk.data.find("tokenizers/punkt")
            else:
                nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    _ensure_nltk()
    return word_tokenize(text)


def is_content_token(token: str) -> bool:
    return any(ch.isalnum() for ch in token)


def remove_stopwords(tokens: list[str]) -> list[str]:
    _ensure_nltk()
    stops = set(stopwords.words("english"))
    return [t for t in tokens if t not in stops]


def preprocess(text: str, *, remove_stops: bool = True) -> str:
    if not isinstance(text, str):
        raise TypeError("preprocess expects a string")

    cleaned = normalize_whitespace(text.lower())
    tokens = [t for t in tokenize(cleaned) if is_content_token(t)]
    if remove_stops:
        tokens = remove_stopwords(tokens)
    return " ".join(tokens)


def preprocess_corpus(
    texts: Iterable[str],
    *,
    remove_stops: bool = True,
) -> list[str]:
    return [preprocess(text, remove_stops=remove_stops) for text in texts]
