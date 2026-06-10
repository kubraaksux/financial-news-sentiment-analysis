"""Text preprocessing pipeline shared across project tasks."""

from __future__ import annotations

import re
from typing import Iterable


def preprocess(text: str) -> str:
    """Apply preprocessing steps to a single text instance."""
    if not isinstance(text, str):
        raise TypeError("preprocess expects a string")

    cleaned = text.lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def preprocess_corpus(texts: Iterable[str]) -> list[str]:
    """Preprocess an iterable of texts."""
    return [preprocess(text) for text in texts]
