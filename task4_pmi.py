#!/usr/bin/env python3
"""Task 4: PMI-based word similarity (Ray).

Implement PMI from scratch (window size = 1). No external PMI/word2vec packages.
Report the 10 most similar words for 10 random vocabulary words.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Set, Tuple

from src.data_utils import ensure_split
from src.preprocess import preprocess_corpus

RANDOM_SEED = 42
WINDOW_SIZE = 1
TOP_K = 10
NUM_RANDOM_WORDS = 10


def tokenize_corpus(texts: Iterable[str]) -> List[List[str]]:
    """Tokenize preprocessed texts (align with Task 2 preprocessing)."""
    raise NotImplementedError("Implement corpus tokenization")


def collect_cooccurrence(
    tokenized_docs: List[List[str]],
    window_size: int = WINDOW_SIZE,
) -> Tuple[Counter, Dict[str, Counter], int]:
    """Count word frequencies and co-occurrences within a symmetric window."""
    raise NotImplementedError("Implement co-occurrence counting")


def compute_pmi_matrix(
    word_counts: Counter,
    cooccurrence: Dict[str, Counter],
    total_tokens: int,
) -> Dict[str, Dict[str, float]]:
    """Compute word-word PMI scores from co-occurrence counts."""
    raise NotImplementedError("Implement PMI matrix computation")


def most_similar_words(
    pmi_matrix: Dict[str, Dict[str, float]],
    word: str,
    top_k: int = TOP_K,
) -> List[Tuple[str, float]]:
    """Return top-k most similar words for a given word."""
    raise NotImplementedError("Implement similarity lookup")


def choose_random_words(vocabulary: Set[str], n: int = NUM_RANDOM_WORDS) -> List[str]:
    """Sample random words reproducibly for the report."""
    raise NotImplementedError("Implement random word sampling")


def main() -> None:
    train_df, _ = ensure_split()
    processed_texts = preprocess_corpus(train_df["text"].tolist())
    tokenized_docs = tokenize_corpus(processed_texts)

    word_counts, cooccurrence, total_tokens = collect_cooccurrence(tokenized_docs)
    pmi_matrix = compute_pmi_matrix(word_counts, cooccurrence, total_tokens)
    vocabulary = set(word_counts.keys())

    selected_words = choose_random_words(vocabulary)
    print("PMI-based most similar words (window size = 1)\n")

    for word in selected_words:
        neighbors = most_similar_words(pmi_matrix, word)
        print(f"{word}:")
        for neighbor, score in neighbors:
            print(f"  {neighbor:20s}  PMI={score:.4f}")
        print()


if __name__ == "__main__":
    main()
