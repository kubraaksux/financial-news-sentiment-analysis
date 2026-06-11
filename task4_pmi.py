#!/usr/bin/env python3
"""Task 4: PMI-based word similarity (Ray).

Implement PMI from scratch (window size = 1). No external PMI/word2vec packages.
Report the 10 most similar words for 10 random vocabulary words.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from operator import itemgetter
from typing import Dict, Iterable, List, Set, Tuple

from src.data_utils import ensure_split
from src.preprocess import preprocess_corpus

RANDOM_SEED = 42
WINDOW_SIZE = 1
TOP_K = 10
NUM_RANDOM_WORDS = 10
MIN_COOCURRENCE = 2


def tokenize_corpus(texts: Iterable[str]) -> List[List[str]]:
    """Tokenize preprocessed texts (align with Task 2 preprocessing)."""
    return [text.split() for text in texts]


def collect_cooccurrence(
    tokenized_docs: List[List[str]],
    window_size: int = WINDOW_SIZE,
) -> Tuple[Counter, Dict[str, Counter], int]:
    """Count word frequencies and co-occurrences within a symmetric window."""
    word_counts = Counter()
    cooccurrence: Dict[str, Counter] = {}
    total_tokens = 0

    for sentence in tokenized_docs:
        total_tokens += len(sentence)
        for i, word in enumerate(sentence):
            word_counts[word] += 1
            for j in range(
                max(0, i - window_size),
                min(len(sentence), i + window_size + 1),
            ):
                if j == i:
                    continue
                neighbor = sentence[j]
                if word not in cooccurrence:
                    cooccurrence[word] = Counter()
                cooccurrence[word][neighbor] += 1

    return word_counts, cooccurrence, total_tokens


def compute_pmi_matrix(
    word_counts: Counter,
    cooccurrence: Dict[str, Counter],
    total_tokens: int,
    min_count: int = MIN_COOCURRENCE,
) -> Dict[str, Dict[str, float]]:
    """Compute word-word PMI scores from co-occurrence counts."""
    total_cooccurrence = sum(
        sum(context.values()) for context in cooccurrence.values()
    )
    if total_cooccurrence == 0 or total_tokens == 0:
        return {}

    pmi_matrix: Dict[str, Dict[str, float]] = {}

    for w1, neighbors in cooccurrence.items():
        p_w1 = word_counts[w1] / total_tokens
        if p_w1 == 0:
            continue

        pmi_matrix[w1] = {}
        for w2, count in neighbors.items():
            if count < min_count:
                continue

            p_w2 = word_counts[w2] / total_tokens
            if p_w2 == 0:
                continue

            p_w1_w2 = count / total_cooccurrence
            ratio = p_w1_w2 / (p_w1 * p_w2)
            if ratio <= 0:
                continue

            pmi_matrix[w1][w2] = math.log2(ratio)

    return pmi_matrix


def most_similar_words(
    pmi_matrix: Dict[str, Dict[str, float]],
    word: str,
    top_k: int = TOP_K,
) -> List[Tuple[str, float]]:
    """Return top-k most similar words for a given word."""
    if word not in pmi_matrix:
        return []

    word_score_list = list(pmi_matrix[word].items())
    word_score_list.sort(key=itemgetter(1), reverse=True)
    return word_score_list[:top_k]


def choose_random_words(
    vocabulary: Set[str],
    word_counts: Counter,
    pmi_matrix: Dict[str, Dict[str, float]],
    n: int = NUM_RANDOM_WORDS,
    min_freq: int = 10,
) -> List[str]:
    """Sample random words reproducibly for the report."""
    random.seed(RANDOM_SEED)
    candidates = sorted(
        word
        for word in vocabulary
        if word_counts[word] >= min_freq
        and len(pmi_matrix.get(word, {})) >= TOP_K
    )
    if len(candidates) < n:
        candidates = sorted(
            word for word in vocabulary if len(pmi_matrix.get(word, {})) >= TOP_K
        )
    return random.sample(candidates, min(n, len(candidates)))


def main() -> None:
    train_df, _ = ensure_split()
    processed_texts = preprocess_corpus(train_df["text"].tolist())
    tokenized_docs = tokenize_corpus(processed_texts)

    word_counts, cooccurrence, total_tokens = collect_cooccurrence(tokenized_docs)
    pmi_matrix = compute_pmi_matrix(word_counts, cooccurrence, total_tokens)
    vocabulary = set(word_counts.keys())

    selected_words = choose_random_words(vocabulary, word_counts, pmi_matrix)
    print("PMI-based most similar words (window size = 1)\n")

    for word in selected_words:
        neighbors = most_similar_words(pmi_matrix, word)
        print(f"{word}:")
        if not neighbors:
            print("  (no neighbors above min co-occurrence threshold)")
        for neighbor, score in neighbors:
            print(f"  {neighbor:20s}  PMI={score:.4f}")
        print()


if __name__ == "__main__":
    main()
