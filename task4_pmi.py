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

# imported packages 
import math
import random

from operator import itemgetter

RANDOM_SEED = 42
WINDOW_SIZE = 1
TOP_K = 10
NUM_RANDOM_WORDS = 10


def tokenize_corpus(texts: Iterable[str]) -> List[List[str]]:
    """Tokenize preprocessed texts (align with Task 2 preprocessing)."""
    #raise NotImplementedError("Implement corpus tokenization")
    tokens = [text.split() for text in texts]
    print(tokens)
    return tokens
    #return [[w for w in text.split() if w.isalpha()] for text in texts]


def collect_cooccurrence(
    tokenized_docs: List[List[str]],
    window_size: int = WINDOW_SIZE,
) -> Tuple[Counter, Dict[str, Counter], int]:
    """Count word frequencies and co-occurrences within a symmetric window."""
    #raise NotImplementedError("Implement co-occurrence counting")
    
    #initialite Counter object for each token
    word_counts = Counter()
    cooccurrence: Dict[str, Counter] = {}
    total_tokens = 0

    for sentence in tokenized_docs:
        total_tokens += len(sentence)
        for i, word in enumerate(sentence):
            word_counts[word] += 1
            # since it is symmetric window. 
            # look left and right by window_size, where window_size = 1
            for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                if j == i:
                    continue
                neighbor = sentence[j]
                if word not in cooccurrence:
                    cooccurrence[word] = Counter()
                cooccurrence[word][neighbor] += 1
        
    #print(cooccurrence)
    return word_counts, cooccurrence, total_tokens


def compute_pmi_matrix(
    word_counts: Counter,
    cooccurrence: Dict[str, Counter],
    total_tokens: int,
) -> Dict[str, Dict[str, float]]:
    """Compute word-word PMI scores from co-occurrence counts."""
    #raise NotImplementedError("Implement PMI matrix computation")

    total_cooccurence = sum(sum(context.values()) for context in cooccurrence.values())
    pmi_matrix: Dict[str, Dict[str, float]] = {}

    for w1, neighbors in cooccurrence.items():

        pmi_matrix[w1] = {}
        for w2, count in neighbors.items():
            # P(W1, W2)
            # count = # of w1 appeared in the context of w2
            p_w1_w2 = count / total_cooccurence
            # P(W1) and P(W2) = how often w_i appears accross all context
            p_w1 = word_counts[w1] / total_cooccurence
            p_w2 = word_counts[w2] / total_cooccurence
            # PMI = log2(P(W1,W2) / (P(W1) * P(W2)))
            #if count < 2:  # for pairs that co-occur only once
            #    continue   # skip its pmi scores, to avoid 
                            # co occuring pairs
            pmi_score = math.log2(p_w1_w2 / (p_w1 * p_w2))
            pmi_matrix[w1][w2] = max(0.0, pmi_score)

            # pmi_matrix[w1][w2] = math.log2(p_w1_w2 / (p_w1 * p_w2))
    return pmi_matrix


def most_similar_words(
    pmi_matrix: Dict[str, Dict[str, float]],
    word: str,
    top_k: int = TOP_K,
) -> List[Tuple[str, float]]:
    """Return top-k most similar words for a given word."""
    #raise NotImplementedError("Implement similarity lookup")
    
    if word not in pmi_matrix:
        return []
    else:
        neighbors = pmi_matrix[word]

    # converts to list of (word, score) tuples
    word_score_list = list(neighbors.items())

    # sort in a descending order based on each word score, to obtain the 
    # top k word and score pairs
    word_score_list.sort(key=itemgetter(1), reverse=True)

    return word_score_list[:top_k]

def choose_random_words(vocabulary: Set[str], n: int = NUM_RANDOM_WORDS) -> List[str]:
    """Sample random words reproducibly for the report."""
    #raise NotImplementedError("Implement random word sampling")
    random.seed(RANDOM_SEED)
    return random.sample(sorted(vocabulary), n)

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
