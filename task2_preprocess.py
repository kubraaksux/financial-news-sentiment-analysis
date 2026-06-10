#!/usr/bin/env python3
"""Task 2: Text Preprocessing (Kübra)."""

from __future__ import annotations

from src.data_utils import ensure_split
from src.preprocess import normalize_whitespace, preprocess, preprocess_corpus


def inspect_examples(train_texts: list[str], n: int = 5) -> None:
    """Print raw vs preprocessed examples for manual inspection."""
    print(f"\nFirst {n} preprocessing examples:\n")
    for text in train_texts[:n]:
        print("RAW: ", text)
        print("PROC:", preprocess(text))
        print("-" * 80)


def _variant_stats(name: str, processed: list[str]) -> None:
    token_lists = [doc.split() for doc in processed if doc]
    vocab = {token for doc in token_lists for token in doc}
    avg_len = sum(len(doc) for doc in token_lists) / max(len(token_lists), 1)
    print(f"{name:32s}  avg tokens/doc: {avg_len:5.1f}  vocab size: {len(vocab)}")


def compare_preprocessing_variants(train_texts: list[str]) -> None:
    """Compare preprocessing variants and note observations for the report."""
    baseline = [normalize_whitespace(text.lower()) for text in train_texts]
    tokenized_keep_stops = preprocess_corpus(train_texts, remove_stops=False)
    tokenized_remove_stops = preprocess_corpus(train_texts, remove_stops=True)

    print("\nPreprocessing variant comparison (training set):\n")
    _variant_stats("baseline (lower + whitespace)", baseline)
    _variant_stats("tokenized (keep stopwords)", tokenized_keep_stops)
    _variant_stats("tokenized + stopword removal", tokenized_remove_stops)

    example = (
        "The major breweries increased their domestic beer sales "
        "by 4.5 per cent last year."
    )
    print("\nBefore/after example (for report):\n")
    print("RAW: ", example)
    print("PROC:", preprocess(example))


def main() -> None:
    train_df, _ = ensure_split()
    processed = preprocess_corpus(train_df["text"].tolist())

    print(f"Processed {len(processed)} training examples")
    inspect_examples(train_df["text"].tolist())
    compare_preprocessing_variants(train_df["text"].tolist())


if __name__ == "__main__":
    main()
