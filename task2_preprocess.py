#!/usr/bin/env python3
"""Task 2: Text Preprocessing (Kübra)."""

from __future__ import annotations

from src.data_utils import ensure_split
from src.preprocess import preprocess, preprocess_corpus


def inspect_examples(train_texts: list[str], n: int = 5) -> None:
    """Print raw vs preprocessed examples for manual inspection."""
    print(f"\nFirst {n} preprocessing examples:\n")
    for text in train_texts[:n]:
        print("RAW: ", text)
        print("PROC:", preprocess(text))
        print("-" * 80)


def compare_preprocessing_variants(train_texts: list[str]) -> None:
    """Compare preprocessing variants and note observations for the report."""
    raise NotImplementedError("Implement preprocessing comparison experiments")


def main() -> None:
    train_df, _ = ensure_split()
    processed = preprocess_corpus(train_df["text"].tolist())

    print(f"Processed {len(processed)} training examples")
    inspect_examples(train_df["text"].tolist())
    compare_preprocessing_variants(train_df["text"].tolist())


if __name__ == "__main__":
    main()
