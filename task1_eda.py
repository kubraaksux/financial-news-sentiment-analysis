#!/usr/bin/env python3
"""Task 1: Exploratory Data Analysis (Ray)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_utils import DEFAULT_DATA_PATH, label_counts, load_dataset

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def plot_label_distribution(df: pd.DataFrame) -> None:
    """Plot class distribution and save to figures/."""
    raise NotImplementedError("Implement label distribution plot")


def analyze_text_lengths(df: pd.DataFrame) -> None:
    """Analyze and visualize text length statistics."""
    raise NotImplementedError("Implement text length analysis")


def explore_vocabulary(df: pd.DataFrame) -> None:
    """Inspect frequent words/n-grams and optional word cloud."""
    raise NotImplementedError("Implement vocabulary and n-gram analysis")


def main() -> None:
    df = load_dataset(DEFAULT_DATA_PATH)
    print(f"Loaded {len(df)} examples")
    print(label_counts(df))

    plot_label_distribution(df)
    analyze_text_lengths(df)
    explore_vocabulary(df)


if __name__ == "__main__":
    main()
