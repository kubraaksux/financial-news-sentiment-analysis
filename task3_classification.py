#!/usr/bin/env python3
"""Task 3: Naive Bayes and feed-forward NN classification (Kübra)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data_utils import LABEL_ORDER, binary_subset, ensure_split
from src.preprocess import preprocess_corpus

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
) -> dict[str, Any]:
    """Compute accuracy, precision, recall, macro F1, and confusion matrix."""
    raise NotImplementedError("Implement evaluation metrics and reporting")


def train_naive_bayes(
    x_train: list[str],
    y_train: np.ndarray,
    x_test: list[str],
    y_test: np.ndarray,
    labels: list[str],
    vectorizer_name: str,
) -> dict[str, Any]:
    """Train Multinomial NB with BoW or TF-IDF features."""
    raise NotImplementedError("Implement Naive Bayes classifier")


def train_feedforward_nn(
    x_train: list[str],
    y_train: np.ndarray,
    x_test: list[str],
    y_test: np.ndarray,
    labels: list[str],
    vectorizer_name: str,
) -> dict[str, Any]:
    """Train a feed-forward neural network classifier."""
    raise NotImplementedError("Implement feed-forward neural network classifier")


def error_analysis(
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    n_examples: int = 10,
) -> None:
    """Inspect misclassified examples for the report."""
    raise NotImplementedError("Implement error analysis")


def run_setting(setting_name: str, train_df, test_df, labels: list[str]) -> None:
    """Run NB and NN experiments for one label setting (multiclass or binary)."""
    print(f"\n{'#' * 80}\nSetting: {setting_name}\n{'#' * 80}")

    x_train = preprocess_corpus(train_df["text"].tolist())
    x_test = preprocess_corpus(test_df["text"].tolist())
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    for vectorizer_name in ("bow", "tfidf"):
        train_naive_bayes(x_train, y_train, x_test, y_test, labels, vectorizer_name)
        train_feedforward_nn(x_train, y_train, x_test, y_test, labels, vectorizer_name)


def main() -> None:
    train_df, test_df = ensure_split()

    run_setting("multiclass", train_df, test_df, LABEL_ORDER)

    binary_train = binary_subset(train_df)
    binary_test = binary_subset(test_df)
    binary_labels = [label for label in LABEL_ORDER if label != "neutral"]
    run_setting("binary (no neutral)", binary_train, binary_test, binary_labels)


if __name__ == "__main__":
    main()
