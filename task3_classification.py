#!/usr/bin/env python3
"""Task 3: Naive Bayes and feed-forward NN classification (Kübra)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from src.data_utils import LABEL_ORDER, binary_subset, ensure_split
from src.evaluation import error_analysis, evaluate_predictions
from src.preprocess import preprocess_corpus


def _build_vectorizer(vectorizer_name: str) -> CountVectorizer | TfidfVectorizer:
    """Vectorize pre-tokenized space-separated strings from Task 2."""
    if vectorizer_name == "bow":
        return CountVectorizer(analyzer=lambda doc: doc.split())
    if vectorizer_name == "tfidf":
        return TfidfVectorizer(analyzer=lambda doc: doc.split())
    raise ValueError(f"Unknown vectorizer: {vectorizer_name!r}")


def train_naive_bayes(
    x_train: list[str],
    y_train: np.ndarray,
    x_test: list[str],
    y_test: np.ndarray,
    labels: list[str],
    vectorizer_name: str,
    setting_name: str = "",
) -> dict[str, Any]:
    """Train Multinomial NB with BoW or TF-IDF features."""
    vectorizer = _build_vectorizer(vectorizer_name)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    classifier = MultinomialNB()
    classifier.fit(x_train_vec, y_train)
    y_pred = classifier.predict(x_test_vec)

    setting_suffix = f" — {setting_name}" if setting_name else ""
    title = f"Naive Bayes ({vectorizer_name.upper()}){setting_suffix}"
    results = evaluate_predictions(y_test, y_pred, labels, title)
    error_analysis(x_test, y_test, y_pred, labels)
    return results


def train_feedforward_nn(
    x_train: list[str],
    y_train: np.ndarray,
    x_test: list[str],
    y_test: np.ndarray,
    labels: list[str],
    vectorizer_name: str,
    setting_name: str = "",
) -> dict[str, Any]:
    """Train a feed-forward neural network classifier."""
    vectorizer = _build_vectorizer(vectorizer_name)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    classifier = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=300,
        random_state=42,
        early_stopping=True,
    )
    classifier.fit(x_train_vec, y_train)
    y_pred = classifier.predict(x_test_vec)

    setting_suffix = f" — {setting_name}" if setting_name else ""
    title = f"Feed-forward NN ({vectorizer_name.upper()}){setting_suffix}"
    results = evaluate_predictions(y_test, y_pred, labels, title)
    error_analysis(x_test, y_test, y_pred, labels)
    return results


def run_setting(setting_name: str, train_df, test_df, labels: list[str]) -> None:
    """Run NB and NN experiments for one label setting (multiclass or binary)."""
    print(f"\n{'#' * 80}\nSetting: {setting_name}\n{'#' * 80}")

    x_train = preprocess_corpus(train_df["text"].tolist())
    x_test = preprocess_corpus(test_df["text"].tolist())
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    for vectorizer_name in ("bow", "tfidf"):
        train_naive_bayes(
            x_train, y_train, x_test, y_test, labels, vectorizer_name, setting_name
        )
        train_feedforward_nn(
            x_train, y_train, x_test, y_test, labels, vectorizer_name, setting_name
        )


def main() -> None:
    train_df, test_df = ensure_split()

    run_setting("multiclass", train_df, test_df, LABEL_ORDER)

    binary_train = binary_subset(train_df)
    binary_test = binary_subset(test_df)
    binary_labels = [label for label in LABEL_ORDER if label != "neutral"]
    run_setting("binary (no neutral)", binary_train, binary_test, binary_labels)


if __name__ == "__main__":
    main()
