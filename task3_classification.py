#!/usr/bin/env python3
"""Task 3: Naive Bayes and feed-forward NN classification (Kübra)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from src.data_utils import LABEL_ORDER, binary_subset, ensure_split
from src.preprocess import preprocess_corpus

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "model"


def _build_vectorizer(vectorizer_name: str) -> CountVectorizer | TfidfVectorizer:
    """Vectorize pre-tokenized space-separated strings from Task 2."""
    if vectorizer_name == "bow":
        return CountVectorizer(analyzer=lambda doc: doc.split())
    if vectorizer_name == "tfidf":
        return TfidfVectorizer(analyzer=lambda doc: doc.split())
    raise ValueError(f"Unknown vectorizer: {vectorizer_name!r}")


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
) -> dict[str, Any]:
    """Compute accuracy, precision, recall, macro F1, and confusion matrix."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\n{title}")
    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  Precision:  {precision:.4f} (macro)")
    print(f"  Recall:     {recall:.4f} (macro)")
    print(f"  F1:         {f1:.4f} (macro)")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    figure_path = FIGURES_DIR / f"{_slugify(title)}_confusion_matrix.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to {figure_path}")

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }


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


def error_analysis(
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    n_examples: int = 10,
) -> None:
    """Inspect misclassified examples for the report."""
    misclassified = [
        (text, true_label, pred_label)
        for text, true_label, pred_label in zip(texts, y_true, y_pred)
        if true_label != pred_label
    ]

    print(f"\nError analysis ({len(misclassified)} misclassified examples):")
    if not misclassified:
        print("  No misclassifications.")
        return

    for text, true_label, pred_label in misclassified[:n_examples]:
        print(f"  TRUE={true_label:8s}  PRED={pred_label:8s}  {text[:120]}{'...' if len(text) > 120 else ''}")


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
