"""Shared evaluation helpers for classification tasks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "model"


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
        print(
            f"  TRUE={true_label:8s}  PRED={pred_label:8s}  "
            f"{text[:120]}{'...' if len(text) > 120 else ''}"
        )
