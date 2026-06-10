#!/usr/bin/env python3
"""Task 5: Fine-tune pre-trained transformer models (Kübra)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.data_utils import LABEL_ORDER, binary_subset, ensure_split

Setting = Literal["multiclass", "binary"]


@dataclass
class TrainingConfig:
    model_name: str = "distilbert-base-uncased"
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    weight_decay: float = 0.01
    max_length: int = 128
    output_dir: Path = Path("models") / "distilbert-financial-sentiment"


def build_label_maps(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def prepare_datasets(setting: Setting):
    """Convert project splits into Hugging Face Dataset objects."""
    raise NotImplementedError("Implement Hugging Face dataset preparation")


def fine_tune_model(config: TrainingConfig, setting: Setting) -> None:
    """Fine-tune a transformer and save metrics for the report."""
    raise NotImplementedError("Implement transformer fine-tuning")


def evaluate_saved_model(config: TrainingConfig, setting: Setting) -> None:
    """Load checkpoint and report accuracy, macro F1, and confusion matrix."""
    raise NotImplementedError("Implement transformer evaluation")


def main() -> None:
    config = TrainingConfig()

    for setting in ("multiclass", "binary"):
        prepare_datasets(setting)
        fine_tune_model(config, setting)
        evaluate_saved_model(config, setting)


if __name__ == "__main__":
    main()
