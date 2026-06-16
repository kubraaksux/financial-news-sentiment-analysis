#!/usr/bin/env python3
"""Task 5: Fine-tune pre-trained transformer models (Kübra)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data_utils import LABEL_ORDER, binary_subset, ensure_split
from src.evaluation import error_analysis, evaluate_predictions

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


def labels_for_setting(setting: Setting) -> list[str]:
    if setting == "binary":
        return [label for label in LABEL_ORDER if label != "neutral"]
    return list(LABEL_ORDER)


def prepare_datasets(
    setting: Setting,
) -> tuple[Dataset, Dataset, list[str], list[str], dict[str, int], dict[int, str]]:
    """Load splits as Hugging Face datasets. Uses raw text (not Task 2 preprocessing)."""
    train_df, test_df = ensure_split()
    labels = labels_for_setting(setting)

    if setting == "binary":
        train_df = binary_subset(train_df)
        test_df = binary_subset(test_df)

    label2id, id2label = build_label_maps(labels)
    test_texts = test_df["text"].tolist()

    train_ds = Dataset.from_dict(
        {
            "text": train_df["text"].tolist(),
            "label": [label2id[label] for label in train_df["label"]],
        }
    )
    test_ds = Dataset.from_dict(
        {
            "text": test_texts,
            "label": [label2id[label] for label in test_df["label"]],
        }
    )
    return train_ds, test_ds, labels, test_texts, label2id, id2label


def _tokenize(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    return dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ),
        batched=True,
    )


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
    }


def _format_for_trainer(dataset: Dataset) -> Dataset:
    dataset = dataset.rename_column("label", "labels")
    return dataset.with_format("torch", columns=["input_ids", "attention_mask", "labels"])


def fine_tune_model(config: TrainingConfig, setting: Setting) -> Path:
    """Fine-tune DistilBERT and save the checkpoint."""
    train_ds, test_ds, _, _, label2id, id2label = prepare_datasets(setting)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_ds = _format_for_trainer(_tokenize(train_ds, tokenizer, config.max_length))
    test_ds = _format_for_trainer(_tokenize(test_ds, tokenizer, config.max_length))

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    output_dir = config.output_dir / setting
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
    )

    print(f"\nFine-tuning {config.model_name} ({setting})")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"Training time: {elapsed / 60:.1f} minutes")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def evaluate_saved_model(config: TrainingConfig, setting: Setting) -> dict[str, float]:
    """Load checkpoint and evaluate on the test split."""
    _, test_ds, labels, test_texts, _, id2label = prepare_datasets(setting)
    output_dir = config.output_dir / setting

    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)

    test_ds = _format_for_trainer(_tokenize(test_ds, tokenizer, config.max_length))

    trainer = Trainer(model=model, processing_class=tokenizer)
    predictions = trainer.predict(test_ds)
    pred_ids = np.argmax(predictions.predictions, axis=-1)
    true_ids = predictions.label_ids

    y_true = np.array([id2label[int(i)] for i in true_ids])
    y_pred = np.array([id2label[int(i)] for i in pred_ids])

    title = f"DistilBERT ({setting})"
    results = evaluate_predictions(y_true, y_pred, labels, title)
    error_analysis(test_texts, y_true, y_pred, labels)
    return results


def main() -> None:
    config = TrainingConfig()

    for setting in ("multiclass", "binary"):
        fine_tune_model(config, setting)
        evaluate_saved_model(config, setting)


if __name__ == "__main__":
    main()
