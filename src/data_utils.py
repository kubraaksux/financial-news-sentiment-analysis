"""Data loading and splitting utilities for Financial PhraseBank."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Sentences_50Agree.txt"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "data" / "split.json"

LABEL_ORDER = ["negative", "neutral", "positive"]


def load_dataset(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the dataset from ``text@label`` lines into a DataFrame."""
    path = Path(path)
    texts: list[str] = []
    labels: list[str] = []

    with path.open(encoding="latin-1") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if "@" not in line:
                raise ValueError(
                    f"Line {line_number} in {path} does not contain '@': {line!r}"
                )
            text, label = line.rsplit("@", maxsplit=1)
            texts.append(text.strip())
            labels.append(label.strip())

    df = pd.DataFrame({"text": texts, "label": labels})
    unknown = sorted(set(df["label"]) - set(LABEL_ORDER))
    if unknown:
        raise ValueError(f"Unexpected labels found: {unknown}")

    return df


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[int], list[int]]:
    """Return stratified train/test DataFrames and their source row indices."""
    indices = list(range(len(df)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df, train_idx, test_idx


def save_split(
    train_indices: Iterable[int],
    test_indices: Iterable[int],
    path: Path | str = DEFAULT_SPLIT_PATH,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
) -> None:
    """Persist train/test row indices for reproducible experiments."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "random_state": random_state,
        "test_size": test_size,
        "train_indices": list(train_indices),
        "test_indices": list(test_indices),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_split(
    df: pd.DataFrame,
    path: Path | str = DEFAULT_SPLIT_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reload a previously saved train/test split."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    train_df = df.iloc[payload["train_indices"]].reset_index(drop=True)
    test_df = df.iloc[payload["test_indices"]].reset_index(drop=True)
    return train_df, test_df


def binary_subset(
    df: pd.DataFrame,
    drop_label: str = "neutral",
) -> pd.DataFrame:
    """Remove one class for binary sentiment experiments."""
    subset = df[df["label"] != drop_label].copy()
    return subset.reset_index(drop=True)


def label_counts(df: pd.DataFrame) -> pd.Series:
    """Return ordered label counts."""
    counts = df["label"].value_counts()
    return counts.reindex(LABEL_ORDER, fill_value=0)


def ensure_split(
    data_path: Path | str = DEFAULT_DATA_PATH,
    split_path: Path | str = DEFAULT_SPLIT_PATH,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data and return train/test split, creating split file if needed."""
    df = load_dataset(data_path)
    split_path = Path(split_path)

    if split_path.exists():
        return load_split(df, split_path)

    train_df, test_df, train_idx, test_idx = stratified_split(
        df,
        random_state=random_state,
    )
    save_split(train_idx, test_idx, split_path, random_state=random_state)
    return train_df, test_df


def main() -> None:
    df = load_dataset()
    print(f"Loaded {len(df)} examples from {DEFAULT_DATA_PATH}")
    print("\nLabel distribution:")
    print(label_counts(df))

    train_df, test_df = ensure_split()
    print(f"\nTrain size: {len(train_df)}")
    print(f"Test size:  {len(test_df)}")
    print("\nTrain label distribution:")
    print(label_counts(train_df))
    print("\nTest label distribution:")
    print(label_counts(test_df))

    binary_df = binary_subset(df)
    print(f"\nBinary subset (no neutral): {len(binary_df)} examples")
    print(label_counts(binary_df))


if __name__ == "__main__":
    main()
