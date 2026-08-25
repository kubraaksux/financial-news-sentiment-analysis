# Financial News Sentiment Analysis

A reproducible comparison of classical machine-learning and transformer approaches for classifying financial-news headlines as **negative**, **neutral**, or **positive**.

## Problem

Financial sentiment is difficult because many headlines express consequences indirectly: legal outcomes, margin changes, percentages, and apparently factual statements can carry sentiment without explicit polarity words. This project studies how feature representations and model families affect performance on those cases.

## Methods compared

- Multinomial Naive Bayes with bag-of-words and TF-IDF
- Feed-forward neural network (MLP) with bag-of-words and TF-IDF
- Fine-tuned `distilbert-base-uncased`
- PMI-based word-similarity analysis
- Multiclass and binary evaluation settings

## Dataset and methodology

The experiments use the **Financial PhraseBank** `Sentences_50Agree` subset. Data is split once using a stratified 80/20 train/test split with seed 42 and stored in `data/split.json` for reproducibility.

The classical pipeline lowercases and tokenizes headlines, removes punctuation-only tokens and English stopwords, and retains numeric tokens because financial quantities are informative. DistilBERT receives raw headlines and uses its native subword tokenizer.

Reported metrics are test accuracy and macro F1. Transformer checkpoints are selected using validation macro F1.

## Results

| Model | Features | Setting | Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| Naive Bayes | BoW | Multiclass | 0.71 | 0.63 |
| Naive Bayes | TF-IDF | Multiclass | 0.68 | 0.42 |
| MLP | BoW | Multiclass | 0.73 | 0.67 |
| MLP | TF-IDF | Multiclass | 0.73 | 0.65 |
| Naive Bayes | BoW | Binary | 0.83 | 0.80 |
| MLP | BoW | Binary | 0.82 | 0.78 |
| DistilBERT | Raw text | Multiclass | **0.84** | **0.82** |
| DistilBERT | Raw text | Binary | **0.96** | **0.95** |

DistilBERT produced the strongest results and raised multiclass negative recall to 0.90, compared with approximately 0.53–0.59 for the MLP variants.

## Error analysis

Recurring errors include:

- Subtle positive statements being classified as neutral or negative
- Legal or financial outcomes being misread when their sentiment is implicit
- Headlines with mixed numeric signals confusing neutral and negative labels
- TF-IDF Naive Bayes collapsing toward the majority neutral class, with very low negative recall

These failures show why macro F1 and class-level recall are more informative than accuracy alone for this imbalanced task.

## Reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.data_utils
python task1_eda.py
python task2_preprocess.py
python task3_classification.py
python task4_pmi.py
python task5_transformers.py
```

Generated figures are written to `figures/`; model checkpoints are written to `models/`. The ACM-style technical report is in `report/main.tex`.

## Individual contribution

This was a collaborative TU Berlin NLP project. My work focused on:

- Text preprocessing and reproducible data handling
- Classical sentiment classification and evaluation
- DistilBERT fine-tuning
- Shared evaluation utilities and experiment logging
- Quantitative comparison and error analysis
- Correcting and stabilizing the PMI scoring implementation

The commit history preserves the collaborative development record.
