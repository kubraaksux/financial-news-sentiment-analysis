# NLP Project 1.2 — Sentiment Analysis of Financial News

TU Berlin · Natural Language Processing · SoSe 2026

**Deadline:** 16 June 2026, 23:55 (GMT+2)

Sentiment classification on financial news headlines from Financial PhraseBank
(`data/Sentences_50Agree.txt`, format `sentence text@label`).

## Task split

| Task | Owner | Script |
|------|-------|--------|
| 1 — EDA | Ray | `task1_eda.py` |
| 2 — Preprocessing | Kübra | `task2_preprocess.py`, `src/preprocess.py` |
| 3 — NB + NN | Kübra | `task3_classification.py` |
| 4 — PMI | Ray | `task4_pmi.py` |
| 5 — Transformers | Kübra | `task5_transformers.py` |

Shared data loading/splitting: `src/data_utils.py`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

NLTK resources (if needed for preprocessing):

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

Train/test split: stratified 80/20, seed 42, saved to `data/split.json`.

```bash
python -m src.data_utils
python task1_eda.py
python task2_preprocess.py
python task3_classification.py
python task4_pmi.py
python task5_transformers.py
```

Figures → `figures/`, model checkpoints → `models/`.

## Layout

```
nlp_project_1_2/
├── data/
├── src/
│   ├── data_utils.py
│   └── preprocess.py
├── task1_eda.py … task5_transformers.py
├── figures/
├── report/main.tex
├── ExperimentSummarySheet_Ex1.xlsx
└── requirements.txt
```

## Submission

- Source code
- Report PDF (ACM template in `report/`, max 4 pages)
- Completed `ExperimentSummarySheet_Ex1.xlsx`
- Zip: `NLP_project_1_2_[GroupName].zip`
