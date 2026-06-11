#!/usr/bin/env python3
"""Task 1: Exploratory Data Analysis (Ray)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_utils import DEFAULT_DATA_PATH, label_counts, load_dataset

# imported packages
import re
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import STOPWORDS, WordCloud
from sklearn.feature_extraction.text import CountVectorizer

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def plot_label_distribution(df: pd.DataFrame) -> None:
    """Plot class distribution and save to figures/."""
    #raise NotImplementedError("Implement label distribution plot")

    # data frame format
    print(df.head(3))
    #                                               text     label
    # 0  According to Gran , the company has no plans t...   neutral
    # 1  Technopolis plans to develop in stages an area...   neutral
    # 2  The international electronic industry company ...  negative
    print(df.columns)
    print(df['text'][0])

    # class_distribution
    counts = df['label'].value_counts()
    plt.figure(figsize=(6, 4))
    counts.plot(kind='bar', color=['steelblue', 'salmon', 'mediumseagreen'])
    plt.title('Sentiment Class Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    for i, v in enumerate(counts):
        plt.text(i, v + 10, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'label_distribution.png', bbox_inches='tight')
    plt.show()

    

def analyze_text_lengths(df: pd.DataFrame) -> None:
    """Analyze and visualize text length statistics."""
    #raise NotImplementedError("Implement text length analysis")
    df = df.copy()
    df['length'] = df['text'].apply(lambda x: len(x.split()))

    # descriptive stats
    print(df.groupby('label')['length'].describe())

    # histogram per class
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, group in df.groupby('label'):
        ax.hist(group['length'], bins='auto', alpha=0.6, label=label)
    ax.set_title('Text Length Distribution by Sentiment')
    ax.set_xlabel('Number of tokens')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'text_lengths.png', bbox_inches='tight')
    plt.show()


def explore_vocabulary(df: pd.DataFrame) -> None:
    """Inspect frequent words/n-grams and optional word cloud."""
    #raise NotImplementedError("Implement vocabulary and n-gram analysis")

    # N_gram
    # top unigram per class
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, label in zip(axes, ['positive', 'negative', 'neutral']):
        subset = df[df['label'] == label]['text'].tolist()
        vec = CountVectorizer(stop_words='english', max_features=15)
        X = vec.fit_transform(subset)
        freqs = np.asarray(X.sum(axis=0)).flatten()
        words = vec.get_feature_names_out()
        top = sorted(zip(words, freqs), key=lambda x: -x[1])[:15]
        words_, counts_ = zip(*top)
        ax.barh(words_, counts_, color='steelblue')
        ax.invert_yaxis()
        ax.set_title(f'Top words — {label}')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'top_unigrams.png', bbox_inches='tight')
    plt.show()


    # top bigram per class 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, label in zip(axes, ['positive', 'negative', 'neutral']):
        subset = df[df['label'] == label]['text'].tolist()
        vec = CountVectorizer(ngram_range=(2, 2), stop_words='english', max_features=15)
        X = vec.fit_transform(subset)
        freqs = np.asarray(X.sum(axis=0)).flatten()
        words = vec.get_feature_names_out()
        top = sorted(zip(words, freqs), key=lambda x: -x[1])[:15]
        words_, counts_ = zip(*top)
        ax.barh(words_, counts_, color='salmon')
        ax.invert_yaxis()
        ax.set_title(f'Top Bigrams — {label}')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'top_bigrams.png', bbox_inches='tight')
    plt.show()

    # cleaning the text data before using it 
    # to generate word cloud
    # remove punction first
    text = ' '.join(df['text'].astype(str).tolist())

    # converting to lower case
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()

    # remove common words such as is, the, etc.
    stopwords = set(STOPWORDS)
    text = ' '.join(word for word in text.split() if word not in stopwords)

    # generating wordcloud
    wordcloud = WordCloud(width=800,
                          height=400,
                          background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Sentence_50Agree.txt Word Cloud')
    plt.savefig(FIGURES_DIR / 'word_cloud.png', bbox_inches='tight')
    plt.show()

def main() -> None:
    df = load_dataset(DEFAULT_DATA_PATH)
    print(f"Loaded {len(df)} examples")
    print(label_counts(df))

    plot_label_distribution(df)
    analyze_text_lengths(df)
    explore_vocabulary(df)


if __name__ == "__main__":
    main()
