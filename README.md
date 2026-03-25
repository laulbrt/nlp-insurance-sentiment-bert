# Insurance Reviews – Sentiment Analysis & NLP

**Author**: Laura LABARTHE
**Course**: Information Retrieval and NLP – 4th year Data/AI Engineering

## Overview

End-to-end NLP pipeline on **insurance product reviews** (34 files, 3,000+ reviews):
- Sentiment classification (positive / negative / neutral)
- Star rating prediction (1–5)
- Topic modeling and zero-shot category detection
- Named entity analysis by insurer and product type

## Pipeline

```
Raw Excel reviews → Preprocessing → EDA → Modeling → Streamlit app
```

## Models Compared

| Model | Task | Notes |
|-------|------|-------|
| TF-IDF + SVC | Sentiment classification | Baseline, fast |
| Word2Vec + ML | Semantic classification | PCA visualization |
| BERT (CamemBERT) | Star rating prediction | Fine-tuned transformer |
| Zero-shot (BART) | Category detection | No labeled data needed |

## Key Results

- BERT achieves best accuracy on star rating prediction
- TF-IDF + SVC provides strong baseline for sentiment
- Topic modeling (LDA) reveals 5 main review themes
- Zero-shot classification identifies 8 insurance categories without training data

## App

Interactive Streamlit app for:
- `app_analysis.py`: review exploration and visualization
- `app_prediction.py`: real-time sentiment and rating prediction

## Stack

`Python` · `BERT/CamemBERT` · `scikit-learn` · `Gensim` · `spaCy` · `Streamlit` · `Pandas` · `Matplotlib`
