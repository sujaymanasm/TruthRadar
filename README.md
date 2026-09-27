# 🛡️ TruthRadar — AI-Powered Fake News Detection and Explainability System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![NLP](https://img.shields.io/badge/NLP-Transformers%20%7C%20spaCy%20%7C%20NLTK-green.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**TruthRadar** is an end-to-end, multi-modal Natural Language Processing system designed to evaluate news articles, headlines, or live URLs and predict whether content is **REAL** or **FAKE**. Unlike single-classifier systems, TruthRadar combines statistical machine learning, deep contextual Transformers, dense semantic similarity, sentiment analysis, entity extraction, and Explainable AI (LIME) into an interactive decision dashboard.

---

## 📌 Key Features

- 📑 **Flexible Inputs**: Analyze pasted articles, separate headline/body pairs, or scrape live web URLs.
- 🔬 **Multi-Signal NLP Ensemble**:
  - **TF-IDF + Logistic Regression**: Statistical n-gram frequency baseline.
  - **DistilBERT**: Deep bidirectional self-attention transformer for semantic nuance.
  - **Sentence Transformers (`all-MiniLM-L6-v2`)**: Dense cosine similarity for Headline-Body consistency (clickbait detection).
  - **NLTK VADER**: Sentiment polarity and sensationalism scoring.
  - **spaCy (`en_core_web_sm`)**: Named Entity Recognition (Persons, Organizations, Geopolitical Entities, Dates, Money).
- 🔍 **Explainable AI (LIME)**: Transparent local word-level attribution charts explaining *why* a classification was made.
- 🌐 **Robust Web Scraper**: Built-in polite HTML article parser with timeout and error handling.
- ⚡ **Production-Ready Streamlit UI**: Caching with `@st.cache_resource`, reactive Plotly charts, confidence tiers, and clean layout.

---

## 🏛️ System Architecture

```
                                  [ User Input ]
                         (Article Text / Headline / URL)
                                        │
                                        ▼
                             [ Ingestion & Scraper ]
                                        │
                                        ▼
                            [ Dual-Path Preprocessing ]
                            ┌───────────┴───────────┐
                            ▼                       ▼
                   [ Statistical Text ]   [ Contextual Natural Text ]
                            │                       │
                            ▼                       ▼
                   [ TF-IDF Vectorizer ]   [ DistilBERT Tokenizer ]
                   (1-2 N-Grams, 10k Feat) (Subword WordPiece Tokens)
                            │                       │
                            ▼                       ▼
                 [ Logistic Regression ]     [ DistilBERT Model ]
                            │                       │
                            └───────────┬───────────┘
                                        │
                                        ▼
                           [ Multi-Signal Ensemble ]
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
     [ Headline-Body ]          [ VADER Sentiment ]          [ spaCy NER ]
     (MiniLM Cosine Sim)       (Emotional Polarity)      (Extracted Entities)
             │                          │                          │
             └──────────────────────────┼──────────────────────────┘
                                        │
                                        ▼
                             [ LIME Explainability ]
                           (Plotly Feature Bar Chart)
                                        │
                                        ▼
                            [ Streamlit Dashboard ]
```

---

## 📁 Project Directory Structure

```
TruthRadar/
├── app.py                      # Main Streamlit web application
├── config.py                   # Central paths, weights, and hyperparameters
├── requirements.txt            # Pinned dependency requirements
├── README.md                   # Comprehensive documentation
├── .gitignore                  # Git ignore rules for models, logs, caches
│
├── data/
│   ├── raw/                    # Directory for raw downloaded datasets
│   ├── processed/              # Cleaned/processed datasets
│   └── sample_news.csv         # Balanced sample dataset for testing & verification
│
├── models/
│   ├── tfidf_model.pkl         # Trained Logistic Regression classifier
│   ├── tfidf_vectorizer.pkl    # Fitted TF-IDF Vectorizer
│   ├── label_encoder.pkl       # Label mapping dictionary
│   └── bert_model/             # Fine-tuned DistilBERT weights & config
│
├── src/
│   ├── __init__.py             # Package initializer
│   ├── preprocessor.py         # Text cleaning and dual-mode tokenization
│   ├── tfidf_classifier.py     # TF-IDF inference engine
│   ├── bert_classifier.py      # DistilBERT PyTorch inference module
│   ├── sentiment_analyzer.py   # NLTK VADER sentiment analyzer
│   ├── headline_checker.py     # Sentence-Transformer semantic similarity
│   ├── ner_analyzer.py         # spaCy Named Entity Recognition
│   ├── explainability.py       # LIME text explainer & Plotly charts
│   ├── fake_score.py           # Weighted ensemble score engine
│   ├── scraper.py              # URL article scraper
│   └── utils.py                # Validation, formatting, and logging utilities
│
├── training/
│   ├── train_tfidf.py          # TF-IDF + Logistic Regression training pipeline
│   └── train_bert.py           # DistilBERT fine-tuning script (GPU/CPU)
│
├── notebooks/
│   └── TruthRadar_Experiment.ipynb # Complete experiment & walkthrough notebook
│
├── tests/
│   ├── test_preprocessor.py    # Text cleaning and validation tests
│   ├── test_sentiment.py       # Sentiment engine tests
│   ├── test_headline_checker.py# Headline-body similarity tests
│   ├── test_fake_score.py      # Fake score ensemble tests
│   └── test_scraper.py         # Web scraper tests
│
└── docs/
    ├── architecture.md         # Detailed architectural documentation
    ├── methodology.md          # Scientific methodology & metrics
    ├── viva_questions.md       # 35 Viva/Interview preparation questions
    └── limitations.md          # System limitations & ethical guidelines
```

---

## 🚀 Quick Start & Installation

### 1. Clone or Open Workspace
```bash
cd "c:\Users\sujay\Downloads\NLP project"
```

### 2. (Optional) Create Virtual Environment
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Language Models (spaCy & NLTK)
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## 🏋️ Model Training

### 1. Train TF-IDF + Logistic Regression Baseline
```bash
python training/train_tfidf.py --data-path data/sample_news.csv
```
- Fits TF-IDF vectorizer (unigrams + bigrams)
- Trains balanced Logistic Regression classifier
- Evaluates Accuracy, Precision, Recall, F1, Confusion Matrix on 20% test split
- Saves artifacts to `models/tfidf_model.pkl` and `models/tfidf_vectorizer.pkl`.

### 2. (Optional) Fine-Tune DistilBERT Deep Learning Classifier
```bash
python training/train_bert.py --data-path data/sample_news.csv --epochs 3 --batch-size 8
```
- Fine-tunes `distilbert-base-uncased` with PyTorch AdamW optimizer and linear warmup
- Supports automatic CUDA GPU acceleration with CPU fallback
- Saves fine-tuned transformer weights to `models/bert_model/`.

---

## 💻 Running the Application

Launch the Streamlit dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

Execute the automated test suite with pytest:
```bash
pytest tests/ -v
```

---

## 📊 Evaluation & Metrics Summary

| Component | Architecture | Primary Metric | Purpose |
| :--- | :--- | :--- | :--- |
| **Statistical Model** | TF-IDF (1-2 ngrams) + LogReg | F1-Score & Accuracy | Lexical keyword and stylistic detection |
| **Contextual Model** | DistilBERT Transformer | Macro F1 & Loss | Deep semantic and context understanding |
| **Consistency Engine** | `all-MiniLM-L6-v2` | Cosine Similarity $[0, 1]$ | Clickbait and title mismatch detection |
| **Sentiment Engine** | VADER Lexicon | Compound Score $[-1, 1]$ | Emotional intensity & sensationalism |
| **Entity Recognizer** | spaCy `en_core_web_sm` | Entity Span Accuracy | Identification of key persons, orgs, places |
| **Explainable AI** | LIME Local Surrogate | Feature Weight Attribution | Transparent word contribution explanations |

---

## ⚖️ Fact-Checking & Academic Disclaimer

> **Important Notice**: TruthRadar is an AI-powered analytical and research tool. Its predictions are probabilistic assessments based on linguistic style, statistical distributions, and semantic alignment. The system **does not** query live ground-truth fact registries and should not be used as definitive proof of truth or falsity. Always verify critical news claims against authoritative, reputable primary sources.

---

## 📜 License
This project is open-source and released under the [MIT License](LICENSE).
