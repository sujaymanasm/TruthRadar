# TruthRadar — System Architecture & Technical Design

## 1. System Overview

**TruthRadar** is a multi-signal Natural Language Processing (NLP) framework designed to identify potentially misleading, unverified, or fabricated news articles. Instead of relying upon a single monolithic classification algorithm, TruthRadar integrates statistical n-gram representations, deep bidirectional transformer embeddings, semantic consistency analysis, lexicon-based sentiment analysis, named entity recognition, and local explainability models into a unified decision dashboard.

---

## 2. High-Level Architectural Diagram

```mermaid
flowchart TD
    UI[Streamlit User Interface] -->|Input: Article / Headline / URL| INGEST[Text Ingestion & Validation]
    INGEST -->|Clean & Format| PREP[Dual-Path Preprocessor]
    
    PREP -->|Normalized Lowercase Alpha| TFIDF_FEAT[TF-IDF Vectorizer (1-2 ngrams)]
    PREP -->|Minimally Cleaned Natural Text| BERT_FEAT[DistilBERT Tokenizer]
    PREP -->|Headline & Body Tokens| DENSE_FEAT[Sentence Transformer MiniLM]
    PREP -->|Raw Sentence Strings| VADER_ENG[NLTK VADER Lexicon]
    PREP -->|Token Stream| SPACY_ENG[spaCy Named Entity Recognizer]
    
    TFIDF_FEAT --> TFIDF_MODEL[Logistic Regression Classifier]
    BERT_FEAT --> BERT_MODEL[Fine-Tuned DistilBERT Classifier]
    DENSE_FEAT --> SIM_MODEL[Cosine Semantic Similarity]
    
    TFIDF_MODEL -->|Probability Score| ENSEMBLE[Calibrated Fake Score Engine]
    BERT_MODEL -->|Probability Score| ENSEMBLE
    SIM_MODEL -->|Alignment Score| ENSEMBLE
    
    TFIDF_MODEL --> LIME[LIME Local Feature Explainer]
    
    ENSEMBLE --> REPORT[TruthRadar Comprehensive Report]
    VADER_ENG --> REPORT
    SPACY_ENG --> REPORT
    LIME --> REPORT
    
    REPORT --> UI
```

---

## 3. Component Breakdown

### 3.1. Ingestion and Preprocessing Pipeline (`src/preprocessor.py`)
- **Dual-Path Token Processing**:
  1. *Statistical Path*: HTML strip, lowercase conversion, punctuation removal, number suppression, whitespace normalization (for TF-IDF).
  2. *Transformer Path*: Minimal cleaning, preserving capitalization, sentence boundaries, and semantic markers for Transformer self-attention.
- **Validation**: Rejects inputs below 40 characters to prevent false high-confidence outputs on insufficient context.

### 3.2. Statistical Baseline: TF-IDF + Logistic Regression (`src/tfidf_classifier.py`)
- Extracts unigrams and bigrams (`(1, 2)` n-grams) up to 10,000 sublinear TF-IDF features.
- Balanced-class Logistic Regression fits an optimal hyperplane with calibrated probability outputs via the sigmoid function:
  $$P(Y = \text{FAKE} \mid X) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

### 3.3. Deep Contextual Transformer: DistilBERT (`src/bert_classifier.py`)
- Employs a 6-layer, 768-hidden dimension DistilBERT model.
- Retains 97% of BERT's language understanding while operating at 60% faster inference speeds and requiring 40% less memory.
- Fine-tuned using cross-entropy loss with PyTorch and Hugging Face.

### 3.4. Headline-Body Consistency Engine (`src/headline_checker.py`)
- Uses `all-MiniLM-L6-v2` dense semantic embeddings (384 dimensions).
- Computes cosine similarity between headline embedding vector $\mathbf{u}$ and body embedding vector $\mathbf{v}$:
  $$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
- Low semantic consistency highlights potential clickbait, exaggeration, or disconnected claims.

### 3.5. Sentiment Polarity Engine (`src/sentiment_analyzer.py`)
- NLTK VADER calculates positive, neutral, negative proportions and a normalized compound score $[-1.0, +1.0]$.
- Identifies emotionally manipulative, sensationalized, or outrage-driven rhetoric.

### 3.6. Named Entity Recognition (`src/ner_analyzer.py`)
- spaCy `en_core_web_sm` extracts geopolitical entities (GPE), organizations (ORG), persons (PERSON), dates (DATE), and monetary values (MONEY).
- Enables users to see key actors and entities mentioned in the claim.

### 3.7. Explainable AI: LIME (`src/explainability.py`)
- Generates locally faithful linear approximations around the specific prediction:
  $$\xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$
- Renders positive/negative word contribution bars via Plotly so non-technical users can see why a decision was reached.

### 3.8. Ensemble Fake Score Engine (`src/fake_score.py`)
- Computes weighted linear combination:
  $$\text{Score}_{\text{Fake}} = w_{\text{TFIDF}} \cdot P_{\text{TFIDF}} + w_{\text{BERT}} \cdot P_{\text{BERT}} + w_{\text{Align}} \cdot (1 - \text{Sim}_{\text{Headline}})$$
- Guarantees $P_{\text{Real}} = 1.0 - P_{\text{Fake}}$ and assigns confidence tiers (High / Medium / Low).
