# TruthRadar — Scientific Methodology & Evaluation Framework

## 1. Experimental Methodology

The development of TruthRadar follows an empirical, data-driven NLP methodology designed to combine the speed and interpretability of statistical classifiers with the contextual depth of Transformer networks.

### 1.1 Dataset Ingestion & Partitioning
- **Data Stratification**: The dataset is split into 80% training and 20% testing subsets with random seed pinning (`random_state=42`) to ensure reproducibility.
- **Leakage Prevention**: Feature extractors (TF-IDF vectorizers, vocabularies, scaling) are strictly fitted on the training split only and applied transform-only on validation/testing sets.

### 1.2 Evaluation Metrics
Given the societal impact of misinformation, model evaluation involves multiple complementary metrics:

1. **Accuracy**:
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Precision (Positive Predictive Value)**:
   $$\text{Precision} = \frac{TP}{TP + FP}$$
3. **Recall (Sensitivity / Detection Rate)**:
   $$\text{Recall} = \frac{TP}{TP + FN}$$
4. **F1-Score (Harmonic Mean)**:
   $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
5. **Confusion Matrix**:
   Visualizes False Positives (Real news misclassified as Fake) and False Negatives (Fake news missed by the system).

---

## 2. Multi-Signal Ensemble Calibration

Rather than treating a single classification score in isolation, TruthRadar leverages a calibrated multi-signal strategy:

| Signal Name | Technology | Underlying Theory | Role in Prediction |
| :--- | :--- | :--- | :--- |
| **Statistical Signal** | TF-IDF + Logistic Regression | N-gram token frequency distribution | Identifies explicit lexical indicators and stylistic wording. |
| **Deep Contextual Signal** | DistilBERT | Multi-head self-attention transformer | Captures semantic coherence, complex syntax, and nuance. |
| **Semantic Consistency** | MiniLM Embeddings | Cosine similarity in vector space | Measures congruence between title and content (clickbait detector). |
| **Sentiment Diagnostic** | NLTK VADER | Lexicon-based polarity scoring | Highlights emotional intensity and sensationalism. |
| **Entity Profiling** | spaCy NER | Transition-based dependency parser | Extracts named actors for transparency. |
| **Local Explainability** | LIME | Perturbation-based surrogate modelling | Explains model reasoning to end users. |

---

## 3. Scientific Distinctions

1. **Classification vs. Verification**: TruthRadar evaluates linguistic patterns, stylistic indicators, and semantic consistency. It does not perform live retrieval against authoritative encyclopedic knowledge bases.
2. **Probabilistic Nature**: All outputs represent estimated probabilities rather than absolute truth determinations.
