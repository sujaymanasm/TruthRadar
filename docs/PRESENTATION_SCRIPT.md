# 🎙️ TruthRadar: 8-Minute Presentation & 2-Minute Q&A Guide
**Course:** Natural Language Processing  
**Evaluator:** Prof. Kuntal Dey  
**Presenter(s):** Sujay M. & Team  
**Live Demo Link:** [https://truthradar-w9h24zcwzxu7hdo6fx9ckq.streamlit.app/](https://truthradar-w9h24zcwzxu7hdo6fx9ckq.streamlit.app/)

---

## ⏱️ Precise 8-Minute Presentation Timeline

```
[00:00 - 01:00] Slide 1: Introduction & Real-World Motivation (1 min)
[01:00 - 02:30] Slide 2: Problem Statement & Why Single Models Fail (1.5 min)
[02:30 - 04:30] Slide 3: Multi-Signal NLP Architecture & Methodology (2 min)
[04:30 - 06:15] LIVE DEMO: Streamlit Web Application Walkthrough (1.75 min)
[06:15 - 07:15] Slide 4: Experimental Results & Benchmark on WELFake (1 min)
[07:15 - 08:00] Slide 5: Limitations, Conclusion & Handoff to Q&A (45 sec)
[08:00 - 10:00] Q&A Session with Prof. Kuntal Dey (2 min)
```

---

### Minute 0:00 - 1:00 | Introduction & Motivation
**[Visual: Title Slide with Project Name, Team Members, and Architecture Diagram]**

> *"Good morning, Professor Kuntal Dey. Today, we are proud to present **TruthRadar**, an explainable, multi-signal NLP framework designed for automated fake news detection.*
>
> *According to MIT research published in Science, digital misinformation spreads nearly six times faster than verified facts. With billions consuming news algorithmically, fake news is no longer just factual errors—it is engineered with subtle rhetorical deception, emotional manipulation, and clickbait headline disparities.*
>
> *Most existing AI models treat this as a pure black-box classification problem. If a model simply says '80% fake' without telling journalists or readers **why**, nobody can truly trust it. TruthRadar solves this by combining five NLP signals with Local Interpretable Model-agnostic Explanations (LIME)."*

---

### Minute 1:00 - 2:30 | Problem Statement & Architecture
**[Visual: Multi-Signal Flowchart: Input -> Preprocessing -> 5 Feature Extractors -> Ensemble -> LIME]**

> *"Our problem statement addresses three critical weaknesses in existing research:*
> 1. ***Lexical Overfitting:*** *Standard n-gram models memorize specific political entities rather than structural deceit.*
> 2. ***Clickbait Disconnect:*** *Malicious publishers write sensationalized, false headlines paired with generic body paragraphs.*
> 3. ***The Interpretability Deficit:*** *End users need token-level transparency.*
>
> *To solve this, TruthRadar uses a hybrid pipeline:*
> - *First, **Lexical Statistics**: TF-IDF unigram and bigrams (10,000 features) with L2 Logistic Regression.*
> - *Second, **Deep Contextual Semantics**: DistilBERT Transformer capturing long-range bidirectional dependencies.*
> - *Third, **Headline-Body Semantic Consistency**: We compute 384-dimensional dense sentence embeddings using Sentence-BERT (MiniLM) and calculate cosine similarity to penalize clickbait discrepancy.*
> - *Fourth, **Affective & Entity Profiling**: VADER sentiment analysis to measure emotional exaggeration, and spaCy Named Entity Recognition to track verified organizations and persons.*
> - *Finally, **Weighted Ensembling** calibrated across all signals to output a single definitive Truth Score."*

---

### Minute 2:30 - 4:30 | Live System Demonstration (The Game Changer)
**[Action: Switch tab to browser and open https://truthradar-w9h24zcwzxu7hdo6fx9ckq.streamlit.app/]**

> *"Rather than presenting static screenshots, we have deployed our complete system live to the cloud.*
>
> *(1) In **Tab 1: Single Article Analysis**, let us paste a sensationalized news claim. Clicking 'Analyze Article' triggers the pipeline in under 50 milliseconds. You can see the gauge: Fake probability, credibility badge, sentiment compound score, and recognized entities.*
>
> *(2) Here is the core scientific contribution: **LIME Explainable AI**. Looking at this interactive chart, the red bars show exact words pushing the prediction toward 'Fake'—words like 'shocking', 'conspiracy', 'unprecedented'—while green bars represent factual credibility anchors like 'spokesperson' or 'official'.*
>
> *(3) In **Tab 2: Headline-Body Consistency**, if we enter a clickbait headline paired with an unrelated body, our Sentence-BERT cosine similarity immediately drops below 0.4, flagging high clickbait divergence.*
>
> *(4) In **Tab 3: Live Web Scraper**, users can simply paste any online news URL; the system fetches the DOM, strips HTML boilerplate, extracts the article, and evaluates it on the fly."*

---

### Minute 4:30 - 6:00 | Experimental Results & Benchmark
**[Visual: Evaluation Metrics Table & WELFake Confusion Matrix]**

> *"To rigorously validate our system, we benchmarked TruthRadar on the **IEEE WELFake dataset**, one of the largest academic benchmarks with 72,134 verified articles.*
>
> *We evaluated on a strict 20% hold-out test set of **14,414 unseen articles**:*
> - *Our tuned TF-IDF model achieved **95.73% Accuracy**, **95.36% Precision**, and **96.37% Recall**.*
> - *The high recall of 96.37% is crucial because in misinformation detection, False Negatives—letting fake news slip through as real—are the most dangerous errors.*
> - *Our full ensemble achieves **96.84% Accuracy** and **96.81% F1-score**, outperforming traditional Naive Bayes and standalone SVM baselines.*
> - *All 19 unit tests in our automated pytest suite achieve 100% pass rates."*

---

### Minute 6:00 - 7:30 | Limitations, Future Scope & Conclusion
**[Visual: Summary Slide with Key Takeaways and Repository Links]**

> *"We also analyzed the system's scientific limitations:*
> 1. ***Domain & Temporal Drift:*** *Models trained on past news corpora require continuous retraining as geopolitical terminology evolves.*
> 2. ***Satire Detection:*** *Subtle irony and satirical outlets like The Onion remain challenging for purely statistical NLP without background world knowledge.*
>
> *For future work, we plan to incorporate real-time knowledge-graph verification and expand to multilingual Indian languages.*
>
> *In conclusion, TruthRadar bridges the gap between state-of-the-art predictive accuracy and explainable human interpretability.*
>
> *Thank you, Professor Dey. We are now eager to take your questions."*

---

## 🎯 Anticipated 2-Minute Q&A Preparation with Prof. Kuntal Dey

### Q1: *"Why use both TF-IDF and BERT? Why not just use BERT alone?"*
> **Answer:** *"Sir, while BERT excels at contextual nuances, it is computationally heavy (~110ms per inference and larger memory footprint). TF-IDF captures global n-gram frequency distributions and lexical markers in just 18ms. Furthermore, LIME explanations on linear models like TF-IDF produce faithful, interpretable token attributions without the high variance often seen when perturbing deep transformer attention layers. Combining them gives us high accuracy with fast inference."*

### Q2: *"How do you handle clickbait where the headline is fake but the body is taken from a real news article?"*
> **Answer:** *"That is exactly why we introduced our Headline-Body Semantic Consistency checker. We pass both through Sentence-BERT (MiniLM-L6-v2) to generate 384-dimensional dense vectors and compute their cosine similarity. If the semantic similarity is low (below 0.45), our ensemble applies an automated clickbait penalty to the fake score, preventing malicious actors from gaming body-only classifiers."*

### Q3: *"How does LIME actually compute the word importance in your app?"*
> **Answer:** *"LIME creates perturbed variations of the input text by randomly masking words, runs each perturbation through the classifier, and weights the samples by their cosine distance to the original text. It then fits an interpretable ridge regression surrogate locally around that prediction. The learned coefficients of that local model represent the exact positive or negative contribution of each word."*

### Q4: *"What dataset did you use and why did you choose it over others like LIAR or ISOT?"*
> **Answer:** *"We selected the IEEE WELFake dataset because it resolves significant biases present in smaller datasets. ISOT and LIAR are smaller (12k to 40k articles) and suffer from source domain leakage. WELFake unifies 72,134 articles across four major benchmark databases, providing a more balanced and diverse linguistic representation."*
