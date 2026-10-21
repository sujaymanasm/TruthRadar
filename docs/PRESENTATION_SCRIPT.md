# 🎙️ TruthRadar: 8-Minute Presentation & 2-Minute Q&A Guide
**Course:** Natural Language Processing  
**Evaluator:** Prof. Kuntal Dey  
**Presenter(s):** Sujay M. & Team  
**Live Demo Link:** [https://truthradar-w9h24zcwzxu7hdo6fx9ckq.streamlit.app/](https://truthradar-w9h24zcwzxu7hdo6fx9ckq.streamlit.app/)

---

## ⏱️ Precise 8-Minute Presentation Timeline

```
[00:00 - 01:00] Slide 1: Introduction & Real-World Motivation (1 min)
[01:00 - 02:30] Slide 2: Multi-Signal NLP Architecture & Psycholinguistics (1.5 min)
[02:30 - 05:00] LIVE DEMO: App Walkthrough, Highlighting, PDF Export & Counterfactuals (2.5 min)
[05:00 - 06:15] Slide 3: Experimental Results & IEEE WELFake Benchmark (1.25 min)
[06:15 - 07:30] Slide 4: Limitations, Conclusion & Future Scope (1.25 min)
[07:30 - 08:00] Hand-off to Q&A (30 sec)
[08:00 - 10:00] Q&A Defense with Prof. Kuntal Dey (2 min)
```

---

### Minute 0:00 - 1:00 | Introduction & Motivation
**[Visual: Title Slide with Project Name, Team Members, and Architecture Diagram]**

> *"Good morning, Professor Kuntal Dey. Today, we are proud to present **TruthRadar**, an explainable, multi-signal NLP framework designed for automated fake news detection and forensic text explainability.*
>
> *Misinformation online spreads six times faster than verified facts. Yet, most automated detectors operate as opaque black boxes—giving a single probability number without linguistic or factual justification. If journalists, researchers, or citizens cannot understand **why** an article is flagged, the model cannot be trusted.*
>
> *TruthRadar solves this by unifying statistical NLP, deep contextual embeddings, psycholinguistic stylometrics, domain credibility intelligence, and Local Interpretable Model-agnostic Explanations (LIME)."*

---

### Minute 1:00 - 2:30 | Problem Statement & Architecture
**[Visual: Multi-Signal Flowchart: Input -> Preprocessing -> 6 Signal Engines -> Calibrated Ensemble -> LIME]**

> *"Our problem statement tackles four core challenges in contemporary NLP:*
> 1. ***Lexical Overfitting:*** *Models memorizing ephemeral political terms instead of structural deceit.*
> 2. ***Clickbait Disparity:*** *Sensational headlines paired with benign body text.*
> 3. ***Affective & Syntactic Manipulation:*** *Use of low readability, excessive shouting, and emotional intensifiers.*
> 4. ***The Interpretability Deficit:*** *Providing token-level evidence.*
>
> *To solve this, TruthRadar combines six distinct analytical signals:*
> - *First, **Lexical Statistics**: 10,000 sublinear TF-IDF n-grams with L2-regularized Logistic Regression.*
> - *Second, **Deep Contextual Semantics**: DistilBERT Transformer for bidirectional contextual representations.*
> - *Third, **Headline-Body Semantic Coherence**: Sentence-BERT (MiniLM) generating 384-dimensional dense vectors to calculate cosine alignment and penalize clickbait discrepancy.*
> - *Fourth, **Psycholinguistic Stylometrics**: Quantifying Flesch Reading Ease, Flesch-Kincaid Grade Level, Type-Token Ratio for vocabulary richness, and a composite Sensationalism Index.*
> - *Fifth, **Domain Credibility Intelligence**: Whitelist and blacklist registries with explicit Satire and Parody detection.*
> - *Sixth, **Token-Level Explainability (LIME)**: Yielding interpretable feature attributions."*

---

### Minute 2:30 - 5:00 | Live System Demonstration (The Game Changer)
**[Action: Switch tab to browser and open https://truthradar-w9h24zcwzxu7hdo6fx9ckq.streamlit.app/]**

> *"Rather than showing slides, we have deployed our complete system live to the cloud.*
>
> *(1) **Instant Analysis & PDF Audit Certificate:** When we analyze a news article, clicking 'Analyze News' computes all signals in under 50 milliseconds. Right at the top, notice this button: **'Download Official Forensic Audit Certificate'**. Clicking it generates a dynamic, publication-grade 1-page PDF forensic report with timestamps, component breakdowns, and attribution weights on the fly.*
>
> *(2) **Inline Forensic Text Reader:** Next to our interactive LIME feature importance chart, look at the **'Inline Forensic Reader'** tab. The article is rendered with token-by-token visual highlights: red highlights indicate deceptive, sensational cues with hover tooltips, and green highlights anchor verified, neutral reporting vocabulary.*
>
> *(3) **Psycholinguistic Diagnostics:** Below, our stylometrics dashboard displays the Sensationalism Index (penalizing all-caps shouting and exclamations), Flesch Reading Ease, and Type-Token Ratio, isolating overused emotional intensifiers like 'shocking' and 'conspiracy'.*
>
> *(4) **Live URL & Satire Intelligence:** In Tab 3, when a URL from a known satirical source like 'The Onion' is entered, our domain intelligence immediately fires a **Satire Alert**, warning that the content is parodic rather than factual.*
>
> *(5) **Counterfactual Rephrasing Sandbox:** At the bottom, we provide a Counterfactual Sandbox. If a user or editor tones down the sensational words, the system immediately recalculates the causal drop in fake probability in real time."*

---

### Minute 5:00 - 6:15 | Experimental Results & Benchmark
**[Visual: Evaluation Metrics Table & WELFake Benchmark]**

> *"We benchmarked TruthRadar on the **IEEE WELFake dataset**, one of the largest peer-reviewed benchmarks with 72,134 verified articles.*
>
> *On a strict 20% hold-out test set of **14,414 unseen articles**:*
> - *Our tuned TF-IDF classifier achieved **95.73% Accuracy**, **95.36% Precision**, and **96.37% Recall**.*
> - *The high recall of 96.37% ensures False Negatives—letting dangerous fake news slip through—are strictly minimized.*
> - *Our full ensemble achieves **96.84% Accuracy** and **96.81% F1-score**, operating at an average latency of ~35 milliseconds per article.*
> - *Our codebase includes **28 automated unit tests** covering all modules with a 100% pass rate."*

---

### Minute 6:15 - 7:30 | Limitations, Future Scope & Conclusion
**[Visual: Limitations & Conclusion Slide]**

> *"We also evaluated our scientific limitations:*
> 1. ***Temporal & Domain Drift:*** *Models trained on historical corpora require continuous active learning as geopolitical topics evolve.*
> 2. ***Knowledge-Graph Grounding:*** *While our system excels at stylistic and semantic pattern recognition, multi-hop fact-checking requires external knowledge-graph triples.*
>
> *For future work, we aim to integrate real-time knowledge-graph verification and expand to Indic languages.*
>
> *In summary, TruthRadar bridges the gap between state-of-the-art predictive performance and transparent, user-facing interpretability.*
>
> *Thank you, Professor Dey. We are now happy to answer your questions."*

---

## 🎯 Anticipated 2-Minute Q&A Preparation with Prof. Kuntal Dey

### Q1: *"Why did you include Stylometric and Readability metrics alongside Deep Learning?"*
> **Answer:** *"Sir, computational linguistics research (such as Vosoughi et al. in Science and Pennebaker’s psycholinguistic work) demonstrates that deceptive articles are engineered for viral emotional contagion. They systematically exhibit lower Flesch-Kincaid reading levels, lower Type-Token Ratios (lexical diversity), and higher affective arousal (exclamations, intensifiers, shouting syntax). Incorporating these stylometric features gives our ensemble orthogonal linguistic signals that pure semantic embeddings can sometimes overlook."*

### Q2: *"How do you handle Satire like 'The Onion', which mimics news style perfectly?"*
> **Answer:** *"Pure text classification models frequently mistake satire for fake news or real news because satire uses clean syntax and deadpan irony. That is why we added our Domain Reputation & Satire Intelligence Engine. When analyzing URLs or known sources, TruthRadar recognizes accredited satirical registries and displays an explicit Satire Alert, explaining that the text is parodic rather than deliberately malicious disinformation."*

### Q3: *"How does your Counterfactual Sandbox contribute to Explainability?"*
> **Answer:** *"Explainable AI is not just about showing static bar charts—it is about causal understanding. By allowing the evaluator to modify specific tokens (e.g. replacing 'SHOCKING' with 'Reported' or removing exclamation marks) and immediately observing the shift in prediction, the user can verify that the model's decision boundaries are sensitive to meaningful semantic cues rather than spurious artifacts."*

### Q4: *"What is the difference between LIME and Attention-based explanation in Transformers?"*
> **Answer:** *"Attention weights show which token representations were attended to in intermediate layers, but extensive research (e.g. Jain & Wallace, 'Attention is not Explanation') shows attention weights do not always correlate with causal feature importance. LIME, by contrast, operates externally as a local surrogate model: it perturbs the input, measures the output change, and computes true mathematical feature attribution."*
