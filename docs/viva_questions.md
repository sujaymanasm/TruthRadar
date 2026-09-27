# TruthRadar — 35 Comprehensive Viva / Interview Questions & Answers

---

### Q1: What is the primary objective of the TruthRadar project?
**Answer:** TruthRadar is an AI-powered fake news detection and explainability system that evaluates news articles, headlines, or URLs using a multi-signal NLP ensemble (TF-IDF, DistilBERT, Sentence-Transformers, VADER sentiment, spaCy NER, and LIME) to provide calibrated authenticity scores and transparent explanations.

### Q2: Why is fake news detection an important NLP problem?
**Answer:** Misinformation spreads rapidly on digital platforms, influencing public health, financial markets, and democratic elections. Automated NLP systems help flag potentially deceptive or manipulative content at scale before it causes harm.

### Q3: Why combine multiple NLP signals instead of relying on a single classifier?
**Answer:** A single model can overfit to specific vocabulary or fail on novel phrasing. Combining statistical n-grams (TF-IDF), deep semantics (DistilBERT), headline consistency, sentiment, and named entities yields a more robust, interpretable assessment with fewer blind spots.

### Q4: What is TF-IDF and how does it work?
**Answer:** Term Frequency-Inverse Document Frequency evaluates word importance in a document relative to a corpus. 
- $\text{TF}(t, d) = \frac{\text{count}(t, d)}{\text{total words in } d}$
- $\text{IDF}(t) = \log\frac{N}{1 + \text{count of docs containing } t}$
- $\text{TF-IDF} = \text{TF} \times \text{IDF}$
It downweights ubiquitous stop words and highlights unique informative keywords.

### Q5: Why choose Logistic Regression for the baseline classifier?
**Answer:** Logistic Regression is computationally efficient, less prone to overfitting on high-dimensional sparse text data, and outputs well-calibrated class probabilities via the sigmoid function ($\sigma(z) = \frac{1}{1 + e^{-z}}$).

### Q6: What is BERT and how does it differ from traditional Word2Vec / TF-IDF?
**Answer:** BERT (Bidirectional Encoder Representations from Transformers) uses multi-head self-attention to process words in relation to all other words in a sentence bidirectionally. Unlike static embeddings (Word2Vec) or bag-of-words (TF-IDF), BERT generates context-dependent representations where word meaning changes based on surrounding context.

### Q7: Why use DistilBERT instead of standard BERT-large?
**Answer:** DistilBERT uses knowledge distillation to retain approximately 97% of BERT-base language comprehension while being 40% smaller and 60% faster, making it ideal for responsive web applications and standard hardware.

### Q8: What is Tokenization in Transformers?
**Answer:** Tokenization splits raw text into subword units (WordPiece in BERT). It handles out-of-vocabulary words by breaking them into frequent subwords (e.g., "unprecedented" $\rightarrow$ "un", "##prece", "##dented") and maps them to token IDs with special tokens like `[CLS]` and `[SEP]`.

### Q9: What is the purpose of Headline-Body Consistency analysis?
**Answer:** Clickbait and sensationalized articles frequently use provocative headlines that are unsupported or contradicted by the body text. Measuring semantic similarity between headline and body helps detect this incongruence.

### Q10: How is semantic similarity calculated between headline and body?
**Answer:** We encode both texts into 384-dimensional dense vectors using the `all-MiniLM-L6-v2` Sentence Transformer and compute their Cosine Similarity: $\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$.

### Q11: What is VADER and why use it for sentiment analysis?
**Answer:** VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based sentiment tool attuned to sentiments expressed in social media and news. It handles punctuation, capitalization, degree modifiers ("extremely"), and negations ("not good").

### Q12: Does negative sentiment prove that an article is fake news?
**Answer:** No. Real news about natural disasters, economic downturns, or conflicts is inherently negative. Sentiment is only an auxiliary signal to detect sensationalist, emotionally charged, or alarmist rhetoric.

### Q13: What is Named Entity Recognition (NER) and why is it included in TruthRadar?
**Answer:** NER identifies real-world entities (people, organizations, locations, dates, monetary values). In TruthRadar, it allows users to quickly see the primary subjects of the news piece and cross-check named individuals or institutions.

### Q14: What is LIME and how does it work?
**Answer:** LIME (Local Interpretable Model-agnostic Explanations) explains black-box model predictions by perturbing the input text (removing random words), observing changes in prediction probabilities, and fitting a simple interpretable linear model locally around the prediction.

### Q15: Why is Explainable AI (XAI) crucial in fake news detection?
**Answer:** Users, journalists, and moderators need to understand *why* an AI flagged an article. Explanations foster trust, reveal algorithmic bias, and help human fact-checkers verify key linguistic patterns.

### Q16: How is the final TruthRadar fake probability calculated?
**Answer:** Through a weighted linear ensemble: 
$\text{Score} = w_{\text{TFIDF}} \cdot P_{\text{TFIDF}} + w_{\text{BERT}} \cdot P_{\text{BERT}} + w_{\text{Align}} \cdot (1 - \text{Similarity})$, where weights sum to 1.0 and adapt dynamically if certain components are unavailable.

### Q17: What is the difference between Precision and Recall?
**Answer:** 
- **Precision:** Of all articles flagged as FAKE, what percentage was actually fake? ($\frac{TP}{TP + FP}$)
- **Recall:** Of all actual fake articles in the dataset, what percentage did the model catch? ($\frac{TP}{TP + FN}$)

### Q18: What is F1-Score and when is it preferred over Accuracy?
**Answer:** F1-score is the harmonic mean of Precision and Recall ($2 \cdot \frac{P \cdot R}{P + R}$). It is preferred over Accuracy when datasets have class imbalance, where predicting only the majority class would yield misleadingly high accuracy.

### Q19: What is a Confusion Matrix?
**Answer:** A $2 \times 2$ table showing True Positives (correctly identified fake), True Negatives (correctly identified real), False Positives (real misclassified as fake), and False Negatives (fake missed by the model).

### Q20: What is data leakage and how is it prevented in TruthRadar?
**Answer:** Data leakage occurs when test set information inadvertently influences model training. We prevent it by performing train-test splits before fitting vectorizers or scalers, fitting only on training data, and evaluating strictly on held-out test data.

### Q21: What is n-gram in NLP?
**Answer:** An n-gram is a contiguous sequence of $n$ items from a text. Unigrams ($n=1$) capture single words ("breaking"), while bigrams ($n=2$) capture two-word phrases ("breaking news", "secret base"), preserving local context.

### Q22: What is Knowledge Distillation in Deep Learning?
**Answer:** A technique where a smaller "student" model (DistilBERT) is trained to mimic the output probability distribution and representations of a larger "teacher" model (BERT), transferring capabilities into a lightweight architecture.

### Q23: Why use Streamlit for the user interface?
**Answer:** Streamlit allows building interactive, reactive, Python-native dashboards with built-in state management, interactive plotting (Plotly), and caching (`@st.cache_resource`) for fast real-time model inference.

### Q24: Why use caching (`@st.cache_resource`) in Streamlit?
**Answer:** Deep learning and NLP models (DistilBERT, Sentence Transformers, spaCy) take seconds to load into memory. `@st.cache_resource` ensures they load once into RAM/VRAM on startup and are reused across all user requests.

### Q25: How does TruthRadar handle web URL scraping?
**Answer:** It uses `requests` and `BeautifulSoup` to fetch HTML, strips boilerplate elements (scripts, navbars, ads), and extracts the headline (`og:title` / `<h1>`) and body paragraphs with timeout protection and polite headers.

### Q26: What happens if a web page uses heavy JavaScript or blocks scraping?
**Answer:** The scraper catches exceptions and HTTP 401/403 status codes, returning a polite message prompting the user to copy and paste the article text manually without crashing the app.

### Q27: How does TruthRadar handle empty or very short inputs?
**Answer:** The `validate_text` module checks that input is non-empty and has at least 40 characters before running inference, showing a friendly notification if criteria are not met.

### Q28: Can TruthRadar determine real-world truth with 100% certainty?
**Answer:** No. NLP models assess linguistic style, emotional tone, and statistical similarities learned from training data. They cannot verify live real-world facts. Fact verification requires checking reputable primary sources.

### Q29: What is the role of the Decision Threshold?
**Answer:** The decision threshold (default $0.50$) is the probability cutoff above which an article is classified as FAKE. Adjusting it allows prioritizing Precision (fewer false alarms) or Recall (catching more fakes).

### Q30: What is Self-Attention in Transformers?
**Answer:** Self-attention computes attention weights between every pair of tokens in a sequence:
$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$, allowing the model to focus on relevant context words regardless of their positional distance.

### Q31: How do you prevent overfitting in the TF-IDF and BERT models?
**Answer:** In TF-IDF: using L2 regularization ($C=1.0$) and minimum document frequency (`min_df`). In BERT: using dropout (0.1), weight decay (0.01 in AdamW), and early stopping/learning rate warmup.

### Q32: What is the difference between extractive and abstractive NLP tasks?
**Answer:** Extractive tasks select existing words/phrases directly from text (e.g., NER, keyword extraction). Abstractive tasks generate new sentences and reformulations (e.g., GPT summarization).

### Q33: How does TruthRadar support CPU-only environments?
**Answer:** PyTorch automatically checks `torch.cuda.is_available()`. If GPU is absent, it executes tensor operations on the CPU and uses lightweight model variants (`distilbert`, `all-MiniLM-L6-v2`) to ensure fast response times.

### Q34: What are the key limitations of the current system?
**Answer:** Lack of real-time knowledge base lookup, potential susceptibility to sophisticated AI-generated text that mimics journalistic tone, and domain-shift when evaluated on news domains not represented in training data.

### Q35: What future enhancements could be added to TruthRadar?
**Answer:** Integration with Google Fact Check Tools API / ClaimReview schema, multimodal analysis of accompanying images (deepfake detection), and graph-based source credibility propagation.
