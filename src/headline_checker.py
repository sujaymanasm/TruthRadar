"""
TruthRadar - Headline-Body Consistency Checker
----------------------------------------------
Calculates semantic alignment and topical congruence between a news headline
and the article body using Dense Semantic Embeddings (Sentence Transformers)
or Asymmetric Keyword Containment & Lede-Paragraph TF-IDF.

Detects potential clickbait, hyperbolic headlines, and mismatched content.
"""

from typing import Dict, Any, Optional
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import config
from src.preprocessor import clean_text
from src.utils import logger

# Lazy singleton for SentenceTransformer model
_embedding_model = None
_embedding_model_loaded = False

# Common English stop words for containment filtering
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves"
}


def _get_sentence_transformer():
    """Initializes and caches SentenceTransformer model with fallback."""
    global _embedding_model, _embedding_model_loaded
    if not _embedding_model_loaded:
        try:
            from sentence_transformers import SentenceTransformer
            # Fast check if weights are available in local cache or loadable
            _embedding_model = SentenceTransformer(
                config.SENTENCE_TRANSFORMER_MODEL,
                device="cpu"
            )
            _embedding_model_loaded = True
            logger.info("SentenceTransformer initialized successfully.")
        except Exception as e:
            logger.info(f"SentenceTransformer not cached ({e}). Using Asymmetric Hybrid Containment.")
            _embedding_model = None
            _embedding_model_loaded = True
    return _embedding_model


def _calculate_asymmetric_containment(headline: str, body: str) -> float:
    """
    Computes asymmetric keyword containment: what percentage of headline
    key content terms appear in the article body.
    """
    # Extract substantive tokens (length > 2, not in stop words)
    raw_tokens = re.findall(r'\b[A-Za-z0-9\'-]+\b', headline.lower())
    content_tokens = [t for t in raw_tokens if len(t) > 2 and t not in STOP_WORDS]
    
    if not content_tokens:
        return 0.70 # Default neutral if headline has no substantive tokens

    body_lower = body.lower()
    matches = sum(1 for t in content_tokens if t in body_lower)
    return float(matches / len(content_tokens))


def _fallback_hybrid_similarity(headline: str, body: str) -> float:
    """
    High-accuracy fallback similarity combining asymmetric containment
    and lead-paragraph cosine similarity (Inverted Pyramid standard in journalism).
    """
    try:
        containment = _calculate_asymmetric_containment(headline, body)
        
        # Lead paragraph (first 3 sentences)
        sentences = [s.strip() for s in re.split(r'[.!?]+', body) if s.strip()]
        lede = " ".join(sentences[:3]) if sentences else body[:400]

        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tfidf_mat = vec.fit_transform([headline, lede])
        lede_sim = float(cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0])
        
        # Weighted hybrid: 60% containment + 40% calibrated lede cosine
        calibrated_sim = (0.60 * containment) + (0.40 * min(1.0, lede_sim * 2.2))
        return float(np.clip(calibrated_sim, 0.05, 1.0))
    except Exception as e:
        logger.warning(f"Error in hybrid similarity: {e}")
        return _calculate_asymmetric_containment(headline, body)


def check_headline_body_consistency(headline: str, body: str) -> Dict[str, Any]:
    """
    Computes semantic similarity between headline and body text.
    """
    clean_h = clean_text(headline)
    clean_b = clean_text(body)

    if not clean_h or not clean_b:
        return {
            "is_available": False,
            "similarity_score": 0.50,
            "similarity_percentage": 50.0,
            "interpretation": "Insufficient data",
            "method": "None",
            "analysis_note": "Both headline and body are required for consistency analysis."
        }

    body_summary = clean_b[:2500]
    model = _get_sentence_transformer()
    method_name = "Dense Embeddings (MiniLM-L6-v2)"

    if model is not None:
        try:
            embeddings = model.encode([clean_h, body_summary], normalize_embeddings=True)
            raw_sim = float(np.dot(embeddings[0], embeddings[1]))
            # Combine neural similarity with keyword containment for robustness
            containment = _calculate_asymmetric_containment(clean_h, clean_b)
            sim = float(0.70 * raw_sim + 0.30 * containment)
            sim = float(np.clip(sim, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Error encoding embeddings: {e}. Falling back to Asymmetric Hybrid.")
            sim = _fallback_hybrid_similarity(clean_h, clean_b)
            method_name = "Asymmetric Containment & Lede-TFIDF"
    else:
        sim = _fallback_hybrid_similarity(clean_h, clean_b)
        method_name = "Asymmetric Containment & Lede-TFIDF"

    percentage = round(sim * 100.0, 1)

    # Interpret similarity level
    if sim >= config.HEADLINE_HIGH_SIMILARITY:
        interpretation = "High consistency"
        note = "The headline directly reflects and aligns with the factual topics covered in the article body."
    elif sim >= config.HEADLINE_MEDIUM_SIMILARITY:
        interpretation = "Medium consistency"
        note = "The headline shares moderate thematic overlap with the article body."
    else:
        interpretation = "Low consistency"
        note = "Topical divergence detected between the headline and body. Possible clickbait or sensational framing."

    return {
        "is_available": True,
        "similarity_score": round(sim, 4),
        "similarity_percentage": percentage,
        "interpretation": interpretation,
        "method": method_name,
        "analysis_note": note
    }
