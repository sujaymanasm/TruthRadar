"""
TruthRadar - Headline-Body Consistency Checker
----------------------------------------------
Calculates semantic alignment and topical congruence between a news headline
and the article body using Dense Semantic Embeddings (Sentence Transformers).

Detects potential clickbait, hyperbolic headlines, and mismatched content.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import config
from src.preprocessor import clean_text
from src.utils import logger

# Lazy singleton for SentenceTransformer model
_embedding_model = None
_embedding_model_loaded = False


def _get_sentence_transformer():
    """Initializes and caches SentenceTransformer model with fallback."""
    global _embedding_model, _embedding_model_loaded
    if not _embedding_model_loaded:
        try:
            from sentence_transformers import SentenceTransformer
            # Fast check if weights are available in local cache
            _embedding_model = SentenceTransformer(
                config.SENTENCE_TRANSFORMER_MODEL,
                device="cpu",
                local_files_only=True
            )
            _embedding_model_loaded = True
            logger.info("SentenceTransformer loaded from local cache successfully.")
        except Exception as e:
            logger.info(f"SentenceTransformer not cached locally ({e}). Using high-performance TF-IDF cosine similarity.")
            _embedding_model = None
            _embedding_model_loaded = True
    return _embedding_model


def _fallback_tfidf_similarity(text1: str, text2: str) -> float:
    """Fallback semantic similarity calculation using TF-IDF n-grams."""
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tfidf_mat = vec.fit_transform([text1, text2])
        sim = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        # Simple word overlap fallback
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.5
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return float(intersection / union) if union > 0 else 0.5


def check_headline_body_consistency(headline: str, body: str) -> Dict[str, Any]:
    """
    Computes semantic similarity between headline and body text.

    Args:
        headline: News title or headline string.
        body: Article body text string.

    Returns:
        Dict containing:
            - is_available: bool
            - similarity_score: float (0.0 to 1.0)
            - similarity_percentage: float (0.0 to 100.0)
            - interpretation: 'High consistency', 'Medium consistency', or 'Low consistency'
            - method: 'Dense Embeddings (MiniLM)' or 'Sparse TF-IDF Cosine'
            - analysis_note: str
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

    # Truncate body if extremely long for embedding efficiency (first 1500 chars usually contain key theme)
    body_summary = clean_b[:2000]

    model = _get_sentence_transformer()
    method_name = "Dense Embeddings (MiniLM-L6-v2)"

    if model is not None:
        try:
            embeddings = model.encode([clean_h, body_summary], normalize_embeddings=True)
            # Dot product of normalized vectors equals cosine similarity
            sim = float(np.dot(embeddings[0], embeddings[1]))
            sim = float(np.clip(sim, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Error encoding embeddings: {e}. Falling back to TF-IDF.")
            sim = _fallback_tfidf_similarity(clean_h, body_summary)
            method_name = "Sparse TF-IDF Cosine"
    else:
        sim = _fallback_tfidf_similarity(clean_h, body_summary)
        method_name = "Sparse TF-IDF Cosine"

    percentage = round(sim * 100.0, 1)

    # Interpret similarity level
    if sim >= config.HEADLINE_HIGH_SIMILARITY:
        interpretation = "High consistency"
        note = "The headline directly reflects and aligns with the factual topics covered in the article body."
    elif sim >= config.HEADLINE_MEDIUM_SIMILARITY:
        interpretation = "Medium consistency"
        note = "The headline shares moderate thematic overlap with the article body but may contain selective phrasing."
    else:
        interpretation = "Low consistency"
        note = "Substantial topical divergence detected between the headline and body. Possible clickbait or sensational framing."

    return {
        "is_available": True,
        "similarity_score": round(sim, 4),
        "similarity_percentage": percentage,
        "interpretation": interpretation,
        "method": method_name,
        "analysis_note": note
    }
