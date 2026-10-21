"""
TruthRadar - Fake Score Multi-Signal Ensemble Engine
---------------------------------------------------
Synthesizes predictions from multiple NLP models (TF-IDF + Logistic Regression,
DistilBERT Transformer, Semantic Headline-Body Alignment, and Sentiment Polarity)
into a unified, calibrated authenticity assessment.
"""

from typing import Dict, Any, Optional
import config
from src.utils import get_confidence_label, logger


def calculate_fake_score(
    tfidf_result: Dict[str, Any],
    bert_result: Optional[Dict[str, Any]] = None,
    headline_result: Optional[Dict[str, Any]] = None,
    sentiment_result: Optional[Dict[str, Any]] = None,
    has_headline: bool = True
) -> Dict[str, Any]:
    """
    Computes weighted ensemble fake news probability and verdict.

    Args:
        tfidf_result: Output dict from TfidfFakeNewsClassifier.predict().
        bert_result: Output dict from BertFakeNewsClassifier.predict().
        headline_result: Output dict from check_headline_body_consistency().
        sentiment_result: Output dict from analyze_sentiment().
        has_headline: Boolean indicating whether an explicit headline was analyzed.

    Returns:
        Dict containing:
            - prediction: 'FAKE NEWS' or 'REAL NEWS'
            - fake_probability: float (0.0 to 1.0)
            - real_probability: float (0.0 to 1.0)
            - fake_percentage: float (0.0 to 100.0)
            - real_percentage: float (0.0 to 100.0)
            - confidence: 'High', 'Medium', or 'Low'
            - component_weights: Dict[str, float]
            - component_scores: Dict[str, float]
            - active_models: list[str]
            - ensemble_mode: str
            - explanation_notes: list[str]
    """
    tfidf_avail = tfidf_result.get("is_available", False)
    bert_avail = bert_result.get("is_available", False) if bert_result else False
    headline_avail = headline_result.get("is_available", False) if (headline_result and has_headline) else False

    # Extract component fake probabilities
    tfidf_prob = tfidf_result.get("fake_probability", 0.50) if tfidf_avail else 0.50
    bert_prob = bert_result.get("fake_probability", 0.50) if bert_avail else 0.50

    # Invert headline similarity into a fake risk signal (lower consistency -> higher fake risk)
    # Scaled moderately so low consistency nudges but does not override classifiers
    if headline_avail:
        sim_score = headline_result.get("similarity_score", 0.70)
        headline_fake_signal = float(1.0 - sim_score)
    else:
        headline_fake_signal = 0.50

    # Determine adaptive dynamic weights based on available active models
    weights = {}
    explanation_notes = []

    if tfidf_avail and bert_avail and headline_avail:
        # Full ensemble mode
        weights["TF-IDF"] = config.TFIDF_WEIGHT
        weights["BERT"] = config.BERT_WEIGHT
        weights["Headline Consistency"] = config.HEADLINE_WEIGHT
        ensemble_mode = "Full Tri-Modal Ensemble (TF-IDF + BERT + Semantic Alignment)"
        
        weighted_fake_prob = (
            weights["TF-IDF"] * tfidf_prob +
            weights["BERT"] * bert_prob +
            weights["Headline Consistency"] * headline_fake_signal
        )
    elif tfidf_avail and bert_avail and not headline_avail:
        # TF-IDF + BERT (no headline)
        weights["TF-IDF"] = config.NO_HEADLINE_TFIDF_WEIGHT
        weights["BERT"] = config.NO_HEADLINE_BERT_WEIGHT
        ensemble_mode = "Dual Deep & Statistical Ensemble (TF-IDF + DistilBERT)"
        
        weighted_fake_prob = (
            weights["TF-IDF"] * tfidf_prob +
            weights["BERT"] * bert_prob
        )
    elif tfidf_avail and not bert_avail and headline_avail:
        # Fallback without BERT
        weights["TF-IDF"] = config.FALLBACK_TFIDF_WEIGHT
        weights["Headline Consistency"] = config.FALLBACK_HEADLINE_WEIGHT
        ensemble_mode = "Baseline Statistical Ensemble (TF-IDF + Semantic Alignment)"
        
        weighted_fake_prob = (
            weights["TF-IDF"] * tfidf_prob +
            weights["Headline Consistency"] * headline_fake_signal
        )
        explanation_notes.append("DistilBERT model weights not detected; weighted ensemble calibrated to TF-IDF baseline.")
    elif tfidf_avail:
        # Only TF-IDF available
        weights["TF-IDF"] = 1.0
        ensemble_mode = "TF-IDF + Logistic Regression Direct Assessment"
        weighted_fake_prob = tfidf_prob
    else:
        # No trained model available
        weights = {}
        ensemble_mode = "Heuristic Uncalibrated Default"
        weighted_fake_prob = 0.50
        explanation_notes.append("No trained model artifacts detected. Run training script to generate predictions.")

    # Normalize probability to 0.0 - 1.0 range
    fake_probability = float(min(max(weighted_fake_prob, 0.0), 1.0))
    real_probability = float(round(1.0 - fake_probability, 4))
    fake_probability = float(round(fake_probability, 4))

    # Determine final categorical prediction
    if fake_probability >= config.DECISION_THRESHOLD:
        prediction = "FAKE NEWS"
    else:
        prediction = "REAL NEWS"

    # Confidence assessment
    confidence = get_confidence_label(fake_probability)

    # Component scores summary
    component_scores = {
        "tfidf_fake_prob": round(tfidf_prob, 4) if tfidf_avail else None,
        "bert_fake_prob": round(bert_prob, 4) if bert_avail else None,
        "headline_similarity": round(headline_result.get("similarity_score", 0.0), 4) if headline_avail else None,
        "sentiment_compound": round(sentiment_result.get("compound", 0.0), 4) if sentiment_result else None
    }

    active_models = list(weights.keys())
    confidence_score = float(max(fake_probability, real_probability))

    return {
        "prediction": prediction,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "fake_percentage": round(fake_probability * 100.0, 1),
        "real_percentage": round(real_probability * 100.0, 1),
        "confidence": confidence,
        "confidence_score": confidence_score,
        "component_weights": weights,
        "component_scores": component_scores,
        "active_models": active_models,
        "ensemble_mode": ensemble_mode,
        "explanation_notes": explanation_notes
    }
