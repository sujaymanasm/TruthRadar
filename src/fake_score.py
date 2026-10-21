"""
TruthRadar - Fake Score Multi-Signal Ensemble Engine
---------------------------------------------------
Synthesizes predictions from multiple NLP models (TF-IDF + Logistic Regression,
DistilBERT Transformer, Semantic Headline-Body Alignment, Sentiment Polarity,
Psycholinguistic Stylometrics, and Domain Reputation) into a unified,
calibrated authenticity assessment.
"""

from typing import Dict, Any, Optional
import config
from src.utils import get_confidence_label, logger


def calculate_fake_score(
    tfidf_result: Dict[str, Any],
    bert_result: Optional[Dict[str, Any]] = None,
    headline_result: Optional[Dict[str, Any]] = None,
    sentiment_result: Optional[Dict[str, Any]] = None,
    has_headline: bool = True,
    domain_result: Optional[Dict[str, Any]] = None,
    stylometrics_result: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Computes weighted ensemble fake news probability and verdict.
    Safely accepts **kwargs to handle any cache/hot-reload mismatch.
    """
    # Fallback to kwargs if passed dynamically
    if domain_result is None and "domain_result" in kwargs:
        domain_result = kwargs["domain_result"]
    if stylometrics_result is None and "stylometrics_result" in kwargs:
        stylometrics_result = kwargs["stylometrics_result"]

    tfidf_avail = tfidf_result.get("is_available", False) if isinstance(tfidf_result, dict) else False
    bert_avail = bert_result.get("is_available", False) if (isinstance(bert_result, dict) and bert_result) else False
    headline_avail = headline_result.get("is_available", False) if (isinstance(headline_result, dict) and headline_result and has_headline) else False

    # Extract component fake probabilities
    tfidf_prob = tfidf_result.get("fake_probability", 0.50) if tfidf_avail else 0.50
    bert_prob = bert_result.get("fake_probability", 0.50) if bert_avail else 0.50

    # Invert headline similarity with non-linear clickbait calibration
    if headline_avail:
        sim_score = headline_result.get("similarity_score", 0.70)
        if sim_score >= 0.50:
            headline_fake_signal = 0.15 # Strong alignment, very low fake risk
        elif sim_score <= 0.20:
            headline_fake_signal = 0.85 # Clickbait / topical mismatch
        else:
            headline_fake_signal = 0.85 - ((sim_score - 0.20) / 0.30) * 0.70
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
        explanation_notes.append("DistilBERT model weights not detected; calibrated to TF-IDF baseline.")
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
        explanation_notes.append("No trained model artifacts detected.")

    # -------------------------------------------------------------
    # Multi-Signal Contextual Calibration (Domain & Stylometrics)
    # -------------------------------------------------------------
    sens_score = stylometrics_result.get("sensationalism_score", 0.0) if isinstance(stylometrics_result, dict) else 0.0
    is_satire = domain_result.get("is_satire", False) if isinstance(domain_result, dict) else False
    trust_tier = domain_result.get("trust_tier", "") if isinstance(domain_result, dict) else ""

    # 1. Verified Journalism Calibration
    if "High Trust" in trust_tier and sens_score < 30.0:
        weighted_fake_prob = weighted_fake_prob * 0.52
        explanation_notes.append("Credibility calibration: Verified Tier-1 Publisher combined with measured, non-sensational phrasing.")

    # 2. Documented Disinformation Calibration
    if "High Risk" in trust_tier:
        weighted_fake_prob = min(1.0, weighted_fake_prob * 1.30 + 0.15)
        explanation_notes.append("Deception risk elevated: Domain is on accredited misinformation/conspiracy registries.")

    # 3. Stylometric Sensationalism Penalty
    if sens_score > 60.0:
        weighted_fake_prob = min(1.0, weighted_fake_prob + 0.10)
        explanation_notes.append("Deception risk elevated: High sensationalism index detected (shouting caps & extreme intensifiers).")

    # Normalize probability to 0.0 - 1.0 range
    fake_probability = float(min(max(weighted_fake_prob, 0.0), 1.0))
    real_probability = float(round(1.0 - fake_probability, 4))
    fake_probability = float(round(fake_probability, 4))

    # Determine final categorical prediction
    if is_satire:
        prediction = "SATIRE / PARODY"
    elif fake_probability >= config.DECISION_THRESHOLD:
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
        "sentiment_compound": round(sentiment_result.get("compound", 0.0), 4) if isinstance(sentiment_result, dict) else None,
        "sensationalism_score": sens_score
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
