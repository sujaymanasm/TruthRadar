"""
TruthRadar - Explainable AI (LIME) Module
-----------------------------------------
Provides transparent, interpretable explanations for predictions made by
the TF-IDF + Logistic Regression fake news classification pipeline.

Uses LIME (Local Interpretable Model-agnostic Explanations) along with
direct model coefficient attribution to visualize feature importance.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
import config
from src.tfidf_classifier import get_tfidf_classifier
from src.preprocessor import preprocess_for_ml
from src.utils import logger


def _get_direct_tfidf_feature_weights(text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Direct model coefficient attribution fallback.
    Inspects which words in the text have the highest positive or negative weights
    in the trained Logistic Regression model.
    """
    classifier_wrapper = get_tfidf_classifier()
    if not classifier_wrapper.is_loaded:
        return []

    model = classifier_wrapper.model
    vectorizer = classifier_wrapper.vectorizer

    cleaned = preprocess_for_ml(text)
    if not cleaned:
        return []

    # Transform text
    vec = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]  # Positive coef -> FAKE, Negative coef -> REAL

    # Non-zero indices in current document
    nonzero_indices = vec.nonzero()[1]
    if len(nonzero_indices) == 0:
        return []

    word_scores = []
    for idx in nonzero_indices:
        word = feature_names[idx]
        tfidf_val = vec[0, idx]
        weight = float(coefficients[idx] * tfidf_val)
        word_scores.append({
            "word": word,
            "weight": round(weight, 4),
            "supports": "FAKE" if weight > 0 else "REAL",
            "abs_weight": abs(weight)
        })

    # Sort by absolute influence
    word_scores.sort(key=lambda x: x["abs_weight"], reverse=True)
    return word_scores[:top_k]


def explain_prediction(text: str, num_features: int = config.LIME_NUM_FEATURES) -> Dict[str, Any]:
    """
    Explains the classification decision for a given news text using LIME.

    Args:
        text: Article or headline text.
        num_features: Number of top contributing words to extract.

    Returns:
        Dict containing:
            - is_available: bool
            - method: 'LIME (Local Explanations)' or 'Direct Coefficient Attribution'
            - explanation_list: List of dicts (word, weight, supports)
            - fake_contributing_words: List of (word, weight)
            - real_contributing_words: List of (word, weight)
            - explanation_summary: str
    """
    classifier_wrapper = get_tfidf_classifier()
    if not classifier_wrapper.is_loaded:
        return {
            "is_available": False,
            "method": "None",
            "explanation_list": [],
            "fake_contributing_words": [],
            "real_contributing_words": [],
            "explanation_summary": "TF-IDF model is not loaded. Train the model to generate explanations."
        }

    try:
        from lime.lime_text import LimeTextExplainer

        # Custom predictor function for LIME: takes list of raw strings, returns proba array (N, 2)
        def predictor_fn(texts):
            return classifier_wrapper.predict_proba_raw(texts)

        explainer = LimeTextExplainer(
            class_names=["REAL", "FAKE"],
            split_expression=r"\W+",
            random_state=config.RANDOM_STATE
        )

        exp = explainer.explain_instance(
            text,
            predictor_fn,
            num_features=num_features,
            num_samples=config.LIME_NUM_SAMPLES,
            labels=[1]  # Explain class 1 (FAKE)
        )

        # List of (word, weight) tuples for FAKE class
        raw_exp_list = exp.as_list(label=1)

        explanation_list = []
        fake_words = []
        real_words = []

        for word, weight in raw_exp_list:
            supports = "FAKE" if weight > 0 else "REAL"
            w_str = str(word)
            item = {
                "word": w_str,
                "weight": round(float(weight), 4),
                "supports": supports,
                "abs_weight": round(abs(float(weight)), 4)
            }
            explanation_list.append(item)
            if weight > 0:
                fake_words.append((w_str, round(float(weight), 4)))
            else:
                real_words.append((w_str, round(float(weight), 4)))

        summary = (
            f"LIME identified {len(fake_words)} feature(s) pushing toward FAKE news classification "
            f"and {len(real_words)} feature(s) supporting REAL news classification."
        )

        return {
            "is_available": True,
            "method": "LIME (Local Interpretable Model-agnostic Explanations)",
            "explanation_list": explanation_list,
            "fake_contributing_words": fake_words,
            "real_contributing_words": real_words,
            "explanation_summary": summary
        }

    except Exception as e:
        logger.warning(f"LIME explanation failed ({e}). Using direct TF-IDF coefficient attribution.")
        direct_weights = _get_direct_tfidf_feature_weights(text, top_k=num_features)
        
        fake_words = [(d["word"], d["weight"]) for d in direct_weights if d["weight"] > 0]
        real_words = [(d["word"], d["weight"]) for d in direct_weights if d["weight"] < 0]

        return {
            "is_available": len(direct_weights) > 0,
            "method": "Direct TF-IDF Model Coefficient Attribution",
            "explanation_list": direct_weights,
            "fake_contributing_words": fake_words,
            "real_contributing_words": real_words,
            "explanation_summary": "Direct linear feature weights computed from model vocabulary."
        }


def create_explanation_chart(explanation_data: Dict[str, Any]) -> Optional[go.Figure]:
    """
    Builds an interactive horizontal bar chart visualizing word contributions using Plotly.

    Green bars = features driving prediction towards REAL.
    Red/Crimson bars = features driving prediction towards FAKE.
    """
    if not explanation_data.get("is_available") or not explanation_data.get("explanation_list"):
        return None

    items = explanation_data["explanation_list"]
    # Sort items by weight for intuitive top-to-bottom bar chart
    items_sorted = sorted(items, key=lambda x: x["weight"])

    words = [it["word"] for it in items_sorted]
    weights = [it["weight"] for it in items_sorted]
    colors = ["#EF4444" if w > 0 else "#10B981" for w in weights]
    hover_texts = [
        f"<b>Word:</b> {w}<br><b>Weight:</b> {val:+.4f}<br><b>Influences toward:</b> {'FAKE' if val > 0 else 'REAL'}"
        for w, val in zip(words, weights)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=words,
        x=weights,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(0,0,0,0.2)", width=1)
        ),
        hovertext=hover_texts,
        hoverinfo="text"
    ))

    fig.update_layout(
        title="<b>Feature Attribution (Word Importance)</b>",
        xaxis_title="Weight / Influence on Fake News Probability",
        yaxis_title="Influential Words / Phrases",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
        height=380,
        shapes=[
            dict(
                type="line",
                x0=0,
                x1=0,
                y0=-0.5,
                y1=len(words) - 0.5,
                line=dict(color="#6B7280", width=1.5, dash="dash")
            )
        ]
    )

    return fig
