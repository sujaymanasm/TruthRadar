"""
Unit tests for src/fake_score.py
"""

import pytest
from src.fake_score import calculate_fake_score


def test_calculate_fake_score_high_fake():
    tfidf_res = {"is_available": True, "fake_probability": 0.88, "real_probability": 0.12}
    bert_res = {"is_available": True, "fake_probability": 0.92, "real_probability": 0.08}
    headline_res = {"is_available": True, "similarity_score": 0.20}  # Low similarity -> high fake signal
    sent_res = {"compound": -0.60, "label": "Negative"}

    res = calculate_fake_score(tfidf_res, bert_res, headline_res, sent_res, has_headline=True)
    assert res["prediction"] == "FAKE NEWS"
    assert res["fake_probability"] > 0.70
    assert abs(res["fake_probability"] + res["real_probability"] - 1.0) < 1e-3
    assert res["confidence"] == "High"


def test_calculate_fake_score_high_real():
    tfidf_res = {"is_available": True, "fake_probability": 0.10, "real_probability": 0.90}
    bert_res = {"is_available": True, "fake_probability": 0.15, "real_probability": 0.85}
    headline_res = {"is_available": True, "similarity_score": 0.85}  # High similarity
    sent_res = {"compound": 0.10, "label": "Positive"}

    res = calculate_fake_score(tfidf_res, bert_res, headline_res, sent_res, has_headline=True)
    assert res["prediction"] == "REAL NEWS"
    assert res["real_probability"] > 0.70
    assert abs(res["fake_probability"] + res["real_probability"] - 1.0) < 1e-3


def test_calculate_fake_score_fallback_mode():
    tfidf_res = {"is_available": True, "fake_probability": 0.30, "real_probability": 0.70}
    bert_res = {"is_available": False}
    headline_res = {"is_available": False}

    res = calculate_fake_score(tfidf_res, bert_res, headline_res, has_headline=False)
    assert res["prediction"] == "REAL NEWS"
    assert res["fake_probability"] == 0.30
    assert res["real_probability"] == 0.70
