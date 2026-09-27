"""
Unit tests for src/sentiment_analyzer.py
"""

import pytest
from src.sentiment_analyzer import analyze_sentiment


def test_sentiment_positive_text():
    text = "The renewable energy project was a remarkable success, bringing joy, hope, and prosperity to thousands."
    res = analyze_sentiment(text)
    assert res["is_available"] is True
    assert res["positive"] > res["negative"]
    assert res["compound"] > 0
    assert res["label"] == "Positive"


def test_sentiment_negative_text():
    text = "A horrific tragedy struck the region causing catastrophic damages, widespread panic, and devastating failures."
    res = analyze_sentiment(text)
    assert res["is_available"] is True
    assert res["negative"] > res["positive"]
    assert res["compound"] < 0
    assert res["label"] == "Negative"


def test_sentiment_empty_text():
    res = analyze_sentiment("")
    assert res["label"] == "Neutral"
    assert res["compound"] == 0.0
