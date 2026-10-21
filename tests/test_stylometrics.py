"""
tests/test_stylometrics.py - Unit tests for stylometrics and readability engine
"""

from src.stylometrics import analyze_stylometrics, count_syllables

def test_count_syllables():
    assert count_syllables("the") == 1
    assert count_syllables("breaking") == 2
    assert count_syllables("unbelievable") >= 4

def test_analyze_stylometrics_basic():
    text = "Scientists at the university announced a new clean energy breakthrough."
    res = analyze_stylometrics(text)
    assert res["word_count"] > 5
    assert res["sentence_count"] >= 1
    assert 0 <= res["type_token_ratio"] <= 1.0
    assert res["sensationalism_score"] < 50

def test_sensationalism_detection():
    fake_text = "SHOCKING! UNBELIEVABLE secret miracle cure exposed!! Big Pharma panic!"
    res = analyze_stylometrics(fake_text)
    assert res["sensationalism_score"] > 25
    assert res["exclamation_count"] >= 2
    assert res["intensifier_count"] >= 2
    assert len(res["intensifiers_found"]) > 0

def test_empty_text():
    res = analyze_stylometrics("")
    assert res["word_count"] == 0
    assert res["type_token_ratio"] == 0.0
