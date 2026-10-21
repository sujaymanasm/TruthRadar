"""
tests/test_domain_trust.py - Unit tests for domain credibility and satire detection
"""

from src.domain_trust import evaluate_domain_trust

def test_satire_domain():
    res = evaluate_domain_trust("https://www.theonion.com/article-some-funny-news")
    assert res["is_satire"] is True
    assert "Satire" in res["trust_tier"]

def test_verified_domain():
    res = evaluate_domain_trust("https://www.reuters.com/world/asia-pacific/")
    assert res["is_satire"] is False
    assert res["trust_score"] > 80
    assert "High Trust" in res["trust_tier"]

def test_flagged_domain():
    res = evaluate_domain_trust("https://infowars.com/breaking-conspiracy")
    assert res["trust_score"] < 30
    assert "Misinformation" in res["trust_tier"] or "High Risk" in res["trust_tier"]

def test_unknown_domain():
    res = evaluate_domain_trust("https://random-news-portal-example.org")
    assert res["trust_tier"] == "Independent / Unrated Domain"

def test_empty_input():
    res = evaluate_domain_trust("")
    assert res["domain"] == "Unknown"
