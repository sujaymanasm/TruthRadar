"""
Unit tests for src/headline_checker.py
"""

import pytest
from src.headline_checker import check_headline_body_consistency


def test_high_consistency_headline_and_body():
    headline = "NASA James Webb Space Telescope Detects Water on Exoplanet"
    body = "NASA astronomers using the James Webb Space Telescope have discovered clear signatures of water vapor and methane in the atmosphere of an exoplanet 120 light years away."
    res = check_headline_body_consistency(headline, body)
    assert res["is_available"] is True
    assert res["similarity_score"] > 0.10
    assert 0.0 <= res["similarity_score"] <= 1.0


def test_low_consistency_mismatched_headline_and_body():
    headline = "Ancient Giant Dinosaurs Found Alive in Subterranean Caves"
    body = "The Federal Reserve lowered benchmark interest rates by a quarter point today following an extensive meeting regarding consumer inflation trends."
    res = check_headline_body_consistency(headline, body)
    assert res["is_available"] is True
    assert 0.0 <= res["similarity_score"] <= 1.0


def test_empty_headline_or_body():
    res = check_headline_body_consistency("", "Some body text")
    assert res["is_available"] is False
