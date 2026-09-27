"""
Unit tests for src/scraper.py
"""

import pytest
from src.scraper import scrape_article_from_url


def test_empty_url():
    res = scrape_article_from_url("")
    assert res["success"] is False
    assert "empty" in res["error"].lower()


def test_invalid_domain_url():
    res = scrape_article_from_url("http://invalid-non-existent-domain-xyz-12345.org")
    assert res["success"] is False
    assert res["error"] is not None
