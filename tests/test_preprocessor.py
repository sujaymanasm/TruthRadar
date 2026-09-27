"""
Unit tests for src/preprocessor.py and src/utils.py
"""

import pytest
from src.preprocessor import clean_text, preprocess_for_ml, clean_html, remove_urls, extract_headline_and_body
from src.utils import validate_text, truncate_text, get_confidence_label, format_percentage


def test_clean_html():
    html_sample = "<p>This is a <b>bold</b> statement &amp; report.</p>"
    result = clean_html(html_sample)
    assert "<b>" not in result
    assert "</p>" not in result
    assert "&amp;" not in result
    assert "&" in result
    assert "bold statement & report" in result


def test_remove_urls():
    url_sample = "Check this link https://example.com/news?id=123 for full story."
    result = remove_urls(url_sample)
    assert "https://example.com" not in result
    assert "Check this link" in result


def test_clean_text_bert_friendly():
    raw = "<h1>Breaking News!</h1> Visit https://fake.com. Dr. Smith said, 'All is well.'   "
    cleaned = clean_text(raw)
    assert "Breaking News!" in cleaned
    assert "Dr. Smith said, 'All is well.'" in cleaned
    assert "https://" not in cleaned
    assert "  " not in cleaned


def test_preprocess_for_ml():
    raw = "<p>Scientists discovered 100 NEW methods! Visit https://link.com</p>"
    cleaned = preprocess_for_ml(raw)
    assert cleaned.islower()
    assert "scientists" in cleaned
    assert "discovered" in cleaned
    assert "methods" in cleaned
    assert "<p>" not in cleaned


def test_validate_text_empty_and_short():
    assert validate_text(None)[0] is False
    assert validate_text("")[0] is False
    assert validate_text("   ")[0] is False
    assert validate_text("Short text.")[0] is False
    
    long_text = "This is a sufficiently long news article body designed for testing validation routines."
    assert validate_text(long_text)[0] is True


def test_extract_headline_and_body():
    multiline = "First Line Headline\n\nThis is the detailed article body that follows the headline."
    h, b = extract_headline_and_body(multiline)
    assert h == "First Line Headline"
    assert "detailed article body" in b


def test_confidence_labels():
    assert get_confidence_label(0.95) == "High"
    assert get_confidence_label(0.05) == "High"
    assert get_confidence_label(0.70) == "Medium"
    assert get_confidence_label(0.52) == "Low"


def test_format_percentage():
    assert format_percentage(0.852) == "85.2%"
    assert format_percentage(85.2) == "85.2%"
