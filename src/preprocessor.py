"""
TruthRadar - Text Preprocessor Module
-------------------------------------
Provides text cleaning, normalization, and tokenization utilities.
Supports dual preprocessing modes:
1. Minimal Preprocessing (for Transformers/BERT) - Preserves casing, grammar, and sentence structure.
2. Full ML Preprocessing (for TF-IDF + Logistic Regression) - Cleans, standardizes, and strips noise.
"""

import re
import html
import unicodedata
from typing import Optional, Tuple
from src.utils import validate_text, logger


def clean_html(text: str) -> str:
    """Removes HTML tags and unescapes HTML entities."""
    if not text:
        return ""
    # Unescape HTML entities (e.g., &amp; -> &, &quot; -> ")
    text = html.unescape(text)
    # Remove HTML tags using regex
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_whitespace(text)


def remove_urls(text: str) -> str:
    """Removes http/https/www URLs from text."""
    if not text:
        return ""
    # Matches http/https URLs and www. URLs
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, " ", text)


def normalize_whitespace(text: str) -> str:
    """Collapses multiple spaces, tabs, and newlines into a single clean space."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_unicode(text: str) -> str:
    """Normalizes accented/special unicode characters into standard ASCII/NFKD."""
    if not text:
        return ""
    # Decompose unicode and keep standard ascii-like characters
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def clean_text(text: Optional[str]) -> str:
    """
    Minimally cleans text while preserving natural punctuation and casing.
    Suitable for BERT tokenization, NER, and Sentiment Analysis.

    Pipeline:
    1. Handle None / empty
    2. Remove HTML tags & unescape entities
    3. Normalize unicode characters
    4. Remove excessive URLs
    5. Normalize whitespaces
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = clean_html(text)
    text = normalize_unicode(text)
    text = remove_urls(text)
    text = normalize_whitespace(text)
    return text


def preprocess_for_ml(text: Optional[str]) -> str:
    """
    Extensively cleans and normalizes text for traditional Bag-of-Words / TF-IDF models.
    
    Pipeline:
    1. Minimal clean (HTML, unicode, URLs)
    2. Lowercase conversion
    3. Punctuation handling (keep alphanumeric and basic spaces)
    4. Remove standalone digits and special symbols
    5. Whitespace normalization
    """
    if not text:
        return ""
    
    cleaned = clean_text(text).lower()
    
    # Replace non-alphanumeric characters with spaces (preserve word boundaries)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    
    # Remove isolated numbers/standalone digits
    cleaned = re.sub(r"\b\d+\b", " ", cleaned)
    
    # Normalize whitespaces
    cleaned = normalize_whitespace(cleaned)
    return cleaned


def extract_headline_and_body(raw_text: str) -> Tuple[str, str]:
    """
    Extracts or separates headline and body from combined text if possible.
    If first line is short, treats it as headline; rest as body.
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return "", ""
    if len(lines) == 1:
        # Only one line/paragraph
        first_sentence = re.split(r"(?<=[.!?])\s+", lines[0])[0]
        if len(first_sentence) < 150 and len(lines[0]) > len(first_sentence):
            return first_sentence, lines[0][len(first_sentence):].strip()
        return lines[0], lines[0]
    
    first_line = lines[0]
    body = " ".join(lines[1:])
    return first_line, body
