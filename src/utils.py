"""
TruthRadar - Utilities Module
-----------------------------
Helper functions for data handling, logging, text validation,
formatting, and metrics calculation.
"""

import re
import logging
from typing import Dict, Any, Tuple
import config

# Configure standardized logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TruthRadar")


def validate_text(text: Any) -> Tuple[bool, str]:
    """
    Validates input text for minimum length and valid string representation.

    Args:
        text: Input text string or object.

    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if text is None:
        return False, "No text provided. Please enter or paste article content."
    
    if not isinstance(text, str):
        text = str(text)
    
    cleaned = text.strip()
    if not cleaned:
        return False, "Input text is empty. Please enter news content to analyze."
    
    if len(cleaned) < config.MIN_TEXT_LENGTH:
        return (
            False,
            f"Input text is too short ({len(cleaned)} chars). "
            f"Please provide at least {config.MIN_TEXT_LENGTH} characters for reliable analysis."
        )
    
    return True, ""


def truncate_text(text: str, max_length: int = config.MAX_TEXT_LENGTH) -> str:
    """
    Truncates overly long text to prevent memory/performance degradation.
    Preserves whole words when possible.
    """
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    # Cut back to last space
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "..."


def get_confidence_label(probability: float) -> str:
    """
    Determines confidence tier based on probability distance from decision boundary (0.50).

    Args:
        probability: Float between 0.0 and 1.0 (typically fake_probability)

    Returns:
        Confidence string: 'High', 'Medium', or 'Low'
    """
    margin = abs(probability - 0.50)
    if margin >= config.CONFIDENCE_HIGH_MARGIN:
        return "High"
    elif margin >= config.CONFIDENCE_MEDIUM_MARGIN:
        return "Medium"
    else:
        return "Low"


def format_percentage(val: float) -> str:
    """Formats float (0-1 or 0-100) into a clean percentage string."""
    if val <= 1.0:
        return f"{val * 100:.1f}%"
    return f"{val:.1f}%"


def clean_url(url: str) -> str:
    """Normalizes and validates a URL string."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url
