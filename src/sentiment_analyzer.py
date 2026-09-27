"""
TruthRadar - Sentiment Analysis Module
--------------------------------------
Uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner)
lexicon-based sentiment engine to evaluate the emotional tone, polarity,
and subjectivity of news articles.

NOTE: Sentiment is treated as an auxiliary diagnostic signal and is
never used alone to classify an article as fake or real.
"""

from typing import Dict, Any, Optional
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from src.utils import logger

# Lazy singleton for VADER analyzer
_vader_analyzer: Optional[SentimentIntensityAnalyzer] = None


def _init_vader() -> Optional[SentimentIntensityAnalyzer]:
    """Ensures vader_lexicon is downloaded and initializes analyzer."""
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            # Check or download NLTK vader_lexicon quietly
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except (LookupError, FileNotFoundError):
                nltk.download("vader_lexicon", quiet=True)
            
            _vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("NLTK VADER Sentiment Analyzer initialized successfully.")
        except Exception as e:
            logger.warning(f"Unable to initialize NLTK VADER: {e}")
            _vader_analyzer = None
    return _vader_analyzer


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyzes sentiment polarity of the provided text using VADER.

    Args:
        text: Article or headline text string.

    Returns:
        Dict containing:
            - is_available: bool
            - positive: float (0.0 to 1.0)
            - negative: float (0.0 to 1.0)
            - neutral: float (0.0 to 1.0)
            - compound: float (-1.0 to 1.0)
            - label: 'Positive', 'Negative', or 'Neutral'
            - tone_description: str
    """
    if not text or not text.strip():
        return {
            "is_available": False,
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "compound": 0.0,
            "label": "Neutral",
            "tone_description": "No text provided for sentiment analysis."
        }

    analyzer = _init_vader()
    if analyzer is None:
        return {
            "is_available": False,
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "compound": 0.0,
            "label": "Neutral",
            "tone_description": "Sentiment analyzer temporarily unavailable."
        }

    try:
        # VADER polarity scores
        scores = analyzer.polarity_scores(text)
        compound = float(scores["compound"])
        pos = float(scores["pos"])
        neg = float(scores["neg"])
        neu = float(scores["neu"])

        # Standard VADER compound score thresholds
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        # Descriptive contextual note
        if neg > 0.25 and compound <= -0.3:
            tone_desc = "Highly negative / sensationalized or alarming tone detected."
        elif pos > 0.25 and compound >= 0.3:
            tone_desc = "Strongly positive / promotional or celebratory tone."
        else:
            tone_desc = "Balanced or objective informational tone."

        return {
            "is_available": True,
            "positive": round(pos, 3),
            "negative": round(neg, 3),
            "neutral": round(neu, 3),
            "compound": round(compound, 3),
            "label": label,
            "tone_description": tone_desc
        }
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}", exc_info=True)
        return {
            "is_available": False,
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "compound": 0.0,
            "label": "Neutral",
            "tone_description": f"Sentiment analysis encountered an error: {e}"
        }
