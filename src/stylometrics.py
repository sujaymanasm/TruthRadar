"""
src/stylometrics.py - Stylometric & Psycholinguistic Text Analysis Engine
------------------------------------------------------------------------
Computes linguistic markers proven to correlate with misinformation:
- Lexical Richness / Diversity (Type-Token Ratio - TTR)
- Readability Indices (Flesch Reading Ease & Flesch-Kincaid Grade Level)
- Sensationalism & Affective Arousal (All-Caps ratio, exclamation density, intensifiers)
- Structural Syntactic Density (Average Sentence Length, Syllable count)
"""

import re
import math
from typing import Dict, Any

# Sensationalist emotional intensifiers frequently overrepresented in deceptive text
SENSATIONAL_INTENSIFIERS = {
    "shocking", "unbelievable", "mind-blowing", "bombshell", "conspiracy",
    "exposed", "secret", "horrifying", "miracle", "disaster", "devastating",
    "unprecedented", "catastrophic", "traitor", "hidden", "panic", "banned",
    "censored", "insane", "nightmare", "furious", "explosive", "urgent"
}

def count_syllables(word: str) -> int:
    """Estimates syllable count using heuristic vowel group identification."""
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    # Remove trailing 'e', 'es', 'ed' if not preceded by single vowel
    word = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', word)
    vowels = re.findall(r'[aeiouy]+', word)
    return max(1, len(vowels))

def analyze_stylometrics(text: str) -> Dict[str, Any]:
    """
    Extracts stylometric and readability metrics from raw or cleaned text.
    
    Returns:
        dict: {
            "word_count": int,
            "sentence_count": int,
            "avg_sentence_length": float,
            "type_token_ratio": float (0-1),
            "flesch_reading_ease": float (0-100),
            "flesch_kincaid_grade": float,
            "sensationalism_score": float (0-100),
            "caps_ratio": float (0-1),
            "exclamation_count": int,
            "intensifier_count": int,
            "intensifiers_found": list,
            "readability_label": str
        }
    """
    if not text or not text.strip():
        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_sentence_length": 0.0,
            "type_token_ratio": 0.0,
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "sensationalism_score": 0.0,
            "caps_ratio": 0.0,
            "exclamation_count": 0,
            "intensifier_count": 0,
            "intensifiers_found": [],
            "readability_label": "N/A"
        }

    raw_text = text.strip()
    
    # Sentences split
    sentences = [s.strip() for s in re.split(r'[.!?]+', raw_text) if s.strip()]
    sentence_count = max(1, len(sentences))
    
    # Words extraction
    words = re.findall(r'\b[A-Za-z0-9\'-]+\b', raw_text)
    word_count = max(1, len(words))
    
    # 1. Lexical Richness (Type-Token Ratio)
    lower_words = [w.lower() for w in words]
    unique_words = set(lower_words)
    ttr = round(len(unique_words) / word_count, 3)
    
    # 2. Syllables calculation
    total_syllables = sum(count_syllables(w) for w in words)
    avg_sentence_length = round(word_count / sentence_count, 2)
    avg_syllables_per_word = total_syllables / word_count
    
    # 3. Flesch Reading Ease Formula:
    # 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    fre = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    fre = max(0.0, min(100.0, round(fre, 1)))
    
    # Flesch-Kincaid Grade Level:
    # 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
    fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
    fk_grade = max(1.0, round(fk_grade, 1))
    
    # Readability Label
    if fre >= 80:
        readability_label = "Very Easy (Grade 5-6)"
    elif fre >= 60:
        readability_label = "Standard / Accessible (Grade 7-8)"
    elif fre >= 40:
        readability_label = "Challenging / College Level"
    else:
        readability_label = "Academic / Complex Dense"
        
    # 4. Sensationalism & Emotional Arousal Markers
    # Exclamation mark density
    exclamation_count = raw_text.count('!')
    question_count = raw_text.count('?')
    
    # All-caps words (ignoring single-letter words like 'I' or 'A')
    caps_words = [w for w in words if len(w) > 1 and w.isupper()]
    caps_ratio = round(len(caps_words) / word_count, 3)
    
    # Intensifiers detection
    found_intensifiers = [w for w in lower_words if w in SENSATIONAL_INTENSIFIERS]
    intensifier_count = len(found_intensifiers)
    
    # Composite Sensationalism Index (0 - 100)
    # Penalizes excessive caps, multiple exclamations, and sensational trigger words
    raw_sensational_score = (
        (caps_ratio * 150) + 
        (min(exclamation_count, 10) * 4) + 
        (min(intensifier_count, 10) * 5)
    )
    sensationalism_score = round(min(100.0, max(0.0, raw_sensational_score)), 1)
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "type_token_ratio": ttr,
        "flesch_reading_ease": fre,
        "flesch_kincaid_grade": fk_grade,
        "sensationalism_score": sensationalism_score,
        "caps_ratio": caps_ratio,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "intensifier_count": intensifier_count,
        "intensifiers_found": list(set(found_intensifiers))[:8],
        "readability_label": readability_label
    }
