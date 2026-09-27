"""
TruthRadar - Named Entity Recognition (NER) Module
--------------------------------------------------
Extracts real-world named entities (Persons, Organizations, Geopolitical Entities,
Dates, Facilities, Events, etc.) from news articles using spaCy.

Provides fallback regex-based entity extraction if the spaCy model is not installed.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict
import re
import config
from src.preprocessor import clean_text
from src.utils import logger

# Lazy singleton for spaCy model
_spacy_nlp = None
_spacy_model_checked = False

# Relevant entity labels for news analysis
TARGET_ENTITY_TYPES = {
    "PERSON": "People, public figures, politicians",
    "ORG": "Companies, agencies, institutions, parties",
    "GPE": "Countries, cities, states, nations",
    "LOC": "Non-GPE geographic locations, mountain ranges, water bodies",
    "DATE": "Absolute or relative dates or periods",
    "EVENT": "Named hurricanes, battles, wars, sports events",
    "MONEY": "Monetary values, including currencies",
    "NORP": "Nationalities or religious or political groups",
    "PRODUCT": "Objects, vehicles, foods, software, hardware"
}


def _get_spacy_nlp():
    """Attempts to load en_core_web_sm spaCy model."""
    global _spacy_nlp, _spacy_model_checked
    if not _spacy_model_checked:
        try:
            import spacy
            try:
                _spacy_nlp = spacy.load(config.SPACY_MODEL_NAME)
                logger.info(f"Loaded spaCy model: {config.SPACY_MODEL_NAME}")
            except OSError:
                logger.warning(
                    f"spaCy model '{config.SPACY_MODEL_NAME}' not found. "
                    f"Run `python -m spacy download {config.SPACY_MODEL_NAME}` for complete NER. "
                    "Using fallback entity extractor in the meantime."
                )
                _spacy_nlp = None
        except ImportError:
            logger.warning("spaCy is not installed. Fallback regex entity extraction will be used.")
            _spacy_nlp = None
        _spacy_model_checked = True
    return _spacy_nlp


def _regex_fallback_ner(text: str) -> Dict[str, List[str]]:
    """Simple regex fallback to extract capitalized proper nouns when spaCy is unavailable."""
    entities = defaultdict(set)
    
    # Extract capitalized proper names (2-3 consecutive capitalized words)
    name_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b"
    names = re.findall(name_pattern, text)
    for name in set(names):
        if len(name.split()) == 2:
            entities["PERSON / ORG"].add(name)
    
    # Extract dates (e.g., January 2024, 2026, Sept 15)
    date_pattern = r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s+\d{4})?|\b20\d{2}\b"
    dates = re.findall(date_pattern, text)
    for d in set(dates):
        entities["DATE"].add(d)

    # Extract monetary amounts
    money_pattern = r"\$\s*\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?|\b\d+\s*dollars\b"
    money = re.findall(money_pattern, text, flags=re.IGNORECASE)
    for m in set(money):
        entities["MONEY"].add(m)

    return {k: sorted(list(v)) for k, v in entities.items()}


def extract_entities(text: str) -> Dict[str, Any]:
    """
    Extracts named entities from article text.

    Args:
        text: Article or headline text.

    Returns:
        Dict containing:
            - is_available: bool
            - engine: 'spaCy (en_core_web_sm)' or 'Pattern-based Fallback'
            - total_entities: int
            - entities_by_type: Dict[str, List[str]]
            - entity_list: List[Dict[str, str]]
            - description: str
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "is_available": False,
            "engine": "None",
            "total_entities": 0,
            "entities_by_type": {},
            "entity_list": [],
            "description": "No text provided for entity extraction."
        }

    nlp = _get_spacy_nlp()

    if nlp is not None:
        doc = nlp(cleaned[:5000])  # Cap at 5000 chars for responsive UI
        entities_by_type = defaultdict(set)
        entity_list = []

        for ent in doc.ents:
            if ent.label_ in TARGET_ENTITY_TYPES:
                ent_text = ent.text.strip()
                if len(ent_text) > 1 and not ent_text.isnumeric():
                    entities_by_type[ent.label_].add(ent_text)
                    entity_list.append({
                        "text": ent_text,
                        "label": ent.label_,
                        "description": TARGET_ENTITY_TYPES.get(ent.label_, ent.label_)
                    })

        # Deduplicate & sort
        formatted_entities = {k: sorted(list(v)) for k, v in entities_by_type.items()}
        total_count = sum(len(v) for v in formatted_entities.values())

        return {
            "is_available": True,
            "engine": f"spaCy ({config.SPACY_MODEL_NAME})",
            "total_entities": total_count,
            "entities_by_type": formatted_entities,
            "entity_list": entity_list,
            "description": f"Extracted {total_count} named entities across {len(formatted_entities)} categories."
        }
    else:
        # Fallback
        fallback_ents = _regex_fallback_ner(cleaned)
        total_count = sum(len(v) for v in fallback_ents.values())
        return {
            "is_available": True,
            "engine": "Pattern-based Fallback (Run `python -m spacy download en_core_web_sm` for full NER)",
            "total_entities": total_count,
            "entities_by_type": fallback_ents,
            "entity_list": [
                {"text": item, "label": label, "description": label}
                for label, items in fallback_ents.items()
                for item in items
            ],
            "description": f"Extracted {total_count} candidate entities via pattern matching."
        }
