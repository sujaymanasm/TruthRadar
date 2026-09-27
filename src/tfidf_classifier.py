"""
TruthRadar - TF-IDF + Logistic Regression Classifier Module
-----------------------------------------------------------
Implements loading, vectorization, inference, and evaluation for
the TF-IDF + Logistic Regression fake news classification baseline.
"""

import os
import joblib
from typing import Dict, Any, Optional, Tuple
import numpy as np
import config
from src.preprocessor import preprocess_for_ml
from src.utils import logger


class TfidfFakeNewsClassifier:
    """
    TF-IDF Vectorizer + Logistic Regression Classifier wrapper.
    Provides probability scores for REAL vs FAKE news classification.
    """

    def __init__(
        self,
        model_path=config.TFIDF_MODEL_PATH,
        vectorizer_path=config.TFIDF_VECTORIZER_PATH,
        label_encoder_path=config.LABEL_ENCODER_PATH,
    ):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.label_encoder_path = label_encoder_path
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self) -> bool:
        """Loads serialized model, vectorizer, and label encoder if they exist."""
        try:
            if (
                os.path.exists(self.model_path)
                and os.path.exists(self.vectorizer_path)
            ):
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                if os.path.exists(self.label_encoder_path):
                    self.label_encoder = joblib.load(self.label_encoder_path)
                self.is_loaded = True
                logger.info("TF-IDF model and vectorizer loaded successfully.")
                return True
            else:
                logger.warning(
                    f"TF-IDF model artifacts not found at {self.model_path} or {self.vectorizer_path}."
                )
                self.is_loaded = False
                return False
        except Exception as e:
            logger.error(f"Error loading TF-IDF artifacts: {e}", exc_info=True)
            self.is_loaded = False
            return False

    def predict_proba_raw(self, texts: list[str]) -> np.ndarray:
        """
        Calculates raw class probabilities for a list of input texts.
        Returns array of shape (N, 2) where col 0 = REAL, col 1 = FAKE.
        """
        if not self.is_loaded or self.model is None or self.vectorizer is None:
            raise RuntimeError("TF-IDF model is not loaded. Train or provide trained model artifacts.")

        cleaned_texts = [preprocess_for_ml(t) for t in texts]
        tfidf_vectors = self.vectorizer.transform(cleaned_texts)
        probabilities = self.model.predict_proba(tfidf_vectors)
        return probabilities

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predicts fake vs real probabilities and label for a single text.

        Returns:
            Dict containing:
                - is_available: bool
                - prediction: 'FAKE' or 'REAL'
                - fake_probability: float (0.0 to 1.0)
                - real_probability: float (0.0 to 1.0)
                - raw_scores: Dict[str, float]
        """
        if not self.is_loaded:
            # Fallback heuristic if model has not been trained yet
            return {
                "is_available": False,
                "prediction": "UNAVAILABLE",
                "fake_probability": 0.50,
                "real_probability": 0.50,
                "error": "TF-IDF model artifacts not found. Please run training/train_tfidf.py first."
            }

        try:
            probas = self.predict_proba_raw([text])[0]
            # Assumes class 0 = REAL, class 1 = FAKE
            real_prob = float(probas[0])
            fake_prob = float(probas[1])

            label = "FAKE" if fake_prob >= config.DECISION_THRESHOLD else "REAL"

            return {
                "is_available": True,
                "prediction": label,
                "fake_probability": round(fake_prob, 4),
                "real_probability": round(real_prob, 4),
                "raw_scores": {
                    "REAL": round(real_prob, 4),
                    "FAKE": round(fake_prob, 4)
                }
            }
        except Exception as e:
            logger.error(f"Prediction error in TF-IDF classifier: {e}", exc_info=True)
            return {
                "is_available": False,
                "prediction": "ERROR",
                "fake_probability": 0.50,
                "real_probability": 0.50,
                "error": str(e)
            }


# Singleton instance helper
_tfidf_instance: Optional[TfidfFakeNewsClassifier] = None


def get_tfidf_classifier(reload: bool = False) -> TfidfFakeNewsClassifier:
    """Returns singleton instance of TfidfFakeNewsClassifier."""
    global _tfidf_instance
    if _tfidf_instance is None or reload:
        _tfidf_instance = TfidfFakeNewsClassifier()
    return _tfidf_instance
