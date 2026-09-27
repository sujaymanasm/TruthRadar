"""
TruthRadar - BERT Classifier Module
-----------------------------------
Implements inference for Transformer-based Fake News classification using
DistilBERT (distilbert-base-uncased) fine-tuned on news authenticity corpora.

Supports automatic GPU/CUDA acceleration with smooth CPU fallback.
Gracefully handles missing or un-trained models without crashing.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
import config
from src.preprocessor import clean_text
from src.utils import logger


class BertFakeNewsClassifier:
    """
    HuggingFace DistilBERT classifier for deep contextual fake news detection.
    """

    def __init__(self, model_dir: Path = config.BERT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.load_status_message = ""
        self._load_model()

    def _load_model(self) -> bool:
        """Loads fine-tuned weights or sets graceful fallback status."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            # Check if custom fine-tuned model exists locally
            has_local_model = (
                self.model_dir.exists()
                and (self.model_dir / "config.json").exists()
                and (
                    (self.model_dir / "model.safetensors").exists()
                    or (self.model_dir / "pytorch_model.bin").exists()
                )
            )

            if has_local_model:
                logger.info(f"Loading local fine-tuned BERT model from {self.model_dir} on {self.device}...")
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                self.load_status_message = "Fine-tuned DistilBERT model active."
                return True
            else:
                self.is_loaded = False
                self.load_status_message = (
                    "Fine-tuned BERT weights not found in models/bert_model/. "
                    "Run `python training/train_bert.py` to fine-tune DistilBERT on your dataset."
                )
                logger.info(self.load_status_message)
                return False

        except Exception as e:
            self.is_loaded = False
            self.load_status_message = f"BERT initialization notice: {e}"
            logger.warning(self.load_status_message)
            return False

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Runs transformer inference to predict fake vs real probabilities.

        Returns:
            Dict containing:
                - is_available: bool
                - prediction: 'FAKE' or 'REAL' (or 'UNAVAILABLE')
                - fake_probability: float (0.0 to 1.0)
                - real_probability: float (0.0 to 1.0)
                - device: 'cuda' or 'cpu'
                - status_message: str
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            return {
                "is_available": False,
                "prediction": "UNAVAILABLE",
                "fake_probability": 0.50,
                "real_probability": 0.50,
                "device": str(self.device),
                "status_message": self.load_status_message
            }

        try:
            cleaned = clean_text(text)
            inputs = self.tokenizer(
                cleaned,
                max_length=config.BERT_MAX_LEN,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

            real_prob = float(probs[0])
            fake_prob = float(probs[1])
            pred_label = "FAKE" if fake_prob >= config.DECISION_THRESHOLD else "REAL"

            return {
                "is_available": True,
                "prediction": pred_label,
                "fake_probability": round(fake_prob, 4),
                "real_probability": round(real_prob, 4),
                "device": str(self.device),
                "status_message": "BERT inference successful."
            }
        except Exception as e:
            logger.error(f"Error during BERT inference: {e}", exc_info=True)
            return {
                "is_available": False,
                "prediction": "ERROR",
                "fake_probability": 0.50,
                "real_probability": 0.50,
                "device": str(self.device),
                "status_message": f"BERT inference error: {e}"
            }


# Singleton helper
_bert_instance: Optional[BertFakeNewsClassifier] = None


def get_bert_classifier(reload: bool = False) -> BertFakeNewsClassifier:
    """Returns singleton instance of BertFakeNewsClassifier."""
    global _bert_instance
    if _bert_instance is None or reload:
        _bert_instance = BertFakeNewsClassifier()
    return _bert_instance
