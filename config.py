"""
TruthRadar - Configuration Module
----------------------------------
Centralized configuration parameters for paths, model hyper-parameters,
ensemble weights, thresholds, and scraping parameters.
"""

from pathlib import Path

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

# Data file paths
SAMPLE_DATASET_PATH = DATA_DIR / "sample_news.csv"
FULL_DATASET_PATH = DATA_DIR / "fake_news.csv"

# Model artifact paths
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_model.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
BERT_MODEL_DIR = MODELS_DIR / "bert_model"

# Ensure essential directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# PRETRAINED MODEL IDENTIFIERS
# =====================================================================
# Hugging Face lightweight transformer for BERT classification
BERT_PRETRAINED_NAME = "distilbert-base-uncased"

# Sentence Transformer for headline-body semantic similarity
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# spaCy language model for Named Entity Recognition
SPACY_MODEL_NAME = "en_core_web_sm"

# =====================================================================
# ENSEMBLE SCORING WEIGHTS & PARAMETERS
# =====================================================================
# The weights determine the contribution of primary model signals to the final score.
# Weights must sum to 1.0 when both TF-IDF and BERT are available.
TFIDF_WEIGHT = 0.35
BERT_WEIGHT = 0.45
HEADLINE_WEIGHT = 0.20

# Fallback weights when BERT is unavailable:
FALLBACK_TFIDF_WEIGHT = 0.70
FALLBACK_HEADLINE_WEIGHT = 0.30

# Fallback when no headline is provided (Headline-Body check skipped):
NO_HEADLINE_TFIDF_WEIGHT = 0.40
NO_HEADLINE_BERT_WEIGHT = 0.60
NO_HEADLINE_FALLBACK_TFIDF = 1.00

# Classification decision threshold (fake probability >= 0.50 -> FAKE)
DECISION_THRESHOLD = 0.50

# Confidence level thresholds based on probability divergence from 0.50
CONFIDENCE_HIGH_MARGIN = 0.30     # >= 80% or <= 20%
CONFIDENCE_MEDIUM_MARGIN = 0.15   # 65%-80% or 20%-35%

# =====================================================================
# HEADLINE CONSISTENCY THRESHOLDS
# =====================================================================
HEADLINE_HIGH_SIMILARITY = 0.65
HEADLINE_MEDIUM_SIMILARITY = 0.40

# =====================================================================
# TEXT PROCESSING & SCRAPING LIMITS
# =====================================================================
MIN_TEXT_LENGTH = 40          # Minimum characters required for reliable analysis
MAX_TEXT_LENGTH = 15000       # Maximum characters processed for safety/performance
MAX_SCRAPER_TIMEOUT = 12      # Seconds for requests timeout
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

# =====================================================================
# LIME EXPLAINABILITY SETTINGS
# =====================================================================
LIME_NUM_FEATURES = 10
LIME_NUM_SAMPLES = 500

# =====================================================================
# TRAINING HYPERPARAMETERS
# =====================================================================
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.20

# TF-IDF Vectorizer settings
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_SUBDIV_C = 1.0
TFIDF_MAX_ITER = 1000

# DistilBERT fine-tuning settings
BERT_MAX_LEN = 256
BERT_BATCH_SIZE = 16
BERT_EPOCHS = 3
BERT_LEARNING_RATE = 2e-5
