"""
TruthRadar - TF-IDF + Logistic Regression Training Script
---------------------------------------------------------
Trains a baseline Fake News Classifier using TF-IDF feature extraction
and Logistic Regression. Evaluates the model on an 80/20 train/test split
and exports artifacts to models/ for deployment.

Usage:
    python training/train_tfidf.py [--data-path data/sample_news.csv]
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import config
from src.preprocessor import preprocess_for_ml
from src.utils import logger


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Loads and standardizes dataset columns and formats."""
    if not data_path.exists():
        logger.warning(f"Specified dataset {data_path} not found. Checking fallback paths...")
        if config.FULL_DATASET_PATH.exists():
            data_path = config.FULL_DATASET_PATH
        elif config.SAMPLE_DATASET_PATH.exists():
            data_path = config.SAMPLE_DATASET_PATH
        else:
            raise FileNotFoundError(
                f"No dataset found at {data_path}, {config.FULL_DATASET_PATH}, or {config.SAMPLE_DATASET_PATH}."
            )

    logger.info(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} total rows. Columns: {list(df.columns)}")

    # Column mapping & standardization
    cols = {c.lower(): c for c in df.columns}
    
    # Identify text column
    text_col = None
    for cand in ["text", "article", "content", "news", "statement", "body"]:
        if cand in cols:
            text_col = cols[cand]
            break

    # Identify title column (optional)
    title_col = None
    for cand in ["title", "headline", "subject"]:
        if cand in cols:
            title_col = cols[cand]
            break

    # Identify label column
    label_col = None
    for cand in ["label", "class", "target", "category", "fake"]:
        if cand in cols:
            label_col = cols[cand]
            break

    if text_col is None or label_col is None:
        raise ValueError(
            f"Dataset must contain text and label columns. Found columns: {list(df.columns)}"
        )

    # Combine title + text if title exists for maximum context
    if title_col and title_col != text_col:
        df["combined_text"] = df[title_col].fillna("").astype(str) + " " + df[text_col].fillna("").astype(str)
    else:
        df["combined_text"] = df[text_col].fillna("").astype(str)

    # Standardize labels to binary integer (0 = REAL, 1 = FAKE)
    def standardize_label(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().upper()
        if s in ["FAKE", "1", "1.0", "TRUE_FAKE", "FALSE", "HOAX", "SPAM"]:
            return 1  # FAKE
        elif s in ["REAL", "0", "0.0", "TRUE_REAL", "TRUE", "FACT", "RELIABLE"]:
            return 0  # REAL
        return np.nan

    df["target"] = df[label_col].apply(standardize_label)
    df = df.dropna(subset=["combined_text", "target"])
    df["target"] = df["target"].astype(int)

    # Drop empty texts
    df = df[df["combined_text"].str.strip().str.len() > 10]
    
    logger.info(f"Dataset after cleaning: {len(df)} samples.")
    logger.info(f"Class distribution: REAL (0) = {(df['target'] == 0).sum()}, FAKE (1) = {(df['target'] == 1).sum()}")
    return df


def train_tfidf_model(data_path: Path):
    """Executes full training pipeline for TF-IDF + Logistic Regression."""
    print("=" * 70)
    print("TruthRadar: TF-IDF + Logistic Regression Training Pipeline")
    print("=" * 70)

    # 1. Load Data
    df = load_dataset(data_path)
    
    # 2. Preprocess text
    print("\n[Step 1/5] Preprocessing text data for ML...")
    df["cleaned_text"] = df["combined_text"].apply(preprocess_for_ml)
    
    # Remove any samples where cleaning left string empty
    df = df[df["cleaned_text"].str.strip().str.len() > 0]

    X = df["cleaned_text"].values
    y = df["target"].values

    # Check class balance
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2:
        raise ValueError(f"Dataset must contain both REAL (0) and FAKE (1) classes. Found only: {unique_classes}")

    # 3. Train-Test Split (80/20 Stratified)
    print("\n[Step 2/5] Splitting dataset (80% Train, 20% Test)...")
    stratify_arg = y if min(counts) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SPLIT_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify_arg
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 4. Feature Extraction: TF-IDF
    print("\n[Step 3/5] Extracting TF-IDF features (n-gram: (1, 2))...")
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        min_df=1,
        stop_words="english"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} features")

    # 5. Model Training: Logistic Regression
    print("\n[Step 4/5] Training Logistic Regression Classifier...")
    classifier = LogisticRegression(
        C=config.TFIDF_SUBDIV_C,
        max_iter=config.TFIDF_MAX_ITER,
        random_state=config.RANDOM_STATE,
        class_weight="balanced"
    )
    classifier.fit(X_train_vec, y_train)

    # 6. Model Evaluation
    print("\n[Step 5/5] Evaluating model performance on Test Set...")
    y_pred = classifier.predict(X_test_vec)
    y_prob = classifier.predict_proba(X_test_vec)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS (TF-IDF + Logistic Regression)")
    print("=" * 50)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print("                Predicted REAL   Predicted FAKE")
    print(f"Actual REAL     {cm[0, 0]:<16} {cm[0, 1]:<16}")
    if cm.shape[0] > 1 and cm.shape[1] > 1:
        print(f"Actual FAKE     {cm[1, 0]:<16} {cm[1, 1]:<16}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["REAL (0)", "FAKE (1)"], zero_division=0))
    print("=" * 50)

    # 7. Save Artifacts
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, config.TFIDF_MODEL_PATH)
    joblib.dump(vectorizer, config.TFIDF_VECTORIZER_PATH)
    
    # Save label mapping for transparency
    label_map = {0: "REAL", 1: "FAKE"}
    joblib.dump(label_map, config.LABEL_ENCODER_PATH)

    print(f"\n[OK] Model successfully saved to: {config.TFIDF_MODEL_PATH}")
    print(f"[OK] Vectorizer successfully saved to: {config.TFIDF_VECTORIZER_PATH}")
    print(f"[OK] Label map saved to: {config.LABEL_ENCODER_PATH}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TF-IDF Fake News Classifier")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(config.SAMPLE_DATASET_PATH),
        help="Path to CSV dataset with text and label columns"
    )
    args = parser.parse_args()
    train_tfidf_model(Path(args.data_path))
