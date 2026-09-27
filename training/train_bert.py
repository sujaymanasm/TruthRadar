"""
TruthRadar - DistilBERT Fine-Tuning Script
------------------------------------------
Fine-tunes a pretrained DistilBERT (distilbert-base-uncased) sequence classifier
for Fake News Detection on an 80/20 train/test split.

Supports automatic GPU/CUDA execution and CPU fallback.
Exports model weights, tokenizer, and config to models/bert_model/.

Usage:
    python training/train_bert.py [--data-path data/sample_news.csv] [--epochs 3] [--batch-size 8]
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
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import config
from src.preprocessor import clean_text
from src.utils import logger
from training.train_tfidf import load_dataset


class NewsTextDataset(Dataset):
    """PyTorch Dataset wrapper for tokenized news articles."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int = 256):
        self.texts = [clean_text(t) for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def train_bert_model(
    data_path: Path,
    epochs: int = config.BERT_EPOCHS,
    batch_size: int = 8,
    lr: float = config.BERT_LEARNING_RATE
):
    """Executes DistilBERT fine-tuning pipeline."""
    print("=" * 70)
    print("TruthRadar: DistilBERT Deep Learning Training Pipeline")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load and prepare dataset
    print("\n[Step 1/6] Loading and standardizing dataset...")
    df = load_dataset(data_path)

    X = df["combined_text"].tolist()
    y = df["target"].tolist()

    # 2. Train-Test Split (80/20 Stratified)
    print("\n[Step 2/6] Splitting dataset (80% Train, 20% Test)...")
    unique_classes, counts = np.unique(y, return_counts=True)
    stratify_arg = y if min(counts) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SPLIT_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify_arg
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3. Tokenizer & DataLoaders
    print(f"\n[Step 3/6] Initializing Tokenizer ({config.BERT_PRETRAINED_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_PRETRAINED_NAME)

    train_dataset = NewsTextDataset(X_train, y_train, tokenizer, max_len=config.BERT_MAX_LEN)
    test_dataset = NewsTextDataset(X_test, y_test, tokenizer, max_len=config.BERT_MAX_LEN)

    # If small dataset, adjust batch size
    actual_batch_size = min(batch_size, len(train_dataset))
    if actual_batch_size < 1:
        actual_batch_size = 1

    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=actual_batch_size, shuffle=False)

    # 4. Model & Optimizer
    print(f"\n[Step 4/6] Initializing Pretrained Model ({config.BERT_PRETRAINED_NAME})...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BERT_PRETRAINED_NAME,
        num_labels=2,
        id2label={0: "REAL", 1: "FAKE"},
        label2id={"REAL": 0, "FAKE": 1}
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # 5. Training Loop
    print(f"\n[Step 5/6] Training DistilBERT across {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} - Average Training Loss: {avg_loss:.4f}")

    # 6. Evaluation
    print("\n[Step 6/6] Evaluating DistilBERT on Test Set...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print("\n" + "=" * 50)
    print("DISTILBERT EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["REAL (0)", "FAKE (1)"], zero_division=0))
    print("=" * 50)

    # 7. Save Fine-Tuned Model & Tokenizer
    output_dir = config.BERT_MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving fine-tuned model and tokenizer to: {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("[OK] DistilBERT fine-tuning pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on Fake News Dataset")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(config.SAMPLE_DATASET_PATH),
        help="Path to CSV dataset"
    )
    parser.add_argument("--epochs", type=int, default=config.BERT_EPOCHS, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=config.BERT_LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    train_bert_model(
        data_path=Path(args.data_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
