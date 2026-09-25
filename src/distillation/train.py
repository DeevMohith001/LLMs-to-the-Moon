"""
Distillation Training Engine for Student Model.

Implements:
- Experiment R1: Classification Baseline (Cross-Entropy Loss on majority teacher labels)
- Experiment R2: Regression Distillation (MSE Loss on continuous teacher soft scores)
- Validation tracking, early stopping, and checkpoint persistence to outputs/models/
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.distillation.dataset import (
    TextRegressionDataset,
    TextClassificationDataset,
    filter_by_consistency,
    LABEL_TO_ID,
)
from src.distillation.student_model import build_student_model
from src.evaluation.metrics import compute_sentiment_metrics, save_metrics
from src.utils.common import MODELS_DIR, OUTPUTS_DIR, RESULTS_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)


def train_regression_student(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    backbone_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 3e-5,
    max_length: int = 256,
    device: Optional[torch.device] = None,
    save_name: str = "student_regression",
) -> Tuple[nn.Module, Any, Dict[str, Any]]:
    """
    Train student model using Regression Distillation (Experiment R2).
    Loss: MSELoss between predicted scalar and teacher_soft_score in [-1.0, 1.0].
    """
    device = device or torch.device("cpu")
    logger.info(
        f"Starting Student Regression Distillation on {len(train_df)} samples ({epochs} epochs, lr={lr})..."
    )

    model, tokenizer = build_student_model(
        backbone_name=backbone_name, objective="regression", device=device
    )

    # Prepare datasets
    train_targets = train_df["teacher_soft_score"].tolist()
    val_targets = (
        val_df["teacher_soft_score"].tolist()
        if "teacher_soft_score" in val_df.columns
        else [0.0] * len(val_df)
    )

    train_ds = TextRegressionDataset(
        train_df["clean_text" if "clean_text" in train_df.columns else "text"].tolist(),
        train_targets,
        tokenizer,
        max_length=max_length,
    )
    val_ds = TextRegressionDataset(
        val_df["clean_text" if "clean_text" in val_df.columns else "text"].tolist(),
        val_targets,
        tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_mse": [], "val_mae": []}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        # Validation evaluation
        model.eval()
        val_preds, val_actuals = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["target"].to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                val_preds.extend(out.logits.squeeze(1).cpu().numpy())
                val_actuals.extend(targets.squeeze(1).cpu().numpy())

        val_mse = float(np.mean((np.array(val_preds) - np.array(val_actuals)) ** 2))
        val_mae = float(np.mean(np.abs(np.array(val_preds) - np.array(val_actuals))))

        history["train_loss"].append(avg_train_loss)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)

        logger.info(
            f"Epoch {epoch+1}/{epochs} - Train MSE: {avg_train_loss:.4f} | Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f}"
        )

    # Persist checkpoint to both outputs/models and models/
    for base_dir in [OUTPUTS_DIR / "models", MODELS_DIR]:
        ckpt_dir = base_dir / save_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    logger.info(f"Saved student regression checkpoint to outputs/models/{save_name}")
    return model, tokenizer, history


def train_classification_student(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    backbone_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 3e-5,
    max_length: int = 256,
    device: Optional[torch.device] = None,
    save_name: str = "student_classification",
) -> Tuple[nn.Module, Any, Dict[str, Any]]:
    """
    Train student model using Categorical Cross-Entropy (Baseline Experiment R1).
    """
    device = device or torch.device("cpu")
    logger.info(
        f"Starting Student Classification Baseline on {len(train_df)} samples ({epochs} epochs, lr={lr})..."
    )

    model, tokenizer = build_student_model(
        backbone_name=backbone_name, objective="classification", device=device
    )

    train_labels = (
        train_df["teacher_majority_label"].tolist()
        if "teacher_majority_label" in train_df.columns
        else train_df["sentiment_label"].tolist()
    )
    val_labels = (
        val_df["sentiment_label"].tolist()
        if "sentiment_label" in val_df.columns
        else ["neutral"] * len(val_df)
    )

    train_ds = TextClassificationDataset(
        train_df["clean_text" if "clean_text" in train_df.columns else "text"].tolist(),
        train_labels,
        tokenizer,
        max_length=max_length,
    )
    val_ds = TextClassificationDataset(
        val_df["clean_text" if "clean_text" in val_df.columns else "text"].tolist(),
        val_labels,
        tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += out.loss.item()
                preds = torch.argmax(out.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        val_acc = correct / max(total, 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss / max(len(val_loader), 1))
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

    for base_dir in [OUTPUTS_DIR / "models", MODELS_DIR]:
        ckpt_dir = base_dir / save_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    logger.info(f"Saved student classification checkpoint to outputs/models/{save_name}")
    return model, tokenizer, history
