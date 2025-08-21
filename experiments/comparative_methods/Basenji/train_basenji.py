#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basenji baseline for plant histone modification prediction (PyTorch).

This script trains and evaluates a Basenji-like CNN model using your existing
data utilities. It is designed to plug into the SeiPlant codebase and compare
against Sei/SeiPlant and other baselines (e.g., DanQ).

Inputs
------
1) FASTA file of tiled genomic windows (one-hot encoding handled by your utils),
   e.g., data/zma/1024_512/zma_1024_512.fa
2) Tag (label) file describing per-window multi-label histone marks,
   e.g., data/zma/1024_512/tag_zma.txt

Assumptions
-----------
- The following utilities are available and compatible:
  - utils.data_utils.load_and_preprocess_data
  - utils.data_utils.split_data_by_chromosome
  - utils.data_utils.create_data_loaders
  - utils.data.NucDataset
- Your Basenji implementation is importable as:
  `from model import BasenjiModel`
  If your path differs, adjust the import accordingly.

Outputs
-------
- Best checkpoint (.pth): saved to --checkpoint (default: <species>_basenji_best.pth)
- Console logs with Train/Val losses and mean AUROC/AUPRC
- Final Test AUROC/AUPRC printed to stdout

Example
-------
python basenji_baseline.py \
  --fasta data/zma/1024_512/zma_1024_512.fa \
  --labels data/zma/1024_512/tag_zma.txt \
  --seq-len 1024 --batch-size 64 --epochs 50 --lr 1e-5 \
  --device cuda:0 --species zma \
  --num-targets 52 \
  --checkpoint zma_basenji_best.pth
"""

import os
import argparse
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# Project utilities (must exist in your repo)
from utils.data_utils import (
    load_and_preprocess_data,
    split_data_by_chromosome,
    create_data_loaders,
)
from utils.data import NucDataset  # Custom PyTorch Dataset

# Your Basenji model (adjust import path if needed)
from model import BasenjiModel


# -----------------------------
# Early Stopping helper
# -----------------------------
class EarlyStopping:
    def __init__(self, patience: int = 10, verbose: bool = True, save_path: str = "best_model.pth"):
        self.patience = patience
        self.verbose = verbose
        self.best_loss = float("inf")
        self.counter = 0
        self.save_path = save_path

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Returns True if training should stop (patience exceeded).
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"Validation loss decreased, model saved to {self.save_path}")
        else:
            self.counter += 1
        return self.counter >= self.patience


# -----------------------------
# Training & Evaluation
# -----------------------------
def _forward_to_window_scores(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass that reduces per-position outputs to per-window scores.

    NOTE: Many Basenji implementations output a track (B, C, L_out) or (B, L_out, C).
    This helper uses a simple global max over the length dimension to get
    per-window scores of shape (B, C). Adjust this if your BasenjiModel
    returns a different shape.
    """
    y = model(x)
    # Accept either (B, C, L) or (B, L, C)
    if y.dim() == 3:
        if y.shape[1] <= y.shape[2]:
            # assume (B, C, L_out) -> max over L_out
            y = y.max(dim=2).values
        else:
            # assume (B, L_out, C) -> max over L_out
            y = y.max(dim=1).values
    elif y.dim() == 2:
        # already (B, C)
        pass
    else:
        raise RuntimeError(f"Unexpected Basenji output shape: {tuple(y.shape)}")
    return y  # (B, C)


def train_basenji(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_targets: int,
    epochs: int = 100,
    lr: float = 1e-5,
    model_path: str = "best_model.pth",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    patience: int = 20,
) -> nn.Module:
    """
    Train Basenji with sigmoid + BCE loss and Adam optimizer.
    Saves the best model (lowest validation loss).
    """
    model = BasenjiModel(num_targets=num_targets).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    early_stopper = EarlyStopping(patience=patience, save_path=model_path)

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        for X_batch, Y_batch in train_loader:
            # Expect X_batch: (B, L, 4) -> permute to (B, 4, L) for Conv1D
            X_batch = X_batch.to(device, dtype=torch.float).permute(0, 2, 1)
            Y_batch = Y_batch.to(device, dtype=torch.float)

            optimizer.zero_grad()
            window_scores = _forward_to_window_scores(model, X_batch)  # logits or scores (B, C)
            # Use sigmoid to map to probabilities for BCE
            loss = criterion(torch.sigmoid(window_scores), Y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        preds, labels = [], []
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device, dtype=torch.float).permute(0, 2, 1)
                Y_batch = Y_batch.to(device, dtype=torch.float)

                window_scores = _forward_to_window_scores(model, X_batch)
                prob = torch.sigmoid(window_scores)
                loss = criterion(prob, Y_batch)
                val_loss += loss.item() * X_batch.size(0)

                preds.append(prob.cpu().numpy())
                labels.append(Y_batch.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        preds = np.vstack(preds)
        labels = np.vstack(labels)

        # AUROC / AUPRC across targets (skip degenerate columns)
        valid_auc, valid_auprc = [], []
        for i in range(labels.shape[1]):
            if np.unique(labels[:, i]).size < 2:
                continue
            try:
                valid_auc.append(roc_auc_score(labels[:, i], preds[:, i]))
                valid_auprc.append(average_precision_score(labels[:, i], preds[:, i]))
            except Exception:
                continue

        auc = float(np.mean(valid_auc)) if valid_auc else -1.0
        auprc = float(np.mean(valid_auprc)) if valid_auprc else -1.0

        print(
            f"[Epoch {epoch + 1}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"AUC: {auc:.4f} | AUPRC: {auprc:.4f}"
        )

        # ---- Early Stopping ----
        if early_stopper.step(val_loss, model):
            print("Early stopping triggered.")
            break

    # Load and return best checkpoint
    best_model = BasenjiModel(num_targets=num_targets).to(device)
    best_model.load_state_dict(torch.load(model_path, map_location=device))
    return best_model


def test_basenji(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[float, float]:
    """
    Evaluate the model on the test set. Returns (mean_AUROC, mean_AUPRC).
    """
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device, dtype=torch.float).permute(0, 2, 1)
            Y_batch = Y_batch.to(device, dtype=torch.float)

            window_scores = _forward_to_window_scores(model, X_batch)
            prob = torch.sigmoid(window_scores)

            preds.append(prob.cpu().numpy())
            labels.append(Y_batch.cpu().numpy())

    preds = np.vstack(preds)
    labels = np.vstack(labels)

    test_auc, test_auprc = [], []
    for i in range(labels.shape[1]):
        if np.unique(labels[:, i]).size < 2:
            continue
        try:
            test_auc.append(roc_auc_score(labels[:, i], preds[:, i]))
            test_auprc.append(average_precision_score(labels[:, i], preds[:, i]))
        except Exception:
            continue

    auc = float(np.mean(test_auc)) if test_auc else -1.0
    auprc = float(np.mean(test_auprc)) if test_auprc else -1.0
    print(f"[Test] AUC: {auc:.4f} | AUPRC: {auprc:.4f}")
    return auc, auprc


# -----------------------------
# CLI Entrypoint
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train/Evaluate Basenji baseline on plant histone marks.")
    parser.add_argument("--fasta", type=str, required=True, help="Path to FASTA file of genomic windows.")
    parser.add_argument("--labels", type=str, required=True, help="Path to tag/label file (multi-label).")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length (default: 1024).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs (default: 50).")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device string, e.g., 'cuda:0' or 'cpu'.")
    parser.add_argument("--species", type=str, default="zma", help="Species code for naming outputs.")
    parser.add_argument("--num-targets", type=int, required=True, help="Number of target labels (histone marks).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to save the best Basenji model.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (default: 20).")
    args = parser.parse_args()

    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = args.checkpoint or f"{args.species}_basenji_best.pth"

    # -----------------------------
    # Load & split data
    # -----------------------------
    x_all, labels, pos, tag_dict = load_and_preprocess_data(args.fasta, args.labels)

    print("Splitting data by chromosome...")
    x_train, y_train, x_val, y_val, x_test, y_test, tag_dict = split_data_by_chromosome(
        x_all, labels, pos, tag_dict
    )

    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        x_train, y_train, x_val, y_val, batch_size=args.batch_size
    )

    # -----------------------------
    # Train
    # -----------------------------
    print("Training Basenji model...")
    best_model = train_basenji(
        train_loader=train_loader,
        val_loader=val_loader,
        num_targets=args.num_targets,
        epochs=args.epochs,
        lr=args.lr,
        model_path=checkpoint_path,
        device=device,
        patience=args.patience,
    )

    # -----------------------------
    # Test
    # -----------------------------
    print("Evaluating on test set...")
    test_dataset = NucDataset(x=x_test, y=y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_auc, test_auprc = test_basenji(best_model, test_loader, device=device)

    # Print a compact summary for logs
    print(f"✅ Basenji Test Summary | AUROC: {test_auc:.4f} | AUPRC: {test_auprc:.4f}")


if __name__ == "__main__":
    main()
