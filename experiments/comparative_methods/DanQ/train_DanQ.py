#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DanQ baseline for plant histone modification prediction (PyTorch).

This script trains and evaluates a DanQ model (CNN + BiLSTM) using your
existing data utilities and loaders. It is designed to slot into the
SeiPlant codebase and compare against the main model.

Inputs
------
1) FASTA file of tiled genomic windows (one-hot will be created by your utils):
   e.g., data/zma/1024_512/zma_1024_512.fa

2) Tag (label) file describing the histone marks per window (multi-label):
   e.g., data/zma/1024_512/tag_zma.txt

   NOTE: Both files are assumed to be compatible with:
     - utils.data_utils.load_and_preprocess_data
     - utils.data_utils.split_data_by_chromosome
     - utils.data_utils.create_data_loaders
     - utils.data.NucDataset

Outputs
-------
- Best checkpoint (.pth): saved to --checkpoint (default: <species>_DanQ_best.pth)
- Console logs with Train/Val losses and mean AUROC/AUPRC
- Final Test AUROC/AUPRC printed to stdout

Example
-------
python danq_baseline.py \
  --fasta data/zma/1024_512/zma_1024_512.fa \
  --labels data/zma/1024_512/tag_zma.txt \
  --seq-len 1024 --batch-size 64 --epochs 50 --lr 1e-5 \
  --device cuda:0 --species zma \
  --checkpoint zma_DanQ_best.pth
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

# If your project is installed as a package, these imports will work as-is.
# Otherwise ensure the repository root is on PYTHONPATH.
from utils.data_utils import (
    load_and_preprocess_data,
    split_data_by_chromosome,
    create_data_loaders,
)
from utils.data import NucDataset  # Custom PyTorch Dataset


# -----------------------------
# Model: DanQ (CNN + BiLSTM)
# -----------------------------
class DanQ(nn.Module):
    def __init__(self, classes: int = 919, seq_len: int = 1024, activation: str = "relu"):
        """
        Parameters
        ----------
        classes : int
            Number of target labels (histone marks or multi-label tasks).
        seq_len : int
            Input sequence length (e.g., 1024).
        activation : str
            Activation type for the conv block ("relu" supported).
        """
        super().__init__()
        if activation != "relu":
            raise ValueError("Unsupported activation function. Use 'relu'.")

        self.seq_len = seq_len
        self.activation = nn.ReLU()

        # Convolutional feature extractor
        self.conv1d = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=19, padding=9),
            self.activation,
            nn.MaxPool1d(kernel_size=13, stride=13),
            nn.Dropout(0.2),
        )

        # After MaxPool1d with stride=13, length becomes seq_len // 13
        seq_len_after_pool = seq_len // 13

        # BiLSTM over the pooled sequence features
        self.lstm = nn.LSTM(
            input_size=320, hidden_size=320,
            num_layers=2, batch_first=True, bidirectional=True
        )  # output dim per position = 640

        # Fully-connected classifier
        self.fc = nn.Sequential(
            nn.Linear(seq_len_after_pool * 640, 925),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(925, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 4, L)
        Returns logits: (B, classes)
        """
        x = self.conv1d(x)            # (B, 320, L')
        x = x.permute(0, 2, 1)        # (B, L', 320) for LSTM
        out, _ = self.lstm(x)         # (B, L', 640)
        out = out.permute(0, 2, 1)    # (B, 640, L')
        out = out.contiguous().view(x.size(0), -1)  # flatten
        out = self.fc(out)            # (B, classes) - logits
        return out


# -----------------------------
# Training & Evaluation
# -----------------------------
def train_danq_pytorch(
    train_loader,
    val_loader,
    device: torch.device,
    num_classes: int,
    seq_len: int = 1024,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 1e-4,
    checkpoint_path: str = "best_danq_model.pth",
    early_stop_patience: int = 10,
) -> nn.Module:
    """
    Train DanQ with BCEWithLogits loss and Adam optimizer.
    Saves the best model (lowest validation loss) to checkpoint_path.
    """
    model = DanQ(classes=num_classes, seq_len=seq_len).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    no_improve_counter = 0

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            # Expect X_batch: (B, L, 4) -> permute to (B, 4, L)
            X_batch = X_batch.to(device, dtype=torch.float).permute(0, 2, 1)
            Y_batch = Y_batch.to(device, dtype=torch.float)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device, dtype=torch.float).permute(0, 2, 1)
                Y_batch = Y_batch.to(device, dtype=torch.float)

                output = model(X_batch)
                loss = criterion(output, Y_batch)
                val_loss += loss.item() * X_batch.size(0)

                all_preds.append(torch.sigmoid(output).cpu().numpy())
                all_labels.append(Y_batch.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Compute mean AUROC/AUPRC across targets (skip degenerate columns)
        valid_auc, valid_auprc = [], []
        for i in range(all_labels.shape[1]):
            if np.unique(all_labels[:, i]).size < 2:
                continue
            try:
                valid_auc.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
                valid_auprc.append(average_precision_score(all_labels[:, i], all_preds[:, i]))
            except Exception:
                continue

        auc = float(np.mean(valid_auc)) if len(valid_auc) > 0 else -1.0
        auprc = float(np.mean(valid_auprc)) if len(valid_auprc) > 0 else -1.0

        print(
            f"[Epoch {epoch + 1}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"AUC: {auc:.4f} | AUPRC: {auprc:.4f}"
        )

        # ---- Checkpointing & Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> 🟢 Improved val_loss. Model saved to {checkpoint_path}")
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            print(f"  -> 🔴 No improvement. Patience: {no_improve_counter}/{early_stop_patience}")

        if no_improve_counter >= early_stop_patience:
            print(f"⛔ Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"✅ Best model achieved val_loss = {best_val_loss:.4f}, saved to {checkpoint_path}")

    # Load the best model weights before returning
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def evaluate_on_test_set(model: nn.Module, test_loader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate on the test set and return (mean_AUROC, mean_AUPRC).
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device, dtype=torch.float).permute(0, 2, 1)
            Y_batch = Y_batch.to(device, dtype=torch.float)

            outputs = model(X_batch)
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(Y_batch.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    test_auc, test_auprc = [], []
    for i in range(all_labels.shape[1]):
        if np.unique(all_labels[:, i]).size < 2:
            continue
        try:
            test_auc.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
            test_auprc.append(average_precision_score(all_labels[:, i], all_preds[:, i]))
        except Exception:
            continue

    auc = float(np.mean(test_auc)) if len(test_auc) > 0 else -1.0
    auprc = float(np.mean(test_auprc)) if len(test_auprc) > 0 else -1.0
    print(f"[Test] AUC: {auc:.4f} | AUPRC: {auprc:.4f}")
    return auc, auprc


# -----------------------------
# Experiment Entrypoint
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train/Evaluate DanQ baseline on plant histone marks.")
    parser.add_argument("--fasta", type=str, required=True, help="Path to FASTA file of genomic windows.")
    parser.add_argument("--labels", type=str, required=True, help="Path to tag/label file (multi-label).")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length (default: 1024).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs (default: 50).")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device string, e.g., 'cuda:0' or 'cpu'.")
    parser.add_argument("--species", type=str, default="zma", help="Species code for naming outputs.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to save the best DanQ model.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    print(f"Using device: {device}")

    # Prepare checkpoint path
    checkpoint_path = args.checkpoint or f"{args.species}_DanQ_best.pth"

    # -----------------------------
    # Load and preprocess data
    # -----------------------------
    x_all, labels, pos, tag_dict = load_and_preprocess_data(args.fasta, args.labels)

    # Split by chromosome to avoid leakage
    print("Splitting data by chromosome...")
    x_train, y_train, x_val, y_val, x_test, y_test, tag_dict = split_data_by_chromosome(
        x_all, labels, pos, tag_dict
    )

    # DataLoaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        x_train, y_train, x_val, y_val, batch_size=args.batch_size
    )

    # -----------------------------
    # Train
    # -----------------------------
    print("Training DanQ model...")
    model = train_danq_pytorch(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=labels.shape[1],
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=checkpoint_path,
        early_stop_patience=10,
    )

    # -----------------------------
    # Test
    # -----------------------------
    print("Evaluating on test set...")
    test_dataset = NucDataset(x=x_test, y=y_test)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    evaluate_on_test_set(model, test_loader, device=device)


if __name__ == "__main__":
    main()
