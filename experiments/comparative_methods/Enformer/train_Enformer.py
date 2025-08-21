#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enformer (tiny/adapted) baseline for plant histone modification prediction (PyTorch).

This script trains and evaluates an Enformer-like model using your existing data
utilities. It is designed to plug into the SeiPlant codebase and compare against
Sei/SeiPlant and other baselines (DanQ, Basenji).

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
- Your Enformer implementation is importable as:
  `from model.enformer_new import Enformer_new`
  If your path differs, adjust the import accordingly.

Outputs
-------
- Best checkpoint (.pth): saved to --checkpoint (default: <species>_<batch>_<lr>_Enformer_best.pth)
- Console logs with Train/Val losses and mean AUROC/AUPRC
- Final Test AUROC/AUPRC printed to stdout

Example
-------
python enformer_baseline.py \
  --fasta data/zma/1024_512/zma_1024_512.fa \
  --labels data/zma/1024_512/tag_zma.txt \
  --seq-len 1024 --batch-size 64 --epochs 100 --lr 1e-3 \
  --device cuda:0 --species zma \
  --checkpoint zma_64_0.001_Enformer_best.pth
"""

import argparse
from typing import Tuple, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

# Project utilities (must exist in your repo)
from utils.data_utils import (
    load_and_preprocess_data,
    split_data_by_chromosome,
    create_data_loaders,
)
from utils.data import NucDataset  # Custom PyTorch Dataset

# Your Enformer model (adjust import if needed)
from model.enformer_new import Enformer_new


# -----------------------------
# Helpers
# -----------------------------
def print_gpu_memory(prefix: str = "", device: Union[torch.device, str, None] = None):
    """Optional GPU memory logger (safe no-op on CPU)."""
    if torch.cuda.is_available():
        if device is None:
            device = torch.device("cuda:0")
        if isinstance(device, str):
            device = torch.device(device)
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(device) / 1024**2    # MB
        print(f"{prefix} GPU Memory on {device} - Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")


def _forward_to_window_scores(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass that reduces per-position outputs to per-window logits of shape (B, C).

    Enformer variants often return:
      - Tensor (B, L_out, C) or (B, C, L_out)
      - Dict with a key like 'pred' or similar mapping to a Tensor above

    We:
      1) Extract the Tensor (if dict)
      2) Global max-pool over the length dimension to get (B, C) logits

    Adjust this function if your Enformer_new returns a different structure.
    """
    y = model(x)
    # If dict-like, try common keys; otherwise, assume y is already a Tensor
    if isinstance(y, dict):
        # Prefer common prediction keys, else take first tensor value
        for key in ("pred", "logits", "outputs", "y", "out"):
            if key in y and torch.is_tensor(y[key]):
                y = y[key]
                break
        if isinstance(y, dict):
            # Fall back: pick the first tensor in dict
            for v in y.values():
                if torch.is_tensor(v):
                    y = v
                    break

    if not torch.is_tensor(y):
        raise RuntimeError("Enformer_new forward output is not a Tensor or dict of Tensors.")

    if y.dim() == 3:
        # (B, L_out, C) or (B, C, L_out)
        if y.shape[1] <= y.shape[2]:
            # Likely (B, L_out, C) -> pool over L_out (dim=1)
            y = y.max(dim=1).values  # (B, C)
        else:
            # Likely (B, C, L_out) -> pool over L_out (dim=2)
            y = y.max(dim=2).values  # (B, C)
    elif y.dim() == 2:
        # Already (B, C)
        pass
    else:
        raise RuntimeError(f"Unexpected Enformer output shape: {tuple(y.shape)}")

    return y  # logits (B, C)


# -----------------------------
# Training & Evaluation
# -----------------------------
def train_enformer_pytorch(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    seq_len: int = 1024,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    checkpoint_path: str = "best_enformer.model",
    early_stop_patience: int = 20,
) -> nn.Module:
    """
    Train Enformer_new with BCEWithLogits loss and Adam optimizer.
    Saves the best model (lowest validation loss) to checkpoint_path.
    """
    model = Enformer_new(num_targets=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    no_improve_counter = 0

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            # Expect X: (B, L, 4) float; Enformer usually accepts (B, L, 4)
            X_batch = X_batch.to(device, dtype=torch.float)
            Y_batch = Y_batch.to(device, dtype=torch.float)

            optimizer.zero_grad()
            logits = _forward_to_window_scores(model, X_batch)  # (B, C) logits
            loss = criterion(logits, Y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        preds, labels = [], []
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device, dtype=torch.float)
                Y_batch = Y_batch.to(device, dtype=torch.float)

                logits = _forward_to_window_scores(model, X_batch)  # (B, C)
                loss = criterion(logits, Y_batch)
                val_loss += loss.item() * X_batch.size(0)

                preds.append(torch.sigmoid(logits).cpu().numpy())
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
        print_gpu_memory(prefix=f"[Epoch {epoch + 1}]", device=device)

        # ---- Early stopping & checkpointing ----
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


def test_enformer(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on the test set. Returns (mean_AUROC, mean_AUPRC).
    """
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device, dtype=torch.float)  # (B, L, 4)
            Y_batch = Y_batch.to(device, dtype=torch.float)  # (B, C)

            logits = _forward_to_window_scores(model, X_batch)  # (B, C)
            prob = torch.sigmoid(logits)

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
    parser = argparse.ArgumentParser(description="Train/Evaluate Enformer baseline on plant histone marks.")
    parser.add_argument("--fasta", type=str, required=True, help="Path to FASTA file of genomic windows.")
    parser.add_argument("--labels", type=str, required=True, help="Path to tag/label file (multi-label).")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length (default: 1024).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs (default: 100).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device string, e.g., 'cuda:0' or 'cpu'.")
    parser.add_argument("--species", type=str, default="zma", help="Species code for naming outputs.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to save the best Enformer model.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (default: 20).")
    args = parser.parse_args()

    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")
    print(f"Using device: {device}")
    print(f"batch_size: {args.batch_size}")
    print(f"learning_rate: {args.lr}")

    # Prepare checkpoint path
    ckpt_default = f"{args.species}_{args.batch_size}_{args.lr}_Enformer_best.pth"
    checkpoint_path = args.checkpoint or ckpt_default

    # -----------------------------
    # Load & split data
    # -----------------------------
    x_all, labels, pos, tag_dict = load_and_preprocess_data(args.fasta, args.labels)

    print("Splitting data by chromosome...")
    x_train, y_train, x_val, y_val, x_test, y_test, _ = split_data_by_chromosome(
        x_all, labels, pos, tag_dict
    )

    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        x_train, y_train, x_val, y_val, batch_size=args.batch_size
    )

    # -----------------------------
    # Train
    # -----------------------------
    print("Training Enformer model...")
    model = train_enformer_pytorch(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=labels.shape[1],
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=checkpoint_path,
        early_stop_patience=args.patience,
    )
    print(f"✅ Training finished. Best checkpoint: {checkpoint_path}")

    # -----------------------------
    # Test
    # -----------------------------
    print("Evaluating on test set...")
    test_dataset = NucDataset(x=x_test, y=y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_auc, test_auprc = test_enformer(model, test_loader, device=device)

    # Compact summary for logs
    print(f"✅ Enformer Test Summary | AUROC: {test_auc:.4f} | AUPRC: {test_auprc:.4f}")


if __name__ == "__main__":
    main()
