"""
Training Module
===============
This module encapsulates training and evaluation routines for CNN-based EEG motor-imagery
classification using stratified k-fold cross-validation.

Key functionalities:
- Data preprocessing: normalization based on training set statistics.
- Stratified k-fold cross-validation for robust evaluation.
- Class weight computation to address class imbalance.
- Training with early stopping based on validation loss.
- Comprehensive performance metrics on both training and test sets.

"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from utils.models import get_model
from utils.evaluate import compute_metrics, print_report


class Trainer:
    """Class for managing model training and evaluation via stratified k-fold cross-validation."""

    def __init__(self, X, y, cfg):
        """Initialize the Trainer with data and configuration.

        Parameters
        ----------
        X : ndarray
            Input data of shape (n_epochs, n_channels, n_times).
        y : array-like
            Ground truth labels.
        cfg : object
            Configuration object containing training parameters.
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_enc = LabelEncoder().fit(y)
        self.X = torch.tensor(X[:, None, ...], dtype=torch.float32)  # Adds channel dimension (N, 1, C, F)
        self.y = torch.tensor(self.label_enc.transform(y), dtype=torch.long)

    def kfold_cv(self):
        """Perform stratified k-fold cross-validation.

        Returns
        -------
        all_metrics : list
            List containing performance metrics for each fold.
        """
        skf = StratifiedKFold(n_splits=self.cfg.k_folds, shuffle=True, random_state=42)
        all_metrics = []
        for fold, (tr, va) in enumerate(skf.split(self.X, self.y), 1):
            metrics = self._run_fold(fold, tr, va)
            all_metrics.append(metrics)
        return all_metrics

    def _run_fold(self, fold, tr_idx, va_idx):
        """Run training and evaluation for a single fold.

        Parameters
        ----------
        fold : int
            Fold number.
        tr_idx : array-like
            Indices for training data.
        va_idx : array-like
            Indices for validation data.

        Returns
        -------
        dict
            Dictionary containing fold number, training, and test metrics.
        """
        # Split data
        X_tr, X_va = self.X[tr_idx], self.X[va_idx]
        y_tr, y_va = self.y[tr_idx], self.y[va_idx]

        # Normalize based on training set statistics
        mu, sigma = X_tr.mean(), X_tr.std()
        X_tr = (X_tr - mu) / sigma
        X_va = (X_va - mu) / sigma

        # Prepare DataLoader
        tr_loader = DataLoader(TensorDataset(X_tr, y_tr),
                               batch_size=self.cfg.batch_size, shuffle=True)
        va_loader = DataLoader(TensorDataset(X_va, y_va),
                               batch_size=self.cfg.batch_size, shuffle=False)

        # Model initialization
        model = get_model(self.cfg.n_classes).to(self.device)

        # Compute class weights for imbalanced classification
        class_w = torch.tensor(
            compute_class_weight("balanced",
                                 classes=np.unique(y_tr.numpy()),
                                 y=y_tr.numpy()),
            dtype=torch.float32, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=class_w)
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.lr)

        # Training loop with early stopping
        best_val, patience = np.inf, 0
        start_time = time.time()
        for epoch in range(1, self.cfg.epochs + 1):
            self._train_one_epoch(model, tr_loader, criterion, optimizer)
            val_loss = self._eval_loss(model, va_loader, criterion)
            if val_loss < best_val:
                best_val, patience = val_loss, 0
            else:
                patience += 1
            if patience >= self.cfg.patience:
                break
        train_duration_ms = (time.time() - start_time) * 1e03

        # Evaluate on training set
        y_tr_true, y_tr_pred = self._inference(model, tr_loader)
        train_metrics = compute_metrics(y_tr_true, y_tr_pred) | {"train_time": train_duration_ms}
        print(f"\n--- Fold {fold} Training Set Evaluation ---")
        print_report(y_tr_true, y_tr_pred)

        # Evaluate on validation/test set
        y_va_true, y_va_pred = self._inference(model, va_loader)
        test_metrics = compute_metrics(y_va_true, y_va_pred)
        print(f"\n--- Fold {fold} Test Set Evaluation ---")
        print_report(y_va_true, y_va_pred)

        return {"fold": fold, "train": train_metrics, "test": test_metrics}

    def _train_one_epoch(self, model, loader, criterion, optimizer):
        """Train model for one epoch."""
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def _eval_loss(self, model, loader, criterion):
        """Evaluate loss on a given dataset."""
        model.eval()
        total_loss, total_samples = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            total_loss += criterion(model(xb), yb).item() * yb.size(0)
            total_samples += yb.size(0)
        return total_loss / total_samples

    @torch.no_grad()
    def _inference(self, model, loader):
        """Run inference and return true and predicted labels."""
        model.eval()
        y_true, y_pred = [], []
        for xb, yb in loader:
            xb = xb.to(self.device)
            logits = model(xb)
            y_true.extend(yb.numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())
        return y_true, y_pred
