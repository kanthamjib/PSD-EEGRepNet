"""
Main Script
===========
This script orchestrates the overall workflow for EEG motor-imagery classification,
including configuration loading, dataset preprocessing, PSD feature extraction,
and model training with stratified k-fold cross-validation.

Steps:
------
1. Load configuration parameters.
2. Load and preprocess EEG data.
3. Compute power spectral density (PSD) features from EEG epochs.
4. Execute training and evaluate model performance.

Notes:
------
- Ensure to replace the dataset loading mechanism (`my_dataset_loader`) with an appropriate
  implementation suitable for the dataset being used.
- Input data dimensions are expected as:
  - `windows_data`: ndarray of shape (n_epochs, n_channels, n_times)
  - `windows_label`: array-like of shape (n_epochs,)

References:
-----------
.. [1] P. Welch, “The use of fast Fourier transform for the estimation of power spectra:
       A method based on time averaging over short, modified periodograms,”
       *IEEE Transactions on Audio and Electroacoustics*, vol. 15, no. 2, pp. 70-73, June 1967.
"""

import yaml
import numpy as np
from mne.time_frequency import psd_array_welch
from pathlib import Path

from utils.train import Trainer


# Load configuration parameters from a YAML file
cfg = type("Cfg", (), yaml.safe_load(Path("config/default.yaml").read_text()))

# Load EEG epochs and labels (replace with custom data loader)
from my_dataset_loader import windows_data, windows_label

# Feature extraction: Compute power spectral density (PSD) for each epoch
psds = []
for epoch in windows_data:
    psd, _ = psd_array_welch(epoch,
                             sfreq=cfg.sample_rate,
                             fmin=cfg.fmin,
                             fmax=cfg.fmax,
                             n_fft=cfg.n_fft,
                             n_overlap=cfg.n_overlap)
    psds.append(psd)
psds = np.stack(psds)  # Final PSD shape: (n_epochs, n_channels, n_freqs)

# Execute training and evaluation via stratified k-fold cross-validation
metrics = Trainer(psds, windows_label, cfg).kfold_cv()

# Extract test set metrics for summary

test_metrics = [result["test"] for result in metrics]

# Display cross-validation summary
print("\n===== Cross-validation Summary (Test Set) =====")
for metric in ("acc", "precision", "recall", "f1", "kappa"):
    metric_values = [m[metric]*100 for m in test_metrics]
    mean_val, std_val = np.mean(metric_values), np.std(metric_values)
    print(f"{metric.capitalize():10}: {mean_val:.2f} ± {std_val:.2f} %")
print("===============================================")
