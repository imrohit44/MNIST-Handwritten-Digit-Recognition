"""
MNIST dataset exploration and inspection.

Dataset properties, class distributions, data-quality checks and sample
visualization. This module only inspects data; it does NOT modify it
(actual preprocessing lives in preprocess_data.py).
Part of the MNIST CNN AAT project (development Phase 3 logic).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.load_data import load_mnist

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "phase3_samples.png"


def dataset_dimensions(X_train, y_train, X_test, y_test):
    """Print basic dataset dimensions."""
    print("\n[1] Dataset Dimensions:")
    print(f"    Training set size: {X_train.shape[0]} images")
    print(f"    Test set size:     {X_test.shape[0]} images")
    print(f"    Image dimensions:  {X_train.shape[1]}x{X_train.shape[2]} pixels")
    print(f"    Number of classes: {len(set(y_train.tolist()))}")


def pixel_range_report(X_train, X_test):
    """Print pixel value statistics."""
    print("\n[2] Pixel Value Analysis:")
    print(f"    Training images dtype: {X_train.dtype}")
    print(f"    Training images min:   {X_train.min()}")
    print(f"    Training images max:   {X_train.max()}")
    print(f"    Training images mean:  {X_train.mean():.2f}")
    return X_train.min(), X_train.max(), X_train.mean()


def quality_checks(X_train, X_test):
    """Check for NaN / Inf values (missing-value style checks)."""
    print("\n[3] Data Quality Checks:")
    train_nan = bool(np.isnan(X_train).any())
    test_nan = bool(np.isnan(X_test).any())
    train_inf = bool(np.isinf(X_train).any())
    test_inf = bool(np.isinf(X_test).any())
    print(f"    - NaN values in training set: {train_nan}")
    print(f"    - NaN values in test set:     {test_nan}")
    print(f"    - Inf values in training set: {train_inf}")
    print(f"    - Inf values in test set:     {test_inf}")
    return not (train_nan or test_nan or train_inf or test_inf)


def class_distribution(y, name="dataset"):
    """Print and return the class distribution of a label vector."""
    print(f"\n    {name} class distribution:")
    distribution = {}
    for digit in range(10):
        count = int(np.sum(y == digit))
        distribution[digit] = count
        print(f"      - Class {digit}: {count} samples ({count / len(y) * 100:.1f}%)")
    return distribution


def plot_sample_digits(X, y, save_path=SAMPLE_FIGURE_PATH):
    """Save a 2x5 grid showing the first occurrence of each digit 0-9."""
    print("\n[6] Sample images visualization...")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for digit in range(10):
        row, col = digit // 5, digit % 5
        idx = np.where(y == digit)[0][0]
        axes[row, col].imshow(X[idx], cmap="gray")
        axes[row, col].set_title(f"Label: {y[idx]}")
        axes[row, col].axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    - Sample images saved to: {save_path}")
    return save_path


def main():
    print("=" * 60)
    print("MNIST CNN - EXPLORE DATA")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist()

    dataset_dimensions(X_train, y_train, X_test, y_test)
    pixel_range_report(X_train, X_test)
    clean = quality_checks(X_train, X_test)

    print("\n[4] Training Set")
    class_distribution(y_train, "Training set")
    print("\n[5] Test Set")
    class_distribution(y_test, "Test set")

    plot_sample_digits(X_train, y_train)

    print("\n" + "=" * 60)
    if clean:
        print("Explore data check: PASSED")
    else:
        print("Explore data check: FAILED")
        raise SystemExit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
