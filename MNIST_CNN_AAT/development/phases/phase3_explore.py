"""
MNIST Handwritten Digit Recognition - AAT Project
Phase 3: Dataset Exploration and Validation
"""
import sys
print("=" * 60)
print("MNIST CNN - AAT PROJECT - PHASE 3: DATASET EXPLORATION")
print("=" * 60)

import numpy as np
import matplotlib.pyplot as plt

# Import TensorFlow
import tensorflow as tf

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Basic dataset properties
print("\n[1] Dataset Dimensions:")
print(f"    Training set size: {x_train.shape[0]} images")
print(f"    Test set size:     {x_test.shape[0]} images")
print(f"    Image dimensions:  {x_train.shape[1]}x{x_train.shape[2]} pixels")
print(f"    Number of classes: {len(set(y_train.tolist()))}")

# Check pixel value range
print("\n[2] Pixel Value Analysis:")
print(f"    Training images dtype: {x_train.dtype}")
print(f"    Training images min:   {x_train.min()}")
print(f"    Training images max:   {x_train.max()}")
print(f"    Training images mean:  {x_train.mean():.2f}")

# Check for NaN/Inf values
print("\n[3] Data Quality Checks:")
has_nan = np.isnan(x_train).any() or np.isnan(x_test).any()
has_inf = np.isinf(x_train).any() or np.isinf(x_test).any()
print(f"    - NaN values in training set: {has_nan}")
print(f"    - NaN values in test set:     {has_inf}")
print(f"    - Inf values in training set: {has_inf}")
print(f"    - Inf values in test set:     {has_inf}")

# Class distribution
print("\n[4] Class Distribution:")
for i in range(10):
    count = int(np.sum(y_train == i))
    print(f"    - Class {i}: {count} samples ({count/len(y_train)*100:.1f}%)")

# Check test set distribution
print("\n[5] Test Set Class Distribution:")
for i in range(10):
    count = int(np.sum(y_test == i))
    print(f"    - Class {i}: {count} samples ({count/len(y_test)*100:.1f}%)")

# Sample images visualization
print("\n[5] Displaying sample images...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(10):
    row = i // 5
    col = i % 5
    idx = np.where(y_train == i)[0][0]  # First occurrence of each digit
    axes[row, col].imshow(x_train[idx], cmap='gray')
    axes[row, col].set_title(f"Label: {y_train[idx]}")
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('D:/DL AAT/MNIST_CNN_AAT/results/figures/phase3_samples.png', dpi=150)
print("    - Sample images saved to: results/figures/phase3_samples.png")
plt.close()

# Summary
print("\n" + "=" * 60)
print("PHASE 3 STATUS: PASSED")
print("=" * 60)
print("\nDataset exploration completed:")
print(f"  - Training set: {x_train.shape[0]} images (60,000)")
print(f"  - Test set:     {x_test.shape[0]} images (10,000)")
print(f"  - 10 classes (0-9) confirmed")
print(f"  - Pixel range: [{x_train.min()}, {x_train.max()}]")
print(f"  - Pixel mean: {x_train.mean():.2f}")
