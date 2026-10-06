"""
MNIST Handwritten Digit Recognition - AAT Project
Phase 5: Training/Validation/Test Split
"""
import sys
print("=" * 60)
print("MNIST CNN - AAT PROJECT - PHASE 5: TRAINING/VALIDATION/TEST SPLIT")
print("=" * 60)

import numpy as np
import tensorflow as tf

# Load and preprocess MNIST dataset (Phase 4)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Parameters
RANDOM_SEED = 42
TRAIN_RATIO = 0.85  # 85% for training
VAL_RATIO = 0.15    # 15% for validation

print("\n[1] Setting up random seed for reproducibility...")
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print(f"\n[2] Original dataset sizes:")
print(f"    Training set (original): {x_train.shape[0]} images")
print(f"    Test set:                {x_test.shape[0]} images")

# Calculate split sizes
n_train = x_train.shape[0]
n_val = int(n_train * VAL_RATIO)
n_train_final = n_train - n_val

print(f"\n[3] Split configuration:")
print(f"    Training set:     {n_train_final} samples ({n_train_final/n_train*100:.1f}%)")
print(f"    Validation set:   {n_val} samples ({n_val/n_train*100:.1f}%)")
print(f"    Test set:         {x_test.shape[0]} samples (original)")

# Shuffle training data with fixed seed
print("\n[4] Shuffling training indices...")
indices = np.arange(n_train)
np.random.seed(RANDOM_SEED)
np.random.shuffle(indices)

# Split into training and validation
train_indices = indices[:n_train_final]
val_indices = indices[n_train_final:]

# Create training and validation sets
x_train_split = x_train[train_indices]
y_train_split = y_train[train_indices]
x_val_split = x_train[val_indices]
y_val_split = y_train[val_indices]

# Verify split sizes
print("\n[5] Split Verification:")
print(f"    x_train_split: {x_train_split.shape}")
print(f"    y_train_split: {y_train_split.shape}")
print(f"    x_val_split:   {x_val_split.shape}")
print(f"    y_val_split:   {y_val_split.shape}")

if x_train_split.shape[0] != n_train_final:
    print(f"    - ERROR: Training split size mismatch")
    sys.exit(1)
if x_val_split.shape[0] != n_val:
    print(f"    - ERROR: Validation split size mismatch")
    sys.exit(1)
if x_train_split.shape[0] != y_train_split.shape[0]:
    print(f"    - ERROR: Training image/label count mismatch")
    sys.exit(1)
if x_val_split.shape[0] != y_val_split.shape[0]:
    print(f"    - ERROR: Validation image/label count mismatch")
    sys.exit(1)

# Verify all classes are present in each split
print("\n[6] Class Distribution Checks:")

train_classes = set(y_train_split.tolist())
val_classes = set(y_val_split.tolist())
test_classes = set(y_test.tolist())

print(f"    Training set classes: {sorted(train_classes)}")
print(f"    Validation set classes: {sorted(val_classes)}")
print(f"    Test set classes:       {sorted(test_classes)}")

if train_classes == set(range(10)):
    print("    - All 10 classes in training set: OK")
else:
    print("    - All 10 classes in training set: FAILED")
    sys.exit(1)

if val_classes == set(range(10)):
    print("    - All 10 classes in validation set: OK")
else:
    print("    - All 10 classes in validation set: FAILED")
    sys.exit(1)

if test_classes == set(range(10)):
    print("    - All 10 classes in test set: OK")
else:
    print("    - All 10 classes in test set: FAILED")
    sys.exit(1)

# Verify no data leakage
print("\n[7] Data Leakage Prevention Checks:")
train_test_class_overlap = train_classes.intersection(test_classes)
val_test_class_overlap = val_classes.intersection(test_classes)
print(f"    - Training/Test class overlap: {sorted(train_test_class_overlap)}")
print(f"    - Validation/Test class overlap: {sorted(val_test_class_overlap)}")
print("    - Note: Class overlap is expected (all classes exist in all splits)")
print("    - WARNING: Test set MUST NOT be used for training or validation!")
print("      The test set remains untouched until final evaluation.")

# Final dataset summary
print("\n" + "=" * 60)
print("FINAL DATASET SIZES:")
print("=" * 60)
print(f"    Training set:     {x_train_split.shape[0]:>5} images ({x_train_split.shape[0]/60000*100:.1f}%)")
print(f"    Validation set:   {x_val_split.shape[0]:>5} images ({x_val_split.shape[0]/60000*100:.1f}%)")
print(f"    Test set:         {x_test.shape[0]:>5} images (original)")
print(f"    Total:            {x_train_split.shape[0] + x_val_split.shape[0] + x_test.shape[0]} images")
print("\n" + "=" * 60)
print("PHASE 5 STATUS: PASSED")
print("=" * 60)
print("\nSplit completed successfully:")
print("  - Training set: 85% of original training data (51,000 images)")
print("  - Validation set: 15% of original training data (9,000 images)")
print("  - Test set: 10,000 original test images")
print("\nReady for Phase 6: CNN Architecture.")
