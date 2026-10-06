"""
MNIST Handwritten Digit Recognition - AAT Project
Phase 4: Data Preprocessing
"""
import sys
print("=" * 60)
print("MNIST CNN - AAT PROJECT - PHASE 4: DATA PREPROCESSING")
print("=" * 60)

import numpy as np
import tensorflow as tf

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("\n[1] Original Data Shapes:")
print(f"    x_train: {x_train.shape}, dtype: {x_train.dtype}")
print(f"    x_test:  {x_test.shape}, dtype: {x_test.dtype}")
print(f"    y_train: {y_train.shape}, dtype: {y_train.dtype}")
print(f"    y_test:  {y_test.shape}, dtype: {y_test.dtype}")

# Step 1: Convert image data to float32
print("\n[2] Converting image data to float32...")
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Step 2: Normalize pixels from 0-255 to 0-1
print("\n[3] Normalizing pixel values (0-255 to 0-1)...")
x_train = x_train / 255.0
x_test = x_test / 255.0

# Verify normalization
print(f"    x_train min after normalization: {x_train.min()}")
print(f"    x_train max after normalization: {x_train.max()}")
print(f"    x_test min after normalization:  {x_test.min()}")
print(f"    x_test max after normalization:  {x_test.max()}")

# Verify normalized values
if x_train.min() == 0.0 and x_train.max() == 1.0 and x_test.min() == 0.0 and x_test.max() == 1.0:
    print("    - Normalization check: OK")
else:
    print("    - Normalization check: FAILED")
    sys.exit(1)

# Step 3: Add grayscale channel dimension (28x28 -> 28x28x1)
print("\n[4] Adding grayscale channel dimension...")
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print(f"    x_train shape after expansion: {x_train.shape}")
print(f"    x_test shape after expansion:  {x_test.shape}")

# Verify final shapes
expected_shape = (60000, 28, 28, 1)
if x_train.shape != expected_shape:
    print(f"    - ERROR: x_train shape mismatch: {x_train.shape} != {expected_shape}")
    sys.exit(1)
else:
    print(f"    - x_train shape: OK (should be {expected_shape})")

expected_shape_test = (10000, 28, 28, 1)
if x_test.shape != expected_shape_test:
    print(f"    - ERROR: x_test shape mismatch: {x_test.shape} != {expected_shape_test}")
    sys.exit(1)
else:
    print(f"    - x_test shape: OK (should be {expected_shape_test})")

# Verify dtype and value ranges
print(f"\n[5] Final Data Verification:")
print(f"    x_train dtype: {x_train.dtype}")
print(f"    x_train min: {x_train.min()}, max: {x_train.max()}")

if x_train.dtype == 'float32':
    print("    - dtype = float32: OK")
else:
    print("    - dtype = float32: FAILED")
    sys.exit(1)

if x_train.min() == 0 and x_train.max() == 1:
    print("    - min = 0, max = 1: OK")
else:
    print("    - min = 0, max = 1: FAILED")
    sys.exit(1)

if x_train.shape == (60000, 28, 28, 1):
    print("    - shape = (60000, 28, 28, 1): OK")
else:
    print("    - shape = (60000, 28, 28, 1): FAILED")
    sys.exit(1)

# Verify labels alignment
print("\n[6] Label Verification:")
print(f"    y_train: {y_train.shape}, dtype: {y_train.dtype}")
print(f"    y_test:  {y_test.shape}, dtype: {y_test.dtype}")
print(f"    Number of unique training labels: {len(set(y_train.tolist()))}")
print(f"    Number of unique test labels:     {len(set(y_test.tolist()))}")

if len(set(y_train.tolist())) == 10 and len(set(y_test.tolist())) == 10:
    print("    - All 10 classes present in both training and test sets: OK")
else:
    print("    - All 10 classes present: FAILED")
    sys.exit(1)

print("\n" + "=" * 60)
print("PHASE 4 STATUS: PASSED")
print("=" * 60)
print("\nPreprocessing completed successfully:")
print("  - Data type: float32")
print("  - Pixel range: [0, 1]")
print("  - Image shape: (N, 28, 28, 1)")
print("  - Labels: 0-9 (10 classes)")
print("\nReady for Phase 5: Training/Validation/Test Split.")
