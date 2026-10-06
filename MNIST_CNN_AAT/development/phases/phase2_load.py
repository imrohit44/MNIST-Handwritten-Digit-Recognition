"""
MNIST Handwritten Digit Recognition - AAT Project
Phase 2: Load MNIST Dataset
"""
import sys
print("=" * 60)
print("MNIST CNN - AAT PROJECT - PHASE 2: LOAD MNIST DATASET")
print("=" * 60)

# Import TensorFlow
import tensorflow as tf

print("\n[1] Loading MNIST dataset from TensorFlow/Keras...")

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("\n[2] Dataset loaded successfully!")

# Print shapes of the datasets
print("\n[3] Dataset shapes:")
print(f"    Training images: {x_train.shape}")
print(f"    Training labels: {y_train.shape}")
print(f"    Test images:     {x_test.shape}")
print(f"    Test labels:     {y_test.shape}")

# Expected shapes
expected_train_images = (60000, 28, 28)
expected_train_labels = (60000,)
expected_test_images = (10000, 28, 28)
expected_test_labels = (10000,)

# Verify dataset dimensions
tests_passed = True
if x_train.shape != expected_train_images:
    print(f"    - ERROR: Training images shape mismatch: {x_train.shape} != {expected_train_images}")
    tests_passed = False
else:
    print(f"    - Training images shape: OK ({expected_train_images})")

if y_train.shape != expected_train_labels:
    print(f"    - ERROR: Training labels shape mismatch: {y_train.shape} != {expected_train_labels}")
    tests_passed = False
else:
    print(f"    - Training labels shape: OK ({expected_train_labels})")

if x_test.shape != expected_test_images:
    print(f"    - ERROR: Test images shape mismatch: {x_test.shape} != {expected_test_images}")
    tests_passed = False
else:
    print(f"    - Test images shape: OK ({expected_test_images})")

if y_test.shape != expected_test_labels:
    print(f"    - ERROR: Test labels shape mismatch: {y_test.shape} != {expected_test_labels}")
    tests_passed = False
else:
    print(f"    - Test labels shape: OK ({expected_test_labels})")

# Verify labels contain only 0-9
print("\n[4] Verifying label values...")
unique_labels = sorted(set(y_train.tolist()) | set(y_test.tolist()))
print(f"    - Unique labels in training set: {sorted(set(y_train.tolist()))}")
print(f"    - Unique labels in test set:     {sorted(set(y_test.tolist()))}")

if set(unique_labels) == set(range(10)):
    print(f"    - Labels 0-9 confirmed: OK")
else:
    print(f"    - ERROR: Unexpected labels found: {unique_labels}")
    tests_passed = False

# Summary
print("\n" + "=" * 60)
if tests_passed:
    print("PHASE 2 STATUS: PASSED")
else:
    print("PHASE 2 STATUS: FAILED")
print("=" * 60)

if not tests_passed:
    sys.exit(1)

print("\nDataset loaded and verified successfully.")
print("Ready for Phase 3: Dataset Exploration.")
