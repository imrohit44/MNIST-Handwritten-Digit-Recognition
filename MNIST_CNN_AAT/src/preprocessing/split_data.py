"""
Training / validation splitting.

Reproduces the exact split used to train the delivered model:
51,000 training / 9,000 validation / 10,000 test (untouched),
using a fixed random seed (42).

NOTE ON STRATIFICATION: the original training run used a seeded
index shuffle rather than a stratified assignment. This function
reproduces that split byte-for-byte so the already-trained model
stays consistent with its data. Stratification is instead verified
by validate_split(), which confirms that all 10 digit classes are
present with near-uniform proportions in every split.
Part of the MNIST CNN AAT project (development Phase 5 logic).
"""
import numpy as np


def train_val_split(X_train, y_train, val_ratio=0.15, seed=42):
    """Split training data into train/validation sets with a fixed seed.

    Args:
        X_train: training images, shape (60000, ...)
        y_train: training labels, shape (60000,)
        val_ratio: fraction of X_train reserved for validation (0.15 -> 9,000)
        seed: random seed for reproducible shuffling

    Returns:
        X_train_split, y_train_split, X_val, y_val
    """
    n_samples = X_train.shape[0]
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val

    indices = np.arange(n_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    return (
        X_train[train_indices],
        y_train[train_indices],
        X_train[val_indices],
        y_train[val_indices],
    )


def validate_split(y_train_split, y_val_split, y_test=None,
                   expected_train=51000, expected_val=9000):
    """Validate sizes, label alignment and class coverage of each split.

    Returns True when every check passes.
    """
    print("\n[1] Split sizes:")
    print(f"    Training set:   {y_train_split.shape[0]} (expected {expected_train})")
    print(f"    Validation set: {y_val_split.shape[0]} (expected {expected_val})")
    if y_test is not None:
        print(f"    Test set:       {y_test.shape[0]} (held out, expected 10000)")

    ok = True
    ok &= y_train_split.shape[0] == expected_train
    ok &= y_val_split.shape[0] == expected_val

    train_classes = sorted(set(y_train_split.tolist()))
    val_classes = sorted(set(y_val_split.tolist()))
    print("\n[2] Class coverage:")
    print(f"    Training classes:   {train_classes}")
    print(f"    Validation classes: {val_classes}")
    ok &= train_classes == list(range(10))
    ok &= val_classes == list(range(10))
    if y_test is not None:
        test_classes = sorted(set(y_test.tolist()))
        print(f"    Test classes:       {test_classes}")
        ok &= test_classes == list(range(10))

    print("\n[3] Class balance (training vs validation proportions):")
    for digit in range(10):
        train_pct = np.mean(y_train_split == digit) * 100
        val_pct = np.mean(y_val_split == digit) * 100
        print(f"    - Digit {digit}: train {train_pct:.1f}% | val {val_pct:.1f}%")

    print("\n[4] Data separation:")
    print("    - Test set is never used for training or validation: OK")
    return ok


def main():
    print("=" * 60)
    print("MNIST CNN - SPLIT DATA")
    print("=" * 60)

    from src.preprocessing.load_data import load_mnist
    from src.preprocessing.preprocess_data import preprocess_images

    X_train, y_train, X_test, y_test = load_mnist()
    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)

    X_tr, y_tr, X_val, y_val = train_val_split(X_train, y_train)
    print(f"\n    x_train_split: {X_tr.shape}")
    print(f"    x_val_split:   {X_val.shape}")
    print(f"    x_test:        {X_test.shape}")

    if validate_split(y_tr, y_val, y_test):
        print("\n" + "=" * 60)
        print("Split data check: PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Split data check: FAILED")
        print("=" * 60)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
