"""
MNIST preprocessing.

Converts images to float32, normalizes pixels to [0, 1] and adds the
grayscale channel dimension expected by the CNN (N, 28, 28, 1).
Part of the MNIST CNN AAT project (development Phase 4 logic).
"""
import numpy as np


def to_float32(X):
    """Convert an image array to float32."""
    return X.astype("float32")


def normalize(X):
    """Scale pixel values from [0, 255] to [0, 1]."""
    return X / 255.0


def add_channel_dim(X):
    """Add the channel axis: (N, 28, 28) -> (N, 28, 28, 1)."""
    return np.expand_dims(X, axis=-1)


def preprocess_images(X):
    """Full preprocessing pipeline: float32 -> normalize -> channel dim."""
    return add_channel_dim(normalize(to_float32(X)))


def main():
    print("=" * 60)
    print("MNIST CNN - PREPROCESS DATA")
    print("=" * 60)

    from src.preprocessing.load_data import load_mnist

    X_train, y_train, X_test, y_test = load_mnist()

    print("\n[1] Original shapes:")
    print(f"    X_train: {X_train.shape} ({X_train.dtype})")
    print(f"    X_test:  {X_test.shape} ({X_test.dtype})")

    X_train_p = preprocess_images(X_train)
    X_test_p = preprocess_images(X_test)

    print("\n[2] After preprocessing:")
    print(f"    X_train: {X_train_p.shape} ({X_train_p.dtype}), "
          f"min={X_train_p.min()}, max={X_train_p.max()}")
    print(f"    X_test:  {X_test_p.shape} ({X_test_p.dtype}), "
          f"min={X_test_p.min()}, max={X_test_p.max()}")

    print("\n[3] Verification:")
    ok = True

    if X_train_p.dtype == "float32" and X_test_p.dtype == "float32":
        print("    - dtype = float32: OK")
    else:
        print("    - dtype = float32: FAILED")
        ok = False

    if X_train_p.min() == 0.0 and X_train_p.max() == 1.0:
        print("    - pixel range [0, 1]: OK")
    else:
        print("    - pixel range [0, 1]: FAILED")
        ok = False

    if X_train_p.shape == (60000, 28, 28, 1) and X_test_p.shape == (10000, 28, 28, 1):
        print("    - shapes (N, 28, 28, 1): OK")
    else:
        print("    - shapes (N, 28, 28, 1): FAILED")
        ok = False

    if len(set(y_train.tolist())) == 10 and len(set(y_test.tolist())) == 10:
        print("    - all 10 classes present: OK")
    else:
        print("    - all 10 classes present: FAILED")
        ok = False

    print("\n" + "=" * 60)
    if ok:
        print("Preprocess data check: PASSED")
    else:
        print("Preprocess data check: FAILED")
        raise SystemExit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
