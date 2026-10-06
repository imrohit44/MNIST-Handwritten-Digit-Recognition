"""
MNIST data loading.

Loads the MNIST handwritten digit dataset from TensorFlow/Keras.
Part of the MNIST CNN AAT project (development Phase 2 logic).
"""
import tensorflow as tf


def load_mnist():
    """Load the MNIST dataset.

    Returns:
        X_train: numpy array, shape (60000, 28, 28), dtype uint8
        y_train: numpy array, shape (60000,), dtype uint8
        X_test:  numpy array, shape (10000, 28, 28), dtype uint8
        y_test:  numpy array, shape (10000,), dtype uint8
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    return X_train, y_train, X_test, y_test


def main():
    print("=" * 60)
    print("MNIST CNN - LOAD DATA")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist()

    print("\n[1] Dataset shapes:")
    print(f"    Training images: {X_train.shape} ({X_train.dtype})")
    print(f"    Training labels: {y_train.shape} ({y_train.dtype})")
    print(f"    Test images:     {X_test.shape} ({X_test.dtype})")
    print(f"    Test labels:     {y_test.shape} ({y_test.dtype})")

    print("\n[2] Verification:")
    ok = True
    ok &= X_train.shape == (60000, 28, 28)
    ok &= y_train.shape == (60000,)
    ok &= X_test.shape == (10000, 28, 28)
    ok &= y_test.shape == (10000,)
    ok &= set(y_train.tolist()) | set(y_test.tolist()) == set(range(10))

    if ok:
        print("    - Shapes and labels 0-9: OK")
        print("\nLoad data check: PASSED")
    else:
        print("    - Shapes or labels: FAILED")
        print("\nLoad data check: FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
