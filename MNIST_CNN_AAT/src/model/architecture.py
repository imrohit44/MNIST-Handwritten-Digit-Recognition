"""
CNN model architecture.

Contains ONLY the model definition. Compilation lives in compile.py
and training lives in training.py.
Part of the MNIST CNN AAT project (development Phase 6 logic).

Architecture (UNCHANGED from the validated training run):
    Input (28x28x1)
    -> Conv2D(32, 3x3, relu) -> MaxPool2D
    -> Conv2D(64, 3x3, relu) -> MaxPool2D
    -> Conv2D(128, 3x3, relu) -> MaxPool2D
    -> Flatten
    -> Dense(128, relu) -> Dropout(0.5)
    -> Dense(10, softmax)

Expected parameter count: 241,546
"""
import sys

import tensorflow as tf

EXPECTED_PARAMETERS = 241546


def build_model():
    """Build and return the compiled-shape CNN (not yet compiled)."""
    model = tf.keras.Sequential([
        # Input layer (28x28x1)
        tf.keras.layers.Input(shape=(28, 28, 1)),

        # Block 1: Conv2D -> ReLU -> MaxPooling
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),

        # Block 2: Conv2D -> ReLU -> MaxPooling
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),

        # Block 3: Conv2D -> ReLU -> MaxPooling
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool3'),

        # Flatten and fully connected layers
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(128, activation='relu', name='fc1'),
        tf.keras.layers.Dropout(0.5, name='dropout'),
        tf.keras.layers.Dense(10, activation='softmax', name='output'),
    ])
    return model


def main():
    print("=" * 60)
    print("MNIST CNN - MODEL ARCHITECTURE")
    print("=" * 60)

    model = build_model()

    print("\n[1] Architecture:")
    print("""    Input: 28x28x1
    Conv2D (32 filters, 3x3) -> ReLU -> MaxPooling
    Conv2D (64 filters, 3x3) -> ReLU -> MaxPooling
    Conv2D (128 filters, 3x3) -> ReLU -> MaxPooling
    Flatten
    Dense (128 neurons) -> ReLU -> Dropout(0.5)
    Dense (10 neurons) -> Softmax""")

    print("\n[2] Model summary:")
    model.summary()

    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"\n[3] Parameter count:")
    print(f"    Total parameters:     {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    print(f"    Non-trainable:        {total_params - trainable_params:,}")

    print(f"\n[4] Output shape: {model.output_shape}")

    print("\n[5] Verification:")
    ok = True

    if total_params == EXPECTED_PARAMETERS:
        print(f"    - parameter count == {EXPECTED_PARAMETERS:,}: OK")
    else:
        print(f"    - parameter count == {EXPECTED_PARAMETERS:,}: FAILED "
              f"(got {total_params:,})")
        ok = False

    if model.output_shape[-1] == 10:
        print("    - output has 10 classes: OK")
    else:
        print("    - output has 10 classes: FAILED")
        ok = False

    print("\n" + "=" * 60)
    if ok:
        print("Architecture check: PASSED")
    else:
        print("Architecture check: FAILED")
        print("=" * 60)
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
