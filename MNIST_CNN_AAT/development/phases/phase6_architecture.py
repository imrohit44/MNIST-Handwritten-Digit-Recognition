"""
MNIST Handwritten Digit Recognition - AAT Project
Phase 6: CNN Architecture
"""
import sys
print("=" * 60)
print("MNIST CNN - AAT PROJECT - PHASE 6: CNN ARCHITECTURE")
print("=" * 60)

import tensorflow as tf

print("\n[1] Defining CNN Model Architecture...")
print("""    Architecture:
    Input: 28x28x1
    Conv2D (32 filters, 3x3) -> ReLU -> MaxPooling
    Conv2D (64 filters, 3x3) -> ReLU -> MaxPooling
    Conv2D (128 filters, 3x3) -> ReLU -> MaxPooling
    Flatten
    Dense (128 neurons) -> ReLU -> Dropout
    Dense (10 neurons) -> Softmax""")

# Build the CNN model
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

    # Flatten and Fully Connected layers
    tf.keras.layers.Flatten(name='flatten'),
    tf.keras.layers.Dense(128, activation='relu', name='fc1'),
    tf.keras.layers.Dropout(0.5, name='dropout'),
    tf.keras.layers.Dense(10, activation='softmax', name='output'),
])

# Print model summary
print("\n[2] Model Summary:")
model.summary()

# Count parameters
total_params = model.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_params = total_params - trainable_params
print(f"\n[3] Parameter Count:")
print(f"    Total parameters:     {total_params:,}")
print(f"    Trainable parameters: {trainable_params:,}")
print(f"    Non-trainable:        {non_trainable_params:,}")

# Check output shape
print(f"\n[4] Output Shape: {model.output_shape}")
if model.output_shape[-1] == 10:
    print("    - Output has 10 classes: OK")
else:
    print("    - Output has 10 classes: FAILED")
    sys.exit(1)

# Perform dummy forward pass with a small batch
print("\n[5] Performing dummy forward pass...")
try:
    dummy_input = tf.random.normal((2, 28, 28, 1))
    dummy_output = model(dummy_input)
    print(f"    Input shape:  {dummy_input.shape}")
    print(f"    Output shape: {dummy_output.shape}")
    print(f"    Output values: {tf.nn.softmax(dummy_output, axis=-1).numpy().round(3).tolist()}")
    print("    - Dummy forward pass: OK")
except Exception as e:
    print(f"    - ERROR: Dummy forward pass failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("PHASE 6 STATUS: PASSED")
print("=" * 60)
print("\nCNN Architecture built successfully:")
print("  - 3 Conv2D layers (32, 64, 128 filters)")
print("  - 3 MaxPooling layers")
print("  - 1 Dense layer (128 neurons)")
print("  - 1 Dropout layer (50%)")
print("  - 1 Output layer (10 neurons, softmax)")
print("\nReady for Phase 7: Model Compilation.")
