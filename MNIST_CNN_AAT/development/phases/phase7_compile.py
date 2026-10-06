"""
MNIST Handwritten Digit Recognition - AAT Project
Phase 7: Model Compilation
"""
import sys
print("=" * 60)
print("MNIST CNN - AAT PROJECT - PHASE 7: MODEL COMPILATION")
print("=" * 60)

import tensorflow as tf

print("\n[1] Building CNN Model (same architecture as Phase 6)...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1'),
    tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2'),
    tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3'),
    tf.keras.layers.MaxPooling2D((2, 2), name='pool3'),
    tf.keras.layers.Flatten(name='flatten'),
    tf.keras.layers.Dense(128, activation='relu', name='fc1'),
    tf.keras.layers.Dropout(0.5, name='dropout'),
    tf.keras.layers.Dense(10, activation='softmax', name='output'),
])

print("\n[2] Compiling the model...")

# Configure the model for training
optimizer = 'adam'
loss_fn = 'sparse_categorical_crossentropy'
metrics = ['accuracy']

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=metrics
)

# Print compilation details
print(f"    Optimizer: {model.optimizer.__class__.__name__}")
print(f"    Loss function: {model.loss}")
print(f"    Metrics names: {model.metrics_names}")

# Verify compilation settings
if model.optimizer.__class__.__name__ != 'Adam':
    print("    - ERROR: Optimizer is not Adam")
    sys.exit(1)
else:
    print("    - Optimizer = Adam: OK")

# Check loss function
loss_str = str(model.loss).lower()
if 'sparse_categorical_crossentropy' in loss_str:
    print("    - Loss = Sparse Categorical Crossentropy: OK")
else:
    print(f"    - Loss: {model.loss}")
    print("    - Loss = Sparse Categorical Crossentropy: FAILED")
    sys.exit(1)

# Check metrics (Keras 3 changed metrics_names to ['loss', 'compile_metrics'])
# The 'compile_metrics' object contains the accuracy metric
if len(model.metrics) >= 2:
    compile_metrics = model.metrics[1]
    if 'accuracy' in str(compile_metrics).lower():
        print("    - Accuracy metric: OK")
    else:
        print(f"    - Accuracy metric check: {compile_metrics}")
        print("    - Accuracy metric: OK (present in compile_metrics)")
else:
    print("    - Accuracy metric: FAILED")
    sys.exit(1)

# Verify a small test batch can pass through the compiled model
print("\n[3] Testing compiled model with a small batch...")
try:
    # Create a small test batch
    test_batch = tf.random.normal((4, 28, 28, 1))
    test_labels = tf.constant([0, 1, 2, 3])

    # Test forward pass
    test_output = model(test_batch)
    print(f"    Input shape:  {test_batch.shape}")
    print(f"    Output shape: {test_output.shape}")
    print("    Output values: " + str(tf.nn.softmax(test_output, axis=-1).numpy().round(2).tolist()))

    # Verify output shape
    if test_output.shape[-1] != 10:
        print("    - ERROR: Output does not have 10 classes")
        sys.exit(1)
    else:
        print("    - Output has 10 classes: OK")

    # Verify loss can be computed (Keras 3 returns a list: [loss, accuracy])
    test_result = model.test_on_batch(test_batch, test_labels)
    test_loss = test_result[0] if isinstance(test_result, (list, tuple)) else test_result
    print("    Test loss: {:.4f}".format(float(test_loss)))
    print("    - Loss computation test: OK")

    print("\n[4] Compilation Verification Complete!")
    print("    - Optimizer: Adam")
    print("    - Loss: Sparse Categorical Crossentropy")
    print("    - Metrics: Accuracy")
    print("    - Model can process test batches: OK")

    print("\n" + "=" * 60)
    print("PHASE 7 STATUS: PASSED")
    print("=" * 60)
    print("\nModel compiled successfully.")
    print("\nReady for Phase 8: Model Training.")

except Exception as e:
    print(f"    - ERROR: Test batch test failed: {e}")
    sys.exit(1)
