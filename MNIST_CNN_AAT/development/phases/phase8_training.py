"""
MNIST Handwritten Digit Recognition - AAT Project
Phase 8: Model Training with Callbacks
"""
import sys
print("=" * 60)
print("MNIST CNN - AAT PROJECT - PHASE 8: MODEL TRAINING")
print("=" * 60)

import os
import numpy as np
import tensorflow as tf

# Paths
MODELS_DIR = r"D:\DL AAT\MNIST_CNN_AAT\models"
RESULTS_DIR = r"D:\DL AAT\MNIST_CNN_AAT\results"
MODEL_PATH = os.path.join(MODELS_DIR, "mnist_cnn_model.keras")
HISTORY_PATH = os.path.join(RESULTS_DIR, "data", "training_history.npy")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "data"), exist_ok=True)

# Reproduce Phase 5 split exactly
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print("\n[1] Loading and preprocessing MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Split: 51,000 train / 9,000 val / 10,000 test (same as Phase 5)
n_train = x_train.shape[0]
n_val = int(n_train * 0.15)
n_train_final = n_train - n_val

indices = np.arange(n_train)
np.random.seed(RANDOM_SEED)
np.random.shuffle(indices)

train_indices = indices[:n_train_final]
val_indices = indices[n_train_final:]

x_train_split = x_train[train_indices]
y_train_split = y_train[train_indices]
x_val_split = x_train[val_indices]
y_val_split = y_train[val_indices]

print(f"    Training set:   {x_train_split.shape[0]} images")
print(f"    Validation set: {x_val_split.shape[0]} images")
print(f"    Test set:       {x_test.shape[0]} images (not used in training)")

if x_train_split.shape[0] != 51000 or x_val_split.shape[0] != 9000:
    print("    - ERROR: Split sizes do not match Phase 5")
    sys.exit(1)
print("    - Split matches Phase 5: OK")

print("\n[2] Building and compiling CNN model (Phases 6 & 7)...")
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
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("    - Model built and compiled: OK")

print("\n[3] Setting up callbacks...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
]
print("    - EarlyStopping (monitor=val_loss, patience=3, restore_best_weights=True)")
print("    - ModelCheckpoint (monitor=val_accuracy, save_best_only=True)")
print(f"    - Model will be saved to: {MODEL_PATH}")

# Training configuration
EPOCHS = 15
BATCH_SIZE = 128

print(f"\n[4] Training the model...")
print(f"    Epochs: {EPOCHS} (with early stopping)")
print(f"    Batch size: {BATCH_SIZE}")
print("-" * 60)

history = model.fit(
    x_train_split, y_train_split,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_val_split, y_val_split),
    callbacks=callbacks,
    verbose=2
)

print("-" * 60)

# Verify training completed
print("\n[5] Verifying training results...")
final_epoch = len(history.history['loss'])
print(f"    Epochs actually run: {final_epoch}")
print(f"    Final training accuracy:   {history.history['accuracy'][-1]:.4f}")
print(f"    Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"    Best validation accuracy:  {max(history.history['val_accuracy']):.4f}")

if final_epoch < 1:
    print("    - ERROR: Training did not run")
    sys.exit(1)
else:
    print("    - Training completed: OK")

best_val_acc = max(history.history['val_accuracy'])
if best_val_acc < 0.90:
    print(f"    - WARNING: Best validation accuracy {best_val_acc:.4f} is below 0.90")
else:
    print(f"    - Best validation accuracy >= 0.90: OK")

# Save history for Phase 10 visualization
np.save(HISTORY_PATH, np.array([
    history.history['loss'],
    history.history['accuracy'],
    history.history['val_loss'],
    history.history['val_accuracy'],
]))
print(f"    - Training history saved to: {HISTORY_PATH}")

# Verify model file was saved
if os.path.exists(MODEL_PATH):
    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"    - Model file saved: {MODEL_PATH} ({size_kb:.1f} KB)")
else:
    print("    - ERROR: Model file was not saved")
    sys.exit(1)

# Quick validation evaluation
print("\n[6] Quick validation evaluation of saved model...")
val_loss, val_acc = model.evaluate(x_val_split, y_val_split, verbose=0)
print(f"    Validation loss: {val_loss:.4f}")
print(f"    Validation accuracy: {val_acc:.4f}")

print("\n" + "=" * 60)
print("PHASE 8 STATUS: PASSED")
print("=" * 60)
print("\nModel training completed successfully:")
print(f"  - Trained for {final_epoch} epochs (early stopping active)")
print(f"  - Best validation accuracy: {best_val_acc:.4f}")
print(f"  - Model saved to: {MODEL_PATH}")
print("\nReady for Phase 9: Model Evaluation.")
