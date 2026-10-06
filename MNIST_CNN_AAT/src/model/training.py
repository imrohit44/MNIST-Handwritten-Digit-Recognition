"""
Model training.

Handles fitting, callbacks (EarlyStopping / ModelCheckpoint), training
history and saving the best model.
Part of the MNIST CNN AAT project (development Phase 8 logic).

Training configuration (UNCHANGED from the validated training run):
    Batch size:            128
    Maximum epochs:        15
    EarlyStopping:         monitor=val_loss, patience=3,
                           restore_best_weights=True
    ModelCheckpoint:       monitor=val_accuracy, save_best_only=True

The delivered model in models/mnist_cnn_model.keras is already trained.
This module only retrain when explicitly requested with --retrain.
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.model.architecture import build_model
from src.model.compile import compile_model
from src.preprocessing.load_data import load_mnist
from src.preprocessing.preprocess_data import preprocess_images
from src.preprocessing.split_data import train_val_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_cnn_model.keras"
HISTORY_PATH = PROJECT_ROOT / "results" / "data" / "training_history.npy"

RANDOM_SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 15


def get_callbacks(model_path=MODEL_PATH):
    """EarlyStopping + ModelCheckpoint callbacks used for training."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


def save_history(history, history_path=HISTORY_PATH):
    """Save the training history as a (4, N) numpy array.

    Rows: loss, accuracy, val_loss, val_accuracy.
    """
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        history_path,
        np.array([
            history.history["loss"],
            history.history["accuracy"],
            history.history["val_loss"],
            history.history["val_accuracy"],
        ]),
    )
    return history_path


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                model_path=MODEL_PATH, history_path=HISTORY_PATH):
    """Train the model with the project's validated configuration."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=get_callbacks(model_path),
        verbose=2,
    )
    save_history(history, history_path)
    return history


def main():
    print("=" * 60)
    print("MNIST CNN - TRAINING")
    print("=" * 60)

    if "--retrain" not in sys.argv:
        print("\nThe delivered model is ALREADY TRAINED and validated:")
        print(f"    Model:  {MODEL_PATH}")
        print("    Result: 99.27% test accuracy (73 / 10,000 misclassified)")
        print("\nRetraining is NOT required for evaluation or reporting.")
        print("\nConfiguration used for the delivered model:")
        print(f"    Batch size:     {BATCH_SIZE}")
        print(f"    Maximum epochs: {MAX_EPOCHS}")
        print("    EarlyStopping:  val_loss, patience 3, restore best weights")
        print("    ModelCheckpoint: val_accuracy, save best only")
        print("\nTo retrain from scratch (overwrites the delivered model):")
        print("    python -m src.model.training --retrain")
        return

    print("\n[1] Retraining requested. Loading and preparing data...")
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    X_train_raw, y_train, X_test, y_test = load_mnist()
    X_train = preprocess_images(X_train_raw)
    X_test = preprocess_images(X_test)
    X_tr, y_tr, X_val, y_val = train_val_split(X_train, y_train)
    print(f"    Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("\n[2] Building and compiling model...")
    model = build_model()
    compile_model(model)

    print("\n[3] Training...")
    history = train_model(model, X_tr, y_tr, X_val, y_val)

    print("\n[4] Training summary:")
    best_epoch = int(np.argmax(history.history["val_accuracy"]))
    print(f"    Epochs run:           {len(history.history['loss'])}")
    print(f"    Best epoch (val acc): {best_epoch + 1}")
    print(f"    Best val accuracy:    {max(history.history['val_accuracy']):.4f}")
    print(f"    Model saved to:       {MODEL_PATH}")
    print(f"    History saved to:     {HISTORY_PATH}")

    print("\n" + "=" * 60)
    print("Training: COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
