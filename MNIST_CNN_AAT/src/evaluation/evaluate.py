"""
Final model evaluation.

Primary module for evaluation work: loads the trained .keras model,
loads the held-out MNIST test set and computes test loss, test
accuracy, predictions, classification report, per-class
precision/recall/F1, confusion matrix and misclassification analysis.

The test set (10,000 images) is NEVER used for training or validation.
Part of the MNIST CNN AAT project (development Phase 9 logic).

Run from the project root:
    python -m src.evaluation.evaluate
"""
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.preprocessing.load_data import load_mnist
from src.preprocessing.preprocess_data import preprocess_images

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_cnn_model.keras"
DATA_DIR = PROJECT_ROOT / "results" / "data"

# Results reported in the project report (validated against every run).
REPORTED = {
    "loss": 0.0210,
    "accuracy": 0.9927,
    "precision_macro": 0.9927,
    "recall_macro": 0.9927,
    "f1_macro": 0.9927,
    "misclassified": 73,
}


def load_trained_model(model_path=MODEL_PATH):
    """Load the trained CNN from disk."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            "The delivered model lives in models/mnist_cnn_model.keras."
        )
    return tf.keras.models.load_model(model_path)


def load_test_set():
    """Load ONLY the MNIST test set, preprocessed for the CNN.

    Returns:
        X_test: (10000, 28, 28, 1) float32 in [0, 1]
        y_test: (10000,) uint8 ground-truth labels
    """
    _, _, X_test, y_test = load_mnist()
    return preprocess_images(X_test), y_test


def evaluate_model(model, X_test, y_test):
    """Compute test loss and test accuracy. Returns {'loss', 'accuracy'}."""
    results = model.evaluate(X_test, y_test, verbose=0)
    if isinstance(results, (list, tuple)):
        loss, accuracy = float(results[0]), float(results[1])
    else:  # single metric configured
        loss, accuracy = float(results), float("nan")
    return {"loss": loss, "accuracy": accuracy}


def predict(model, X_test):
    """Return (predicted labels, class probabilities)."""
    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    return predictions, probabilities


def compute_metrics(y_test, y_pred):
    """Macro and per-class precision / recall / F1 plus error counts."""
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )
    precision_per_class, recall_per_class, f1_per_class, support = (
        precision_recall_fscore_support(y_test, y_pred, average=None)
    )
    return {
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_per_class": np.asarray(precision_per_class),
        "recall_per_class": np.asarray(recall_per_class),
        "f1_per_class": np.asarray(f1_per_class),
        "support": np.asarray(support),
        "misclassified": int(np.sum(np.asarray(y_test) != np.asarray(y_pred))),
    }


def classification_report_table(y_test, y_pred):
    """sklearn classification report with human-readable class names."""
    target_names = [f"Digit {i}" for i in range(10)]
    return classification_report(
        y_test, y_pred,
        labels=list(range(10)),
        target_names=target_names,
        digits=4,
    )


def compute_confusion_matrix(y_test, y_pred):
    """10x10 confusion matrix (rows = true, columns = predicted)."""
    return confusion_matrix(y_test, y_pred, labels=list(range(10)))



def print_confusion_matrix(cm):
    """Print the 10x10 confusion matrix as a table."""
    print("        " + "".join(f"pred {d:<6}" for d in range(10)))
    for digit in range(10):
        row = f"true {digit:<4} "
        row += "".join(f"{int(cm[digit, d]):<10}" for d in range(10))
        print(row)


def print_per_class_metrics(metrics):
    """Print per-class precision, recall, F1 and support."""
    print(f"{'Digit':<7}{'Precision':<12}{'Recall':<12}{'F1-score':<12}{'Support':<8}")
    print("-" * 51)
    for digit in range(10):
        print(
            f"{digit:<7}"
            f"{metrics['precision_per_class'][digit]:<12.4f}"
            f"{metrics['recall_per_class'][digit]:<12.4f}"
            f"{metrics['f1_per_class'][digit]:<12.4f}"
            f"{int(metrics['support'][digit]):<8}"
        )


def verify_reported_results(metrics):
    """Compare measured metrics against the reported project results."""
    print("\n[8] Verifying reported results:")
    checks = [
        ("Test loss", REPORTED["loss"], metrics["loss"], "abs", 5e-4),
        ("Test accuracy", REPORTED["accuracy"], metrics["accuracy"], "round", 4),
        ("Macro precision", REPORTED["precision_macro"], metrics["precision_macro"], "round", 4),
        ("Macro recall", REPORTED["recall_macro"], metrics["recall_macro"], "round", 4),
        ("Macro F1", REPORTED["f1_macro"], metrics["f1_macro"], "round", 4),
        ("Misclassified", REPORTED["misclassified"], metrics["misclassified"], "exact", None),
    ]
    ok = True
    for name, reported, measured, mode, param in checks:
        if mode == "abs":
            match = abs(reported - measured) < param
        elif mode == "round":
            match = round(float(measured), param) == reported
        else:
            match = reported == measured
        status = "OK" if match else "MISMATCH"
        ok &= match
        print(f"    - {name:<16} reported={reported:<10} "
              f"measured={round(float(measured), 4)}  [{status}]")
    return ok


def save_artifacts(metrics, cm, y_pred, out_dir=DATA_DIR):
    """Save machine-readable evaluation artifacts to results/data/."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        out_dir / "eval_metrics.npy",
        np.array([
            metrics["accuracy"],
            metrics["precision_macro"],
            metrics["recall_macro"],
            metrics["f1_macro"],
            metrics["loss"],
        ]),
    )
    np.save(out_dir / "confusion_matrix.npy", cm)
    np.save(out_dir / "y_pred.npy", y_pred)
    return out_dir


def main():
    print("=" * 60)
    print("MNIST CNN - FINAL MODEL EVALUATION")
    print("=" * 60)

    print("\n[1] Loading trained model...")
    model = load_trained_model()
    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"    - Loaded: {MODEL_PATH} ({size_kb:.1f} KB)")

    print("\n[2] Loading test set (held out from training)...")
    X_test, y_test = load_test_set()
    print(f"    - X_test: {X_test.shape}, y_test: {y_test.shape}")
    print("    - Test set is separate from training/validation: OK")

    print("\n[3] Evaluating on the test set...")
    evaluation = evaluate_model(model, X_test, y_test)
    print(f"    - Test loss:     {evaluation['loss']:.4f}")
    print(f"    - Test accuracy: {evaluation['accuracy']:.4f} "
          f"({evaluation['accuracy'] * 100:.2f}%)")

    print("\n[4] Generating predictions...")
    y_pred, probabilities = predict(model, X_test)
    print(f"    - Predictions: {y_pred.shape}, probabilities: {probabilities.shape}")

    metrics = compute_metrics(y_test, y_pred)
    metrics.update(evaluation)

    print("\n[5] Classification report:")
    print(classification_report_table(y_test, y_pred))

    print("\n[6] Macro averages:")
    print(f"    - Precision: {metrics['precision_macro']:.4f}")
    print(f"    - Recall:    {metrics['recall_macro']:.4f}")
    print(f"    - F1-score:  {metrics['f1_macro']:.4f}")

    print("\n[7] Per-class metrics:")
    print_per_class_metrics(metrics)

    cm = compute_confusion_matrix(y_test, y_pred)
    print("\n[7b] Confusion matrix (rows = true, columns = predicted):")
    print_confusion_matrix(cm)

    total = len(y_test)
    errors = metrics["misclassified"]
    print(f"\n    Misclassified: {errors} / {total} "
          f"({errors / total * 100:.2f}%)")

    verified = verify_reported_results(metrics)

    print("\n[9] Saving artifacts to results/data/...")
    out_dir = save_artifacts(metrics, cm, y_pred)
    print(f"    - Saved: {out_dir / 'eval_metrics.npy'}")
    print(f"    - Saved: {out_dir / 'confusion_matrix.npy'}")
    print(f"    - Saved: {out_dir / 'y_pred.npy'}")

    print("\n" + "=" * 60)
    if verified:
        print("Evaluation check: PASSED (matches reported results)")
    else:
        print("Evaluation check: FAILED (measured values differ from report)")
        print("=" * 60)
        raise SystemExit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
