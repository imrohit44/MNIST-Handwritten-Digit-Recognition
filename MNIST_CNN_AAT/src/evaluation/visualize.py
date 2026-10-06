"""
Result visualization.

Generates the six project figures from saved artifacts and the trained
model. Visualization is kept out of the model and preprocessing modules.
Part of the MNIST CNN AAT project (development Phase 10 logic).

Figures produced in results/figures/:
    training_curves.png             Loss & accuracy curves
    sample_predictions.png          5x5 grid of test predictions
    misclassified_examples.png      Wrongly classified digits
    confusion_matrix.png            Raw confusion matrix
    confusion_matrix_normalized.png Row-normalized confusion matrix
    per_class_accuracy.png          Per-digit accuracy bar chart

Run from the project root:
    python -m src.evaluation.visualize
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from src.preprocessing.load_data import load_mnist
from src.preprocessing.preprocess_data import preprocess_images

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_cnn_model.keras"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = RESULTS_DIR / "data"
FIGURES_DIR = RESULTS_DIR / "figures"

HISTORY_PATH = DATA_DIR / "training_history.npy"
CM_PATH = DATA_DIR / "confusion_matrix.npy"
Y_PRED_PATH = DATA_DIR / "y_pred.npy"

RANDOM_SEED = 42


def _require(path, hint):
    """Fail with a clear message when a required artifact is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing: {path}\n{hint}")
    return Path(path)


def plot_training_curves(history_path=HISTORY_PATH,
                          save_path=FIGURES_DIR / "training_curves.png"):
    """Plot training/validation loss and accuracy curves."""
    train_loss, train_acc, val_loss, val_acc = np.load(_require(
        history_path, "Run training first (python -m src.model.training --retrain)."))
    epochs = range(1, len(train_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, train_loss, "b-o", label="Training Loss", markersize=4)
    ax1.plot(epochs, val_loss, "r-o", label="Validation Loss", markersize=4)
    ax1.set_title("Model Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, "g-o", label="Training Accuracy", markersize=4)
    ax2.plot(epochs, val_acc, "c-o", label="Validation Accuracy", markersize=4)
    ax2.set_title("Model Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Training History - Loss & Accuracy", fontsize=14)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    - Saved: {save_path}")
    return save_path


def plot_sample_predictions(model, X_test, y_test, n=25,
                            save_path=FIGURES_DIR / "sample_predictions.png"):
    """Plot a grid of random test images with true vs predicted labels."""
    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    np.random.seed(RANDOM_SEED)
    sample_indices = np.random.choice(len(y_test), n, replace=False)
    side = int(np.sqrt(n))
    fig, axes = plt.subplots(side, side, figsize=(12, 12))

    for axis, idx in zip(axes.flat, sample_indices):
        axis.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        correct = y_pred[idx] == y_test[idx]
        axis.set_title(
            f"T:{y_test[idx]}\nP:{y_pred[idx]}",
            fontsize=9,
            color="green" if correct else "red",
        )
        axis.axis("off")

    plt.suptitle("Sample Predictions (Green=Correct, Red=Wrong)", fontsize=14, y=1.01)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    - Saved: {save_path}")
    return save_path


def plot_misclassified(model, X_test, y_test, max_examples=9,
                       save_path=FIGURES_DIR / "misclassified_examples.png"):
    """Plot up to 9 misclassified test images."""
    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    error_indices = np.where(y_pred != y_test)[0]

    if len(error_indices) == 0:
        print("    - No misclassified examples found, figure skipped.")
        return None

    n = min(max_examples, len(error_indices))
    side = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(side, side, figsize=(8, 8))
    axes_flat = np.atleast_1d(axes).ravel().tolist()

    for axis, idx in zip(axes_flat, error_indices[:n]):
        axis.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        axis.set_title(f"True: {y_test[idx]}\nPred: {y_pred[idx]}",
                       fontsize=10, color="red")
        axis.axis("off")
    for axis in axes_flat[n:]:
        axis.axis("off")

    plt.suptitle("Misclassified Examples", fontsize=14)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    - Saved: {save_path} ({len(error_indices)} total errors)")
    return save_path


def plot_confusion_matrix(cm,
                          save_path=FIGURES_DIR / "confusion_matrix.png"):
    """Plot the raw confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=range(10), yticklabels=range(10),
        ax=ax, cbar_kws={"label": "Count"},
    )
    ax.set_title("Confusion Matrix - MNIST CNN", fontsize=14)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    - Saved: {save_path}")
    return save_path


def plot_confusion_matrix_normalized(cm,
                                     save_path=FIGURES_DIR / "confusion_matrix_normalized.png"):
    """Plot the row-normalized confusion matrix heatmap."""
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Greens",
        xticklabels=range(10), yticklabels=range(10),
        ax=ax, cbar_kws={"label": "Proportion"},
    )
    ax.set_title("Normalized Confusion Matrix - MNIST CNN", fontsize=14)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    - Saved: {save_path}")
    return save_path


def plot_per_class_accuracy(cm,
                            save_path=FIGURES_DIR / "per_class_accuracy.png"):
    """Plot per-digit accuracy bar chart from the confusion matrix."""
    class_acc = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(10), class_acc, color="steelblue", edgecolor="black")
    ax.set_title("Per-Class Accuracy", fontsize=14)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.95, 1.0)
    ax.set_xticks(range(10))
    ax.grid(True, axis="y", alpha=0.3)
    for bar, acc_val in zip(bars, class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{acc_val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    - Saved: {save_path}")
    return save_path


def main():
    print("=" * 60)
    print("MNIST CNN - VISUALIZATION")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    saved = []

    print("\n[1] Training curves...")
    saved.append(plot_training_curves())

    print("\n[2] Loading test set...")
    X_train_raw, y_train, X_test_raw, y_test = load_mnist()
    X_test = preprocess_images(X_test_raw)
    print(f"    - X_test: {X_test.shape}")

    print("\n[3] Loading trained model...")
    _require(MODEL_PATH, "Train the model first (python -m src.model.training --retrain).")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\n[4] Sample predictions...")
    saved.append(plot_sample_predictions(model, X_test, y_test))

    print("\n[5] Misclassified examples...")
    saved.append(plot_misclassified(model, X_test, y_test))

    print("\n[6] Confusion matrices...")
    cm = np.load(_require(
        CM_PATH, "Run evaluation first (python -m src.evaluation.evaluate)."))
    saved.append(plot_confusion_matrix(cm))
    saved.append(plot_confusion_matrix_normalized(cm))

    print("\n[7] Per-class accuracy...")
    saved.append(plot_per_class_accuracy(cm))

    print("\n[8] Visualization summary:")
    all_ok = True
    for figure in saved:
        if figure and Path(figure).exists():
            size_kb = Path(figure).stat().st_size / 1024
            print(f"    [OK] {Path(figure).name} ({size_kb:.1f} KB)")
        else:
            all_ok = False
            print(f"    [MISSING] {figure}")

    print("\n" + "=" * 60)
    if all_ok:
        print("Visualization check: PASSED")
    else:
        print("Visualization check: FAILED")
        print("=" * 60)
        raise SystemExit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
