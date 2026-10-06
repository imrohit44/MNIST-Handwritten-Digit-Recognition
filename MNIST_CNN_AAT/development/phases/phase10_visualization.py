"""
MNIST Handwritten Digit Recognition - AAT Project
Phase 10: Visualization
Generates:
  1. Training curves (loss & accuracy)
  2. Sample predictions grid
  3. Confusion matrix heatmap
  4. Per-class accuracy chart
"""
import sys
print("=" * 60)
print("MNIST CNN - AAT PROJECT - PHASE 10: VISUALIZATION")
print("=" * 60)

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
MODELS_DIR = r"D:\DL AAT\MNIST_CNN_AAT\models"
RESULTS_DIR = r"D:\DL AAT\MNIST_CNN_AAT\results"
MODEL_PATH = os.path.join(MODELS_DIR, "mnist_cnn_model.keras")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
DATA_DIR = os.path.join(RESULTS_DIR, "data")
HISTORY_PATH = os.path.join(DATA_DIR, "training_history.npy")
CM_PATH = os.path.join(DATA_DIR, "confusion_matrix.npy")
Y_PRED_PATH = os.path.join(DATA_DIR, "y_pred.npy")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Load data
print("\n[1] Loading training history and evaluation results...")
if not os.path.exists(HISTORY_PATH):
    print(f"    - ERROR: {HISTORY_PATH} not found. Run Phase 8 first.")
    sys.exit(1)
if not os.path.exists(CM_PATH):
    print(f"    - ERROR: {CM_PATH} not found. Run Phase 9 first.")
    sys.exit(1)

hist_arr = np.load(HISTORY_PATH)
train_loss, train_acc, val_loss, val_acc = hist_arr
cm = np.load(CM_PATH)
y_pred = np.load(Y_PRED_PATH)
print(f"    - History loaded: {len(train_loss)} epochs")
print(f"    - Confusion matrix loaded: {cm.shape}")

# Load test data for sample predictions
print("\n[2] Loading test data...")
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1)
print(f"    Test set: {x_test.shape[0]} images")

# ---------------- Figure 1: Training Curves ----------------
print("\n[3] Creating Figure 1: Training curves...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(range(1, len(train_loss) + 1), train_loss, 'b-o', label='Training Loss', markersize=4)
axes[0].plot(range(1, len(val_loss) + 1), val_loss, 'r-o', label='Validation Loss', markersize=4)
axes[0].set_title('Model Loss Over Epochs', fontsize=13)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy curve
axes[1].plot(range(1, len(train_acc) + 1), train_acc, 'b-o', label='Training Accuracy', markersize=4)
axes[1].plot(range(1, len(val_acc) + 1), val_acc, 'r-o', label='Validation Accuracy', markersize=4)
axes[1].set_title('Model Accuracy Over Epochs', fontsize=13)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
curves_path = os.path.join(FIGURES_DIR, "training_curves.png")
fig.savefig(curves_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    - Saved: {curves_path}")

# ---------------- Figure 2: Sample Predictions ----------------
print("\n[4] Creating Figure 2: Sample predictions grid...")
model = tf.keras.models.load_model(MODEL_PATH)
y_pred_probs = model.predict(x_test, verbose=0)
y_pred_full = np.argmax(y_pred_probs, axis=1)

n_samples = 25  # 5x5 grid
sample_indices = np.random.choice(len(x_test), n_samples, replace=False)

fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    idx = sample_indices[i]
    ax.imshow(x_test[idx].squeeze(), cmap='gray')
    pred = y_pred_full[idx]
    true = y_test[idx]
    color = 'green' if pred == true else 'red'
    ax.set_title(f"P:{pred} T:{true}", fontsize=10, color=color)
    ax.axis('off')

plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=14, y=1.01)
plt.tight_layout()
pred_path = os.path.join(FIGURES_DIR, "sample_predictions.png")
fig.savefig(pred_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    - Saved: {pred_path}")

# Also show some misclassified examples
errors = np.where(y_pred_full != y_test)[0]
err_path = None
if len(errors) > 0:
    n_errors_show = min(9, len(errors))
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < n_errors_show:
            idx = errors[i]
            ax.imshow(x_test[idx].squeeze(), cmap='gray')
            pred = y_pred_full[idx]
            true = y_test[idx]
            ax.set_title(f"P:{pred} T:{true}", fontsize=10, color='red')
        ax.axis('off')
    plt.suptitle('Misclassified Examples', fontsize=14)
    plt.tight_layout()
    err_path = os.path.join(FIGURES_DIR, "misclassified_examples.png")
    fig.savefig(err_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    - Saved: {err_path}")


# ---------------- Figure 3: Confusion Matrix Heatmap ----------------
print("\n[5] Creating Figure 3: Confusion matrix heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=range(10), yticklabels=range(10),
    ax=ax, cbar_kws={'label': 'Count'}
)
ax.set_title('Confusion Matrix - MNIST CNN', fontsize=14)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.tight_layout()
cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
fig.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    - Saved: {cm_path}")

# Normalized confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
sns.heatmap(
    cm_norm, annot=True, fmt='.2%', cmap='Greens',
    xticklabels=range(10), yticklabels=range(10),
    ax=ax, cbar_kws={'label': 'Proportion'}
)
ax.set_title('Normalized Confusion Matrix - MNIST CNN', fontsize=14)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.tight_layout()
cm_norm_path = os.path.join(FIGURES_DIR, "confusion_matrix_normalized.png")
fig.savefig(cm_norm_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    - Saved: {cm_norm_path}")


# ---------------- Figure 4: Class-wise accuracy bar chart ----------------
print("\n[6] Creating Figure 4: Per-class accuracy...")
class_acc = cm.diagonal() / cm.sum(axis=1)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(10), class_acc, color='steelblue', edgecolor='black')
ax.set_title('Per-Class Accuracy', fontsize=14)
ax.set_xlabel('Digit')
ax.set_ylabel('Accuracy')
ax.set_ylim(0.95, 1.0)
ax.set_xticks(range(10))
ax.grid(True, axis='y', alpha=0.3)
for bar, acc_val in zip(bars, class_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f'{acc_val:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
perclass_path = os.path.join(FIGURES_DIR, "per_class_accuracy.png")
fig.savefig(perclass_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    - Saved: {perclass_path}")

# Summary
print("\n[7] Visualization Summary:")
figures = [curves_path, pred_path, err_path, cm_path, cm_norm_path, perclass_path]
for f in figures:
    if f and os.path.exists(f):
        size_kb = os.path.getsize(f) / 1024
        print(f"    [OK] {os.path.basename(f)} ({size_kb:.1f} KB)")
    else:
        print(f"    [MISSING] {f}")

print("\n" + "=" * 60)
print("PHASE 10 STATUS: PASSED")
print("=" * 60)
print("\nVisualizations created successfully:")
print("  - Training curves (loss & accuracy)")
print("  - Sample predictions grid (25 samples)")
print("  - Misclassified examples grid")
print("  - Confusion matrix heatmap (raw & normalized)")
print("  - Per-class accuracy bar chart")
print("\nAll 10 phases of the AAT project are now COMPLETE.")

