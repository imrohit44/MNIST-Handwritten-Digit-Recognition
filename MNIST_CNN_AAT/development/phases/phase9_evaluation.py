"""
MNIST Handwritten Digit Recognition - AAT Project
Phase 9: Model Evaluation
Evaluates the trained model on the held-out test set (10,000 images)
using accuracy, precision, recall, F1-score, and confusion matrix.
"""
import sys
print("=" * 60)
print("MNIST CNN - AAT PROJECT - PHASE 9: MODEL EVALUATION")
print("=" * 60)

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Paths
MODELS_DIR = r"D:\DL AAT\MNIST_CNN_AAT\models"
RESULTS_DIR = r"D:\DL AAT\MNIST_CNN_AAT\results"
MODEL_PATH = os.path.join(MODELS_DIR, "mnist_cnn_model.keras")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "data"), exist_ok=True)

# Reproduce Phase 5 split for consistency
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Load and preprocess MNIST (same as earlier phases)
print("\n[1] Loading MNIST dataset (test split only for evaluation)...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1)
print(f"    Test set: {x_test.shape[0]} images")

# Load trained model
print("\n[2] Loading trained model...")
if not os.path.exists(MODEL_PATH):
    print(f"    - ERROR: Model not found at {MODEL_PATH}")
    print("    - Run Phase 8 first.")
    sys.exit(1)
model = tf.keras.models.load_model(MODEL_PATH)
size_kb = os.path.getsize(MODEL_PATH) / 1024
print(f"    - Loaded: {MODEL_PATH} ({size_kb:.1f} KB)")

# Evaluate on test set
print("\n[3] Evaluating model on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"    Test loss:     {test_loss:.4f}")
print(f"    Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

if test_acc < 0.90:
    print("    - WARNING: Test accuracy below 0.90")
else:
    print("    - Test accuracy >= 0.90: OK")

# Generate predictions
print("\n[4] Generating predictions for all test samples...")
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
print(f"    Predictions shape: {y_pred.shape}")

# Compute metrics
print("\n[5] Computing evaluation metrics...")
acc = accuracy_score(y_test, y_pred)
prec_macro = precision_score(y_test, y_pred, average='macro')
rec_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')
prec_weighted = precision_score(y_test, y_pred, average='weighted')
rec_weighted = recall_score(y_test, y_pred, average='weighted')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print("\n    Overall Metrics:")
print(f"      Accuracy:           {acc:.4f} ({acc*100:.2f}%)")
print(f"      Precision (macro):   {prec_macro:.4f}")
print(f"      Recall (macro):      {rec_macro:.4f}")
print(f"      F1-score (macro):    {f1_macro:.4f}")
print(f"      Precision (weighted): {prec_weighted:.4f}")
print(f"      Recall (weighted):    {rec_weighted:.4f}")
print(f"      F1-score (weighted):  {f1_weighted:.4f}")

# Per-class report
print("\n[6] Per-class classification report:")
report = classification_report(
    y_test, y_pred,
    target_names=[f"Digit {i}" for i in range(10)],
    digits=4,
)
print(report)

# Confusion matrix
print("[7] Confusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print("    " + "     " + "  ".join(f"{i:>4}" for i in range(10)))
for i in range(10):
    row = "    " + f"{i:>3}: " + "  ".join(f"{v:>4}" for v in cm[i])
    print(row)

# Verify confusion matrix
total_cm = cm.sum()
correct = np.trace(cm)
print(f"\n    Total samples in confusion matrix: {total_cm}")
print(f"    Correctly classified (diagonal):    {correct}")
if total_cm != x_test.shape[0]:
    print("    - ERROR: Confusion matrix total mismatch")
    sys.exit(1)
else:
    print("    - Confusion matrix total matches test set: OK")

# Save metrics for reporting
np.save(
    os.path.join(RESULTS_DIR, "data", "eval_metrics.npy"),
    np.array([acc, prec_macro, rec_macro, f1_macro, test_loss]),
)
np.save(os.path.join(RESULTS_DIR, "data", "confusion_matrix.npy"), cm)
np.save(os.path.join(RESULTS_DIR, "data", "y_pred.npy"), y_pred)
print(f"    - Metrics saved to {RESULTS_DIR}/data")

# Final checks
print("\n[8] Final verification...")
errors = np.where(y_pred != y_test)[0]
print(f"    Misclassified samples: {len(errors)} / {x_test.shape[0]}")
if len(errors) > 0:
    print(f"    First few error indices: {errors[:10].tolist()}")

if acc >= 0.95:
    print("    - Test accuracy >= 0.95: OK")
else:
    print(f"    - WARNING: Test accuracy {acc:.4f} below 0.95")

print("\n" + "=" * 60)
print("PHASE 9 STATUS: PASSED")
print("=" * 60)
print("\nEvaluation complete:")
print(f"  - Test Accuracy:  {acc*100:.2f}%")
print(f"  - F1-score (macro): {f1_macro:.4f}")
print(f"  - Misclassified:  {len(errors)} / {x_test.shape[0]}")
print("\nReady for Phase 10: Visualization.")
