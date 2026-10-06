# MNIST Handwritten Digit Recognition - CNN (AAT Project)

A complete 10-phase Convolutional Neural Network project for handwritten digit
recognition on the MNIST dataset, built with Python and TensorFlow/Keras.

The repository is organized into clean reusable source code (`src/`), the trained
model (`models/`), final results (`results/`), the original development phase
scripts (`development/`) and evaluation handoff material (`handoff/`).

## Overview

- Dataset: MNIST (70,000 grayscale 28x28 handwritten digit images)
- Model: 3-layer Convolutional Neural Network (241,546 parameters)
- Final test accuracy: **99.27%** (73 / 10,000 misclassified)
- Status: training, evaluation and visualization all COMPLETE and validated

## Problem Statement

Handwritten digit recognition is a classic image classification problem.
Scanned or handwritten digits must be recognized despite wide variation in
stroke style, thickness and alignment. The goal of this project is to design and
train a Convolutional Neural Network that classifies MNIST images into one of
ten classes (digits 0-9) with high accuracy, and to evaluate it thoroughly with
per-class metrics, a confusion matrix and error analysis.

## Objectives

1. Set up the Python / TensorFlow deep learning environment
2. Load and explore the MNIST dataset
3. Preprocess the data (normalization, channel dimension)
4. Split data into training, validation and test sets
5. Design a CNN architecture for digit classification
6. Compile and train the model with callbacks
7. Evaluate the trained model with accuracy, precision, recall and F1
8. Visualize training behaviour, predictions and the confusion matrix
9. Analyse errors and document strengths, weaknesses and conclusions

## Dataset

| Property | Value |
|---|---|
| Dataset | MNIST (via `tf.keras.datasets.mnist`) |
| Training images | 60,000 |
| Test images | 10,000 (held out) |
| Image size | 28 x 28 pixels, grayscale |
| Classes | 10 (digits 0-9) |
| Pixel range | 0-255 (raw), normalized to 0-1 |

Splits used by the project:

- Training: 51,000 images
- Validation: 9,000 images
- Test: 10,000 images (never used during training or validation)

The split is reproduced deterministically with random seed 42.

## Methodology

1. **Environment setup** - Python 3.13, TensorFlow 2.21 (Keras 3), scikit-learn, matplotlib, seaborn
2. **Data loading** - MNIST loaded through TensorFlow/Keras
3. **Exploration** - shapes, pixel range, class distributions, missing-value checks, sample images
4. **Preprocessing** - float32 conversion, normalization to [0, 1], channel dimension (28x28x1)
5. **Splitting** - 85/15 train/validation split with seed 42; test set held out
6. **Modeling** - 3-block CNN built with `tf.keras.Sequential`
7. **Training** - Adam optimizer with early stopping and best-checkpoint saving
8. **Evaluation** - loss, accuracy, precision, recall, F1, confusion matrix
9. **Visualization** - six figures covering training, predictions and errors

## CNN Architecture

```
Input (28x28x1)
  -> Conv2D(32, 3x3, relu) -> MaxPool2D
  -> Conv2D(64, 3x3, relu) -> MaxPool2D
  -> Conv2D(128, 3x3, relu) -> MaxPool2D
  -> Flatten
  -> Dense(128, relu) -> Dropout(0.5)
  -> Dense(10, softmax)
Total params: 241,546
```

Defined in `src/model/architecture.py` (`build_model()`). The architecture is
fixed - it is the one used to train the delivered model.

## Training Configuration

- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Metrics:** Accuracy
- **Batch size:** 128
- **Maximum epochs:** 15
- **EarlyStopping:** monitor `val_loss`, patience 3, restore best weights
- **ModelCheckpoint:** monitor `val_accuracy`, save best only
- **Result:** best weights at epoch 9, training stopped at epoch 12
- **Model saved to:** `models/mnist_cnn_model.keras`
- **History saved to:** `results/data/training_history.npy`

## Final Results

| Metric | Value |
|---|---|
| Test Accuracy | **99.27%** |
| Test Loss | 0.0210 |
| Precision (macro) | 0.9927 |
| Recall (macro) | 0.9927 |
| F1-score (macro) | 0.9927 |
| Misclassified | 73 / 10,000 |

Every run of `python -m src.evaluation.evaluate` compares its measured metrics
against these reported values and reports PASS/FAIL.

## Visualizations

All figures live in `results/figures/`:

| Figure | File | Description |
|---|---|---|
| Sample digits | `phase3_samples.png` | First example of each digit |
| Training curves | `training_curves.png` | Loss & accuracy vs epoch |
| Predictions | `sample_predictions.png` | 25-sample grid (green = correct, red = wrong) |
| Misclassified | `misclassified_examples.png` | Wrongly classified digits |
| Confusion matrix | `confusion_matrix.png` | Raw counts heatmap |
| Normalized CM | `confusion_matrix_normalized.png` | Row-normalized heatmap |
| Per-class accuracy | `per_class_accuracy.png` | Bar chart per digit |

## Project Structure

```
MNIST_CNN_AAT/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── src/                         # Clean reusable source code
│   ├── preprocessing/           # load_data, explore_data, preprocess_data, split_data
│   ├── model/                   # architecture, compile, training
│   └── evaluation/              # evaluate, visualize
├── models/
│   └── mnist_cnn_model.keras    # Trained model (best checkpoint)
├── results/
│   ├── figures/                 # Human-readable figures (7 PNG files)
│   └── data/                    # Machine-readable artifacts (4 NPY files)
├── development/
│   └── phases/                  # Original 10 phase scripts (development history)
└── handoff/
    ├── RAAGHAV_HANDOFF.md       # Evaluation handoff document
    └── evaluation_checklist.md  # Step-by-step evaluation checklist
```

| Directory | Contains |
|---|---|
| `src/` | Reusable implementation refactored from the phase scripts |
| `models/` | The trained `.keras` model - do not overwrite |
| `results/figures/` | Visual outputs only (PNG) |
| `results/data/` | Numerical artifacts only (NPY) |
| `development/phases/` | Original phase-by-phase validation scripts |
| `handoff/` | Material for the evaluation handoff (Raaghav) |

No source code lives in `results/`, and no figures live outside `results/`.

## Development Phases

The original scripts that built and validated the project are preserved in
`development/phases/`:

| Phase | Script | Description | Status |
|---|---|---|---|
| 1 | `phase1_env.py` | Environment & library versions | PASSED |
| 2 | `phase2_load.py` | Load MNIST (60k train / 10k test) | PASSED |
| 3 | `phase3_explore.py` | Class distribution, sample images | PASSED |
| 4 | `phase4_preprocess.py` | Normalize [0,1], reshape to 28x28x1 | PASSED |
| 5 | `phase5_split.py` | 51,000 / 9,000 / 10,000 split (seed 42) | PASSED |
| 6 | `phase6_architecture.py` | 3-conv CNN design (241,546 params) | PASSED |
| 7 | `phase7_compile.py` | Adam + sparse categorical crossentropy | PASSED |
| 8 | `phase8_training.py` | Training with EarlyStopping + ModelCheckpoint | PASSED |
| 9 | `phase9_evaluation.py` | Accuracy, precision, recall, F1, CM | PASSED |
| 10 | `phase10_visualization.py` | Curves, predictions, heatmaps | PASSED |

The phase scripts are the development/testing history of the project; the
maintained implementation lives in `src/`.

## How to Run

Install dependencies (once):

```bash
pip install -r requirements.txt
```

Run all modules from the project root (`MNIST_CNN_AAT/`):

```bash
# Data pipeline
python -m src.preprocessing.load_data
python -m src.preprocessing.explore_data
python -m src.preprocessing.preprocess_data
python -m src.preprocessing.split_data

# Model
python -m src.model.architecture
python -m src.model.compile
python -m src.model.training            # shows config only - does NOT retrain

# Evaluation (already complete)
python -m src.evaluation.evaluate
python -m src.evaluation.visualize

# Optional: full retraining (OVERWRITES the delivered model!)
python -m src.model.training --retrain
```

Phase scripts can still be run directly, for example:

```bash
python development/phases/phase9_evaluation.py
```

**Note:** the delivered model is already trained. Do not run `--retrain`
unless you intentionally want to reproduce the training run.

## Environment

- Python 3.13 (Windows)
- TensorFlow 2.21.0 / Keras 3
- scikit-learn 1.7.2, matplotlib 3.10.3, seaborn 0.13.2
- CPU training (TensorFlow does not support GPU natively on Windows for TF >= 2.11)

## Team Responsibilities

### Rohit

- Dataset preparation
- Preprocessing
- CNN architecture
- Compilation
- Training
- Training analysis

### Raaghav

- Final model evaluation
- Per-class metrics
- Confusion matrix analysis
- Prediction/error analysis
- Strengths and weaknesses
- Conclusion

Raaghav should start from `handoff/RAAGHAV_HANDOFF.md` and
`handoff/evaluation_checklist.md`.

## Limitations

- The model was trained on CPU only; a full training run takes several minutes.
- MNIST is a small, clean, centered dataset - the model is not evaluated on
  harder or real-world datasets (e.g. EMNIST, scanned documents).
- Results come from a single run with a fixed seed (42); no hyperparameter
  search or cross-validation was performed.
- Small rotations, scaling or noise in input images can reduce accuracy.
- 73 test images remain misclassified, mostly ambiguous handwriting.

## Conclusion

The 3-layer CNN trained for this project reaches **99.27% test accuracy** with
a test loss of **0.0210** and macro precision, recall and F1 of **0.9927**,
misclassifying only 73 of 10,000 test images. Training was stable thanks to
early stopping and best-checkpoint selection, and the confusion matrix shows
consistent performance across all ten digits. Together with the training
curves, sample predictions, misclassified examples and per-class accuracy
charts, the project demonstrates a complete, reproducible deep learning
workflow from environment setup through evaluation and analysis.


