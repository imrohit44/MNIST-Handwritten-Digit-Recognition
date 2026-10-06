# Raaghav - Evaluation Handoff

## Project status

| Item                | Status |
|---------------------|--------|
| Training complete   | YES    |
| Model validated     | YES    |
| Test evaluation     | YES    |
| Visualization       | YES    |

The model is **already trained and validated**. Do not retrain unless
required for verification.

## Model

- **Architecture:** 3-layer CNN (Conv2D 32/64/128 + MaxPool, Dense 128, Dropout 0.5, Dense 10 softmax)
- **Parameters:** 241,546
- **Model file:** `models/mnist_cnn_model.keras`
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy

## Training configuration (do not change)

- Batch size: 128
- Maximum epochs: 15 (stopped at epoch 12, best weights at epoch 9)
- EarlyStopping: `val_loss`, patience 3, restore best weights
- ModelCheckpoint: `val_accuracy`, save best only

## Final results

| Metric             | Value      |
|--------------------|------------|
| Test Accuracy      | 99.27%     |
| Test Loss          | 0.0210     |
| Macro Precision    | 0.9927     |
| Macro Recall       | 0.9927     |
| Macro F1-score     | 0.9927     |
| Misclassified      | 73 / 10,000 |

## What Raaghav should work on

1. Verify final test evaluation
2. Generate/verify per-class precision
3. Generate/verify per-class recall
4. Generate/verify per-class F1
5. Analyze confusion matrix
6. Analyze misclassified images
7. Write strengths and weaknesses
8. Prepare conclusion
9. Prepare report sections
10. Prepare viva explanation

## Tools for the evaluation work

Run everything from the project root (`MNIST_CNN_AAT/`):

```bash
# Full evaluation: loss, accuracy, classification report,
# per-class metrics, confusion matrix, verification of reported results
python -m src.evaluation.evaluate

# Regenerate all six figures in results/figures/
python -m src.evaluation.visualize
```

`python -m src.evaluation.evaluate` prints macro AND per-class
precision/recall/F1, the 10x10 confusion matrix, the misclassified
count and a check table comparing measured values against the
reported results above. It also refreshes the artifacts in
`results/data/`.

## Key paths

| Item                      | Path                                    |
|---------------------------|-----------------------------------------|
| Trained model             | `models/mnist_cnn_model.keras`          |
| Evaluation artifacts      | `results/data/`                         |
| Figures                   | `results/figures/`                      |
| Evaluation source code    | `src/evaluation/evaluate.py`            |
| Visualization source code | `src/evaluation/visualize.py`           |
| Development phase scripts | `development/phases/`                   |

## Important rules

- The test set (10,000 images) is held out - never use it for training.
- Do **not** retrain the model (`python -m src.model.training --retrain`
  overwrites the validated model).
- Do **not** change the reported results: 99.27% accuracy, 0.0210 loss,
  0.9927 macro precision/recall/F1, 73/10,000 misclassified.
