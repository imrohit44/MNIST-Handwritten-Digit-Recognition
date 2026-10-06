# Evaluation Checklist

Work through this list in order. Run everything from the project root
(`MNIST_CNN_AAT/`).

- [ ] Load trained model (`models/mnist_cnn_model.keras`)
- [ ] Load test data (10,000 held-out MNIST images)
- [ ] Evaluate test accuracy (expect 99.27%)
- [ ] Generate classification report
- [ ] Record per-class precision
- [ ] Record per-class recall
- [ ] Record per-class F1
- [ ] Generate confusion matrix
- [ ] Identify most confused classes
- [ ] Review misclassified examples (73 / 10,000)
- [ ] Verify reported 99.27% accuracy
- [ ] Prepare analysis (strengths and weaknesses)
- [ ] Prepare conclusion

## Quick start

```bash
python -m src.evaluation.evaluate    # steps 1-7, 11 in one run
python -m src.evaluation.visualize   # confusion matrix + misclassified figures
```

Figures to review:

- `results/figures/confusion_matrix.png`
- `results/figures/confusion_matrix_normalized.png`
- `results/figures/per_class_accuracy.png`
- `results/figures/misclassified_examples.png`
