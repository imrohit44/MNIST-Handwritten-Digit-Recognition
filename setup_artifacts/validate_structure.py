"""Post-restructure static validation for the MNIST CNN AAT project.

Checks: Python syntax, model/artifact/figure presence, README path
validity, stale path references, duplicate files and preservation of
all important project files. This is a repository check - it does NOT
retrain or modify anything.
"""
import ast
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\DL AAT\MNIST_CNN_AAT")

print("=" * 60)
print("RESTRUCTURE VALIDATION")
print("=" * 60)

# Test 1: Python syntax
py_files = sorted(ROOT.rglob("*.py"))
syntax_errors = []
for f in py_files:
    try:
        ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
    except (SyntaxError, UnicodeDecodeError) as e:
        syntax_errors.append(f"{f.relative_to(ROOT)}: {e}")
print(f"\n[Test 1] Python syntax: {len(py_files) - len(syntax_errors)}/{len(py_files)} PASSED")
for err in syntax_errors:
    print(f"    ERROR {err}")

# Test 2: model file exists
model = ROOT / "models" / "mnist_cnn_model.keras"
model_ok = model.exists() and model.stat().st_size > 1_000_000
size = model.stat().st_size if model.exists() else 0
print(f"\n[Test 2] Model file exists: {'PASSED' if model_ok else 'FAILED'} ({size} bytes)")

# Test 6: artifacts readable
print("\n[Test 6] Result artifacts readable:")
artifacts_ok = True
for name in ["training_history.npy", "confusion_matrix.npy",
             "eval_metrics.npy", "y_pred.npy"]:
    path = ROOT / "results" / "data" / name
    try:
        arr = np.load(path)
        print(f"    PASSED {name}: shape={arr.shape}, dtype={arr.dtype}")
    except Exception as exc:  # noqa: BLE001
        artifacts_ok = False
        print(f"    FAILED {name}: {exc}")

# Test 7: figures present
print("\n[Test 7] Result figures:")
figures_ok = True
figures = ["phase3_samples.png", "training_curves.png", "sample_predictions.png",
           "misclassified_examples.png", "confusion_matrix.png",
           "confusion_matrix_normalized.png", "per_class_accuracy.png"]
for name in figures:
    path = ROOT / "results" / "figures" / name
    if path.exists():
        print(f"    PASSED {name}: {path.stat().st_size} bytes")
    else:
        figures_ok = False
        print(f"    FAILED {name}: missing")

# Test 8: README paths
print("\n[Test 8] README path references:")
readme = (ROOT / "README.md").read_text(encoding="utf-8")
path_tokens = set(re.findall(
    r"(?:src|models|results|development|handoff)[/A-Za-z0-9_.-]*\.(?:py|md|txt|keras|png|npy)",
    readme))
path_tokens.add("requirements.txt")
readme_ok = True
for token in sorted(path_tokens):
    if (ROOT / token).exists():
        print(f"    PASSED {token}")
    else:
        readme_ok = False
        print(f"    FAILED {token}: not found")
# python -m module targets
for mod in sorted(set(re.findall(r"python -m (src(?:\.\w+)+)", readme))):
    target = ROOT / (mod.replace(".", "/") + ".py")
    if target.exists():
        print(f"    PASSED module {mod}")
    else:
        readme_ok = False
        print(f"    FAILED module {mod}: {target} not found")

# Test 9: stale path scan across repo
print("\n[Test 9] Stale path scan:")
stale_patterns = [
    r"MNIST_CNN_AAT_phase",
    r"results[/\\]phase3_samples",
    r"results[/\\]training_history\.npy",
    r"results[/\\]eval_metrics\.npy",
    r"results[/\\]confusion_matrix\.npy",
    r"results[/\\]y_pred\.npy",
]
stale_ok = True
for pattern in stale_patterns:
    hits = []
    for f in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.md")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(pattern, text):
            line = text.count("\n", 0, match.start()) + 1
            hits.append(f"{f.relative_to(ROOT)}:{line}")
    if hits:
        stale_ok = False
        print(f"    FAILED pattern '{pattern}': {hits}")
    else:
        print(f"    PASSED pattern '{pattern}': no hits")

# Test 10: duplicates + important files
print("\n[Test 10] Duplicate file scan:")
names = [f.name for f in ROOT.rglob("*") if f.is_file()]
dupes = {n: c for n, c in Counter(names).items() if c > 1}
if dupes:
    print(f"    FAILED duplicates: {dupes}")
else:
    print(f"    PASSED no duplicate file names ({len(names)} files)")

important = [
    "README.md", "requirements.txt", "models/mnist_cnn_model.keras",
    "src/preprocessing/load_data.py", "src/preprocessing/explore_data.py",
    "src/preprocessing/preprocess_data.py", "src/preprocessing/split_data.py",
    "src/model/architecture.py", "src/model/compile.py", "src/model/training.py",
    "src/evaluation/evaluate.py", "src/evaluation/visualize.py",
    "handoff/RAAGHAV_HANDOFF.md", "handoff/evaluation_checklist.md",
]
important += [f"results/figures/{f}" for f in figures]
important += [f"results/data/{f}" for f in
              ["training_history.npy", "confusion_matrix.npy",
               "eval_metrics.npy", "y_pred.npy"]]
important += [f"development/phases/phase{i}_{s}.py" for i, s in [
    (1, "env"), (2, "load"), (3, "explore"), (4, "preprocess"), (5, "split"),
    (6, "architecture"), (7, "compile"), (8, "training"), (9, "evaluation"),
    (10, "visualization")]]
missing = [f for f in important if not (ROOT / f).exists()]
if missing:
    print(f"    FAILED missing important files: {missing}")
else:
    print(f"    PASSED all {len(important)} important files present")

print("\n" + "=" * 60)
summary_ok = (not syntax_errors) and model_ok and artifacts_ok and figures_ok \
    and readme_ok and stale_ok and not dupes and not missing
print(f"STATIC VALIDATION: {'ALL PASSED' if summary_ok else 'FAILURES ABOVE'}")
print("=" * 60)
