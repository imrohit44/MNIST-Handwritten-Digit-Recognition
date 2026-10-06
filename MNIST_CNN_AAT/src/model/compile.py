"""
Model compilation.

Responsible only for compiling the CNN built by architecture.py.
Part of the MNIST CNN AAT project (development Phase 7 logic).

Configuration (UNCHANGED from the validated training run):
    Optimizer:  Adam
    Loss:       Sparse Categorical Crossentropy
    Metrics:    Accuracy
"""
import tensorflow as tf

from src.model.architecture import build_model

OPTIMIZER = "adam"
LOSS = "sparse_categorical_crossentropy"
METRICS = ["accuracy"]


def compile_model(model, optimizer=OPTIMIZER, loss=LOSS, metrics=None):
    """Compile the model and return it."""
    if metrics is None:
        metrics = list(METRICS)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def main():
    print("=" * 60)
    print("MNIST CNN - MODEL COMPILATION")
    print("=" * 60)

    model = build_model()
    compile_model(model)

    print("\n[1] Compilation configuration:")
    print(f"    Optimizer: {model.optimizer.__class__.__name__}")
    print(f"    Loss:      {LOSS}")
    print(f"    Metrics:   {METRICS}")

    print("\n[2] Verification:")
    ok = True
    if model.optimizer.__class__.__name__ == "Adam":
        print("    - optimizer is Adam: OK")
    else:
        print("    - optimizer is Adam: FAILED")
        ok = False

    # Keras 3: model.metrics_names is ['loss', 'compile_metrics'] before any
    # training/evaluation step, so verify via the loss attribute instead.
    if model.loss == LOSS:
        print(f"    - loss is {LOSS}: OK")
    else:
        print(f"    - loss is {LOSS}: FAILED")
        ok = False

    print("\n" + "=" * 60)
    if ok:
        print("Compile check: PASSED")
    else:
        print("Compile check: FAILED")
        raise SystemExit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
