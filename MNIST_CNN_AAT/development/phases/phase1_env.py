"""
MNIST Handwritten Digit Recognition - AAT Project
Phase 1: Environment Setup
"""
import sys
print("=" * 60)
print("MNIST CNN - AAT PROJECT - PHASE 1: ENVIRONMENT SETUP")
print("=" * 60)

# Check Python version
print(f"\n[1] Python Version: {sys.version.split()[0]}")
print(f"    Python executable: {sys.executable}")

# Import required libraries
print("\n[2] Checking required library imports...")

try:
    import tensorflow as tf
    print(f"    - TensorFlow: {tf.__version__}")
except ImportError as e:
    print(f"    - ERROR: TensorFlow import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"    - NumPy: {np.__version__}")
except ImportError as e:
    print(f"    - ERROR: NumPy import failed: {e}")
    sys.exit(1)

try:
    import matplotlib
    print(f"    - Matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"    - ERROR: Matplotlib import failed: {e}")
    sys.exit(1)

try:
    import sklearn
    print(f"    - Scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"    - ERROR: Scikit-learn import failed: {e}")
    sys.exit(1)

try:
    import keras
    print(f"    - Keras: {keras.__version__}")
except ImportError as e:
    print(f"    - Keras may be bundled with TensorFlow: {e}")

# Check hardware acceleration (GPU)
print("\n[3] Checking hardware acceleration...")
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"    - Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
    else:
        print("    - No GPU detected. Running on CPU.")
except Exception as e:
    print(f"    - GPU check warning: {e}")

# Check TensorFlow version compatibility
print("\n[4] Checking TensorFlow functionality...")
try:
    # Test basic TensorFlow functionality
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print(f"    - TensorFlow is functional: matrix multiplication test passed")
    print(f"      Result: {c.numpy().tolist()}")
except Exception as e:
    print(f"    - ERROR: TensorFlow functionality test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("PHASE 1 STATUS: PASSED")
print("=" * 60)
print("\nAll required libraries imported successfully.")
print("Environment is ready for MNIST CNN project.")
