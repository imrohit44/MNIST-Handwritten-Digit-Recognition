import sys

print('Python version:', sys.version)

# Check what's installed
try:
    import tensorflow
    print('tensorflow:', tensorflow.__version__)
except ImportError as e:
    print('TensorFlow import error:', e)

import numpy as np
print('NumPy:', np.__version__)

import matplotlib
print('Matplotlib:', matplotlib.__version__)

import sklearn
print('Scikit-learn:', sklearn.__version__)

