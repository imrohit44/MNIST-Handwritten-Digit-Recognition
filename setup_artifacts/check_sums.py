import hashlib, os, sys

files = [
    'D:/DL AAT/tensorflow_cpu.whl',
    'D:/DL AAT/tensorflow.whl',
    'D:/DL AAT/tf.whl',
    'D:/DL AAT/check_sums.py'
]

for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        with open(f, 'rb') as fh:
            h = hashlib.sha256(fh.read()).hexdigest()
        print(f'{f}: {size} bytes, SHA256: {h[:16]}...')
    else:
        print(f'{f}: NOT FOUND')
