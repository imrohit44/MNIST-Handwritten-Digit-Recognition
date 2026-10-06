import sys
import subprocess

wheel_path = 'D:/DL_AAT/tf.whl'

# Debug: print the repr of the path to see exactly what's passed
print('repr of path:', repr(wheel_path))

# Try with Popen directly to see the exact command
cmd = [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', wheel_path]
print('cmd:', cmd)
result = subprocess.run(cmd, capture_output=True, text=True)
print('STDOUT:', result.stdout)
print('STDERR:', result.stderr)
print('Return code:', result.returncode)

sys.exit(result.returncode)
