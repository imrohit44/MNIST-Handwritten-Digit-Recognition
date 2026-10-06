data = open('D:/DL AAT/tensorflow.whl', 'rb').read(4)
print('magic:', data.hex(), data)
print('size:', __import__('os').path.getsize('D:/DL AAT/tensorflow.whl'))
