import cv2
import numpy as np
import os


path = './train.txt'

file_names = os.listdir(path)
file_names = sorted(file_names, key=lambda x: int(x[5:-4]))
print(file_names)

count = 0
for name in file_names:
    img = cv2.imread(path+name, cv2.IMREAD_GRAYSCALE)

    count += np.count_nonzero(img)

img = cv2.imread(path+file_names[0], cv2.IMREAD_GRAYSCALE)

r, c = img.shape

total = r * c * len(file_names)

print(count)
print(total)

print(count / total)

