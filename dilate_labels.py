import cv2
import os
import numpy as np

path = '/Users/whyguu/Documents/docker-volume/gago-segnet/satellite_data/'

file_names = os.listdir(path+'thin_labels/')

file_names = sorted(file_names, key=lambda x: np.int(x[5:-4]))
print(file_names)
kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))

count = 0
for name in file_names:
    img = cv2.imread(path+'thin_labels/'+name, cv2.IMREAD_GRAYSCALE)
    img = cv2.dilate(img, kernel=kernel, iterations=3)
    cv2.imwrite(path+'labels/'+name, img)
'''
cv2.imshow('haha', img*255)
cv2.imshow('xixi', img1*255)

while 1:
    if cv2.waitKey(0) == 27:
        break

cv2.destroyAllWindows()

'''


