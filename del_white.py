import numpy as np
import cv2
import os


data_path = './train_data/'
file_name = './train.txt'
with open(os.path.abspath(file_name), 'r') as file:
    count = 0
    while True:
        line = file.readline(1000)
        if not line:
            break
        names = line.split(' ')
        print(names)
        img = cv2.imread(data_path + names[1][:-1], cv2.IMREAD_GRAYSCALE)  # names[1][:-1] remove '\n'
        tp_pos = np.sum(img > 0)

        if not tp_pos:
            os.remove(os.path.join(data_path, names[0]))
            os.remove(os.path.join(data_path, names[1][0:-1]))
            count += 1

    print(count)


