import os
import sys
import cv2

prifix1 = '/workspace/training_data/val/'
prifix2 = '/workspace/training_data/valannot/'

fw = open('val1.txt', 'w')
with open('val.txt', 'r') as file:
    line = file.readline(1000)
    while line:
        ln = line.split(' ')

        a = ln[0].split('/')
        a = prifix1 + a[-1]
        fw.write(a+' ')

        a = ln[1].split('/')
        a = prifix2 + a[-1]
        fw.write(a)

        line = file.readline(1000)

fw.close()

img = cv2.imread('5701.tif', cv2.IMREAD_ANYCOLOR)

cv2.imwrite('5701.png', img)
