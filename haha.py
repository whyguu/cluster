import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
import tifffile as tiff


with open('/Users/whyguu/PycharmProjects/hed-tf/data/weights/vgg16.npy', 'rb') as file:
    a = np.load(file, encoding='latin1')
    print(type(a))
# img = tiff.imread('/Users/whyguu/Desktop/trainrrr_data/image_0.tif')
#
# print(img.shape)
# print(img.dtype)
# print(np.max(img[:,:,3]))
# cv2.imshow('haha', 255*img[10000:10300, 10000:10500])
#
# while 1:
#     if cv2.waitKey(0) == 27:
#         break
# cv2.destroyAllWindows()


# path = '/Users/whyguu/Documents/docker-volume/gago-segnet/road/'
# path1 = '/Users/whyguu/Documents/docker-volume/gago-segnet/road_data/'
#
#
# file_names = os.listdir(path+'sat')
# file_names = sorted(file_names, key=lambda x: int(x[9:-4]))
#
# map_names = os.listdir(path+'map')
# map_names = sorted(map_names, key=lambda x: int(x[9:-4]))
# print(file_names)
# print(len(file_names))
'''
count = 0
for name in file_names:
    os.rename(path+'sat/'+name, path+'sat/satImage_'+str(count)+'.png')
    count += 1

count = 0
for name in map_names:
    os.rename(path+'map/'+name, path+'map/satImage_'+str(count)+'.png')
    count += 1

'''

# print(file_names)
# print(len(file_names))
# print(map_names)
# print(len(map_names))
# for name in file_names:
#     img = cv2.imread(path+'sat/'+name)
#     img = cv2.resize(img, (512, 512))
#     cv2.imwrite(path1+'sat/'+name, img)
#
# for name in map_names:
#     img_map = cv2.imread(path + 'map/' + name)
#     img_map = cv2.resize(img_map, (512, 512))
#     cv2.imwrite(path1 + 'map/' + name, img_map)
#
# 0.46397 690.443 rd


# mx = np.ones((len(ls), 2), np.float32)
# yy = np.zeros((len(ls), 1), np.float32)
# for i in range(len(ls)):
#     mx[i, 0] = ls[i][0]
#     yy[i, 0] = ls[i][1]
# plt.plot(mx[:, 0], yy[:, 0])
# plt.show()

# ls = [(370, 862), [369, 862], [370, 862], [368, 861], [367, 861], [366, 860], [365, 860], [364, 859], [363, 859], [362, 858], [362, 859], [361, 859], [362, 857], [361, 860], [362, 856], [360, 861], [361, 855], [360, 862], [361, 854], [359, 863], [361, 853]]
# # 0.347801 732.748 rd
# def sci_line_param(mx, yy):
#
#     reg = linear_model.LinearRegression(fit_intercept=True)
#     reg.fit(mx, yy)
#     print(reg.coef_)  # k
#     print(reg.intercept_)  # b
#     plt.plot(mx[:, 0], yy[:, 0])
#     plt.show()
#
#
# mx = np.ones((len(ls), 1), np.float32)
# yy = np.zeros((len(ls), 1), np.float32)
# for i in range(len(ls)):
#     mx[i, 0] = ls[i][0]
#     yy[i, 0] = ls[i][1]
#
#
# sci_line_param(ls,mx,yy)
#
# x = np.arange(360, 370)
# y1 = 0.3478089*x + 732.75366211
# y2 = 2.32009*x + (-2288.08)
#
#
# plt.plot(mx[:, 0], yy[:, 0])
# plt.hold
# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.show()


# # -*- coding: utf-8 -*-
# """
# Created on Tue Dec  5 09:31:02 2017
# @author: kuan
# """
# from __future__ import division, unicode_literals, print_function, with_statement
#
# from osgeo import gdal
# import gdal
# import numpy as np
# import sys, os, time
#
# gdal.UseExceptions()
#
# if len(sys.argv) < 4:
#     print(
#         'How to use this stacking tool: \n   python merge_tiff.py filename_1 filename_2 ... filename_n filename_stacked ')
#     sys.exit()
#
# # print(sys.argv)
#
#
#
# outvrt = '/tmp/stacked.vrt'
# outtif = sys.argv[-1]
# tifs = sys.argv[1:-1]
# print('Files to be stacked:', tifs)
# print('Stacked file to create: ', outtif)
# outds = gdal.BuildVRT(outvrt, tifs, separate=True)
# outds = gdal.Translate(outtif, outds)
