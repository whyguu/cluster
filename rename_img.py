import numpy as np
import cv2
import os


path = '/Users/whyguu/Documents/docker-volume/gago-segnet/stallite_data/'

if os.path.exists(path+'data/.DS_store'):
    os.remove(path+'data/.DS_store')
if os.path.exists(path+'labels/.DS_store'):
    os.remove(path + 'labels/.DS_store')

datafiles = os.listdir('data/')
labelsfiles = os.listdir('labels/')

count = 0
for name in datafiles:
    os.rename('data/'+name, 'luhe_'+str(count)+'.png')
    os.rename(path+'labels/'+name[0:-3]+'tif', 'luhe_'+str(count)+'.tif')
    count += 1

