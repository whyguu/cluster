#coding:utf-8
#GAO Bin

from __future__ import print_function
import os,sys
import tensorflow as tf
import numpy as np
import argparse
import glob
from PIL import Image
import skimage.io as io

_PREFIX = {
  'train': 'train',
  'valid': 'valid',
  'test': 'test'
}

IMAGE_PATH = './data'

#生成整数型数据
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = value))

#生成字符串型数据
def _byte_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def glob_dir(wildcard):
    return len(glob.glob(wildcard))

#将图片存储到tfreocrd中：
def convert_to_tfrecord(name,num,input_dir,save_dir = '.'):
    '''convert all images and labels to one tfrecord file.
        Args:
            images: list of image directories, string type
            labels: list of labels, int type
            save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
            name: the name of tfrecord file, string type, e.g.: 'train'
        Return:
            no return
        Note:
            converting needs some time, be patient...
        '''

    filename = os.path.join(save_dir,name + '.tfrecords')
    n_samples = int(len(os.listdir(input_dir)) / 2)
    #if np.shape(images)[0] != n_samples:
     #   raise ValueError('Image size %d does not match label size %d.' % (images.shape[0],n_samples))

    #Create a tf writer
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start...')

    for i in range(1,n_samples):
        try:
            fim = os.path.join(input_dir,'%s_%d_im' % (_PREFIX[name], i) + '.jpg')
            flb = os.path.join(input_dir,'%s_%d_lb' % (_PREFIX[name], i) + '.jpg')

            '''im = cv2.imread(fim)
            lb = cv2.imread(flb)
            im_raw = im.tostring()
            #print(im_raw)
            lb_raw = np.int64(lb)
            #print('lb_raw',lb)'''

            image = io.imread(fim)  # type(image) must be array!
            label = io.imread(flb)
            image_raw = image.tostring()
            label_raw = label.tostring()

            example = tf.train.Example(features = tf.train.Features(feature = {
                'label': _byte_feature(label_raw),
                'image_raw': _byte_feature(image_raw)
            }))
            writer.write(example.SerializeToString())
        except Exception as e:
            print('Could not read image:',i)
            print(e)
    writer.close()
    print('Transform done!')

def main():
    #FLAGS = parser.parse_args()
    n_samples = {
        i: glob_dir(os.path.join(IMAGE_PATH, j + '*')) / 2 for i, j in _PREFIX.items()
    }

    print('Number of samples:',n_samples)
    for i,j in n_samples.items():
        if not j == 0:
            convert_to_tfrecord(i,j,IMAGE_PATH)


if __name__ == '__main__':
    main()
