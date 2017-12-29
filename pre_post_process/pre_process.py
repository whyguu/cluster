import cv2
import numpy as np
import os
import sys
from skimage.exposure import equalize_hist
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy import misc


class PreProcess(object):

    @staticmethod
    def eq_hist(src_path, dst_path, ext):

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        # img = cv2.Canny(image=img, threshold1=30, threshold2=70)
        # cv2.imwrite('canny_3070.jpg', img)
        # a = cv2.calcHist([img], [0], mask=None, histSize=[256], ranges=[0.0, 255.0])
        # b = cv2.calcHist([img], [1], mask=None, histSize=[256], ranges=[0.0, 255.0])
        # c = cv2.calcHist([img], [2], mask=None, histSize=[256], ranges=[0.0, 255.0])
        # plt.subplot(311)
        # plt.plot(a)
        # plt.subplot(312)
        # plt.plot(b)
        # plt.subplot(313)
        # plt.plot(c)
        # plt.show()
        # print(img.shape)

        names = os.listdir(src_path)
        names = [name for name in names if ext in name]

        for name in names:
            print(name)
            img = cv2.imread(os.path.join(src_path, name))
            img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
            img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
            img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

            cv2.imwrite(os.path.join(dst_path, name), img)

            # img[:, :, 0] = equalize_hist(img[:, :, 0])
            # img[:, :, 1] = equalize_hist(img[:, :, 1])
            # img[:, :, 2] = equalize_hist(img[:, :, 2])
            # img = img[:,:,(2,1,0)]
            # imsave('eq2.jpg', img)

    @staticmethod
    def stretch(bands, lower_percent=2, higher_percent=98, bits=8):
        if bits not in [8, 16]:
            print('error ! dest image must be 8bit or 16bits !')
            return
        out = np.zeros_like(bands, dtype=np.float32)
        n = bands.shape[2]
        for i in range(n):
            a = 0
            b = 1
            c = np.percentile(bands[:, :, i], lower_percent)
            d = np.percentile(bands[:, :, i], higher_percent)
            if d-c == 0:
                out[:, :, i] = 0
                continue
            t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
            out[:, :, i] = np.clip(t, a, b)
        if bits == 8:
            return np.uint8(out.astype(np.float32)*255)
        else:
            return np.uint16(out.astype(np.float32)*65535)

    @staticmethod
    def images_stretch(src_path, dst_path, ext):
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        names = os.listdir(src_path)
        names = [name for name in names if ext in name]
        for name in names:
            print(name)
            img = cv2.imread(os.path.join(src_path, name))
            img = PreProcess.stretch(img)
            cv2.imwrite(os.path.join(dst_path, name), img)

    @staticmethod
    def random_cut_image(img_path, label_path, cut_size=(224, 224), cut_num=8000, save_path=None, stretch=True, use_infrared=False):
        """
        :param img_path: image
        :param label_path:
        :param stretch:
        :param cut_size: (row, col)
        :param cut_num:
        :param save_path:
        :param use_infrared: 使用红外波段
        :return:
        """
        img = tiff.imread(img_path)
        if not use_infrared:
            img = img[:, :, 0:-1]
        label = tiff.imread(label_path)
        if label.dtype == np.uint16:
            label = np.uint8(label)
        if np.max(label) == 1:
            label *= 255
        image_name_prefix = 'image'
        label_name_prefix = 'label'
        row, col = label.shape
        for it in range(cut_num):
            r = np.random.randint(low=0, high=row-cut_size[0])
            c = np.random.randint(low=0, high=col-cut_size[1])
            tp_img = img[r:r+cut_size[0], c:c+cut_size[1], :]
            tp_label = label[r:r+cut_size[0], c:c+cut_size[1]]
            if stretch:
                tp_img = PreProcess.stretch(tp_img)
            tiff.imsave(os.path.join(save_path, '{:0{:d}d}'.format(it, len(str(cut_num)))+'_'+image_name_prefix+'.tif'), tp_img)
            tiff.imsave(os.path.join(save_path, '{:0{:d}d}'.format(it, len(str(cut_num)))+'_'+label_name_prefix+'.tif'), tp_label)
            print('done_{:0{:d}d}.png'.format(it, len(str(cut_num))))

        print('done !')

    @staticmethod
    def calc_bin_class_balance_ratio(file_name, data_path):
        count_neg = np.int64(0)
        count_pos = np.int64(0)
        num_pos = np.int64(0)
        num_neg = np.int64(0)

        with open(os.path.abspath(file_name), 'r') as file:
            while True:
                line = file.readline(1000)
                if not line:
                    break
                names = line.split(' ')
                print(names)
                img = cv2.imread(data_path+names[1][:-1], cv2.IMREAD_GRAYSCALE)  # names[1][:-1] remove '\n'
                tp_pos = np.sum(img > 0)
                tp_neg = np.sum(img == 0)
                if tp_pos != 0:
                    num_pos += tp_pos
                    count_pos += 1
                if tp_neg != 0:
                    num_neg += tp_neg
                    count_neg += 1
        r, c = img.shape
        ratio_pos = 1.0*num_pos / (count_pos*r*c)
        ratio_neg = 1.0*num_neg / (count_neg*r*c)
        pos_weight = ratio_neg / ratio_pos
        print(count_neg, count_pos)
        print('ratio_pos=', ratio_pos)
        print('ratio_neg=', ratio_neg)
        print('pos_weight=', pos_weight)

    @staticmethod
    def calc_rgb_mean(file_name, data_path):
        with open(os.path.abspath(file_name), 'r') as file:
            count = 0
            mean = np.zeros(3, np.int64)
            while True:
                line = file.readline(1000)
                if not line:
                    break
                names = line.split(' ')
                # print(names)
                img = cv2.imread(data_path + names[0])
                count += 1
                mean += np.sum(img, axis=(0, 1)).astype(int)
            r, c = img.shape[0:-1]
            print(r, c, count)
            haha = mean / (1.0*r*c*count)
        print('b, g, r = ', haha)


if __name__ == '__main__':
    # src_path = '/Users/whyguu/Desktop/RGB/common-region-lap/11/'
    # dst_path = '/Users/whyguu/Desktop/RGB/common-region-lap/11-eq-hist/'
    # img_path = './hh_image.tif'
    # label_path = './hh_label.tif'
    # save_path = './train_data'
    # img = tiff.imread(img_path)
    # label = tiff.imread(label_path)
    # PreProcess.random_cut_image(img_path, label_path, cut_size=(224, 224), cut_num=8000, save_path=save_path)
    #

    sys.path.insert(0, './')

    data_path = './train_data/'
    file_name = './train.txt'
    PreProcess.calc_bin_class_balance_ratio(file_name=file_name, data_path=data_path)
    PreProcess.calc_rgb_mean(file_name=file_name, data_path=data_path)
