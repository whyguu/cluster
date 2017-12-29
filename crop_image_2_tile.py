import cv2
import numpy as np
import os


def crop_img(img_path, store_path, base_name, width, height):
    img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
    h, w = img.shape[0:2]
    count = 0
    print('h = ', h, 'w = ', w)
    for tph in range(h // height):
        for tpw in range(w // width):
            p = np.percentile(img[height*tph:height*(tph+1), width*tpw:width*(tpw+1), 0], 20)
            if p > 0:
                cv2.imwrite(store_path+base_name+'_crop_{:d}.jpg'.format(count), img[height*tph:height*(tph+1), width*tpw:width*(tpw+1)])
                count += 1
    print('{:d} images have been cropped !'.format(count))


def crop_lap(img_path, store_path, base_name, width, height):
    ratio = 0.5
    step_x = int(width*ratio)
    step_y = int(height*ratio)
    img = cv2.imread(img_path)
    h, w = img.shape[0:2]
    img_pad = np.zeros((step_y*(h//step_y+2), step_x*(w//step_x+2), 3), np.uint8)
    img_pad[step_y:step_y+h, step_x:step_x+w, :] = img

    pd_h, pd_w = img_pad.shape[0:2]
    for tph in range(pd_h // step_y - 1):
        for tpw in range(pd_w // step_x - 1):
            cv2.imwrite(store_path+base_name+'_{:d}_{:d}.jpg'.format(tph, tpw),
                        img_pad[step_y*tph:step_y*tph+height, step_x*tpw:step_x*tpw+width])


if __name__ == '__main__':

    src_path = '/Users/whyguu/Desktop/RGB/common-region/9794_1.tif'
    dst_path = '/Users/whyguu/Desktop/RGB/common-region-lap/11/'
    crop_lap(src_path, dst_path, '11', 512, 512)

