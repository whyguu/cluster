import os
import numpy as np


if __name__ == '__main__':
    image_path = './train_data/'
    save_path = './'

    tag_len = 4
    file_names = os.listdir(image_path)
    img_names = [name for name in file_names if '.tif' in name and 'image' in name]
    label_names = [name for name in file_names if '.tif' in name and 'label' in name]

    img_names = sorted(img_names, key=lambda x: int(x[0:tag_len]))
    label_names = sorted(label_names, key=lambda x: int(x[0:tag_len]))
    print(img_names)
    print(label_names)

    if len(img_names) != len(label_names):
        print('image number not equal label number !!!')
        exit(0)

    with open(save_path+'train.txt', 'w') as file:
        for it in range(len(img_names)):
            file.write(img_names[it] + ' ')
            file.write(label_names[it]+'\r\n')







