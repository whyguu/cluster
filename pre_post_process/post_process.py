import numpy as np
import os
import cv2
from skimage import morphology
from sklearn import linear_model


class PostPrecess(object):
    def __init__(self):
        pass

    @staticmethod
    def skeleton(bw_img):
        # 提取骨架
        # 不改变原图
        skeleton_img = morphology.skeletonize(bw_img > 0)  # 输入应为二值图像,输出的数组中是bool值
        skeleton_img = skeleton_img.astype(np.uint8)  # int64
        return skeleton_img*255

    @staticmethod
    def spur_clear(bw_img, iteration=4):
        """
        不改变原图
        去除毛刺的具体实现
        :param bw_img: 二值图像
        :param iteration: 循环迭代次数
        :return: 去除毛刺的图像
        """
        img = bw_img > 0
        img = img.astype(np.uint8)
        # img:像素值为０和１
        row, col = img.shape
        for _ in range(iteration):
            for r in range(1, row-1):
                for c in range(1, col-1):
                    if img[r, c] == 1:
                        # 判断８邻域内像素点＝１的个数
                        tp = np.sum(img[r - 1:r + 2, c - 1:c + 2])
                        if tp <= 2:
                            img[r, c] = 0

        return img*255

    @staticmethod
    def noisy_points_clear(img):
        rg = 3
        img = img>0
        for r in range(1, img.shape[0] - 1):
            for c in range(1, img.shape[1] - 1):
                if img[r, c] == 1:
                    # 判断 邻域内像素点＝１的个数
                    tp1 = np.sum(img[r-rg:r+rg+1, c-rg:c+rg+1])
                    tp2 = np.sum(img[r-rg+1:r+rg, c-rg+1:c+rg])
                    tp3 = np.sum(img[r-rg+2:r+rg-1, c-rg+2:c+rg-1])
                    # print(tp1, tp2, tp3)
                    if tp1-tp2 < 1e-5 or tp2-tp3 < 1e-5:
                        img[r - rg:r + rg+1, c - rg:c + rg+1] = 0
        return img.astype(np.uint8)*255

    @staticmethod
    def lsd(img):
        # 不改变原图
        rt_img = np.zeros(img.shape, dtype=np.uint8)
        lsd = cv2.createLineSegmentDetector()  # parameters can be set
        lines = lsd.detect(img)
        # lsd.drawSegments(rt_img, lines)
        for line in lines[0]:
            x0 = int(round(line[0][0]))
            y0 = int(round(line[0][1]))
            x1 = int(round(line[0][2]))
            y1 = int(round(line[0][3]))
            cv2.line(rt_img, (x0, y0), (x1, y1), 255, 1, cv2.LINE_8)
        return rt_img

    @staticmethod
    def hough_line_p(img):
        # 不改变原图
        rt_img = np.zeros(img.shape, dtype=np.uint8)
        lines = cv2.HoughLinesP(img, rho=2, theta=3.14/30, threshold=20)
        # print(lines)
        for line in lines:
            x0 = int(round(line[0][0]))
            y0 = int(round(line[0][1]))
            x1 = int(round(line[0][2]))
            y1 = int(round(line[0][3]))
            cv2.line(rt_img, (x0, y0), (x1, y1), 255, 1, cv2.LINE_8)
        return rt_img

    @staticmethod
    def morphology(img, method, kernel_size, iteration=1):
        # 不改变原图
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        img = cv2.morphologyEx(src=img, op=method, kernel=kernel, iterations=iteration)
        return img

    @staticmethod
    def line_grow(img):
        ls = []
        max_points = 20  # 最大增长点数
        range_y = 3  # y跨度
        range_x = 2  # x跨度
        max_error = 5  # 直线拟合的 mse error
        """
        :param img:
        """
        def region_grow():
            ls.clear()
            ls.append(anchor)
            count = 0
            iteration = 0
            while len(ls) != iteration and count < max_points:
                point = ls[iteration]
                x_min = max(point[0] - 1, 0)
                x_max = min(point[0] + 1, img.shape[1] - 1)
                y_min = max(point[1] - 1, 0)
                y_max = min(point[1] + 1, img.shape[0] - 1)
                for it_x in range(x_min, x_max+1):
                    for it_y in range(y_min, y_max+1):
                        if img[it_y, it_x] == img[anchor[1], anchor[0]]:
                            if [it_x, it_y] not in ls:
                                ls.append([it_x, it_y])
                                count += 1
                iteration += 1
            # print(ls)
            return ls

        def line_param():
            mx = np.ones((len(ls), 2), np.float32)
            yy = np.zeros((len(ls), 1), np.float32)
            for i in range(len(ls)):
                mx[i, 0] = ls[i][0]
                yy[i, 0] = ls[i][1]
            k, b = np.dot(np.linalg.pinv(mx), yy)
            # direction
            sm = np.sum(yy[:, 0])/yy.shape[0]
            if sm < anchor[1]:
                direc = 'd'
            else:
                direc = 'u'
            sm = np.sum(mx[:, 0]) / yy.shape[0]
            if sm < anchor[0]:
                direc = 'r'+direc
            else:
                direc = 'l'+direc
            # error
            error = np.sum(np.power(np.dot(mx, [k, b])-yy, 2))/len(ls)

            return k[0], b[0], error, direc

        def sci_line_param():
            mx = np.ones((len(ls), 2), np.float32)
            yy = np.zeros((len(ls), 1), np.float32)
            for i in range(len(ls)):
                mx[i, 0] = ls[i][0]
                yy[i, 0] = ls[i][1]
            reg = linear_model.LinearRegression(fit_intercept=True)
            reg.fit(mx, yy)
            print(reg.coef_)  # k
            print(reg.intercept_)  # b

        for row in range(1, img.shape[0]-1):
            for col in range(1, img.shape[1]-1):
                # find end-point
                if img[row, col] == 0:
                    continue
                if np.sum(np.uint8(img[row - 1:row + 2, col - 1:col + 2] > 0)) != 2:
                    continue
                # trace the end-point
                # region grow
                anchor = (col, row)
                # print(img[anchor[1], anchor[0]])
                region_grow()
                if len(ls) < 6:
                    # print(ls)
                    continue
                k, b, error, direc = line_param()
                if error > max_error:
                    continue
                print(k, b, error, direc)
                print(ls)
                if abs(k) > 1e5 and direc[1] == 'd':  # a vertical line
                    for it in range(row+5, img.shape[0]):
                        if img[it, col] > 0:
                            cv2.line(img, (col, row), (col, it), 255, 1, cv2.LINE_4)
                            break
                elif abs(k) > 1e5 and direc[1] == 'u':
                    for it in range(row-5, -1, -1):
                        if img[it, col] > 0:
                            cv2.line(img, (col, row), (col, it), 255, 1, cv2.LINE_4)
                            break
                else:
                    if direc[0] == 'r':
                        start = col+3
                        end = img.shape[1]
                        step = 1
                    elif direc[0] == 'l':
                        start = col-3
                        end = -1
                        step = -1
                    for it in range(start, end, step):
                        y = k * it + b
                        y = int(y)
                        if y <= 0:
                            cv2.line(img, (col, row), (it, 0), 255, 1, cv2.LINE_4)
                            break
                        elif y >= img.shape[0]-1:
                            cv2.line(img, (col, row), (it, img.shape[0]-1), 255, 1, cv2.LINE_4)
                            break
                        elif np.any(img[y-range_y:y+range_y+1, it-range_x:it+1+range_x]):
                            xixi = np.argmax(img[y-range_y:y+range_y+1, it])
                            cv2.line(img, (col, row), (it, y+1+xixi-range_y), 255, 1, cv2.LINE_4)
                            break
                    # print((col, row), (it, y))

        return img


if __name__ == '__main__':
    path = '../xixi_black.png'
    path1 = '../xixi.jpg'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    img = PostPrecess.skeleton(img)
    cv2.imwrite('skeleton.png', img)
    img = PostPrecess.morphology(img, cv2.MORPH_CLOSE, (5, 5), 3)
    img = PostPrecess.noisy_points_clear(img)
    cv2.imwrite('noisy.png', img)
    img = PostPrecess.line_grow(img)
    cv2.imwrite('grow.png', img)
    # iter_num = 2
    # img1 = img = PostPrecess.bw_spur_clear(img, iteration=2)
    # for i in range(iter_num):
    #     # img = PostPrecess.bw_spur_clear(img1, iteration=1)
    #     img2 = PostPrecess.morphology(img1, cv2.MORPH_DILATE, (5, 5), iteration=3)
    #     img1 = PostPrecess.skeleton(img2)
    #     if i == iter_num-1:
    #         cv2.imwrite('skeleton.png', img1)
    #         cv2.imwrite('dilate.png', img2)
    #         # cv2.imwrite('spur_clear.png', img)

    # example for lsd
    # img = PostPrecess.hough_line_p(img)
    # img = PostPrecess.lsd(img)
    # cv2.imwrite('hough.png', img)



