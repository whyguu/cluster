import cv2
import numpy as np


class Analysis(object):
    def __init__(self):
        pass

    @staticmethod
    def canny_analysis(filename=None):

        def on_canny_analysis(x=None):
            # print('position:', x)
            threshold1 = cv2.getTrackbarPos('minVal', win_name)
            threshold2 = cv2.getTrackbarPos('maxVal', win_name)
            print(threshold1, threshold2)
            if threshold1 > threshold2:
                print('threshold1 {} should be no less than threshold2 {}'.format(threshold1, threshold2))
                cv2.setTrackbarPos('minVal', win_name, threshold2)
                return

            edges = cv2.Canny(img, threshold1=threshold1, threshold2=threshold2, apertureSize=3, L2gradient=True)
            edges_resize = cv2.resize(edges, dsize=(1000, int(600*img.shape[0]/img.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imshow(win_name, edges_resize)
        img = cv2.imread(filename)
        win_name = 'analysis_an_image_by_canny'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(win_name, 600, 600*int(img.shape[0] / img.shape[1]))
        # track bar
        cv2.createTrackbar('minVal', win_name, 100, 1000, on_canny_analysis)
        cv2.createTrackbar('maxVal', win_name, 200, 1000, on_canny_analysis)

        on_canny_analysis()
        while 1:
            k = cv2.waitKey(1)
            if k == 27:
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    path = '/Users/whyguu/Desktop/QTH_band3.tif'

    # while 1:
    #     cv2.imshow('haha', img)
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27:
    #         break
    # cv2.destroyAllWindows()
    Analysis.canny_analysis(path)


