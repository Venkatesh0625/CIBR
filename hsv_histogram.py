import cv2, numpy as np
from math import *


def hscColorHistpgram(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rows, cols, _ = img_hsv.shape

    h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]

    HLevels, SLevels, VLevels = 8, 2, 2
    HMax, SMax, VMax = h.max(), s.max(), v.max()

    hsvColorHistorgram = np.zeros((8, 2, 2))

    index = np.zeros((rows * cols, 3), dtype=int)
    count = 0

    quantizedValueForH = np.zeros((rows, cols))
    quantizedValueForS = np.zeros((rows, cols))
    quantizedValueForV = np.zeros((rows, cols))

    for row in range(rows):
        for col in range(cols):
            quantizedValueForH[row, col] = ceil(HLevels * h[row, col] / HMax)
            quantizedValueForS[row, col] = ceil(SLevels * s[row, col] / SMax)
            quantizedValueForV[row, col] = ceil(VLevels * v[row, col] / VMax)

            index[count, 0] = quantizedValueForH[row, col]
            index[count, 1] = quantizedValueForS[row, col]
            index[count, 2] = quantizedValueForV[row, col]
            count += 1

    index = np.subtract(index, np.ones(index.shape, dtype=int))

    for row in range(rows * cols):
        if -1 not in index[row]:
            x, y, z = index[row]
            hsvColorHistorgram[x, y, z] += 1

    hsvColorHistorgram.shape = (32, 1)
    hsvColorHistorgram = hsvColorHistorgram / sum(hsvColorHistorgram)

