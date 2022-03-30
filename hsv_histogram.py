import cv2, math
import numpy as np

img = cv2.imread("img.png ")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

rows, cols, p = img_hsv.shape
h, s, v = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]
hist_h = cv2.calcHist([img_hsv],[0],None,[256],[0,256])
hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
hist_v = cv2.calcHist([v],[0],None,[256],[0,256])

numberOfLevelsForH = 8
numberOfLevelsForS = 2
numberOfLevelsForV = 2

maxValueForH = h.max()
maxValueForS = s.max()
maxValueForV = v.max()

hsvColorHistorgram = np.zeros((8, 2, 2))

index = np.zeros((rows*cols, 3), dtype=int)
count = 0

quantizedValueForH = np.zeros((rows, cols))
quantizedValueForS = np.zeros((rows, cols))
quantizedValueForV = np.zeros((rows, cols))

for row in range(rows):
    for col in range(cols):
        quantizedValueForH[row, col] = math.ceil(numberOfLevelsForH * h[row, col]/maxValueForH)
        quantizedValueForS[row, col] = math.ceil(numberOfLevelsForS * s[row, col]/maxValueForS)
        quantizedValueForV[row, col] = math.ceil(numberOfLevelsForV * v[row, col]/maxValueForV)
        # print(numberOfLevelsForH * h[row, col]/maxValueForH, numberOfLevelsForS * s[row, col]/maxValueForS, numberOfLevelsForV * v[row, col]/maxValueForV)
        
        index[count, 0] = quantizedValueForH[row, col]
        index[count, 1] = quantizedValueForS[row, col]
        index[count, 2] = quantizedValueForV[row, col]
        count += 1

index = np.subtract(index, np.ones(index.shape, dtype=int))

print("max", index.max())


for row in range(rows*cols):
    if index[row, 0] == -1 or index[row, 1] == -1 or index[row, 2] == -1:
        continue

    hsvColorHistorgram[index[row, 0], index[row, 1], index[row, 2]] = hsvColorHistorgram[index[row, 0], index[row, 1], index[row, 2]] + 1

hsvColorHistorgram.shape = (32, 1)
hsvColorHistorgram =  hsvColorHistorgram / sum(hsvColorHistorgram)
