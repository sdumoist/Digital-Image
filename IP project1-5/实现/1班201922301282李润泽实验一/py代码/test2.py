import cv2 as cv
import numpy as np

imgSrcA = cv.imread('./picture/a.png', cv.IMREAD_UNCHANGED)
imgSrcBg = cv.imread('./picture/bg.png', cv.IMREAD_UNCHANGED)
# cv.imshow('alpha src', imgSrcA)

rows, cols, channels = imgSrcA.shape
B, G, R, A = cv.split(imgSrcA)
imgAlpha = np.zeros([rows, cols, 3], dtype='uint8')

for k in range(3):
    imgAlpha[:, :, k] = A[:, :]

# cv.imshow('alpha channel', imgAlpha)

# 不包含alpha通道
imgSrcA = cv.imread('./picture/a.png')

imgMix = np.zeros([rows, cols, 3], dtype='uint8')
for i in range(rows):
    for j in range(cols):
        alpha = A[i][j] / 255
        beta = 1 - alpha
        for k in range(3):
            temp = int(imgSrcA[i, j, k]*alpha + imgSrcBg[i, j, k]*beta)
            if temp > 255:
                imgMix[i, j, k] = 255
            elif temp < 0:
                imgMix[i, j, k] = 0
            else:
                imgMix[i, j, k] = temp

cv.imshow('mixed image', imgMix)
cv.waitKey(0)
cv.destroyAllWindows()
