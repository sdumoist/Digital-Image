import numpy as np
import cv2 as cv
import math


def Scale(imgSrc, imgDst, scaleRow, scaleCol):
    """
    双线性插值法,来调整图片尺寸

    :param imgSrc: 原始图片
    :param imgDst: 调整后的目标图片的尺寸
    :param scaleRow: x的倍数变化，eg. 0.5; 1; 2
    :param scaleCol: y的倍数变化，eg. 0.5; 1; 2
    :return: time 执行时间
    """
    timeBegin = cv.getTickCount()  # 记录开始时间
    rowsSrc, colsSrc, channels = imgSrc.shape
    rowsDst = int(rowsSrc*scaleRow)
    colsDst = int(colsSrc*scaleCol)
    # i：纵坐标y，j：横坐标x
    # 缩放因子，scaleCol 和 scaleRow，和函数的参数不一样
    scaleCol = colsSrc / colsDst
    scaleRow = rowsSrc / rowsDst

    for i in range(rowsDst):
        for j in range(colsDst):
            srcX = float((j + 0.5) * scaleCol - 0.5)
            srcY = float((i + 0.5) * scaleRow - 0.5)

            # 向下取整，代表靠近源点的左上角的那一点的行列号
            srcXint = math.floor(srcX)
            srcYint = math.floor(srcY)

            # 取出小数部分，用于构造权值
            srcXfloat = srcX - srcXint
            srcYfloat = srcY - srcYint

            if srcXint + 1 == colsSrc or srcYint + 1 == rowsSrc:
                imgDst[i, j, :] = imgSrc[srcYint, srcXint, :]
                continue
            imgDst[i, j, :] = (1. - srcYfloat) * (1. - srcXfloat) * imgSrc[srcYint, srcXint, :] + \
                               (1. - srcYfloat) * srcXfloat * imgSrc[srcYint, srcXint + 1, :] + \
                               srcYfloat * (1. - srcXfloat) * imgSrc[srcYint + 1, srcXint, :] + \
                               srcYfloat * srcXfloat * imgSrc[srcYint + 1, srcXint + 1, :]

    timeEnd = cv.getTickCount()  # 记录结束时间
    time = (timeEnd - timeBegin) / cv.getTickFrequency()  # 计算总时间
    imgDst.shape = imgDst.shape
    imgDst = imgDst.copy()
    return time


def makeImgDst(imgSrc, scaleRow, scaleCol):
    """
    返回一个imgDst，用于Scale中的第二个参数

    :param imgSrc: 原始图片
    :param scaleRow: x的倍数变化，eg. 0.5; 1; 2
    :param scaleCol: y的倍数变化，eg. 0.5; 1; 2
    :return: imgDst 输出图像
    """
    rowsSrc, colsSrc, channels = imgSrc.shape
    rowsDst = int(rowsSrc*scaleRow)
    colsDst = int(colsSrc*scaleCol)
    imgDst = np.zeros((rowsDst, colsDst, channels), dtype='uint8')
    return imgDst


# x和y轴的缩放倍数，该实验可尝试：x = 1.44, y = 0.64
x = float(input('The scale of X: '))
y = float(input('The scale of Y: '))
# 输入图像
imgSrc = cv.imread('./picture/lab2.png')
# 输出图像
imgDst = makeImgDst(imgSrc, x, y)
# 缩放
time = Scale(imgSrc, imgDst, x, y)

nameImg = 'scale x: %.2f scale y: %.2f' % (x, y)
cv.imshow('source', imgSrc)
cv.imshow(nameImg, imgDst)
print('\nSuccessful!!!')
print('Scale time: %.4f s' % time)
cv.waitKey(0)
cv.destroyAllWindows()
