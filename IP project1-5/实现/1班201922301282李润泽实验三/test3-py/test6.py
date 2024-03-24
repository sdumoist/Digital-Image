import cv2 as cv
import numpy as np
import math


def jbf(D, C, W, sigma_f, sigma_g):
    """
    brief: 实现图像的联合双边滤波处理

    :param D: 输入图像
    :param C: 引导图像
    :param W: 滤波窗口大小
    :param sigma_f: spatial kernel 标准差
    :param sigma_g: range kernel 标准差
    :return: 元组(JBF后的图像，执行时间)
    """
    timeBegin = cv.getTickCount()  # 记录开始时间
    # 让变量更清楚一些
    imgSrc = D
    imgGuide = C
    size = W // 2 * 2 + 1  # 重新调整滤波器大小，保证为奇数
    sigmaSpatial = sigma_f
    sigmaRange = sigma_g

    # 两次采样后，可能会导致图像的shape不一致，因此选择最小值
    rowsSrc, colsSrc, channels = imgSrc.shape
    rowsGuide, colsGuide, channels = imgGuide.shape
    rows = min(rowsSrc, rowsGuide)
    cols = min(colsSrc, colsGuide)
    imgSrc = imgSrc[:rows, :cols, :]
    imgGuide = imgGuide[:rows, :cols, :]

    # Spatial Kernel
    kernelSpatial = np.zeros([size, size], dtype=np.float_)
    center = size // 2  # 将滤波器分为size x size个小方格，中心点为center，坐标为(0, 0)
    for i in range(size):
        for j in range(size):
            x = i - center  # 方格的横坐标
            y = center - j  # 方格的纵坐标
            kernelSpatial[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigmaSpatial ** 2))
            #print(kernelSpatial[i, j], end=' ')  # 打印Spatial模板，与下一行共同使用
        #print('')  # 打印Spatial模板，与上一行共同使用

    # Range Kernel
    # 提前计算 2*sigmaRange**2代入公式，对于这个计算会提升25%的速度，但由于其本身时间仅为0.0004311s
    # 加速后也仅为0.0002932s，因此意义不大
    kernelRange = np.zeros([256], dtype=np.float_)
    for i in range(256):
        kernelRange[i] = np.exp(-(i ** 2) / (2 * sigmaRange ** 2))
        #print(kernelRange[i], end=' ')  # 打印Range模板

    # 输出的图像
    imgDst = np.zeros([rows, cols, channels], dtype='uint8')

    # 边界填充，imgFillSrc为对原图的边界扩充，imgFillGuide为对引导图的边界扩充
    border = center  # 需要添加的边界大小
    imgFillSrc = cv.copyMakeBorder(imgSrc, border, border, border, border,
                                    borderType=cv.BORDER_REPLICATE)
    imgFillGuide = cv.copyMakeBorder(imgGuide, border, border, border, border,
                                    borderType=cv.BORDER_REPLICATE)
    # 为了便于计算，改为np.int
    imgFillSrc = imgFillSrc.astype(np.int_)
    imgFillGuide = imgFillGuide.astype(np.int_)

    # matSpatial就是kernelSpatial的三维形式
    matSpatial = np.zeros([size, size, channels], dtype=np.float_)
    for k in range(channels):
        matSpatial[:, :, k] = kernelSpatial
    # matRange 储存距离滤波器内核对imgFillGuide的结果
    matRange = np.zeros([size, size, channels], dtype=np.float_)

    # 开始滤波操作
    for i in range(rows):
        for j in range(cols):
            # 计算matRange。colorDiff是灰度差
            colorDiff = np.abs(imgFillGuide[i:i + size, j:j + size, :] - imgFillGuide[i + center, j + center, :])
            matRange = kernelRange[colorDiff]
            # matMulti 储存空间滤波器和内核滤波器相乘的结果
            matMulti = np.multiply(matSpatial, matRange)
            # sumMat为matMulti元素的和，matMulti的shape为(size, size, channels)，sumMat的shape为(channels,)
            sumMat = np.sum(np.sum(matMulti, axis=0), axis=0)
            matMulti = np.multiply(matMulti, imgFillSrc[i:i+size, j:j+size, :])
            temp = np.sum(np.sum(matMulti, axis=0), axis=0) / sumMat
            # 限制在0~255范围
            temp = np.minimum(temp, 255)
            temp = np.maximum(temp, 0)
            imgDst[i, j, :] = temp

    timeEnd = cv.getTickCount()  # 记录结束时间
    time = (timeEnd - timeBegin) / cv.getTickFrequency()  # 计算总时间
    return imgDst, time


def sampling(imgSrc, scaleRow, scaleCol):
    """
    双线性插值法,来调整图片尺寸

    :param imgSrc: 原始图片
    :param scaleRow: x的倍数变化，eg. 0.5; 1; 2
    :param scaleCol: y的倍数变化，eg. 0.5; 1; 2
    :return: 元组(缩放后的图像，执行时间)
    """
    timeBegin = cv.getTickCount()  # 记录开始时间
    rowsSrc, colsSrc, channels = imgSrc.shape
    rowsDst = int(rowsSrc*scaleRow)
    colsDst = int(colsSrc*scaleCol)
    dst_img = np.zeros((rowsDst, colsDst, channels), dtype='uint8')
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
                dst_img[i, j, :] = imgSrc[srcYint, srcXint, :]
                continue
            dst_img[i, j, :] = (1. - srcYfloat) * (1. - srcXfloat) * imgSrc[srcYint, srcXint, :] + \
                               (1. - srcYfloat) * srcXfloat * imgSrc[srcYint, srcXint + 1, :] + \
                               srcYfloat * (1. - srcXfloat) * imgSrc[srcYint + 1, srcXint, :] + \
                               srcYfloat * srcXfloat * imgSrc[srcYint + 1, srcXint + 1, :]

    timeEnd = cv.getTickCount()  # 记录结束时间
    time = (timeEnd - timeBegin) / cv.getTickFrequency()  # 计算总时间
    return dst_img, time


size = int(input('Please input the size of kernel: '))
sigmaSpatial = float(input('Please input the value of Spatial sigma: '))
sigmaRange = float(input('Please input the value of Range sigma'))
"""
不断的优化后，下面的时间已经大为减少，如最后一个451秒，现在也只需要11.28秒，最开始的一个19秒，变为6.52秒
参数（按照上面顺序）：
    3  1  1： 19秒，感觉几乎没变化
    5  10 10：30秒，很轻微的变化
    7  10 20：45秒，明显的变化
    8  10 20：57秒，明显变化，还行
    9  10 20：72秒，很不错，真实
    13 13 30：131秒，非常漂亮
    25 12.5 50：451秒，有点假
"""

imgSrc = cv.imread('./picture/2_2.png')
imgDownSamp, timeDown = sampling(imgSrc, 0.5, 0.5)  # 这里可以修改下采样率
imgUpSamp, timeUp = sampling(imgDownSamp, 2, 2)  # 这里可以修改上采样率
imgGuide = imgUpSamp

imgJBF, timeJBF = jbf(imgSrc, imgGuide, size, sigmaSpatial, sigmaRange)

print('DownSamp image(s):', timeDown)
print('UpSamp image(s):', timeUp)
print('JBF image(s):', timeJBF)

cv.imshow('source image', imgSrc)
cv.imshow('guide image', imgGuide)
cv.imshow('jbf image', imgJBF)

cv.waitKey(0)
cv.destroyAllWindows()
