import numpy as np
import cv2 as cv


def anamorphic(imgSrc):
    """
    根据实验要求的公式实现图像变形
    为了尽可能的减少运行时间，所有的循环和各种运算都尽可能地使用了Numpy，
    因此在该函数中不包含任何循环，时间从4.6s优化到了0.076s

    :param imgSrc: 输入图像
    :return: imgDst time 变形后的图像 时间
    """
    timeBegin = cv.getTickCount()  # 记录开始时间
    rows, cols, channels = imgSrc.shape

    coordinateDst = np.zeros([rows, cols, 2])  # 储存输出图像中心归一化坐标
    coordinateSrc = np.zeros([rows, cols, 2])  # 储存输入图像中心归一化坐标
    coordinateFinal = np.zeros([rows, cols, 2])  # 储存输入图像的原坐标

    # 输出图像中心归一化
    coordinateDst[:, :, 1] = np.arange(cols)
    coordinateDst[:, :, 0] = np.arange(rows).reshape(-1, 1)
    coordinateDst = (coordinateDst - 0.5 * np.array([rows, cols])) / (0.5 * np.array([rows, cols]))

    # 实现映射
    r = np.sqrt(np.sum(coordinateDst ** 2, axis=2))  # 计算所有的r
    theta = (1 - r) ** 2  # 计算所有的theta
    coordinateSrc[:, :, 0] = np.cos(theta) * coordinateDst[:, :, 0] - np.sin(theta) * coordinateDst[:, :, 1]
    coordinateSrc[:, :, 1] = np.sin(theta) * coordinateDst[:, :, 0] + np.cos(theta) * coordinateDst[:, :, 1]
    # 计算r的掩码，此时的r的值只是为0和1
    r = np.expand_dims(r, 2).repeat(2, axis=2)  # 先复制并扩充一个维度
    r[r < 1] = -1
    r[r >= 1] = 0
    r[r == -1] = 1
    coordinateFinal += np.multiply(r, coordinateSrc)  # 对应于 f^-1 中的“otherwise”
    # 下面三句交换0 和 1
    r[r == 1] = -1
    r[r == 0] = 1
    r[r == -1] = 0
    coordinateFinal += np.multiply(r, coordinateDst)  # 对应于 f^-1 中的“if r >= 1”
    coordinateFinal = (coordinateFinal + 1) * np.array([0.5*rows, 0.5*cols])
    coordinateFinal = coordinateFinal.astype(np.int64)  # 不能为np.uint8，可以试一试

    x = np.arange(rows).repeat(cols)  # x 为 [0, 0, 0, ..., 2, 2, 2, ..., rows-1]
    y = np.tile(np.arange(cols), rows)  # y 为 [0, 1, 2, ..., cols-1, 0, 1, 2, ...cols-1, ...]
    # imgDst为输出图像
    imgDst = imgSrc[coordinateFinal[x, y, 0], coordinateFinal[x, y, 1], :].reshape(rows, cols, channels)

    timeEnd = cv.getTickCount()  # 记录结束时间
    time = (timeEnd - timeBegin) / cv.getTickFrequency()  # 计算总时间
    return imgDst, time  # 0.076 s


imgSrc = cv.imread('./picture/lab2.png')
imgDst, time = anamorphic(imgSrc)
cv.imshow('imgSrc', imgSrc)
cv.imshow('imgDst', imgDst)
print('Successful!!!')
print('time: %.4f s' % time)
cv.waitKey(0)
cv.destroyAllWindows()
