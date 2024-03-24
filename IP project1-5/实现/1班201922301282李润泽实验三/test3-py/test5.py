import cv2 as cv
import numpy as np


def Gaussian(inputImage, outputImage, sigma):
    timeBegin = cv.getTickCount()
    # 产生二维高斯滤波器kernel，行列不分离
    size = int(int(6*sigma-1)//2*2+1)  # 高斯滤波器大小为：size x size，size为奇数
    kernel = np.zeros([size, size], dtype=np.int64)
    center = size//2  # 将滤波器分为size x size个小方格，中心点为center，坐标为(0, 0)
    normal = 1/(np.exp(-(2*center*center)/(2*(sigma*sigma))))  # 用于整数化
    sumAll = 0  # 模板参数总和
    for i in range(size):
        for j in range(size):
            x = i-center  # 方格的横坐标
            y = center-j  # 方格的纵坐标
            kernel[i, j] = int(np.exp(-(x*x+y*y)/(2*sigma*sigma)) * normal)
            sumAll += kernel[i, j]
            # print(kernel[i, j], end=' ')  # 打印模板，与下一行共同使用
        # print('')  # 打印模板，与上一行共同使用
    # 对图像inputImage增添
    border = center  # 需要添加的边界大小
    transImage = cv.copyMakeBorder(inputImage, border, border, border, border,
                                   borderType=cv.BORDER_REPLICATE)  # 复制最边缘像素
    # 开始平滑操作
    rows, cols, channels = inputImage.shape
    for i in range(rows):
        for j in range(cols):
            for k in range(channels):
                temp = np.sum(np.multiply(transImage[i:i+size, j:j+size, k], kernel)) // sumAll
                if temp < 0:
                    temp = 0
                elif temp > 255:
                    temp = 255
                outputImage[i, j, k] = temp
    timeEnd = cv.getTickCount()
    time = (timeEnd-timeBegin)/cv.getTickFrequency()
    return time


def SeperateGaussian(inputImage, outputImage, sigma):
    timeBegin = cv.getTickCount()
    # 产生一维高斯滤波器kernel，行列分离
    size = int(int(6*sigma-1)//2*2+1)  # 高斯滤波器大小为：size x size，size为奇数
    kernel = np.zeros([size], dtype=np.int64)
    center = size//2  # 将滤波器分为size x size个小方格，中心点为center，坐标为(0, 0)
    normal = 1/(np.exp(-center*center/(2*(sigma*sigma))))  # 用于整数化
    sumAll = 0  # 模板参数总和
    for i in range(size):
        kernel[i] = int(np.exp(-(i-center)*(i-center)/(2*sigma*sigma)) * normal)
        sumAll += kernel[i]
        #print(kernel[i], end=' ')  # 打印模板
    kernelRow = kernel
    kernelCol = np.resize(kernel, (size, 1))
    #print(kernelCol)
    # 对图像inputImage增添
    border = center  # 需要添加的边界大小
    transImage = cv.copyMakeBorder(inputImage, border, border, border, border,
                                   borderType=cv.BORDER_REPLICATE)  # 复制最边缘像素
    # 开始平滑操作
    rows, cols, channels = inputImage.shape
    # 对行操作
    for j in range(cols):
        for k in range(channels):
            temp = np.sum(np.multiply(transImage[:, j:j+size, k], kernelRow), axis=1) // sumAll
            transImage[:, j+border, k] = temp
    # 对列操作
    for i in range(rows):
        for k in range(channels):
            temp = np.sum(np.multiply(transImage[i:i + size, border:cols + border, k], kernelCol), axis=0) // sumAll
            outputImage[i, :, k] = temp

    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    return time


sig = float(input('Please input the value of sigma: '))
print('Please choose the mode of Gaussian: ')
print('     Type 1 if you want to Gaussian')
print('     Type 2 if you want to SeperateGaussian')
print('     Type 3 if yuu want to both')
flag = int(input('The mode is: '))

imgSrc = cv.imread('./picture/a.jpg')  # (481, 641, 3)
imgDst = np.zeros(list(imgSrc.shape), dtype='uint8')

time1 = 0  # 行列不分离的时间
time2 = 0  # 行列分离的时间
time = 0  # 两种方式的时间差
#print(imgSrc.shape)

if flag == 1:
    time = Gaussian(imgSrc, imgDst, sig)
elif flag == 2:
    time = SeperateGaussian(imgSrc, imgDst, sig)
elif flag == 3:
    time1 = Gaussian(imgSrc, imgDst, sig)
    time2 = SeperateGaussian(imgSrc, imgDst, sig)

strSigma = 'Gaussian image(sigma: ' + str(sig) + ')'
cv.imshow('source image', imgSrc)
cv.imshow(strSigma, imgDst)
saveSigma = str(sig) + '.png'
cv.imwrite(saveSigma, imgDst)

print("Successful!!!")
if flag == 1 or flag == 2:
    print('time(s):', time)
elif flag == 3:
    print('time1(s):', time1)
    print('time2(s):', time2)
    print('time2-time1 =', time2-time1)

cv.waitKey(0)
cv.destroyAllWindows()
