import numpy as np
import cv2 as cv
from skimage import measure, color
import scipy
# import matplotlib.pyplot as plt


def preTreat(imgSrc):
    """
    通过图像边缘灰度增强和光照补偿对原图像进行预处理
    :param imgSrc: 输入图像
    :return: 预处理后的图像
    """
    imgDst = edgeEnhance(imgSrc)  # 边缘增强
    imgDst = lightCompensation(imgDst)  # 光照补偿
    return imgDst


def edgeEnhance(imgSrc):
    """
    图像边缘灰度增强
    :param imgSrc: 输入图像
    :return: 边缘灰度增强图像
    """
    imgSrcGauss = cv.GaussianBlur(imgSrc, (3, 3), 0)  # 对原图像进行高斯平滑
    imgSrcSharp = cv.Laplacian(imgSrcGauss, cv.CV_16S)  # 计算拉普拉斯
    imgSrcSharp = cv.convertScaleAbs(imgSrcSharp)  # 计算绝对值，并将结果转换无符号8位类型
    imgDst = cv.add(imgSrc, imgSrcSharp)  # 原图像与边缘图像相加
    return imgDst


def lightCompensation(imgSrc):
    """
    用GrayWorld算法进行光照补偿
    :param imgSrc: 输入图像
    :return: 光照补偿后的图像
    """
    rows, cols, channels = imgSrc.shape
    avgRGB = np.sum(np.sum(imgSrc, axis=0), axis=0) / (rows * cols)  # 计算每个通道的平均灰度值
    avgGray = np.sum(avgRGB) / channels  # 三通道的总体平均灰度值
    avgCoeff = avgGray / avgRGB  # 三通道的增益系数
    imgDst = np.zeros(imgSrc.shape, dtype=np.uint8)
    imgTemp = imgSrc * avgCoeff
    imgDst[:, :, :] = np.minimum(imgTemp, 255)
    # cv.imshow('lightCompensation', imgDst)
    return imgDst


def faceDetection(imgSrc, imgSrcPreTreate):
    """
    人脸检测
    :param imgSrc: 原始图像
    :param imgSrcPreTreate: 预处理后的图像
    :return: 人脸检测后的图像
    """
    imgSrcYCrCb = cv.cvtColor(imgSrcPreTreate, cv.COLOR_BGR2YCR_CB)
    imgSrcYCrCb = cv.GaussianBlur(imgSrcYCrCb, (5, 5), 0)
    imgSrcYCrCbVice = imgSrcYCrCb  # 二值化时 imgSrcYCrCb 被改变了，这里保留个副本
    rows, cols, channels = imgSrcYCrCb.shape

    # 二值化
    imgSrcYCrCb[imgSrcYCrCb[:, :, 0] < 70] = 0
    imgTemp = np.zeros((rows, cols), dtype=np.uint8)

    imgTempCr = np.zeros((rows, cols), dtype=np.uint8)
    imgTemp[133 <= imgSrcYCrCb[:, :, 1]] = 1
    imgTempCr[imgSrcYCrCb[:, :, 1] <= 173] = 1
    imgTempCr = np.multiply(imgTempCr, imgTemp)

    imgTemp[:, :] = 0

    imgTempCb = np.zeros((rows, cols), dtype=np.uint8)
    imgTemp[77 <= imgSrcYCrCb[:, :, 2]] = 1
    imgTempCb[imgSrcYCrCb[:, :, 2] <= 127] = 1
    imgTempCb = np.multiply(imgTempCb, imgTemp)

    imgBinary = np.multiply(imgTempCr, imgTempCb)
    imgBinary[imgBinary == 1] = 255
    imgBinary = cv.convertScaleAbs(imgBinary)
    # cv.imshow('imgBinary', imgBinary)

    # 形态学处理
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))  # 结构元
    faceMorph = cv.morphologyEx(imgBinary, cv.MORPH_OPEN, kernel)  # 开运算
    # cv.imshow('faceMorphOpen', faceMorph)
    faceMorph = cv.morphologyEx(faceMorph, cv.MORPH_CLOSE, kernel)  # 闭运算
    # cv.imshow('faceMorphClose', faceMorph)

    # 连通区域标记
    faceLabel = measure.label(faceMorph, connectivity=2)  # 八邻域标记
    # faceLabelColor = color.label2rgb(faceLabel, bg_label=0)  # 根据不同标记显示不同的颜色，更清晰看到不同连通域
    # cv.imshow('faceLabelColor', faceLabelColor)

    # 人脸“三庭五眼”的值进行筛选 从0.6到2
    countFaces = 0  # 人脸总数
    count = 0  # 符合的连通域总数
    imgDst = imgSrc.copy()
    for region in measure.regionprops(faceLabel):  # measure.regionprops()测量标记的图像区域的属性
        minRow, minCol, maxRow, maxCol = region.bbox  # 外接边界框的坐标
        if (maxCol - minCol) / rows > 1 / 20 and (maxRow - minRow) / cols > 1 / 15:  # 检测足够大的连通区域
            ratio = (maxRow - minRow) / (maxCol - minCol)  # 计算“三庭五眼”的值
            if 0.6 < ratio < 2.0:  # 符合条件的ratio
                count += 1
                # 下面注释掉的是做实验时方便，可以逐个查看指定连通域的结果
                # 对于Oracle1.jpg，count从1到4，Oracle2.jpg，count从1到2
                # if count == 3:
                #     eyeJudge(region, faceMorph, imgSrc, imgDst)
                if eyeJudge(region, faceMorph, imgSrc, imgDst):  # 人眼检测
                    countFaces += 1
                    imgDst = cv.rectangle(imgDst, (minCol, minRow), (maxCol, maxRow), (0, 255, 0), 2)
    # print('maybe faces: ', count)
    # print('faces: ', countFaces)
    return imgDst


def eyeJudge(region, faceMorph, imgSrc, imgDst):
    """
    人眼检测
    :param region: 人脸候选区
    :param faceMorph: 形态学处理的图像
    :param imgSrc: 原始图像
    :param imgDst: 人脸检测的结果图像
    :return: 包含人眼返回True，反之返回False
    """
    minRow, minCol, maxRow, maxCol = region.bbox  # region 连通域外接边界框的坐标，也就是人脸候选区
    regionFace = faceMorph[minRow:maxRow, minCol:maxCol]  # 废弃不用，效果不好，用后面的 regionThreshold 代替

    # 眼睛粗略定位
    imgGray = sobelEdge(imgSrc)  # Sobel得到边缘
    _, imgThreshold = cv.threshold(imgGray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # 对边缘采用大津法二值处理
    # 下面的动态阈值废弃，效果不好
    # imgThreshold = cv.adaptiveThreshold(imgGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    # cv.imshow('imgThreshold', imgThreshold)
    regionThreshold = imgThreshold[minRow:maxRow, minCol:maxCol]  # 对应的人脸候选区
    rows, cols = regionThreshold.shape
    # 垂直积分投影
    numVerBlack = np.zeros([cols], dtype=np.int_)  # 黑色像素值的数量
    for i in range(cols):
        numVerBlack[i] = rows - np.sum(regionThreshold[:, i]) // 255  # 白色值为255，求和除以255就是白色的数量，做差就是黑色的数量
    # 水平积分投影
    numHorBlack = np.zeros([rows], dtype=np.int_)  # 黑色像素值的数量
    for i in range(rows):
        numHorBlack[i] = cols - np.sum(regionThreshold[i, :]) // 255
    # 作图
    imgVer = np.zeros([rows, cols], dtype=np.uint8)
    imgVer[:, :] = np.array([255])
    for i in range(cols):
        imgVer[rows - numVerBlack[i]:, i] = 0
    imgHor = np.zeros([rows, cols], dtype=np.uint8)
    imgHor[:, :] = np.array([255])
    for i in range(rows):
        imgHor[i, cols - numHorBlack[i]:] = 0
    # 放在一起显示，如果不想显示，下面5句都可以注释掉
    # board = np.zeros([2*rows, 2*cols], dtype=np.uint8)
    # board[:rows, :cols] = regionThreshold
    # board[:rows, cols:] = imgHor
    # board[rows:, :cols] = imgVer
    # board[rows:, cols:] = np.array([255])
    # 下面的4个cv.imshow()中，推荐最后一个，等价于前面3个
    # cv.imshow('regionThreshold', regionThreshold)
    # cv.imshow('imgVer', imgVer)
    # cv.imshow('imgHor', imgHor)
    # cv.imshow('board', board)  # 推荐这个，效果好
    # 进行定位
    # 对水平灰度积分投影进行函数拟合，并根据拟合的曲线求得极值点
    x = np.arange(rows)
    y = numHorBlack
    z1 = np.polyfit(x, y, 15)  # 最小二乘法15次多项式拟合（调参）
    p1 = np.poly1d(z1)  # 拟合后的公式
    yvals = p1(x)
    numPeaks = scipy.signal.find_peaks(yvals, distance=10)  # 找极大值点
    # 找极小值点，在没找到直接找极小值点的函数比如find_valleys后，突然意识到yvals取反valley就变为peak了。。。
    numValleys = scipy.signal.find_peaks(-yvals, distance=10)  # 找极小值点
    # 绘制曲线
    # plot1 = plt.plot(x, y, 'o', label='original')
    # plot2 = plt.plot(x, yvals, 'r', label='curve fit')
    # plt.xlabel('xaxis')
    # plt.ylabel('yaxis')
    # plt.legend(loc=1)
    # plt.title('numHorBlack')
    # for i in range(len(numPeaks[0])):
    #     plt.plot(numPeaks[0][i], yvals[numPeaks[0][i]], '*', markersize=10)
    # for i in range(len(numValleys[0])):
    #     plt.plot(numValleys[0][i], yvals[numValleys[0][i]], '*', markersize=10)
    # plt.show()

    minIndex60 = np.argmin(yvals[numValleys[0][numValleys[0] < rows * 0.6]])  # rows*60%中的最低谷，根据投影图得到的针对该实验的个人结论
    verUp = numValleys[0][minIndex60 - 1] if minIndex60 - 1 >= 0 else 0  # 粗略定位得到的上边界，最低谷的左边波谷
    verDown = numValleys[0][minIndex60 + 1] if minIndex60 + 1 < rows else rows - 1  # 粗略定位得到的下边界，最低谷的右边波谷

    # 对垂直灰度积分投影进行处理
    x = np.arange(cols)
    y = numVerBlack
    z1 = np.polyfit(x, y, 15)  # 最小二乘法15次多项式拟合（调参）
    p1 = np.poly1d(z1)
    yvals = p1(x)
    numValleys = scipy.signal.find_peaks(-yvals, distance=10)
    # 绘制曲线，下面注释掉的是可以看到的绘图。
    # plot1 = plt.plot(x, y, 'o', label='original')
    # plot2 = plt.plot(x, yvals, 'r', label='curve fit')
    # plt.xlabel('xaxis')
    # plt.ylabel('yaxis')
    # plt.legend(loc=1)
    # plt.title('numVerBlack')
    # for i in range(len(numValleys[0])):
    #     plt.plot(numValleys[0][i], yvals[numValleys[0][i]], '*', markersize=10)
    # plt.show()

    min1 = numValleys[0][np.argmin(yvals[numValleys[0]])]  # 在波谷中找最小值
    yvals[min1] = 10000
    min2 = numValleys[0][np.argmin(yvals[numValleys[0]])]  # 在波谷中找第二小的值
    verLeft = min(min1, min2)  # 粗略定位得到的左边界
    verRight = max(min1, min2)  # 粗略定位得到的右边界

    regionThreshold = regionThreshold[verUp:verDown, verLeft:verRight]  # 粗略定位区域

    # 眼睛精确定位
    # 圆圈
    circles = cv.HoughCircles(regionThreshold, cv.HOUGH_GRADIENT, 1.5, 32, param1=200, param2=14, minRadius=1,
                              maxRadius=8)
    """
    参数设计记录：调参）
    1. param2 = 13，效果是Orical1.jpg中从左到右第三张脸的头发上有圈
    regionThreshold, cv.HOUGH_GRADIENT, 1.5, 32, param1=180, param2=13, minRadius=1, maxRadius=8 

    Waiting for you ...

    """
    if circles is not None:
        circles = circles[0, :, :]  # 提取为二维
        circles = np.uint16(np.around(circles))  # 四舍五入，取整
        for i in circles[:]:
            cv.circle(imgDst[:, :, :], (i[0] + minCol + verLeft, i[1] + minRow + verUp), i[2], (255, 0, 0), 3)  # 画圆
            cv.circle(imgDst[:, :, :], (i[0] + minCol + verLeft, i[1] + minRow + verUp), 2, (255, 0, 0), 1)  # 画圆心
        if len(circles[:]) >= 2:  # 2个以上圆即认定为人脸
            return True
    return False


def sobelEdge(image):
    """
    Sobel算子边缘提取
    :param image: 输入图像
    :return: 处理后的图像
    """
    blur = cv.GaussianBlur(image, (3, 3), 0)  # 高斯去噪
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)  # 转灰度图像，尝试过采用直方图均衡化增加对比度但效果不好
    gradx = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    x = cv.convertScaleAbs(gradx)  # 垂直边缘
    # cv.imshow('x', x)
    grady = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    y = cv.convertScaleAbs(grady)  # 水平边缘
    # cv.imshow('y', y)
    result = cv.addWeighted(x, 0.5, y, 0.5, 0)  # 整幅图的 result
    # cv.imshow('sobelResult', result)
    return y  # 发现y的效果会更好，获得水平边缘。因为正脸中眼睛是水平的，因此效果会更好。返回 result 也可以自己尝试一下。


strName = 'Orical2'  # 图像名字
strType = '.jpg'  # 图像类型
imgSrc = cv.imread('./picture/' + strName + strType)  # 读取原始图像
imgSrcPreTreat = preTreat(imgSrc)  # 对图像进行预处理
imgDst = faceDetection(imgSrc, imgSrcPreTreat)  # 肤色提取
cv.imshow('imgSrc', imgSrc)
# cv.imshow('imgSrcTreat', imgSrcPreTreat)
cv.imshow('imgDst', imgDst)
# cv.imwrite(strName+'Final'+strType, imgDst)
cv.waitKey(0)
cv.destroyAllWindows()
