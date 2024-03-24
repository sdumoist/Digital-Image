import cv2 as cv
from PIL import Image, ImageSequence
import numpy as np

"""
总体思想是读取图像之后，把四张图像合在一起显示
"""

img1 = cv.imread('./picture/Img1.png')
img2 = cv.imread('./picture/Img2.jpg')
img3 = cv.imread('./picture/Img3.bmp')

imageSrc = [img1, img2, img3]  # 原图像
imageDst = []  # 保存resize后的图像
for i in range(3):
    imageDst.append(np.zeros([]))
size = 300  # resize后每个图像的height == size, width == size

for i in range(3):
    imageDst[i] = cv.resize(imageSrc[i], (size, size))  # 使所有图像大小相同，显示美观

board = np.zeros([2*size, 2*size, 3], dtype='uint8')  # 用于存放合并图像的大图像

# 先处理简单的三类图像，直接进行替换。我们将board分为4个等大的区域，可以理解为4个ROI
board[0:size, 0:size] = imageDst[0]
board[0:size, size:] = imageDst[1]
board[size:, 0:size] = imageDst[2]

# 处理gif图像。OpenCV由于版权原因无法处理gif图像
picName = './picture/Img4.gif'
im = Image.open(picName)  # 打开图像
flag = False
while True:
    # 循环显示每一帧
    for frame in ImageSequence.Iterator(im):
        frame = frame.convert('RGB')  # frame是PIL.Image对象，转成RGB格式
        cvFrame = np.array(frame)
        img4 = cv.cvtColor(cvFrame, cv.COLOR_RGB2BGR)  # OpenCV默认的彩色图像的颜色空间是BGR
        img4 = cv.resize(img4, (size, size))
        board[size:, size:] = img4  # 替换
        cv.imshow('exp1_1', board)
        time = 10  # 单位ms，通过这个可以改变gif显示的速度
        if cv.waitKey(time) & 0xFF == 27:  # 按Esc结束
            flag = True
    if flag:
        break

cv.destroyAllWindows()
