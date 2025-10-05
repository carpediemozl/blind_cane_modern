import numpy as np
import cv2 as cv

img1 = np.zeros((1080, 720, 3), np.uint8)  # 生成一个空灰度图像
img2 = np.zeros((1080, 720, 3), np.uint8)  # 生成一个空灰度图像

# 矩形左上角和右上角的坐标，绘制一个绿色矩形
ptLeftTop1 = (259, 236)
ptRightBottom1 = (390, 389)
ptLeftTop2 = (310, 267)
ptRightBottom2 = (339, 359)
point_color = (0, 255, 0)  # BGR
point_color2 = (255, 0, 0)  # BGR
thickness = 1
lineType = 4
cv.rectangle(img1, ptLeftTop1, ptRightBottom1, point_color, thickness, lineType)
cv.rectangle(img1, ptLeftTop2, ptRightBottom2, point_color2, thickness, lineType)
# 259 390 236 389
# 310 339 267 359
# ptLeftTop1  ptRightBottom1
cover = 29 * (359-267) +20043
print(cover)
cv.namedWindow("AlanWang")
# cv.namedWindow("AlanWang2")
cv.imshow('AlanWang', img1)
# cv.imshow('AlanWang2', img2)
cv.waitKey (1000) # 显示 10000 ms 即 10s 后消失
cv.destroyAllWindows()