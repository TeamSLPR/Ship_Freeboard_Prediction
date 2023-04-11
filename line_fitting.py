import time
import cv2
import numpy as np

img = cv2.imread("result1.jpg")
img1 = cv2.imread("img1/3.jpg")
b, g, r = cv2.split(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY)
binary = np.array(binary)
h, w = binary.shape

temp = int(w/50)

start = time.time()

result = []
result1 = []
for k in range(100-1):
    x = []
    p = []
    for i in range(h - 1):
        y = int(temp/2)
        if(binary[i][y * (k+1)]) == 0:
            if(binary[i+1][(k+1) * y]) == 255:
                x.append(i)
                x.append((k+1) * y)
        if(binary[i][y * (k+1)]) == 255:
            if (binary[i + 1][(k + 1) * y]) == 0:
                p.append(i)
                p.append((k+1) * y)
    if len(x) != 0:
        result.append(x)
    if len(p) != 0:
        result1.append(p)
    # 画采样直线的语句
    cv2.line(img1, (int(k * temp), int(0)), (int(k * temp), int(h)),(0, 0, 255), 2, 16)

print(result)
print(result1)

# avg1 = 0
# i = 0
# for r in result:
#     avg1 = avg1 + r[0]
#     i = i + 1
# avg1 = avg1/i
# print(avg1)
# avg2 = 0
# j = 0
# for r in result1:
#     avg2 = avg2 + r[0]
#     j = j + 1
# avg2 = avg2/i
#
# print(avg2)

x_x = []
y_y = []
for r in result:
    x_x.append(r[1])
    y_y.append(r[0])
    point1 = (r[1], r[0])
    # cv2.circle(img1, point1, 8, (0, 255, 255), -1)
x_2 = []
y_2 = []



for r in result1:
    x_2.append(r[1])
    y_2.append(r[0])
    point1 = (r[1], r[0])
    # cv2.circle(img1, point1, 8, (0, 255, 255), -1)


# 最小二乘法拟合直线
# def Fun(x, a1,a2): # 定义拟合函数形式
#     return a1*x + a2
#
# x_x = np.array(x_x)
# y_y = np.array(y_y)
# para = np.polyfit(x_x, y_y, deg=1)
# print(para)
# y_fitted = Fun(x_x, para[0], para[1])
#
# x_2 = np.array(x_2)
# y_2 = np.array(y_2)
# para = np.polyfit(x_2, y_2, deg=1)
# y_fitted1 = Fun(x_2, para[0], para[1])
#
#
# cv2.line(img1, (int(x_x[0]), int(y_fitted[0])), (int(x_x[len(x_x)-1]), int(y_fitted[len(y_fitted)-1])),(0, 0, 255), 2, 16)
# cv2.line(img1, (int(x_2[0]), int(y_fitted1[0])), (int(x_2[len(x_x)-1]), int(y_fitted1[len(y_fitted)-1])),(0, 0, 255), 2, 16)
# end = time.time()
# print(end - start)

cv2.imwrite("3.jpg", img1)
