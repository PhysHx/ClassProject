import numpy as np
import matplotlib.pyplot as plt
from math import *

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体，若系统没有可尝试其他字体如 'SimSun'（宋体）
# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

g = 9.8
m1, m2, m3 = 1.0, 2.0, 3.5
mu1, mu2 = 0.1, 0.8
alpha = 30.0/180.0*pi
N=364


data = np.zeros((N, 4))
F1 = (m1+m2)*g*cos(alpha)
F2 = m2*g*cos(alpha)

i=0
for t in np.linspace(0, 90, N):
    alpha = t/180.0*pi
    F1 = (m1+m2)*g*cos(alpha)
    F2 = m2*g*cos(alpha)
    a1=0
    a2=0
    a3=0
    F=m3*g
    f2=m2*g*sin(alpha)
    f1=F-f2-m1*g*sin(alpha)
    if f1>= mu1*F1 and f2<=mu2*F2:
        f1 = mu1*F1
        a1 = (m3*g-m1*g*sin(alpha)-m2*g*sin(alpha)-f1)/(m1+m3)
        a2 = a1
        a3 = a1
    elif f1>= mu1*F1 and f2>=mu2*F2:
        f1 = mu1*F1
        f2 = mu2*F2
        a2 = (f2-m2*g*sin(alpha))/m2
        a1 = (m3*g-f2-f1-m1*g*sin(alpha))/(m1+m3)
        a3 = a1
    data[i, 0] = alpha*180.0/pi
    data[i, 1] = a1
    data[i, 2] = a2
    data[i, 3] = a3
    i += 1
    if alpha == 30.0/180.0*pi:
        print('a1=', a1)
        print('a2=', a2)
        print('a3=', a3)

plt.plot(data[:, 0], data[:, 1], label='a1')
plt.plot(data[:, 0], data[:, 2], label='a2')
plt.plot(data[:, 0], data[:, 3], label='a3')
plt.xlabel('alpha/°')
plt.ylabel('a/(m/s^2)')
plt.legend()
plt.show()