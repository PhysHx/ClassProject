import numpy as np
from math import *
import matplotlib.pyplot as plt

def func(x):
    return 1/(1+4*x*x)

def newton_interpolation(node_x, node_y):
    #判断输入合法性
    if len(node_x) != len(node_y):
        print('newton_interpolation: SIZE ERROR!')
        return
    
    times=len(node_x)-1
    #计算差商表
    coeff = np.zeros((times+1,times+1))
    for j in range(times+1):
        coeff[j][0]=node_y[j]
    for i in range(1,times+1):
        for j in range(1,i+1):
            coeff[i][j]=(coeff[i][j-1]-coeff[i-1][j-1])/(node_x[i]-node_x[i-j])
    print(coeff)
    return np.diagonal(coeff)   #返回系数
    

#定义用于整理字符串公式中符号的函数
def sym(x):
    if x >= 0:
        return '+'
    return ''
def symr(x):
    if x >= 0:
        return '-'
    return '+'


#以字符串形式输出并返回插值公式
def newton_poly(coeff,node_x):
    poly = str(coeff[0])
    for i in range(1,len(coeff)):
        poly = poly + sym(coeff[i]) + str(coeff[i])
        for j in range(i):
            poly = poly + '*(x' + symr(node_x[j]) + str(node_x[j]).replace('-','') +')'
    print('p(x) = ' + poly) 
    return(poly)



nodex=[1,1.1,1.2,1.3,1.4,1.5]    #创建节点#指定节点个数
nodey=[1,1.23368,1.55271,1.99372,2.61170,3.49034]            

coeff=newton_interpolation(nodex,nodey)      #计算系数
p = newton_poly(coeff,nodex)                    #输出并储存字符串


result = np.zeros((4,99))
result[0] = np.linspace(-5,5,99)    #均匀插入节点
result[1] = func(result[0])         #计算准确值
x = result[0]
result[2] = eval(p)                 #将字符串转为可执行代码并计算
result[3] = result[2]-result[1]     #计算误差

x=1.25
print(eval(p))