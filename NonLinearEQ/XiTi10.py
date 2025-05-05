from math import *
import numpy as np

def dF(x):
    h=0.001
    return (F(x+h)-F(x-h))/(2*h)  # 中心差分法计算导数



def newton(F, x0, tol=1e-15, max_iter=100):
    """
    牛顿迭代法求解非线性方程 F(x) = 0

    参数:
    F: 函数，目标方程 F(x)
    x0: 初始猜测值
    tol: 收敛判定的容差
    max_iter: 最大迭代次数

    返回:
    根的近似值
    """
    x = x0
    for i in range(max_iter):
        fx = F(x)
        fpx = dF(x)
        if abs(fpx) < 1e-12:  # 避免除以零
            raise ValueError("导数接近零，无法继续迭代")
        x_new = x - fx / fpx
        if abs(x_new - x) < tol:  # 判断是否收敛
            print(f"牛顿法迭代次数: {i+1}")
            return x_new
        x = x_new
    raise ValueError("迭代未收敛，达到最大迭代次数")



def basic(F, x0, tol=1e-15, max_iter=100):
    """
    基本迭代法求解非线性方程 F(x) = 0
    
    注意使用基本迭代法时，需要函数 F(x) 变形为 x + F(x), 即F(x)具有x的量纲

    参数:
    F: 函数，目标方程 F(x)
    x0: 初始猜测值
    tol: 收敛判定的容差
    max_iter: 最大迭代次数

    返回:
    根的近似值
    """
    x = x0
    for i in range(max_iter):
        x_new = F(x)+x  # 迭代公式
        if abs(x_new - x) < tol:  # 判断是否收敛
            print(f"基本法迭代次数: {i+1}")
            return x_new
        x = x_new
    raise ValueError("迭代未收敛，达到最大迭代次数")



def Js(F_s,x):
    if len(F_s(x)) != len(x):
        raise ValueError("F_s(x) 的元素个数与 x 的维度不匹配，无法计算雅可比矩阵")
    h = 0.0001  # 用于数值计算雅可比矩阵的微小扰动
    n = len(x)
    J = np.zeros((n, n))
    for i in range(n):
        x_perturb = x.copy()
        x_perturb[i] += h
        J[:, i] = (F_s(x_perturb) - F_s(x)) / h  # 数值计算偏导数
    return J



def newton_s(F, x0, tol=1e-15, max_iter=100):
    """
    牛顿迭代法求解非线性方程组 F(x) = 0

    参数:
    F: 函数，目标方程组 F(x)，返回值为向量
    x0: 初始猜测值，向量
    tol: 收敛判定的容差
     max_iter: 最大迭代次数

    返回:
    根的近似值，向量
    """
    x = x0
    for i in range(max_iter):
        Fx = F(x)
        Jx = Js(F, x)  # 计算雅可比矩阵
        try:
            delta_x = -np.linalg.solve(Jx, Fx)  # 解线性方程组 J(x) * delta_x = -F(x)
        except np.linalg.LinAlgError:
            raise ValueError("雅可比矩阵不可逆，无法继续迭代")
        x_new = x + delta_x
        if np.linalg.norm(delta_x, ord=2) < tol:  # 判断是否收敛
            print(f"牛顿法迭代次数: {i+1}")
            return x_new
        x = x_new
    raise ValueError("迭代未收敛，达到最大迭代次数")



def basic_s(F, x0, tol=1e-15, max_iter=100):
    """
    基本迭代法求解非线性方程组 F(x) = 0

    注意使用基本迭代法时，需要函数 F(x) 变形为 x + F(x), 即F(x)具有x的量纲

    参数:
    F: 函数，目标方程组 F(x)，返回值为向量
    x0: 初始猜测值，向量
    tol: 收敛判定的容差
     max_iter: 最大迭代次数

    返回:
    根的近似值，向量
    """
    x = x0
    for i in range(max_iter):
        x_new = F(x)+x  # 迭代公式
        if np.linalg.norm(x_new - x, ord=2) < tol:  # 判断是否收敛
            print(f"基本法迭代次数: {i+1}")
            return x_new
        x = x_new
        if  max(x) > 1e100:
            raise ValueError("数值溢出，基本迭代未收敛") 
    raise ValueError("迭代未收敛，达到最大迭代次数")
    



#==========================使用============================
#求解范德瓦尔斯方程-----------------------------------------
def F(V):
    R = 8.314
    a = 3.59
    b = 0.0427
    P = 500000
    T = 300
    #return (P+a/(V**2))*(V-b)-R*T  # 范德瓦尔斯方程
    return R*T/P - a/(P*V**2)*(V-b)+b-V  # 范德瓦尔斯方程的变形

try:
    root = newton(F, 1.0)
    print(f"范德瓦尔斯方程牛顿迭代法解得V = {root} m^3/mol")
except ValueError as e:
    print(e)
    
try:
    root = basic(F, 1.0)
    print(f"范德瓦尔斯方程基本迭代法解得V = {root} m^3/mol")
except ValueError as e:
    print(e)
    
    
#力学方程组------------------------------------------
def F1(x1,x2): #弹簧拉伸为正
    return 100*x1 + 150*x1**3 
def F2(x1,x2):
    return 120*(x2-x1)+200*(x2-x1)**3
def F3(x1,x2):
    return 140*x2 + 250*x2**3
g = 9.8
m1 = 10
m2 = 8

def F(x):
    return np.array([F1(x[0],x[1]) - m1*g - F2(x[0],x[1]), F2(x[0],x[1]) - F3(x[0],x[1]) - m2*g])  # 力学方程组
try:
    root = newton_s(F, [1.0,1.0])
    print(f"非线性力学方程组牛顿迭代法解得[x1 x2] = {root} m")
except ValueError as e:
    print(e)
    
try:
    root = basic_s(F, [1.0,1.0])
    print(f"非线性力学方程组基本迭代法解得[x1 x2] = {root} m")
except ValueError as e:
    print(e)