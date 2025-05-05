from math import *
import numpy as np

def newton(F, x0, tol=1e-5, max_iter=100):
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
            print(f"迭代次数: {i+1}")
            return x_new
        x = x_new
    raise ValueError("迭代未收敛，达到最大迭代次数")


def newton_s(F, x0, tol=1e-5, max_iter=100):
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
            print(f"迭代次数: {i+1}")
            return x_new
        x = x_new
    raise ValueError("迭代未收敛，达到最大迭代次数")


def dF(x):
    h=0.01
    return (F(x+h)-F(x-h))/(2*h)  # 中心差分法计算导数


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








# 主程序部分
if __name__ == "__main__":

    # 定义目标函数 F(x) 和其导数 F'(x)
    def F(x):
        return x**2 - x

    # 初始猜测值
    x0 = 1.5

    try:
        root = newton(F, x0)
        print(f"方程的根为: {root}")
    except ValueError as e:
        print(e)
        
        
    # 定义非线性方程组 F(x) 和雅可比矩阵 J(x)
    def Fs(x):
        return np.array([3*x[0]-cos(x[1]*x[2])-1/2, x[0]**2-81*(x[1]+0.1)**2+sin(x[2])+1.06, exp(-x[0]*x[1])+20*x[2]+(10*pi-3)/3])
    
    x0 = np.array([1.0, 1.0, 1.0])  # 初始猜测值
    try:
        root = newton_s(Fs, x0)
        print(f"方程组的根为: {root}")
    except ValueError as e:
        print(e)
    
    print('65210720 侯旭')