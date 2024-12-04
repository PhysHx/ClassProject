import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre

#对积分方程进行Simpson数值求解的类
class IntegralSimpsonSolver:
    def __init__(self, kernel_function, rhs_function, a, b, m):
        self.kernel_function = kernel_function # 核函数
        self.rhs_function = rhs_function # 右侧非齐次项
        self.a = a
        self.b = b
        self.m = m

    def Simpson(self, f, a, b, m):
        h = (b - a) / (2*m)
        t = np.linspace(a, b, 2*m+1)
        y = f(t)
        S = h / 3 * (y[0] + y[-1] + 4 * sum(y[1:2*m:2]) + 2 * sum(y[2:2*m-1:2]))
        return S
    
    #y(x)=∫[a,b]K(x,t)y(t)dt+f(x)
    def solve1(self, x_values, tol=1e-6, max_iterations=1000):
        print("Simpson method 1")
        y_values = np.zeros_like(x_values)
        y_old = np.ones_like(x_values)
        iteration = 0
        while np.max(np.abs(y_values - y_old)) > tol and iteration < max_iterations:
            y_old = y_values.copy()
            for i, x in enumerate(x_values):
                integrand = lambda t: self.kernel_function(t, x) * np.interp(t, x_values, y_old)
                integral = self.Simpson(integrand, self.a, self.b, self.m)
                y_values[i] = integral + self.rhs_function(x)
            iteration += 1
        print(f"Iteration {iteration}, max error: {np.max(np.abs(y_values - y_old))}")
        if iteration == max_iterations:
            print("Warning: Maximum number of iterations reached. The solution may not have converged.")
        return y_values

    #0=f(x)+∫[a,b]K(x,t)y(t)dt的求解
    def solve2(self, x_values, tol=1e-6, max_iterations=1000):
        print("Simpson method 2")
        g_values = np.zeros_like(x_values)
        g_old = np.ones_like(x_values)
        iteration = 0
        while np.max(np.abs(g_values - g_old)) > tol and iteration < max_iterations:
            g_old = g_values.copy()
            for i, x in enumerate(x_values):
                integrand = lambda t: self.kernel_function(t, x) * np.interp(t, x_values, g_old)
                integral = self.Simpson(integrand, self.a, self.b, self.m)
                g_values[i] = self.rhs_function(x) + 0.001*integral
            iteration += 1
        print(f"Iteration {iteration}, max error: {np.max(np.abs(g_values - g_old))}")
        if iteration == max_iterations:
            print("Warning: Maximum number of iterations reached. The solution may not have converged.")
        return g_values



#对积分方程进行高斯求积公式离散积分项求解的类
class IntegralGaussSolver:
    def __init__(self, kernel_function, rhs_function, a, b, n):
        self.kernel_function = kernel_function
        self.rhs_function = rhs_function
        self.a = a
        self.b = b
        self.n = n
        self.nodes, self.weights = roots_legendre(n)  # 高斯求积节点和权重

    def gauss_quadrature(self, f, a, b):
        # 将节点和权重从 [-1, 1] 变换到 [a, b]
        t = 0.5 * (self.nodes + 1) * (b - a) + a
        w = 0.5 * (b - a) * self.weights
        return np.sum(w * f(t))

    def solve1(self, x_values, tol=1e-6, max_iterations=1000):
        print("Gauss method 1")
        y_values = np.zeros_like(x_values)
        y_old = np.ones_like(x_values)
        iteration = 0
        while np.max(np.abs(y_values - y_old)) > tol and iteration < max_iterations:
            y_old = y_values.copy()
            for i, x in enumerate(x_values):
                integrand = lambda t: self.kernel_function(t, x) * np.interp(t, x_values, y_old)
                integral = self.gauss_quadrature(integrand, self.a, self.b)
                y_values[i] = integral + self.rhs_function(x)
            iteration += 1
        print(f"Iteration {iteration}, max error: {np.max(np.abs(y_values - y_old))}")
        if iteration == max_iterations:
            print("Warning: Maximum number of iterations reached. The solution may not have converged.")
        return y_values

    def solve2(self, x_values, tol=1e-6, max_iterations=1000):
        print("Gauss method 2")
        y_values = np.zeros_like(x_values)
        y_old = np.ones_like(x_values)
        iteration = 0
        while np.max(np.abs(y_values - y_old)) > tol and iteration < max_iterations:
            y_old = y_values.copy()
            for i, x in enumerate(x_values):
                integrand = lambda t: self.kernel_function(t, x) * np.interp(t, x_values, y_old)
                integral = self.gauss_quadrature(integrand, self.a, self.b)
                y_values[i] = 0.001 * integral + self.rhs_function(x)
            iteration += 1
        print(f"Iteration {iteration}, max error: {np.max(np.abs(y_values - y_old))}")
        if iteration == max_iterations:
            print("Warning: Maximum number of iterations reached. The solution may not have converged.")
        return y_values





# 设置参数
a = 0
b = 1
m = 100
x_values_1 = np.linspace(a, b, m)

c=0
d=10
x_values_2 = np.linspace(c, d, m)

# 创建Simpson求解器实例并求解积分方程1
solver_Simpson_1 = IntegralSimpsonSolver(lambda t,x: t-x, lambda x:np.exp(2*x) + ((np.exp(2)-1)/2)*x - (np.exp(2)+1)/4 , a, b, m)
y_values_Simpson_1 = solver_Simpson_1.solve1(x_values_1)


# 创建Simpson求解器实例并求解积分方程2
solver_Simpson_2 = IntegralSimpsonSolver(lambda t,x: np.log(1e-10 + (np.cos(x)-np.cos(t))**2+(np.sin(x)-np.sin(t))**2), lambda x: 2*np.cos(x) , c, d, m)
y_values_Simpson_2 = solver_Simpson_2.solve2(x_values_2)

# 创建Gauss求解器实例并求解积分方程1
solver_Gauss_1 = IntegralGaussSolver(lambda t,x: t-x, lambda x:np.exp(2*x) + ((np.exp(2)-1)/2)*x - (np.exp(2)+1)/4 , a, b, m)
y_values_Gauss_1 = solver_Gauss_1.solve1(x_values_1)

# 创建Gauss求解器实例并求解积分方程2
solver_Gauss_2 = IntegralGaussSolver(lambda t,x: np.log(1e-10 + (np.cos(x)-np.cos(t))**2+(np.sin(x)-np.sin(t))**2), lambda x: 2*np.cos(x) , c, d, m)
y_values_Gauss_2 = solver_Gauss_2.solve2(x_values_2)




# 绘制图像
plt.figure()
plt.subplot(4, 2, 1)
plt.plot(x_values_1, y_values_Simpson_1, label="Simpson")
plt.plot(x_values_1, np.exp(2*x_values_1), label="Exact", linestyle=":")
plt.legend()
plt.subplot(4, 2, 2)
plt.plot(x_values_1, y_values_Simpson_1-np.exp(2*x_values_1), label="Simpson Error")
plt.fill_between(x_values_1, 0, y_values_Simpson_1-np.exp(2*x_values_1),  color='red', alpha=0.2)
plt.legend()
plt.subplot(4, 2, 3)
plt.plot(x_values_2, y_values_Simpson_2, label="Simpson")
plt.plot(x_values_2, 2*np.cos(x_values_2), label="Exact", linestyle=":")
plt.legend()
plt.subplot(4, 2, 4)
plt.plot(x_values_2, y_values_Simpson_2-2*np.cos(x_values_2), label="Simpson Error")
plt.fill_between(x_values_2, 0, y_values_Simpson_2-2*np.cos(x_values_2),  color='red', alpha=0.2)
plt.subplot(4, 2, 5)
plt.plot(x_values_1, y_values_Gauss_1, label="Gauss")
plt.plot(x_values_1, np.exp(2*x_values_1), label="Exact", linestyle=":")
plt.legend()
plt.subplot(4, 2, 6)
plt.plot(x_values_1, y_values_Gauss_1-np.exp(2*x_values_1), label="Gauss Error")
plt.fill_between(x_values_1, 0, y_values_Gauss_1-np.exp(2*x_values_1),  color='red', alpha=0.2)
plt.legend()
plt.subplot(4, 2, 7)
plt.plot(x_values_2, y_values_Gauss_2, label="Gauss")
plt.plot(x_values_2, 2*np.cos(x_values_2), label="Exact", linestyle=":")
plt.legend()
plt.subplot(4, 2, 8)
plt.plot(x_values_2, y_values_Gauss_2-2*np.cos(x_values_2), label="Gauss Error")
plt.fill_between(x_values_2, 0, y_values_Gauss_2-2*np.cos(x_values_2),  color='red', alpha=0.2)
plt.tight_layout()
plt.legend()
plt.show()