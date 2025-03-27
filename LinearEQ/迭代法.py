import numpy as np


# Jacobi迭代法
def jacobi(A,b,N=150,Norm=1.0e-8): 
    rows, cols = A.shape
    x_old = np.zeros(cols)
    x = np.zeros(cols)

    for k in range(N):
        for i in range(rows):
            x[i] = (b[i] - np.sum([A[i,j]*x_old[j] for j in range(cols) if j != i]))/A[i,i]
        
        if np.linalg.norm(x-x_old) < Norm:
            print('结果收敛，迭代次数：',k+1)
            return x
        if k == N-1:
            print('迭代次数已达到最大值，未收敛')
        x_old = x.copy()
        
        
# Gauss-Seidel迭代法
def GS(A,b,N=150,Norm=1.0e-10):        
    rows, cols = A.shape
    x_old = np.zeros(cols)
    x = np.zeros(cols)

    for k in range(N):
        for i in range(rows):
            x[i] = (b[i] - np.sum([A[i,j]*x[j] for j in range(0,i)]) - np.sum([A[i,j]*x_old[j] for j in range(i+1,cols)]))/A[i,i]

        if np.linalg.norm(x-x_old) < Norm:
            print('结果收敛，迭代次数：',k+1)
            return x
        if k == N-1:
            print('迭代次数已达到最大值，未收敛')
        x_old = x.copy()



A = np.array([[10, -1,-2], [-1, 10,-2], [-1, -1, 5]])   #系数矩阵
b = np.array([72,83,42])                                #非齐次项
#N = 100         #最大迭代次数,默认为150
#Norm = 1e-10    #迭代终止条件,默认为1.0e-8

print('Jacobi迭代法：',jacobi(A,b))
print('Gauss-Seidel迭代法：',GS(A,b))