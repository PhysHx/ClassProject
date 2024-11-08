import numpy as np
import matplotlib.pyplot as plt

#读取数据
f=open('Interpolation\\data')
lines = f.readlines()
f.close()
direction_line = [] #记录标签行数
Line_split = []     #记录数据_分割后的清洗过程
direction = []      #记录标签

for i in range(len(lines)):
    Line_split.append(lines[i].split())
    if Line_split[i][0].replace(':' and ':','') == 'direction':
        direction_line.append(i)
        direction.append(Line_split[i][1])


dir_num = len(direction_line)   #标签数量
data_num = []                   #每组数据数量
for i in range(len(direction_line)-1):
    data_num.append(direction_line[i+1] - direction_line[i])
num = max(data_num)             #最大数据数量
#创建数据数组并完成数据整理
data=np.full((dir_num,num-1,2),np.NaN)
print(lines)
for i in range(dir_num):
    for n in range(num-1):
        data[i][n][0] = eval(Line_split[direction_line[i]+n+1][0])
        data[i][n][1] = eval(Line_split[direction_line[i]+n+1][1])
#删除多余变量

#数据整理完成
print(data)
print(num,direction)

def Lagrange(x,nodex,nodey):
    Sum = 0
    for k in range(len(nodex)):
        Lk = 1
        for i in range(len(nodex)):
            if i == k:
                continue
            Lk = Lk*(x-nodex[i])/(nodex[k]-nodex[i])
        Sum = Sum + Lk*nodey[k]
    return Sum

      
#高斯消元法解线性方程组
def gauss_jordan(A, b):
    n = len(b)
    M = [A[i] + [b[i]] for i in range(n)]
    
    for i in range(n):
        # Make the diagonal contain all 1's
        div = M[i][i]
        for j in range(n + 1):
            M[i][j] /= div
        
        # Make the other rows contain 0's
        for k in range(n):
            if k != i:
                mult = M[k][i]
                for j in range(n + 1):
                    M[k][j] -= mult * M[i][j]
    
    return [M[i][-1] for i in range(n)]                   


#三次样条插值
def Cublic_Spline(nodex,nodey):
    n=len(nodex)
    if n != len(nodey):
        print('Error: nodex and nodey must be the same length!')
        return None
    
    h = np.zeros(n-1)
    for i in range(n-1):
        h[i] = nodex[i+1] - nodex[i]
        
    lam = np.zeros(n-1)
    miu = np.zeros(n-1)
    d = np.zeros(n-2)
    for i in range(n-2):
        lam[i] = h[i]/(h[i]+h[i+1])
        miu[i] = 1-lam[i]
        d[i] = 6/(h[i]+h[i+1])*((nodey[i+2]-nodey[i+1])/h[i+1]-(nodey[i+1]-nodey[i])/h[i])
        
    A=np.zeros((n-1,n-1))
    np.fill_diagonal(A,2)
    for i in range(n-2):
        A[i][i+1] = lam[i]
        A[i+1][i] = miu[i]
    print(A)
    print(d)
    M=[0]
    M=M + gauss_jordan(A,d)
    M.append(0)
    print(M)
    return M

def Spline(x,nodex,nodey,M):
    n = len(nodex)
    for i in range(n-1):
        if x >= nodex[i] and x < nodex[i+1]:
            h = nodex[i+1] - nodex[i]
            a = (nodex[i+1]-x)/h
            b = (x-nodex[i])/h
            return a*nodey[i]+b*nodey[i+1]+h**2/6*((a**3-a)*M[i]+(b**3-b)*M[i+1])
    return None
        
plt_n=1000
plt_x = np.linspace(0,1,plt_n)
plt_y = np.full((dir_num, plt_n),np.NaN)

'''
#Lagrange插值
for n in range(dir_num):
    for XAxisNumber in range(len(plt_x)):
        plt_y[n][XAxisNumber] = Lagrange(plt_x[XAxisNumber], data[n,:,0],data[n,:,1])


#分段Lagrange二次插值
for n in range(dir_num):
    for i in range(len(plt_x)):
        for NodexNum in range(len(data[n,:,0])-2):
            if plt_x[i] >= data[n,NodexNum,0] and plt_x[i] < data[n,NodexNum+2,0]:
                plt_y[n][i] = Lagrange(plt_x[i], data[n,NodexNum:NodexNum+2,0],data[n,NodexNum:NodexNum+2,1])
                break
            else:
                continue
'''

m=[]
for i in range(dir_num):
    m = Cublic_Spline(data[i,:,0],data[i,:,1])
    for j in range(len(plt_x)):
        plt_y[i][j] = Spline(plt_x[j],data[i,:,0],data[i,:,1],m)

plt.figure()
Color=['b','r','g','m','y','c']
for i in range(dir_num):
    plt.plot(plt_x, plt_y[i],label=direction[i],color=Color[i])
    plt.plot(data[i,:,0],data[i,:,1], label='Origin',linestyle=':',color=Color[i])
    
plt.legend()
plt.show()