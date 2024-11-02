import numpy as np
import matplotlib.pyplot as plt

#读取数据
f=open('data')
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

plt_x = np.linspace(0,1,100)
plt_y = np.full((dir_num,100),np.NaN)


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
                         


plt.figure()
Color=['b','r','g','m','y','c']
for i in range(dir_num):
    plt.plot(plt_x, plt_y[i],label=direction[i],color=Color[i])
    plt.plot(data[i,:,0],data[i,:,1], label='Origin',linestyle=':',color=Color[i])
plt.legend()
plt.show()
