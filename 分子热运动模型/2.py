import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import *
from tqdm import tqdm
import pandas as pd

#定义粒子类
class Particle(object):
    def __init__(self,r=np.array([0.0,0.0]), v=np.array([0.0,0.0]),R=0):
        self.r=r#粒子位置
        self.v=v#粒子速度
        self.R=R#粒子半径
    
    def get_v(self):#获取粒子速度大小
        return np.linalg.norm(self.v)
    
    def distance(self,p):#获取该粒子与p粒子的距离
        return np.linalg.norm(self.r-p.r)
    
    def absr(self):#获取与原点距离
        return np.linalg.norm(self.r)

    def angle(self,p):#获取该粒子相对p的单位向量
        return (self.r-p.r)/np.linalg.norm(self.r-p.r)

    def move(self,dt):
        self.r=self.r+self.v*dt 
        if (self.r[0]+self.R>=L and self.v[0]>0)or (self.r[0]-self.R<=0 and self.v[0]<0):
            self.v[0]=-self.v[0]
        elif (self.r[1]+self.R>=L and self.v[1]>0) or (self.r[1]-self.R<=0 and self.v[1]<0):
            self.v[1]=-self.v[1]       

class para(object):#定义参数
    def __init__(self,L=1000,var=1,mean=0,n=1000,dt=10,nsteps=1000,R=5,col=False,average_v=False):
        self.L=L
        self.var=var#方差
        self.mean=mean#均值
        self.n=n#粒子数
        self.dt=dt#时间间隔，单位ms
        self.nsteps=nsteps#运行步数
        self.R=R
        #设定tasks
        self.col=col#碰撞
        self.average_v=average_v#平均速率
    def list(self):#获取参数列表
        return [self.L,self.var,self.mean,self.n,self.dt,self.nsteps,self.R,self.col,self.average_v]
    
def collision(p1,p2):
    vs1=np.dot(p1.v,p1.angle(p2))*p1.angle(p2)
    vs2=np.dot(p2.v,p1.angle(p2))*p1.angle(p2)
    if np.dot(p1.angle(p2),vs1-vs2)>=0:
        return p1,p2
    vt1=p1.v-vs1
    vt2=p2.v-vs2
    vs1,vs2=vs2,vs1
    p1.v=vt1+vs1
    p2.v=vt2+vs2
    return p1,p2
    


#初始化和计算粒子
def begin(demo):
    #建立粒子对象
    global particles
    particles =[] 
    if demo=='2':
        for i in range(n):
            r0=L*np.random.rand(2)
            #v0=np.sqrt(0.1*(r0[0]/L)**1.5) * np.random.randn(2)
            if r0[0]>=0.55*L:
                v0=np.sqrt(0.4) * np.random.randn(2)
            else:
                v0=np.sqrt(0.02) * np.random.randn(2)
            particles.append(Particle(r=r0,v=v0,R=R))
    elif demo=='3':
        for i in range(n):
            r0=L*np.random.rand(2)
            if r0[1]>=0.4*L and r0[1]<=0.6*L and r0[0]<=0.5*L:
                v0=np.sqrt(0.02) * np.random.randn(2)
                v0[0]=v0[0]+1
            else:
                v0=np.sqrt(0.02) * np.random.randn(2)
            particles.append(Particle(r=r0,v=v0,R=R))

    else: 
        for i in range(n):
            particles.append(Particle(r=L*np.random.rand(2), v=np.sqrt(var) * np.random.randn(2) + mean,R=R))
   
   #逐帧运算
    PX=[]
    PY=[]
    for i in tqdm(range(nsteps)):
        tasks(i,col=col)
        XJ=[]
        YJ=[]
        for j in range(n):
            particles[j].move(dt)
            XJ.append(particles[j].r[0])
            YJ.append(particles[j].r[1])
        PX.append(XJ)
        PY.append(YJ)    
    return PX,PY



def tasks(frame,col=True):#定义操作
    if col==True:#碰撞
        for i in range(n):
            for j in range(i):
                if particles[i].distance(particles[j])<=particles[i].R+particles[j].R:
                    particles[i],particles[j]=collision(particles[i],particles[j])
    return


def outinfo(frame,average_v=False):#定义输出
    if average_v==True:#平均速率
        sum_v=0
        for i in range(n):
            sum_v=sum_v+particles[i].get_v()
        return sum_v/n
    return 0
        

# 定义更新函数
def update(frame):
    sc.set_offsets(np.c_[PX[frame-1], PY[frame-1]])
    return sc,

def plot():
    fig, ax = plt.subplots()
    ax.set_xlim((0, L))
    ax.set_ylim((0, L))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    global sc
    sc=ax.scatter(0,0,s=((1000*R/L)/fig.dpi*72)**2)
    # 定义动画对象
    ani = animation.FuncAnimation(fig, update, interval=dt,frames=range(1, nsteps), blit=False)
    plt.show()

def save(X,Y):
    way=input('请输入路径：')
    data=pd.DataFrame(data=[X,Y])
    data.to_csv(way)
    print("储存完成！")
            
def read():
    way=input('请输入路径：')
    X=[]
    Y=[]
    data=pd.read_csv(way)
    XY=data.values.tolist()
    for i in tqdm(range(1,len(XY[0]))):
        X.append(eval(XY[0][i]))
        Y.append(eval(XY[1][i]))
    print('读取完成！')
    return X,Y
    

while(1):
    print("====================分子热运动模型======================")
    print("本模型以点模拟二维平面容器中分子运动")
    print("通过均匀分布随机数设定分子位置，通过正态分布设定分子初始速度")
    print("可以设定是否启用分子间碰撞、可以设定分子半径、容器线度等参数")
    print("本模型基于Python")
    print("!受计算机性能影响，请勿在启用碰撞时设定较高的分子数!")
    print("===================Designed by Hou Xu===================\n\n")

    #默认粒子参数
    L=1000
    var=1#方差
    mean=0#均值
    n=1000#粒子数
    dt=10#时间间隔，单位ms
    nsteps=1000#运行步数
    R=5
    #设定tasks
    col=False#碰撞
    average_v=False#平均速率

    print("输入1使用自定义参数，输入2模拟输运现象，0读取计算数据，否则使用默认参数：",end='')
    demo=input()
    if demo=='1':
        in_L=eval(input('输入容器边长：'))
        in_n=eval(input('输入粒子数：'))
        in_R=eval(input('输入粒子半径(输入0忽略半径与碰撞)：'))
        in_dt=eval(input('输入帧间隔(ms)：'))
        in_var=eval(input('输入速度方差：'))
        in_mean=eval(input('输入速度均值：'))
        L,n,dt,var,mean=in_L,in_n,in_dt,in_var,in_mean
        if in_R!=0:
            R=in_R
            in_col=input('是否开启碰撞？（输入1开启，否则关闭）：')
            if in_col=='1':
                col=True
        if input('是否输出平均速率？（输入1开启，否则关闭）：')=='1':
            average_v=True
    elif demo=='2':
        L=1000
        n=5000
        dt=10
        col=True
        average_v=False
        R=5
        var='存在梯度'
        print("***模拟一种热传导，两侧气体温度不均匀（运动平均速率不同）***")
    elif demo=='3':
        L=1000
        n=1000
        dt=10
        col=True
        average_v=False
        R=6
        var='吹气'
        print("***模拟吹气***")
    elif demo=='0':
        print("读取文件")
    else:
        print("*使用默认参数*")

    if demo!='0':
        #显示参数
        print("【参数】")
        print("容器线度：{}\n粒子数：{}\n帧间隔：{}ms\n粒子半径：{}\n速度方差：{}\n速度均值：{}".format(L,n,dt,R,var,mean))
        print("【功能】")
        print("碰撞：{}\n平均速度输出：{}".format(col,average_v))
        PX,PY=begin(demo)
        if input('是否储存数据？（输入1储存，否则不储存）：')=='1':
            save(PX,PY)
    else:
        PX,PY=read()

    plot()
    del demo