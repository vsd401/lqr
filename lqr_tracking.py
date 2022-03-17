#! /usr/bin/python
"""

Path tracking simulation with LQR steering control and PID speed control.



"""
import numpy as np #numpy为开源数值计算拓展工具，存储与处理大型多维矩阵
import math #math 提供了很多对浮点数的数学运算函数
import matplotlib.pyplot as plt #matplotlib是python的绘图库，是python最常用的可视化工具之一，其可以创建2D,3D表格
import unicycle_model #导入单轨模型（单轮车模型）
from pycubicspline import pycubicspline #从pycubicspline包中导入该类
from matplotrecorder import matplotrecorder
import scipy.linalg as la #Scipy是一组专门解决科学计算中各种标准问题域的包，.linalg扩展了由numpy.linalg提供的线性代数例程和矩阵分解功能

Kp = 1.0  # speed proportional gain

# LQR parameter
Q = np.eye(4) #np.eye()为生成对角阵，k=正数表示高对角，负数表示低对角
R = np.eye(1)

animation = True #动画相关
# animation = False

#matplotrecorder.donothing = True

#定义了PIDcontrol函数，其包括目标与当前两个参数，返回给出a，为Kp*目标减去当前
def PIDControl(target, current):
    a = Kp * (target - current)

    return a

#定义了pi_2_pi函数，其包括角度参数，当角度大于180度，则角度为角度-360度，若角度小于-180度，则为+360度，保证角度在正负180度之间
def pi_2_pi(angle):
    while (angle > math.pi):
        angle = angle - 2.0 * math.pi

    while (angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle

#定义了solve_DARE函数，其参数包括A,B,Q,R,求解一个离散时间代数黎卡迪方程，A是状态量，B是控制量，用的数值计算
def solve_DARE(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 150
    eps = 0.01
    #la.inv是求矩阵的逆
    for i in range(maxiter):
        Xn = A.T * X * A - A.T * X * B * \
                           la.inv(R + B.T * X * B) * B.T * X * A + Q#离散时间黎卡提方程，X即为P
        if (abs(Xn - X)).max() < eps: #Xn与X的差的最大值小于0.01时，把Xn赋给X，即为迭代变化不大
            X = Xn
            break
        X = Xn

    return Xn

#定义dlqr函数，参数包括A,B,Q,R，解决离散时间的LQR控制器，先列出状态空间表达式以及惩罚方程，后求解黎卡迪方程
def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k] 
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_DARE(A, B, Q, R)#求得P

    # compute the LQR gain，matrix为矩阵库，计算LQR增益
    K = np.matrix(la.inv(B.T * X * B + R) * (B.T * X * A))#通过P得到K

    eigVals, eigVecs = la.eig(A - B * K) #计算矩阵特征值和特征向量，返出特征值，x导=Ac*x的Ac
    
    return K, X, eigVals

#定义lqr_steering_control函数，参数为state,cx,cy,cyaw,ck,pe,pth_e
def lqr_steering_control(state, cx, cy, cyaw, ck, pe, pth_e):
    ind, e = calc_nearest_index(state, cx, cy, cyaw)#找到此时state对应的x,y在cx,cy中的点位以及最小距离

    k = ck[ind]#找到ck中对应最小距离的点位
    v = state.v#v设定为此时的速度
    th_e = pi_2_pi(state.yaw - cyaw[ind])#th_e设定为当前横摆角与样条曲线拟合出的对应点位的横摆角差值

    A = np.matrix(np.zeros((4, 4)))#设定状态空间表达式中的A
    A[0, 0] = 1.0
    A[0, 1] = unicycle_model.dt
    A[1, 2] = v
    A[2, 2] = 1.0
    A[2, 3] = unicycle_model.dt
    # print(A)

    B = np.matrix(np.zeros((4, 1)))#设定状态空间表达式中的B
    B[3, 0] = v / unicycle_model.L

    K, _, _ = dlqr(A, B, Q, R)#求解黎卡提方程，得到全反馈控制系数K

    x = np.matrix(np.zeros((4, 1)))#建立一四行一列的空矩阵

    x[0, 0] = e#e是最小距离
    x[1, 0] = (e - pe)/unicycle_model.dt#pe是上一时间的最小距离
    x[2, 0] = th_e#th_e是横摆角的差值
    x[3, 0] = (th_e - pth_e)/unicycle_model.dt#pth_e是上一时间点横摆角速度差值

    ff = math.atan2(unicycle_model.L * k, 1)#k为车辆转动半径的曲率
    fb = pi_2_pi((-K * x)[0, 0])#-Kx为控制量

    delta = ff + fb#得到控制量结果前轮转角

    return delta, ind, e, th_e#输出前轮转角，对应样条曲线点位以及最小距离，对应横摆角差值。

#定义calc_nearest_index函数，其包括state,cx,cy,cyaw四个参量
def calc_nearest_index(state, cx, cy, cyaw):#输入当前状态state,三次样条拟合得到的cx与cy（目标），以及cyaw
    dx = [state.x - icx for icx in cx]#dx等于当前的state-cx中的每一个icx
    dy = [state.y - icy for icy in cy]#dy等于当前的state-cy中的每一个icy

    d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]#计算目前的state和所有cx,cy之间的距离

    mind = min(d)#找到其中最小的d

    ind = d.index(mind)#查找mind在d中的位置，0起，找到了距离当前状态x,y最近的cx以及cy

    dxl = cx[ind] - state.x#最近的cx与cy与当前的state.x,y做差
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))#当前的横摆角，减去dyl与dxl的反切值
    if angle < 0:#如果角度小于0
        mind *= -1#距离设为负数，将目标点与实际点之间的连线角度与横摆角进行对比，以正负值距离进行区分,区分路径方向

    return ind, mind#输出位置代表指示ind以及距离mind

#定义闭环预测
def closed_loop_prediction(cx, cy, cyaw, ck, speed_profile, goal):
    T = 500.0  # max simulation time
    goal_dis = 0.3#最终与目标间距的误差衡量值
    stop_speed = 0.05

    state = unicycle_model.State(x=-0.0, y=-0.0, yaw=0.0, v=0.0)#定义初始x,y,yaw以及速度v

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]

    t = [0.0]
    target_ind = calc_nearest_index(state, cx, cy, cyaw)#得到此时的目标点位以及距离

    e, e_th  = 0.0, 0.0

    while T >= time:#开始循环迭代
        dl, target_ind, e, e_th = lqr_steering_control(state, cx, cy, cyaw, ck, e, e_th)#dl为前轮转角，target_ind为目标点位，e为实际与目标点位之间的距离，e_th为实际与目标点位间横摆角差

        ai = PIDControl(speed_profile[target_ind], state.v)#ai为该目标点位预计的速度与实际速度的差值
        # state = unicycle_model.update(state, ai, di)
        state = unicycle_model.update(state, ai, dl)#针对差值以及控制量前轮转角输入uncycle_model(代表车辆)进行状态更新，dl为前轮转角
        time = time + unicycle_model.dt#时间也进行下一个点位的顺承

        # check goal，检验x,y是否到达设定值，设定目的地
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        if math.sqrt(dx ** 2 + dy ** 2) <= goal_dis:#到目标就停止，输出goal
            print("Goal")
            break

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        if target_ind % 1 == 0 and animation:#绘制样条图像以及实际控制图像
            plt.cla()
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("speed[km/h]:" + str(round(state.v * 3.6, 2)) +
                      ",target index:" + str(target_ind))
            plt.pause(0.0001)
            matplotrecorder.save_frame()  # save each frame

    plt.close()
    return t, x, y, yaw, v#到达目的地后返出时间,x,y,yaw,v


def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)

    direction = 1.0

    # Set stop point
    for i in range(len(cx) - 1):
        dyaw = cyaw[i + 1] - cyaw[i]
        switch = math.pi / 4.0 <= dyaw < math.pi / 2.0

        if switch:
            speed_profile[i] = 0#把满足switch的设成0
        else:
            speed_profile[i] = target_speed#不满足的switch设成正速度
        
    speed_profile[-1] = 0.0#最后停车

    #  flg, ax = plt.subplots(1)
    #  plt.plot(speed_profile, "-r")
    #  plt.show()

    return speed_profile


def main():
    print("LQR steering control tracking start!!")
    ax = [0.0, 6.0, 12.5, 10.0, 7.5, 3.0, -1.0]
    ay = [0.0, -3.0, -5.0, 6.5, 3.0, 5.0, -2.0]
    goal = [ax[-1], ay[-1]]

    cx, cy, cyaw, ck, s = pycubicspline.calc_2d_spline_interpolation(ax, ay, num=100)#对给定部分点进行三次样条曲线拟合，0.1s取一个点进行拟合，最后得到cx,cy,cyaw,ck,s
    target_speed = 10.0 / 3.6#不太明白这个target_speed的意义？？？

    sp = calc_speed_profile(cx, cy, cyaw, target_speed)#计算出各个点位所需求的速度

    t, x, y, yaw, v = closed_loop_prediction(cx, cy, cyaw, ck, sp, goal)#得到到达目标后的t,x,y,yaw,v，完成整个控制过程

    if animation:
        matplotrecorder.save_movie("animation.gif", 0.1)  # gif is ok.

    flg, _ = plt.subplots(1)
    plt.plot(ax, ay, "xb", label="input")
    plt.plot(cx, cy, "*", label="spline")
    plt.plot(x, y, "-g", label="tracking")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    flg, ax = plt.subplots(1)
    plt.plot(s, [math.degrees(iyaw) for iyaw in cyaw], "-r", label="yaw")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("yaw angle[deg]")

    flg, ax = plt.subplots(1)
    plt.plot(s, ck, "-r", label="curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    flg, ax = plt.subplots(1)
    plt.plot(t, [iv * 3.6 for iv in v], "-r")
    plt.xlabel("Time[s]")
    plt.ylabel("Speed[km/h]")
    plt.grid(True)
    plt.show()
  
    plt.show()
    

if __name__ == '__main__':
    main()
