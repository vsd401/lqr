#! /usr/bin/python
"""

Path tracking simulation with LQR steering control and PID speed control.

author: Atsushi Sakai

"""
import numpy as np
import math
import matplotlib.pyplot as plt
import unicycle_model
from pycubicspline import pycubicspline
from matplotrecorder import matplotrecorder
import scipy.linalg as la

Kp = 1.0  # speed proportional gain

# LQR parameter
Q = np.eye(4)
R = np.eye(1)

# animation = True
animation = False

#matplotrecorder.donothing = True


def PIDControl(target, current):
    a = Kp * (target - current)

    return a


def pi_2_pi(angle):
    while (angle > math.pi):
        angle = angle - 2.0 * math.pi

    while (angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle


def solve_DARE(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 150
    eps = 0.01

    for i in range(maxiter):
        Xn = A.T * X * A - A.T * X * B * \
                           la.inv(R + B.T * X * B) * B.T * X * A + Q
        if (abs(Xn - X)).max() < eps:
            X = Xn
            break
        X = Xn

    return Xn


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_DARE(A, B, Q, R)

    # compute the LQR gain
    K = np.matrix(la.inv(B.T * X * B + R) * (B.T * X * A))

    eigVals, eigVecs = la.eig(A - B * K)

    return K, X, eigVals


def lqr_steering_control(state, cx, cy, cyaw, ck, pe, pth_e):
    ind, e = calc_nearest_index(state, cx, cy, cyaw)

    k = ck[ind]
    v = state.v
    th_e = pi_2_pi(state.yaw - cyaw[ind])

    A = np.matrix(np.zeros((4, 4)))
    A[0, 0] = 1.0
    A[0, 1] = unicycle_model.dt
    A[1, 2] = v
    A[2, 2] = 1.0
    A[2, 3] = unicycle_model.dt
    # print(A)

    B = np.matrix(np.zeros((4, 1)))
    B[3, 0] = v / unicycle_model.L

    K, _, _ = dlqr(A, B, Q, R)

    x = np.matrix(np.zeros((4, 1)))

    x[0, 0] = e
    x[1, 0] = (e - pe)/unicycle_model.dt
    x[2, 0] = th_e
    x[3, 0] = (th_e - pth_e)/unicycle_model.dt

    ff = math.atan2(unicycle_model.L * k, 1)
    fb = pi_2_pi((-K * x)[0, 0])

    delta = ff + fb

    return delta, ind, e, th_e


def calc_nearest_index(state, cx, cy, cyaw):
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]

    d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


def SW_smooth(SW_arr,a,B):
    for i in range(a):
        SW_arr.pop(0)
        SW_arr.insert(len(SW_arr), B)
        C = np.array(SW_arr)
    # return sum(C)/len(C), C.tolist()
    return np.dot(C, [0.25, 0.25, 0.25, 0.25]), C.tolist()


def closed_loop_prediction(cx, cy, cyaw, ck, speed_profile, goal):
    T = 500.0  # max simulation time
    goal_dis = 0.3
    stop_speed = 0.05

    state = unicycle_model.State(x=-0.0, y=-0.0, yaw=0.0, v=0.0)

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    SW = [0.0]
    SW_arr = [0.0, 0.0, 0.0, 0.0]
    target_ind = calc_nearest_index(state, cx, cy, cyaw)

    e, e_th  = 0.0, 0.0

    while T >= time:
        dl, target_ind, e, e_th = lqr_steering_control(state, cx, cy, cyaw, ck, e, e_th)
        SW_, SW_arr = SW_smooth(SW_arr,1,dl)  # 第二个参数的意义：长度为A的SW_arr，如果设置为1，则滚动更新一步，则相当于把前A步的历史方向盘结果进行了平均平滑，滚动A步则说明就用当前的结果，也就是没有进行平滑
        dl = SW_
        ai = PIDControl(speed_profile[target_ind], state.v)
        # state = unicycle_model.update(state, ai, di)
        state = unicycle_model.update(state, ai, dl)

        if abs(state.v) <= stop_speed:
            target_ind += 1

        time = time + unicycle_model.dt

        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        if math.sqrt(dx ** 2 + dy ** 2) <= goal_dis:
            print("Goal")
            break

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        SW.append(dl)

        if target_ind % 1 == 0 and animation:
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
    return t, x, y, yaw, v, SW


def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)

    direction = 1.0

    # Set stop point
    for i in range(len(cx) - 1):
        dyaw = cyaw[i + 1] - cyaw[i]
        switch = math.pi / 4.0 <= dyaw < math.pi / 2.0

        if switch:
            direction *= -1

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

        if switch:
            speed_profile[i] = 0.0

    speed_profile[-1] = 0.0

    #  flg, ax = plt.subplots(1)
    #  plt.plot(speed_profile, "-r")
    #  plt.show()

    return speed_profile


def main():
    print("LQR steering control tracking start!!")
    ax = [0.0, 6.0, 12.5, 10.0, 7.5, 3.0, -1.0]
    ay = [0.0, -3.0, -5.0, 6.5, 3.0, 5.0, -2.0]
    goal = [ax[-1], ay[-1]]

    cx, cy, cyaw, ck, s = pycubicspline.calc_2d_spline_interpolation(ax, ay)
    target_speed = 10.0 / 3.6

    sp = calc_speed_profile(cx, cy, cyaw, target_speed)

    t, x, y, yaw, v, SW = closed_loop_prediction(cx, cy, cyaw, ck, sp, goal)

    if animation:
        matplotrecorder.save_movie("animation.gif", 0.1)  # gif is ok.

    flg, _ = plt.subplots(1)
    plt.plot(ax, ay, "xb", label="input")
    plt.plot(cx, cy, "-r", label="spline")
    plt.plot(x, y, "-g", label="tracking")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()



    flg, ax = plt.subplots(1)
    plt.plot(t, [math.degrees(iyaw) for iyaw in yaw], "-r")
    plt.grid(True)
    plt.legend()
    plt.xlabel("Time[s]")
    plt.ylabel("yaw angle[deg]")

    flg, ax = plt.subplots(1)
    plt.plot(s, ck, "-r", label="curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    flg, ax = plt.subplots(1)
    plt.plot(t, [math.degrees(iyaw) for iyaw in SW], "-r")
    plt.grid(True)
    plt.legend()
    plt.xlabel("time[s]")
    plt.ylabel("SteeringAngle [deg]")

    plt.show()


if __name__ == '__main__':
    main()
