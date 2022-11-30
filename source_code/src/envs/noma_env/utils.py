from sympy import symbols, solve, Eq
import math
import numpy as np
from src.envs.noma_env.state3 import UAVState, CarState, HumanState, Building
import logging


def compute_theta(dpx, dpy, dpz):
    ''''''
    #
    # dpx>0 dpy>0theta
    # dpx<0 dpy>0theta
    # dpx<0 dpy<0theta
    # dpx>0 dpy<0theta
    theta = math.atan(dpy / (dpx + 1e-8))

    #  2022/1/10  y ~
    x1, y1 = 0, 0
    x2, y2 = dpx, dpy
    ang1 = np.arctan2(y1, x1)
    ang2 = np.arctan2(y2, x2)
    # theta = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    theta = (ang1 - ang2) % (2 * np.pi)  # theta in [0, 2*pi]

    return theta


# def compute_elevation(dpx, dpy, dpz):
#     ''''''
#     elevation = math.atan(dpz / (math.sqrt(dpx * dpx + dpy * dpy) + 1e-8))
#     return elevation


def compute_distance(obj1, obj2):
    if type(obj1) in [list, tuple]:
        assert len(obj1) == 3
        A, B = obj1, obj2
    else:
        assert type(obj1) in [UAVState, CarState, HumanState, Building]
        A = obj1.px, obj1.py, obj1.pz
        B = obj2.px, obj2.py, obj2.pz
    return math.sqrt(
        (A[0] - B[0]) ** 2 +
        (A[1] - B[1]) ** 2 +
        (A[2] - B[2]) ** 2
    )

def w2db(x):
    return 10 * math.log(x, 10)

def db2w(x):
    return math.pow(10, x/10)




import signal
def set_timeout(num):
    def wrap(func):
        def handle(signum, frame):  #  SIGALRM the interrupted stack frame.
            raise RuntimeError

        def to_do(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)  #
                signal.alarm(num)  #  num
                r = func(*args, **kwargs)
                signal.alarm(0)  #
                return r
            except RuntimeError as e:
                pass

        return to_do

    return wrap



def judge_intersection(obj1, obj2, buildings, mode):
    if type(obj1) in [tuple, list] and type(obj2) in [tuple, list]:
        x1, y1, z1 = obj1
        x2, y2, z2 = obj2
    else:
        assert type(obj1) in [HumanState, UAVState, CarState, Building]
        assert type(obj2) in [HumanState, UAVState, CarState, Building]
        x1, y1, z1 = obj1.px, obj1.py, obj1.pz
        x2, y2, z2 = obj2.px, obj2.py, obj2.pz

    if (x1, y1) == (x2, y2):  # /LoS
        if mode == 'collide':
            return False
        else:
            return True

    for i, building in enumerate(buildings):
        x0, y0, r, h = building.px, building.py, building.r, building.pz
        # A: y2-y1
        # B: -(x2-x1)
        # C: (x2-x1)*y1 - (y2-y1)*x1
        centerOfCircle_to_line_dis = math.fabs(
            (y2 - y1) * x0 - (x2 - x1) * y0 + (x2 - x1) * y1 - (y2 - y1) * x1
        ) / math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        if centerOfCircle_to_line_dis >= r: continue  # judge stage 1

        @set_timeout(2)
        def time_limited_solve(x0, y0, x1, y1, x2, y2, r):
            # k:-A/B
            # b:-C/B
            x = symbols('x')
            k = -(y2 - y1) / -(x2 - x1 + 1e-16)  # 0 1e-16
            b = -((x2 - x1) * y1 - (y2 - y1) * x1) / -(x2 - x1 + 1e-16)  # 0 1e-16
            x_inters = solve(Eq((x - x0) ** 2 + (k * x + b - y0) ** 2, r ** 2), x)
            return x_inters

        x_inters = time_limited_solve(x0, y0, x1, y1, x2, y2, r)

        try:
            x_low, x_high = min(x1, x2), max(x1, x2)
            if not (x_low < x_inters[0] < x_high or x_low < x_inters[1] < x_high): continue  # 
            height1 = (x_inters[0] - x1) / (x2 - x1) * (z2 - z1) + z1
            height2 = (x_inters[1] - x1) / (x2 - x1) * (z2 - z1) + z1
            # print(f'In judge_LoS_channel(), solving the intersection！height1={height1}, height2={height2}, h={h}')
            if height1 > h and height2 > h: continue  # judge stage 2
        except Exception as e:  # len(x_inters) < 2,
            logging.info(f'waring: solve Timed out! error detail is: {e}')
            continue

        if mode == 'collide':
            return True  # mode=='collide'buildingtrue
        else:
            return False  # mode=='LoS'buildingFalse
    if mode == 'collide':
        return False
    else:
        return True


def judge_intersection_2(obj1, obj2, buildings, mode):

    if type(obj1) in [tuple, list] and type(obj2) in [tuple, list]:
        x1, y1, z1 = obj1
        x2, y2, z2 = obj2
    else:
        assert type(obj1) in [HumanState, UAVState, CarState, Building]
        assert type(obj2) in [HumanState, UAVState, CarState, Building]
        x1, y1, z1 = obj1.px, obj1.py, obj1.pz
        x2, y2, z2 = obj2.px, obj2.py, obj2.pz

    if (x1, y1) == (x2, y2):  # /LoS
        if mode == 'collide':
            return False
        else:
            return True

    for i, building in enumerate(buildings):
        x0, y0, r, h = building.px, building.py, building.r, building.h
        # A: y2-y1
        # B: -(x2-x1)
        # C: (x2-x1)*y1 - (y2-y1)*x1
        centerOfCircle_to_line_dis = math.fabs(
            (y2 - y1) * x0 - (x2 - x1) * y0 + (x2 - x1) * y1 - (y2 - y1) * x1
        ) / math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        if centerOfCircle_to_line_dis >= r: continue  # judge stage 1

        # k:-A/B
        # b:-C/B
        k = -(y2 - y1) / -(x2 - x1 + 1e-16)  # 0 1e-16
        b = -((x2 - x1) * y1 - (y2 - y1) * x1) / -(x2 - x1 + 1e-16)  # 0 1e-16
        # (xv, yv)
        # k2b2 , k2 = B/A, (x0, y0), b2 = y0 - k*x0
        k2 = -(x2 - x1) / (y2 - y1 + 1e-16)
        b2 = y0 - k2 * x0
        # xv = (b2-b) / (k-k2)  yv = (k*b2-k2*b) / (k-k2)
        xv = (b2 - b) / (k - k2)
        yv = (k * b2 - k2 * b) / (k - k2)

        x_low, x_high = min(x1, x2), max(x1, x2)
        y_low, y_high = min(y1, y2), max(y1, y2)
        if (x2 - x0) ** 2 + (y2 - y0) ** 2 > r ** 2 and (x1 - x0) ** 2 + (y1 - y0) ** 2 > r ** 2:
            continue

        @set_timeout(2)
        def solve_x_inters(x0, y0, r, k, b):
            x = symbols('x')
            x_inters = solve(Eq((x - x0) ** 2 + (k * x + b - y0) ** 2, r ** 2), x)
            return x_inters

        @set_timeout(2)
        def solve_y_inters(x0, y0, r, x1):
            y = symbols('y')
            y_inters = solve(Eq((x1 - x0) ** 2 + (y - y0) ** 2, r ** 2), y)
            return y_inters


        try:
            if x1 == x2:
                y_inters = solve_y_inters(x0, y0, r, x1)
                height1 = (y_inters[0] - y1) / (y2 - y1) * (z2 - z1) + z1
                height2 = (y_inters[1] - y1) / (y2 - y1) * (z2 - z1) + z1
            else:
                x_inters = solve_x_inters(x0, y0, r, k, b)
                height1 = (x_inters[0] - x1) / (x2 - x1) * (z2 - z1) + z1
                height2 = (x_inters[1] - x1) / (x2 - x1) * (z2 - z1) + z1


            if height1 > h and height2 > h: continue  # judge stage 3
        except Exception as e:  # len(x_inters) < 2,
            logging.info(f'waring: solve Timed out! error detail is: {e}')
            continue

        if mode == 'collide':
            return True  # mode=='collide'buildingtrue
        else:
            return False  # mode=='LoS'buildingFalse
    if mode == 'collide':
        return False
    else:
        return True

def true_or_false(mode):
    if mode == 'collide':  # mode=='collide'buildingtrue
        return True
    else:
        return False

def not_true_or_false(mode):
    if mode == 'collide':  # mode=='collide'buildingFalse
        return False
    else:
        return True

def judge_intersection_3(obj1, obj2, buildings, mode):

    if type(obj1) in [tuple, list] and type(obj2) in [tuple, list]:
        x1, y1, z1 = obj1
        x2, y2, z2 = obj2
    else:
        assert type(obj1) in [HumanState, UAVState, CarState, Building]
        assert type(obj2) in [HumanState, UAVState, CarState, Building]
        x1, y1, z1 = obj1.px, obj1.py, obj1.pz
        x2, y2, z2 = obj2.px, obj2.py, obj2.pz

    if (x1, y1) == (x2, y2):  # /LoS
        if mode == 'collide':
            return False
        else:
            return True

    for i, building in enumerate(buildings):
        x0, y0, r, h = building.px, building.py, building.r, building.h

        #
        if (x2 - x0) ** 2 + (y2 - y0) ** 2 > r ** 2 and (x1 - x0) ** 2 + (y1 - y0) ** 2 > r ** 2:
            continue

        #
        if (x2 - x0) ** 2 + (y2 - y0) ** 2 <= r ** 2 and (x1 - x0) ** 2 + (y1 - y0) ** 2 <= r ** 2:
            if z1 > h and z2 > h:
                continue
            else:
                return true_or_false(mode)

        #
        if (x1 - x0) ** 2 + (y1 - y0) ** 2 <= r ** 2 and z1 < h:
            return true_or_false(mode)
        if (x2 - x0) ** 2 + (y2 - y0) ** 2 <= r ** 2 and z2 < h:
            return true_or_false(mode)


        @set_timeout(2)
        def solve_x_inters(x0, y0, r, k, b):
            x = symbols('x')
            x_inters = solve(Eq((x - x0) ** 2 + (k * x + b - y0) ** 2, r ** 2), x)
            return x_inters

        @set_timeout(2)
        def solve_y_inters(x0, y0, r, x1):
            y = symbols('y')
            y_inters = solve(Eq((x1 - x0) ** 2 + (y - y0) ** 2, r ** 2), y)
            return y_inters

        x_low, x_high = min(x1, x2), max(x1, x2)
        y_low, y_high = min(y1, y2), max(y1, y2)
        try:
            flag = 0
            if x1 == x2:
                y_inters = solve_y_inters(x0, y0, r, x1)
                if y_low <= y_inters[0] <= y_high:
                    height1 = (y_inters[0] - y1) / (y2 - y1) * (z2 - z1) + z1
                    if height1 <= h:
                        flag = 1
                if y_low <= y_inters[1] <= y_high:
                    height2 = (y_inters[1] - y1) / (y2 - y1) * (z2 - z1) + z1
                    if height2 <= h:
                        flag = 1
            else:
                # k:-A/B
                # b:-C/B
                k = -(y2 - y1) / -(x2 - x1 + 1e-16)  # 0 1e-16
                b = -((x2 - x1) * y1 - (y2 - y1) * x1) / -(x2 - x1 + 1e-16)  # 0 1e-16
                x_inters = solve_x_inters(x0, y0, r, k, b)
                if x_low <= x_inters[0] <= x_high:
                    height1 = (x_inters[0] - x1) / (x2 - x1) * (z2 - z1) + z1
                    if height1 <= h:
                        flag = 1
                if x_low <= x_inters[1] <= x_high:
                    height2 = (x_inters[1] - x1) / (x2 - x1) * (z2 - z1) + z1
                    if height2 <= h:
                        flag = 1

            if flag == 0: continue  # judge stage 3
        except Exception as e:  # len(x_inters) < 2,
            # logging.info(f'waring: solve Timed out! error detail is: {e}')
            continue

        # flag == 1
        return true_or_false(mode)
    return not_true_or_false(mode)


def judge_intersection_4(obj1, obj2, buildings, mode):

    if type(obj1) in [tuple, list] and type(obj2) in [tuple, list]:
        x1, y1, z1 = obj1
        x2, y2, z2 = obj2
    else:
        assert type(obj1) in [HumanState, UAVState, CarState, Building]
        assert type(obj2) in [HumanState, UAVState, CarState, Building]
        x1, y1, z1 = obj1.px, obj1.py, obj1.pz
        x2, y2, z2 = obj2.px, obj2.py, obj2.pz

    if (x1, y1) == (x2, y2):  # /LoS
        if mode == 'collide':
            return False
        else:
            return True

    for i, building in enumerate(buildings):
        x0, y0, r, h = building.px, building.py, building.r, building.h

        # A: y2-y1
        # B: -(x2-x1)
        # C: (x2-x1)*y1 - (y2-y1)*x1
        centerOfCircle_to_line_dis = math.fabs(
            (y2 - y1) * x0 - (x2 - x1) * y0 + (x2 - x1) * y1 - (y2 - y1) * x1
        ) / math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        # >r
        if centerOfCircle_to_line_dis > r: continue

        x_low, x_high = min(x1, x2), max(x1, x2)
        y_low, y_high = min(y1, y2), max(y1, y2)
        #
        if (x2 - x0) ** 2 + (y2 - y0) ** 2 > r ** 2 and (x1 - x0) ** 2 + (y1 - y0) ** 2 > r ** 2:
            # k:-A/B
            # b:-C/B
            k = -(y2 - y1) / -(x2 - x1 + 1e-16)  # 0 1e-16
            b = -((x2 - x1) * y1 - (y2 - y1) * x1) / -(x2 - x1 + 1e-16)  # 0 1e-16
            k2 = -(x2 - x1) / (y2 - y1 + 1e-16)
            b2 = y0 - k2 * x0
            xv = (b2 - b) / (k - k2)
            yv = (k * b2 - k2 * b) / (k - k2)
            #
            # if x_low <= xv <= x_high and y_low <= yv <= y_high:
            if (x_low <= xv <= x_high or x_low == x_high) and y_low <= yv <= y_high:
                return true_or_false(mode)
            else:
                continue

        #
        if (x2 - x0) ** 2 + (y2 - y0) ** 2 <= r ** 2 and (x1 - x0) ** 2 + (y1 - y0) ** 2 <= r ** 2:
            if z1 > h and z2 > h:
                continue
            else:
                return true_or_false(mode)

        #
        if (x1 - x0) ** 2 + (y1 - y0) ** 2 <= r ** 2 and z1 < h:
            return true_or_false(mode)
        if (x2 - x0) ** 2 + (y2 - y0) ** 2 <= r ** 2 and z2 < h:
            return true_or_false(mode)


        @set_timeout(2)
        def solve_x_inters(x0, y0, r, k, b):
            x = symbols('x')
            x_inters = solve(Eq((x - x0) ** 2 + (k * x + b - y0) ** 2, r ** 2), x)
            return x_inters

        @set_timeout(2)
        def solve_y_inters(x0, y0, r, x1):
            y = symbols('y')
            y_inters = solve(Eq((x1 - x0) ** 2 + (y - y0) ** 2, r ** 2), y)
            return y_inters


        try:
            flag = 0
            if x1 == x2:
                y_inters = solve_y_inters(x0, y0, r, x1)
                if y_low <= y_inters[0] <= y_high:
                    height1 = (y_inters[0] - y1) / (y2 - y1) * (z2 - z1) + z1
                    if height1 <= h:
                        flag = 1
                if y_low <= y_inters[1] <= y_high:
                    height2 = (y_inters[1] - y1) / (y2 - y1) * (z2 - z1) + z1
                    if height2 <= h:
                        flag = 1
            else:
                x_inters = solve_x_inters(x0, y0, r, k, b)
                if x_low <= x_inters[0] <= x_high:
                    height1 = (x_inters[0] - x1) / (x2 - x1) * (z2 - z1) + z1
                    if height1 <= h:
                        flag = 1
                if x_low <= x_inters[1] <= x_high:
                    height2 = (x_inters[1] - x1) / (x2 - x1) * (z2 - z1) + z1
                    if height2 <= h:
                        flag = 1

            if flag == 0: continue  # judge stage 3
        except Exception as e:  # len(x_inters) < 2,
            # logging.info(f'waring: solve Timed out! error detail is: {e}')
            continue

        # flag == 1
        return true_or_false(mode)
    return not_true_or_false(mode)


def consume_uav_energy(fly_time, v):
    '''
    :param fly_time:
    :param v: velocity of the UAV, m/s
    '''

    # configs
    # Pu = 0.5  # the average transmitted power of each user, W,  e.g. mobile phone
    P0 = 79.8563  # blade profile power, W
    P1 = 88.6279  # derived power, W
    U_tips = 120  # tip speed of the rotor blade of the UAV,m/s
    v0 = 4.03  # the mean rotor induced velocity in the hovering state,m/s
    d0 = 0.6  # fuselage drag ratio
    rho = 1.225  # density of air,kg/m^3
    s0 = 0.05  # the rotor solidity
    A = 0.503  # the area of the rotor disk, m^2


    Power_flying = P0 * (1 + 3 * v ** 2 / U_tips ** 2) + \
                   P1 * np.sqrt((np.sqrt(1 + v ** 4 / (4 * v0 ** 4)) - v ** 2 / (2 * v0 ** 2))) + \
                   0.5 * d0 * rho * s0 * A * v ** 3

    # Power_hovering = P0 + P1  # v=0P0+P1
    return fly_time * Power_flying

if __name__ == '__main__':
    print(db2w(-9))

    # print(w2db(1))  # 0
    # print(w2db(1000))  # 30
    # print(db2w(20))  # 100

    # == test consume_uav_energy ==
    # fly_time = 5  # s
    # hover_time = 0
    # v = range(0, 20, 2)
    # e = []
    # for velocity in v:
    #     e1 = consume_uav_energy(fly_time, hover_time, velocity)
    #     print(f' {velocity} m/s,  {e1} J.')
    #     e.append(e1)
    # import matplotlib.pyplot as plt
    # plt.plot(v, e)
    # plt.show()

    # print(consume_uav_energy(5, 20) * 100)
