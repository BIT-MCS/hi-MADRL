import numpy as np
from src.envs.noma_env.utils import *

def compute_LoS_prob(env_config, dis, height):
    '''
    :param dis: poiuav3D
    :param height: uav
    :return: LoS
    '''
    psi = env_config['psi']
    beta = env_config['beta']
    theta = np.arcsin(height / dis) / np.pi * 180  # theta

    LoS_prob = 1 / (1 + psi * np.exp(-beta * (theta - psi)))
    return LoS_prob


def compute_channel_gain_G2A(env_config, dis, aA=None):
    '''
    :param dis: poiuavuavugv3D
    :return: G2A
    '''
    height = env_config['uav_init_height']
    if aA is None:
        aA = env_config['aA']
    nLoS = db2w(env_config['nLoS'])
    nNLoS = db2w(env_config['nNLoS'])
    Los_prob = compute_LoS_prob(env_config, dis, height)
    channel_gain_G2A = Los_prob * nLoS * dis ** (-aA) + \
                       (1 - Los_prob) * nNLoS * dis ** (-aA)
    return channel_gain_G2A


def compute_channel_gain_G2G(env_config, dis, aG=None):
    '''
    :param dis: poiugv2D
    :param aG: aG
    :return: G2G
    '''
    if aG is None:
        aG = env_config['aG']
    channel_gain_G2G = max(dis, 1.0) ** (-aG)
    return channel_gain_G2G


def compute_capacity_G2A(env_config, i_dis, j_dis, co_factor=1):
    '''
    :param i_dis: poi iuav3D
    :param j_dis: poi juav3D
    :param co_factor: 
    :return: G2A
    '''
    n0 = env_config['noise0_density']
    B0 = env_config['bandwidth_subchannel'] / co_factor
    P = env_config['p_poi']
    Gi_G2A = compute_channel_gain_G2A(env_config, i_dis)
    Gj_G2A = compute_channel_gain_G2A(env_config, j_dis)
    sinr = Gi_G2A * P / (n0 * B0 + Gj_G2A * P)
    Ri_G2A = B0 * np.log2(1 + sinr)
    return sinr, Ri_G2A


def compute_capacity_RE(env_config, uav_ugv_dis, i_2d_dis, j_2d_dis, co_factor=1):
    '''
    :param uav_ugv_dis: uavugv3D
    :param i_2d_dis: poi iugv2Ddeprecated
    :param j_2d_dis: poi jugv2D
    :return: RE
    '''
    n0 = env_config['noise0_density']
    B0 = env_config['bandwidth_subchannel'] / co_factor
    P_poi = env_config['p_poi']  # w
    P_uav = env_config['p_uav']  # w
    # Gi_G2G = compute_channel_gain_G2G(env_config, i_2d_dis)
    Gj_G2G = compute_channel_gain_G2G(env_config, j_2d_dis)
    G_RE = compute_channel_gain_G2A(env_config, uav_ugv_dis)  # debug:
    # sinr = (G_RE * P_uav + Gi_G2G * P_poi) / (n0 * B0 + Gj_G2G * P_poi)
    sinr = (G_RE * P_uav) / (n0 * B0 + Gj_G2G * P_poi)
    Ri_RE = B0 * np.log2(1 + sinr)
    return sinr, Ri_RE


def compute_capacity_G2G(env_config, j_2d_dis, co_factor=1):
    '''
    :param j_2d_dis: poi j ugv2D
    :return: G2G
    '''

    n0 = env_config['noise0_density']
    B0 = env_config['bandwidth_subchannel'] / co_factor
    P = env_config['p_poi']
    Gj_G2G = compute_channel_gain_G2G(env_config, j_2d_dis)
    sinr = Gj_G2G * P / (n0 * B0)
    Rj_G2G = B0 * np.log2(1 + sinr)
    return sinr, Rj_G2G


if __name__ == '__main__':
    from config.env_config_noma import env_config

    # receiver determinationuser association

    i_pos = (0, 0, 0)
    j_pos = (0, 500, 0)
    uav_pos = (0, 250, 100)
    ugv_pos = (0, 250, 0)

    sinr_G2A, R_G2A = compute_capacity_G2A(env_config,
                                 compute_distance(i_pos, uav_pos),
                                 compute_distance(j_pos, uav_pos))

    sinr_RE, R_RE = compute_capacity_RE(env_config,
                               compute_distance(uav_pos, ugv_pos),
                               compute_distance(i_pos, ugv_pos),
                               compute_distance(j_pos, ugv_pos),
                               )

    sinr_G2G, R_G2G = compute_capacity_G2G(env_config, compute_distance(j_pos, ugv_pos))

    print('poi j G2G', R_G2G)
    print('poi i G2A', min(R_G2A, R_RE))
    # 
    # R_G2G 94667086.81227656
    # R_G2A 19999975.618556052
    # R_RE 468539953.2692103


    # G2AG2G2D
    # _2d_dis = np.linspace(1, 10, 1000)
    # _3d_dis = np.sqrt(_2d_dis ** 2 + env_config['uav_init_height'] ** 2)
    #
    # gain_G2A = compute_channel_gain_G2A(env_config, _3d_dis)
    # gain_G2G = compute_channel_gain_G2G(env_config, _2d_dis)
    #
    # plt.plot(_2d_dis, gain_G2A)
    # # plt.plot(_2d_dis, gain_G2G)
    # plt.legend(['gain_G2A', 'gain_G2G'])
    # plt.show()

    # G2G
    # +-----------+-------------+
    # | 2d(m) | G2G |
    # +-----------+-------------+
    # | 1         | 1.0         |
    # | 2         | 0.0625      |
    # | 10        | 0.0001      |
    # +-----------+-------------+
    # G2A
    # +---------+-------------+
    # | (m) | G2G |
    # +---------+-------------+
    # | 1       | 9.99e-5     |
    # | 10      | 9.90e-5     |
    # | 100     | 4.84e-5     |
    # | 500     | 4.97e-7     |
    # +---------+-------------+


