import math

env_config = {
    #
    'name': 'base',
    'num_timestep': 100,
    'theta_range': (0, 2*math.pi),
    # 'elevation_range': (-math.pi/2, math.pi/2),
    'max_x': 2000,
    'max_y': 2000,
    'max_z': 100,
    'uav_init_height': 50,  # 50m

    # deprecated
    'max_r': 40,  # 44
    'space_between_building': 100,
    'max_human_movelength': 0,  #

    #
    'max_uav_energy': 1500000,  # 1500KJ
    'max_car_energy': 2000000,  # 2000KJ
    'num_uav': 4,
    'num_car': 2,
    'num_human': 100,
    'num_building': 0,
    'num_special_human': 10,
    'num_special_building': 0,
    'max_data_amount': 3e9,  # 3000000000
    # env.step，th=0.2, throughput5e7, 1e920timestep。throughput,
    'tc': 10,
    'tm': 10,
    'v_uav': 20,  # m/s
    'v_car': 10,  # m/s

    #
    'power_tx': 20,  # dBm
    'noise0': -70,  # dBm
    'rho0': -50,  # dB
    'bandwidth_subchannel': 2e7,  # Hz for each sub channel, NOMA3.125e5 Hz
    # noma
    'num_subchannel': 3,  # subchannel
    'noise0_density': 5e-20,  # W/Hz
    'aA': 2,  # G2A pass loss factor
    'aG': 4,  # G2G pass loss factor
    'nLoS': 0,  # dB, 1w
    'nNLoS': -20,  # dB, 0.01w
    'psi': 9.6,
    'beta': 0.16,
    'p_uav': 3,  # w, 34.7dbm
    'p_poi': 0.1,  # w, 20dbm
    'sinr_demand': 0.2,
    'is_uav_competitive': False,

    # multi-agent
    'obs_range': None

}
