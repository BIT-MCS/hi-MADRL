import copy
import torch
import numpy as np
import osmnx as ox
import networkx as nx
import os
import os.path as osp
import pandas as pd
import gym
from gym import spaces
from datetime import datetime
from src.config.model_config import model_config
from src.envs.noma_env.state3 import UAVState, CarState, HumanState, JointState
from src.envs.noma_env.utils import *
from src.envs.noma_env.noma_utils import *
from src.envs.roadmap_env.roadmap_utils import Roadmap

project_dir = osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__))))  # base.py

# def print(*args, **kwargs):  #
#     pass

count = 0


class BaseRoadmapEnv():

    def __init__(self, rllib_env_config):
        global count
        print(f'{count}env...')
        count += 1

        self.args = rllib_env_config['args']
        env_config = rllib_env_config['my_env_config']
        self.env_config = env_config
        self.output_dir = self.args.output_dir

        # debug
        self.reward_scale = self.args.reward_scale

        # == roadmap
        self.roadmap_dir = self.args.roadmap_dir
        self.G = ox.load_graphml(self.roadmap_dir + '/map.graphml')
        self.rm = Roadmap(self.args.dataset)
        self.type2_act_dim = self.args.type2_act_dim
        self.gr = self.args.gr
        # ==
        self.use_eoi = self.args.use_eoi
        self.use_copo = self.args.use_copo
        # rllib_env_config.getNone
        self.partial_obs = rllib_env_config['partial_obs']
        self.use_reward_shaping = rllib_env_config['use_reward_shaping']
        self.energy_factor = rllib_env_config['energy_factor']
        self.consider_return_to_zero_list = rllib_env_config['consider_return_to_zero_list']
        self.use_shared_parameters = rllib_env_config.get('use_shared_parameters')  # single agent
        self.add_svo_in_obs = rllib_env_config.get('add_svo_in_obs')  # IPPO
        self.debug_use_nei_max_distance = rllib_env_config.get('debug_use_nei_max_distance')  # IPPO
        self.use_ball_svo = rllib_env_config.get('use_ball_svo')  # IPPO
        # validate
        if self.use_copo:
            assert self.add_svo_in_obs is not None
            assert self.use_ball_svo is not None
            assert self.debug_use_nei_max_distance is not None

        self.max_x = env_config['max_x']
        self.max_y = env_config['max_y']
        self.max_z = env_config['max_z']  # task region's height
        self.max_uav_energy = env_config['max_uav_energy']
        self.max_car_energy = env_config['max_car_energy']
        self.theta_range = env_config['theta_range']
        self.num_uav = env_config['num_uav']
        self.num_car = env_config['num_car']
        self.num_human = env_config['num_human']
        self.num_timestep = env_config['num_timestep']
        self.num_subchannel = env_config['num_subchannel']
        self.sinr_demand = env_config['sinr_demand']
        self.is_uav_competitive = env_config['is_uav_competitive']

        self.num_agent = self.num_uav + self.num_car
        self.agent_name_list = [f'uav{i + 1}' for i in range(self.num_uav)] + \
                               [f'car{i + 1}' for i in range(self.num_car)]

        uav_state_dim = model_config['gnn_config']['uav_state_dim']
        car_state_dim = model_config['gnn_config']['car_state_dim']
        human_state_dim = model_config['gnn_config']['human_state_dim']
        self.obs_dim = uav_state_dim * env_config['num_uav'] + \
                       car_state_dim * env_config['num_car'] + \
                       human_state_dim * env_config['num_human']

        # obs and act spaces
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.obs_dim,))
        self.share_observation_space = self.observation_space
        self.mappo_share_observation_space = spaces.Box(0.0, 1.0, shape=(self.obs_dim * self.num_agent,))  # mappoconcatccobs
        uav_act_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Dict(
            {
                'uav': uav_act_space,
                'car': spaces.Discrete(self.type2_act_dim)  #
            }
        )


        # ==== start for CoPO ==== #
        self.config = dict()
        self.config.update(  # in SVOEnv
            dict(
                # neighbours_distance=20,
                # Two mode to compute utility for each vehicle:
                # "linear": util = r_me * svo + r_other * (1 - svo), svo in [0, 1]
                # "angle": util = r_me * cos(svo) + r_other * sin(svo), svo in [0, pi/2]
                # "angle" seems to be more stable!
                svo_mode="angle",
                svo_dist="uniform",  # "uniform" or "normal"
                svo_normal_std=0.3,  # The initial STD of normal distribution, might change by calling functions.
                return_native_reward=False,
                include_ego_reward=False,
            )
        )

        self.config['svo_dist'] = rllib_env_config.get('svo_dist', 'uniform')  # let it work in copo_validate
        self.config['svo_normal_std'] = rllib_env_config.get('svo_normal_std', 0.3)  # let it work in copo_validate
        self.config['return_native_reward'] = rllib_env_config.get('return_native_reward', False)  # let it work in copo_validate
        self.nei_dis_scale = self.args.nei_dis_scale

        self.config["neighbours_distance"] = min(env_config['max_x'], env_config['max_y']) * self.nei_dis_scale  # copo
        # ==== end for CoPO ========

        self.device = torch.device('cpu')

        # self.human_mat.shape = (num_human, num_timestep+1, 5)
        # this two mat are **read-only**
        self.human_mat = self.init_humans(self.args.setting_dir)

        # --- for noadmap ---
        # timestepactiondst(lon,lat)
        self.wipe_last_things()
        self.car_cur_node = [None for _ in range(self.num_car)]  #

        # --- some variable(including entities) which change during explore, so they **must** be reset in self.reset
        self.global_timestep = None
        self.uavs = None
        self.cars = None
        self.humans = None
        self.saved_uav_trajs = None
        self.saved_th_actions = None  # deprecated
        # self.state_for_train = None
        # self.next_state_for_train = None
        # self.collect_data_amount = None
        self.total_data_amount = None
        self.return_to_zero_list = None
        self.loss_ratio_list = None
        self.collect_data_ratio_list = None
        self.energy_consumption_ratio_list = None
        self.nei_matrix = None  # step
        # roadmap


        # debug

        '''variables that used in self.callback'''
        self.best_train_reward = -float('inf')
        self.best_eval_reward = -float('inf')

    def set_device(self, device):
        self.device = device

    def wipe_last_things(self):
        self.last_dst_node = [[None for _ in range(self.action_space['car'].n)] for _ in range(self.num_car)]
        self.last_length = [[None for _ in range(self.action_space['car'].n)] for _ in range(self.num_car)]
        self.last_dst_lonlat = [[None for _ in range(self.action_space['car'].n)] for _ in range(self.num_car)]

    def obj(self, i):
        # agent
        obj = self.uavs[i] if i < self.num_uav else self.cars[i - self.num_uav]
        return obj

    def is_uav(self, i):
        # agent
        return True if i < self.num_uav else False

    def init_humans(self, setting_dir):
        setting_absdir = osp.join(project_dir, setting_dir)
        human_df = pd.read_csv(os.path.join(setting_absdir, 'human.csv'))

        human_mat = np.expand_dims(human_df[human_df['id'] == 0].values[:, 2:], axis=0)  # idt
        for id in range(1, self.num_human):
            subDf = human_df[human_df['id'] == id]
            human_mat = np.concatenate((human_mat, np.expand_dims(subDf.values[:, 2:], axis=0)), axis=0)
        return human_mat  # human_mat.shape = (num_human, num_timestep, 4) 4px, py, pz, theta

    def reset_uavs(self):
        px = self.max_x / 2
        py = self.max_y / 2
        pz = self.env_config['uav_init_height']
        uavs = [UAVState(px=px, py=py, pz=pz, theta=0, energy=self.max_uav_energy, env_config=self.env_config)
                for _ in range(self.num_uav)]
        return uavs

    def reset_cars(self):
        px = self.max_x / 2
        py = self.max_y / 2
        # OK
        init_node = ox.distance.nearest_nodes(self.G, *self.rm.pygamexy2lonlat(px, py))
        px, py = self.rm.lonlat2pygamexy(self.G.nodes[init_node]['x'], self.G.nodes[init_node]['y'])
        cars = [CarState(px=px, py=py, theta=0, energy=self.max_car_energy, env_config=self.env_config)
                for _ in range(self.num_car)]
        self.car_cur_node = [init_node for _ in range(self.num_car)]  # init
        return cars

    def reset_humans(self):
        humans = []
        for id in range(self.num_human):
            tuple = self.human_mat[id, 0, :]  # human_mat.shape =
            humans.append(HumanState(px=tuple[0], py=tuple[1], pz=tuple[2], theta=tuple[3],
                                     data=self.env_config['max_data_amount'], env_config=self.env_config))

        return humans

    def compute_total_data_amount(self):
        total_data_amount = 0
        for obj in self.humans:
            total_data_amount += obj.data
        return total_data_amount

    def flatten_state(self, state):
        batch_dim = 1  # rllibbatch
        state_list = []
        for s in state:
            state_list.append(s.view(batch_dim, -1))

        flatted_state = torch.cat(state_list, dim=1).to(self.device)
        flatted_state = flatted_state.squeeze(0)  # (x, )squeeze
        return flatted_state.numpy()

    def reset(self):
        # ===step1.reset the **must** entities
        self.wipe_last_things()
        self.nei_matrix = np.zeros([self.num_agent, self.num_agent])
        self.global_timestep = 0
        self.uavs = self.reset_uavs()
        self.cars = self.reset_cars()
        self.humans = self.reset_humans()
        self.saved_uav_trajs = [np.zeros((self.num_timestep + 1, 3)) for _ in range(self.num_uav)]
        self.saved_car_trajs = [np.zeros((self.num_timestep + 1, 3)) for _ in range(self.num_car)]
        for id in range(self.num_uav):
            self.saved_uav_trajs[id][0, :] = (self.uavs[id].px, self.uavs[id].py, self.uavs[id].pz)
        for id in range(self.num_car):
            self.saved_car_trajs[id][0, :] = (self.cars[id].px, self.cars[id].py, self.cars[id].pz)
        self.total_data_amount = self.compute_total_data_amount()
        self.return_to_zero_list = []  # poisinr

        self.saved_th_actions = [np.zeros(self.num_timestep + 1, ) for _ in range(len(self.agent_name_list))]
        self.loss_ratio_list = np.zeros(len(self.agent_name_list))  # episodeuav
        self.collect_data_ratio_list = np.zeros(len(self.agent_name_list))  # episodeuvcollect_data_ratio
        self.energy_consumption_ratio_list = np.zeros(len(self.agent_name_list))
        # self.avail_act_prop_list = [np.zeros(self.num_timestep + 1, ) for _ in range(self.num_car)]  # debug
        self.epmean_utpt = []
        self.epmean_ctpt = []
        self.ep_succ_i_uav_dis = []
        self.ep_succ_j_car_dis = []
        self.ep_succ_uav_car_dis = []
        # roadmap
        state = JointState(self.uavs, self.cars, self.humans)

        return self._get_reset_return(state)

    def check_action_in_space(self, action_dict):
        for i, key in enumerate(action_dict.keys()):
            if self.is_uav(i):
                action_dict[key] = np.clip(action_dict[key],
                                           self.action_space['uav'].low[0],
                                           self.action_space['uav'].high[0])
        return action_dict

    def step(self, action_dict):
        # action_dict{'uav1': 12, 'uav2': 4}
        # action_dict{'uav1': [0.5, 0.2], 'uav2': [0.1, 0.9]}

        # (OK) uavclipmappoagent
        action_dict = self.check_action_in_space(action_dict)

        self.nei_matrix = np.zeros([self.num_agent, self.num_agent])
        self.global_timestep += 1
        rewards, infos = dict(), dict()
        for agent_name in self.agent_name_list:
            rewards[agent_name] = 0
            infos[agent_name] = {}
        # == step1. users and UVs move ==
        # action_dictdst
        uav_move_dict, car_dst_dict, car_length_dict = self._get_move_dict_from_action_dict(action_dict)
        rewards = self._movement(uav_move_dict, car_dst_dict, car_length_dict, rewards)
        # == step2. collect data ==
        rewards, infos, _, _ = self._collect(rewards, infos)
        return rewards, infos

    def _get_reset_return(self, state):
        return self._get_step_return(state)

    def _get_step_return(self, state):
        # agentobsnp.array()shape=(obs_dim, )
        obs = {agent_name: self.get_obs_for_agent(self.obj(i), state)
               for i, agent_name in enumerate(self.agent_name_list)}
        return obs

    def get_obs_for_agent(self, obj, state):
        # OK partial_obsrllib_env_configcopopartial_obs
        s = copy.deepcopy(state)
        if self.partial_obs:  #
            for obj2 in s.uav_states + s.car_states + s.human_states:
                if compute_distance(obj, obj2) > self.env_config['obs_range']:
                    obj2.set_zero()
        agent_obs = self.flatten_state(s.to_tensor(add_batchsize_dim=False, device=self.device, normalize=True))
        del s
        return agent_obs

    def shaping1(self, reward, pos, scale=None):
        if not self.use_reward_shaping:
            return reward
        if scale is None:
            scale = 1 / (self.num_timestep * self.num_uav * 2 * self.num_subchannel)  # uavreward shaping
        center_pos = (self.max_x / 2, self.max_y / 2, 0)
        bound = self.max_x / 4
        factor = compute_distance(pos, center_pos) - bound  # poi, poi
        factor = np.clip(factor, -bound, bound) / bound * scale  # factor[-scale, scale]
        reward += 0.001 * factor  # poireward, poireward
        return reward

    def _get_move_dict_from_action_dict(self, action_dict):
        '''uav and car have different logic now!'''
        uav_move_dict, car_dst_dict, car_length_dict = dict(), dict(), dict()
        for i, (agent_name, action) in enumerate(action_dict.items()):
            if self.is_uav(i):
                theta, velocity = action[0], action[1]
                theta *= self.env_config['theta_range'][1]  # scaling
                velocity *= self.env_config['v_uav'] if self.is_uav(i) else self.env_config['v_car']  # scaling
                radius = velocity * self.env_config['tm']
                # print(f'for agent {agent_name}, v = {velocity}, theta = {theta}')  # debug
                dpx, dpy = radius * math.cos(theta), radius * math.sin(theta)
                obj = self.obj(i)
                cur_pos = (obj.px + dpx, obj.py + dpy, obj.pz)
                if 0 <= cur_pos[0] <= self.max_x and 0 <= cur_pos[1] <= self.max_y and 0 <= cur_pos[2] <= self.max_z:
                    uav_move_dict[agent_name] = (dpx, dpy)
                else:  #
                    uav_move_dict[agent_name] = (0.0, 0.0)
            else:
                id = i - self.num_uav
                assert self.last_dst_node[id][action] is not None  # 4/11[id][action]
                assert self.last_dst_lonlat[id][action] is not None
                assert self.last_length[id][action] is not None
                self.car_cur_node[id] = self.last_dst_node[id][action]
                car_dst_dict[agent_name] = self.rm.lonlat2pygamexy(*self.last_dst_lonlat[id][action])
                car_length_dict[agent_name] = self.last_length[id][action]
        self.wipe_last_things()
        return uav_move_dict, car_dst_dict, car_length_dict

    def _movement(self, uav_move_dict, car_dst_dict, car_length_dict, rewards):
        for id, human in enumerate(self.humans):
            tuple = self.human_mat[id, 0, :]
            human.px = tuple[0]
            human.py = tuple[1]
            human.pz = tuple[2]
            human.theta = tuple[3]

        for i, (agent_name, (dpx, dpy)) in enumerate(uav_move_dict.items()):
            # OK , _get_move_dict_from_action_dict
            radius = math.sqrt(dpx ** 2 + dpy ** 2)
            velocity = radius / self.env_config['tm']
            obj = self.uavs[i]
            obj.px += dpx
            obj.py += dpy
            obj.theta = compute_theta(dpx, dpy, 0)
            # OK deal with energy consumption  max_uav_energy
            e = consume_uav_energy(fly_time=self.env_config['tm'], v=velocity)
            obj.energy -= e
            self.energy_consumption_ratio_list[i] += (e / self.max_uav_energy)
            rewards[agent_name] -= (self.energy_factor * e / self.max_uav_energy)

        for j, (agent_name, (px, py)) in enumerate(car_dst_dict.items()):
            obj = self.cars[j]
            dpx = px - obj.px
            dpy = py - obj.py
            obj.theta = compute_theta(dpx, dpy, 0)
            obj.px = px
            obj.py = py
            velocity = car_length_dict[agent_name] / self.env_config['tm']
            e = consume_uav_energy(fly_time=self.env_config['tm'], v=velocity)
            obj.energy -= e
            self.energy_consumption_ratio_list[j + self.num_uav] += (e / self.max_car_energy)
            rewards[agent_name] -= (self.energy_factor * e / self.max_car_energy)

        for id in range(self.num_uav):
            self.saved_uav_trajs[id][self.global_timestep, :] = (self.uavs[id].px, self.uavs[id].py, self.uavs[id].pz)
        for id in range(self.num_car):
            self.saved_car_trajs[id][self.global_timestep, :] = (self.cars[id].px, self.cars[id].py, self.cars[id].pz)

        return rewards

    def relay_association(self):
        '''
        relay
        :return: relay_dict, {0: 1, 1: 1, 2: 1, 3: 1}
        '''
        dis_mat = [[compute_distance(uav, car) for car in self.cars] for uav in self.uavs]
        argmin = np.argmin(dis_mat, axis=-1)  #
        relay_dict = {uav_id: car_id for uav_id, car_id in enumerate(argmin)}
        return relay_dict

    def receiver_determin(self):
        '''
        poiuavcar
        '''
        sorted_uav_access = dict()
        sorted_car_access = dict()

        for i, uav in enumerate(self.uavs):
            dis_list = np.array([max(1, compute_distance(uav, human)) for human in self.humans])
            if self.consider_return_to_zero_list:
                dis_list[self.return_to_zero_list] = float('inf')  # return_to_zero_list, 
            sorted_uav_access[i] = np.argsort(dis_list)[:self.num_subchannel]
        for i, car in enumerate(self.cars):
            dis_list = np.array([max(1, compute_distance(car, human)) for human in self.humans])
            if self.consider_return_to_zero_list:
                dis_list[self.return_to_zero_list] = float('inf')
            sorted_car_access[i] = np.argsort(dis_list)[:self.num_subchannel]
        return sorted_uav_access, sorted_car_access

    def _collect(self, rewards, infos):
        debug_poiid = 37

        # step1. greedy1) UAV-BS 2)
        relay_dict = self.relay_association()
        sorted_uav_access, sorted_car_access = self.receiver_determin()
        self._nei_determin(relay_dict)

        # step2. poi, 1） 2）reward 3）info
        uav_throughput_list = []  # for test
        car_throughput_list = []  # for test
        serve_poi_list = {}
        inter_poi_list = {}  # sinr
        for uav_id, car_id in relay_dict.items():
            # print(f'# uav {uav_id}  car {car_id}  #')
            uav, car = self.uavs[uav_id], self.cars[car_id]
            for channel in range(self.num_subchannel):
                # print(f'## {channel} ##')
                # ==  ==
                poi_i, poi_j = sorted_uav_access[uav_id][channel], sorted_car_access[car_id][channel]
                human_i, human_j = self.humans[poi_i], self.humans[poi_j]
                if self.is_uav_competitive:
                    co_factor = list(relay_dict.values()).count(car_id)
                    assert 1 <= co_factor <= self.num_uav
                else:
                    co_factor = 1
                # print('human_i', human_i.px, human_i.py)
                # print('human_j', human_j.px, human_j.py)
                # print('uav', uav.px, uav.py)
                # print('car', car.px, car.py)
                # uav, car, human_i, human_j poi！
                sinr_G2A, R_G2A = compute_capacity_G2A(self.env_config,
                                                       compute_distance(human_i, uav),
                                                       compute_distance(human_j, uav),
                                                       co_factor=co_factor)
                sinr_RE, R_RE = compute_capacity_RE(self.env_config,
                                                    compute_distance(uav, car),
                                                    compute_distance(human_i, car),  # 3/12 deprecated！
                                                    compute_distance(human_j, car),
                                                    co_factor=co_factor)
                sinr_G2G, R_G2G = compute_capacity_G2G(self.env_config,
                                                       compute_distance(human_j, car),
                                                       co_factor=co_factor)

                sinr_i, sinr_j = min(sinr_G2A, sinr_RE), sinr_G2G  # ok
                capacity_i, capacity_j = min(R_G2A, R_RE), R_G2G
                # print(f"poi i {poi_i}G2Asinr_i={sinr_i}, capacity={'%.1e' % capacity_i}")
                # print(f'poi j {poi_j}G2Gsinr_j={sinr_j}, capacity={capacity_j}')
                # ==  ==
                # for i
                throughput_i = min(capacity_i, human_i.data)
                agent_name = self.agent_name_list[uav_id]
                if sinr_i < self.sinr_demand:  #
                    # print(f'sinr_i={sinr_i}！')
                    penalty = throughput_i / self.total_data_amount
                    rewards[agent_name] -= self.shaping1(penalty, (human_i.px, human_i.py, 0))
                    self.loss_ratio_list[uav_id] += 1  #
                    inter_poi_list[poi_i] = uav_id
                else:
                    # == debug ==
                    succ_dis = compute_distance(human_i, uav)
                    # print(f"sinr_i={sinr_i}uav {uav_id}throughput={'%.1e' % throughput_i}")
                    # print('human i  UAV = ', succ_dis)
                    self.ep_succ_i_uav_dis.append(succ_dis)
                    self.ep_succ_uav_car_dis.append(compute_distance(uav, car))
                    # ==
                    human_i.data -= throughput_i
                    if self.consider_return_to_zero_list and human_i.data == 0:  # return_to_zero_list
                        self.return_to_zero_list.append(poi_i)
                    r = throughput_i / self.total_data_amount
                    rewards[agent_name] += self.shaping1(r, (human_i.px, human_i.py, 0))
                    uav_throughput_list.append(throughput_i)
                    serve_poi_list[poi_i] = uav_id
                    self.collect_data_ratio_list[uav_id] += r

                # for j
                throughput_j = min(capacity_j, human_j.data)
                agent_name = self.agent_name_list[car_id + self.num_uav]  # self.num_uavinspection~
                if sinr_j < self.sinr_demand:  #
                    # print(f'sinr_j={sinr_j}！')
                    penalty = throughput_j / self.total_data_amount
                    rewards[agent_name] -= self.shaping1(penalty, (human_j.px, human_j.py, 0))
                    self.loss_ratio_list[car_id + self.num_uav] += 1  #
                    inter_poi_list[poi_j] = car_id + self.num_uav
                else:
                    # == debug ==
                    succ_dis = compute_distance(human_j, car)
                    # print(f'sinr_j={sinr_j}car {car_id}throughput={throughput_j}')
                    # print('human j  CAR = ', succ_dis)
                    self.ep_succ_j_car_dis.append(succ_dis)
                    self.ep_succ_uav_car_dis.append(compute_distance(uav, car))
                    # ==
                    human_j.data -= throughput_j
                    if self.consider_return_to_zero_list and human_j.data == 0:  # return_to_zero_list
                        self.return_to_zero_list.append(poi_j)
                    r = throughput_j / self.total_data_amount
                    rewards[agent_name] += self.shaping1(r, (human_j.px, human_j.py, 0))
                    car_throughput_list.append(throughput_j)
                    serve_poi_list[poi_j] = car_id + self.num_uav
                    self.collect_data_ratio_list[car_id + self.num_uav] += r

        utpt = np.mean(uav_throughput_list) if len(uav_throughput_list) != 0 else 0.0
        ctpt = np.mean(car_throughput_list) if len(car_throughput_list) != 0 else 0.0
        self.epmean_utpt.append(utpt)
        self.epmean_ctpt.append(ctpt)
        # print('')
        # print('', utpt)
        # print('', ctpt)
        # print(':', np.mean(np.concatenate([uav_throughput_list, car_throughput_list])))
        # print(':', len(serve_poi_list))
        # print(':', len(inter_poi_list))

        return rewards, infos, serve_poi_list, inter_poi_list

    def _nei_determin(self, relay_dict):

        for uav_id, car_id in relay_dict.items():
            self.nei_matrix[uav_id][self.num_uav + car_id] = 1
            self.nei_matrix[self.num_uav + car_id][uav_id] = 1

        ''''''
        for uav1_id in range(self.num_uav):
            for uav2_id in range(uav1_id + 1, self.num_uav):
                if compute_distance(self.uavs[uav1_id], self.uavs[uav2_id]) < self.config["neighbours_distance"]:
                    self.nei_matrix[uav1_id][uav2_id] = 1
                    self.nei_matrix[uav2_id][uav1_id] = 1

        ''''''
        for car1_id in range(self.num_car):
            for car2_id in range(car1_id + 1, self.num_car):
                if compute_distance(self.cars[car1_id], self.cars[car2_id]) < self.config["neighbours_distance"]:
                    self.nei_matrix[self.num_uav + car1_id][self.num_uav + car2_id] = 1
                    self.nei_matrix[self.num_uav + car2_id][self.num_uav + car1_id] = 1

    def callback(self, total_steps, writer, episode_reward, last_info, svo_dict, is_evaluate):
        '''rllibMyCallBack
        ！vecenvdummyenv

        trainevalepisodeis_evaluateenv
        1. save best train & eval model
        2. record something in output.txt and svo.txt
        3. add metrics to tensorboard
        '''
        print('Inside env.callback...')
        tag = 'eval' if is_evaluate else 'train'

        '''step1. tensorboardrewardmetric'''
        loss_ratio_list = []
        collect_data_ratio_list = []
        energy_consumption_ratio_list = []
        have_nei_freq_list = []
        # avail_act_prop_list = []
        fairness = None
        uav_util_factor = None
        for k in self.agent_name_list:
            loss_ratio_list.append(last_info[k]['loss_ratio'])
            collect_data_ratio_list.append(last_info[k]['collect_data_ratio'])
            energy_consumption_ratio_list.append(last_info[k]['energy_consumption_ratio'])
            have_nei_freq_list.append(last_info[k].get('have_nei_freq', 0.0))  # IPPO metric0
            # if k.startswith('car'):
            #     avail_act_prop_list.append(last_info[k]['avail_act_prop'])
            if fairness is None:
                fairness = last_info[k]['fairness']
            else:
                assert fairness == last_info[k]['fairness']
            if uav_util_factor is None:
                uav_util_factor = last_info[k]['uav_util_factor']
            else:
                assert uav_util_factor == last_info[k]['uav_util_factor']

        writer.add_scalar(f'{tag}/episode_reward', episode_reward, total_steps)
        writer.add_scalar(f'{tag}/loss_ratio', np.mean(loss_ratio_list), total_steps)  # >0
        writer.add_scalar(f'{tag}/collect_data_ratio', np.sum(collect_data_ratio_list), total_steps)  # 0-1
        writer.add_scalar(f'{tag}/energy_consumption_ratio', np.mean(energy_consumption_ratio_list), total_steps)  # 0-1
        writer.add_scalar(f'{tag}/fairness', fairness, total_steps)
        writer.add_scalar(f'{tag}/uav_util_factor', uav_util_factor, total_steps)

        writer.add_scalar(f'{tag}/have_nei_freq', np.mean(have_nei_freq_list), total_steps)  # 0-1
        # writer.add_scalar(f'{tag}/avail_act_prop', np.mean(avail_act_prop_list), total_steps)
        '''step2. save best trajs'''
        save = False
        if is_evaluate and episode_reward > self.best_eval_reward:
            self.best_eval_reward = episode_reward
            save = True
        if not is_evaluate and episode_reward > self.best_train_reward:
            self.best_train_reward = episode_reward
            save = True
        if save:
            save_traj_dir = osp.join(self.output_dir, f'{tag}_saved_trajs')
            logging_path = os.path.join(self.output_dir, f'{tag}_output.txt')
            if not osp.exists(save_traj_dir):
                os.makedirs(save_traj_dir)

            # human_trajs = [self.human_mat[id, 0:self.global_timestep + 1, 0:3]
            #                for id in range(self.num_human)]  # shape = (num_timestep+1, 3)
            # t=0！
            human_trajs = [np.tile(self.human_mat[id, 0, 0:3], (self.num_timestep + 1, 1)) for id in range(self.num_human)]

            np.savez(osp.join(save_traj_dir, f'eps{total_steps}.npz'), np.array(human_trajs), np.array(self.saved_uav_trajs), np.array(self.saved_car_trajs))
            np.savez(osp.join(save_traj_dir, 'eps_best.npz'), np.array(human_trajs), np.array(self.saved_uav_trajs), np.array(self.saved_car_trajs))

            with open(logging_path, 'a') as f:
                f.write('[' + datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S') + ']\n')
                f.write(f"best trajs have been changed in ts={total_steps}. "
                        f"best_{tag}_reward: {'%.3f' % self.best_eval_reward if tag == 'eval' else '%.3f' % self.best_train_reward} "
                        f"collect_data_ratio: {'%.3f' % np.sum(collect_data_ratio_list)} "
                        f"loss_ratio: {'%.3f' % np.mean(loss_ratio_list)} "
                        f"energy_consumption_ratio: {'%.3f' % np.mean(energy_consumption_ratio_list)} "
                        + '\n'
                        )
                if 'svo' in svo_dict.keys():  # copo1
                    svos = svo_dict['svo']
                    f.write(f'svo degree corresponds to best model: {np.round([svo.item() * 90 for svo in svos], 3)}' + '\n')
                else:  # copo2
                    phis, thetas = svo_dict['phi'], svo_dict['theta']
                    f.write(f'phi degree corresponds to best model: {np.round([phi.item() * 90 for phi in phis], 3)}' + '\n')
                    f.write(f'theta degree corresponds to best model: {np.round([theta.item() * 360 for theta in thetas], 3)}' + '\n')

        return save

    def summarize_fairness(self):
        Dp = self.env_config['max_data_amount']
        fenzi = sum([((Dp - human.data) / Dp) for human in self.humans]) ** 2
        fenmu = self.num_human * sum([((Dp - human.data) / Dp) ** 2 for human in self.humans]) + 1e-2
        fairness = fenzi / fenmu
        return fairness

    def summarize_uav_util_factor(self):
        uav_util_factor = (np.mean(self.epmean_utpt) + np.mean(self.epmean_ctpt)) / np.mean(self.epmean_ctpt)
        return uav_util_factor

    def render(self, mode='human'):
        pass

    def close(self):
        pass









