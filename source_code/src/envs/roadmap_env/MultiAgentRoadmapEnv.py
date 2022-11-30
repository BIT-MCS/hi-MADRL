# numpy.arraydf
import json
import numpy as np
import osmnx as ox
import networkx as nx
from src.envs.roadmap_env.BaseRoadmapEnv import BaseRoadmapEnv
from src.envs.noma_env.state3 import JointState
from src.envs.noma_env.noma_utils import *
from src.envs.noma_env.action_space import car_env_actions


# def print(*args, **kwargs):  #
#     pass

class MultiAgentRoadmapEnv(BaseRoadmapEnv):

    def __init__(self, env_config):
        super().__init__(env_config)
        if self.use_copo:
            if self.add_svo_in_obs:  # svo
                self.obs_dim += 2 if self.use_ball_svo else 1
            else:
                # print("current algo is COPO, but we don't add svo in obs.")
                pass

        postfix = '' if self.gr == 0 else '_grid' + str(self.gr)
        with open(self.roadmap_dir + f'/pair_dis_dict{postfix}.json', 'r') as f:
            self.pair_dis_dict = json.load(f)

    def get_available_action_for_each_car_type2(self):
        num_action = self.action_space['car'].n
        mask = np.zeros((self.num_car, num_action))


        # step2. pair_dis_dictsorted10mask=1
        for id in range(self.num_car):
            #    index
            src_node = self.car_cur_node[id]
            #
            assert (self.cars[id].px, self.cars[id].py) == self.rm.lonlat2pygamexy(self.G.nodes[src_node]['x'], self.G.nodes[src_node]['y'])

            pairs = list(zip(self.pair_dis_dict[str(src_node)].keys(), self.pair_dis_dict[str(src_node)].values()))
            near_set = sorted(pairs, key=lambda x: x[1])[:num_action]  # 1key. act_sortkey

            for act in range(num_action):
                dst_node, length = near_set[act]
                dst_node = int(dst_node)
                if length != np.inf:
                    mask[id][act] = 1
                    self.last_dst_node[id][act] = dst_node
                    self.last_length[id][act] = length
                    self.last_dst_lonlat[id][act] = (self.G.nodes[dst_node]['x'], self.G.nodes[dst_node]['y'])

        return mask

    def get_mask(self):
        mask = self.get_available_action_for_each_car_type2()
        return mask

    def reset(self):
        obs = super().reset()
        return obs, self.get_mask()

    def step(self, action_dict):
        # action_dict{'uav1': [0.5, 0.2], 'uav2': [0.1, 0.9], 'car1': 5}
        rewards, infos = super().step(action_dict)
        if self.reward_scale != 1:
            print('reward_scale!')
            for key in rewards.keys():
                rewards[key] *= self.reward_scale
        state = JointState(self.uavs, self.cars, self.humans)
        done = True if self.global_timestep >= self.num_timestep else False
        dones = {"__all__": done}
        if done:  # episodeinfo
            for i, agent_name in enumerate(self.agent_name_list):
                infos[agent_name] = {'loss_ratio': self.loss_ratio_list[i] / (self.num_timestep * self.num_subchannel * 2),
                                     'collect_data_ratio': self.collect_data_ratio_list[i],
                                     'energy_consumption_ratio': self.energy_consumption_ratio_list[i],
                                     # agent
                                     'fairness': self.summarize_fairness(),
                                     'uav_util_factor': self.summarize_uav_util_factor(),
                                     # 'avail_act_prop': np.mean(self.avail_act_prop_list[i-self.num_uav]) if not self.is_uav(i) else 0
                                     }

            # print('', np.sum(self.collect_data_ratio_list))
            # print('', np.mean(self.loss_ratio_list / (self.num_timestep * self.num_subchannel * 2)))
            # print('', np.mean(self.energy_consumption_ratio_list[i]))
            # print('avail_act_prop: ', np.mean(self.avail_act_prop_list))
            # print('i-uav:', np.mean(self.ep_succ_i_uav_dis))
            # print('j-car:', np.mean(self.ep_succ_j_car_dis))
            # print('uav-car:', np.mean(self.ep_succ_uav_car_dis))

        return self._get_step_return(state), self.get_mask(), rewards, dones, infos







