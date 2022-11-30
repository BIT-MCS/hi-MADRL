# 
# 1.env（num_uav）
# 2.callback

import copy
import torch
import osmnx as ox
import networkx as nx
import os
import os.path as osp
import pandas as pd
from datetime import datetime
from src.config.model_config import model_config
from src.envs.noma_env.state3 import UAVState, CarState, HumanState, JointState
from src.envs.noma_env.utils import *
from src.envs.noma_env.noma_utils import *
from src.envs.roadmap_env.roadmap_utils import Roadmap
from src.envs.roadmap_env.BaseRoadmapEnv import BaseRoadmapEnv
from tools.macro.macro import *

project_dir = osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__))))  # base.py

# def print(*args, **kwargs):  #
#     pass

class DummyEnv(BaseRoadmapEnv):
    def __init__(self, rllib_env_config):
        super().__init__(rllib_env_config)

    def callback(self, total_steps, writer, episode_reward,
                 saved_uav_trajs, saved_car_trajs,
                 last_info, svo_dict, is_evaluate, test=False):
        '''
        callback saved_uav_trajssaved_car_trajs
        '''

        tag = 'eval' if is_evaluate else 'train'
        '''step1. tensorboardrewardmetric'''
        loss_ratio_list = []
        collect_data_ratio_list = []
        energy_consumption_ratio_list = []
        have_nei_freq_list = []
        fairness = None
        for k in self.agent_name_list:
            loss_ratio_list.append(last_info[k]['loss_ratio'])
            collect_data_ratio_list.append(last_info[k]['collect_data_ratio'])
            energy_consumption_ratio_list.append(last_info[k]['energy_consumption_ratio'])
            have_nei_freq_list.append(last_info[k].get('have_nei_freq', 0.0))  # IPPO metric0
            if fairness is None:
                fairness = last_info[k]['fairness']
            else:
                assert fairness == last_info[k]['fairness']

        if test:
            logging_path = os.path.join(self.output_dir, f'test_output.txt')
            with open(logging_path, 'a') as f:
                f.write('[' + datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S') + ']\n')
                f.write(f"ts={total_steps}. "
                        f"test_reward: {'%.3f' % episode_reward} "
                        f"collect_data_ratio: {'%.3f' % np.sum(collect_data_ratio_list)} "
                        f"loss_ratio: {'%.3f' % np.mean(loss_ratio_list)} "
                        f"fairness: {'%.3f' % fairness} "
                        f"energy_consumption_ratio: {'%.3f' % np.mean(energy_consumption_ratio_list)} "
                        + '\n'
                        )
            print(
                f":",
                f"test_reward: {'%.3f' % episode_reward} ",
                f"collect_data_ratio: {'%.3f' % np.sum(collect_data_ratio_list)} ",
                f"loss_ratio: {'%.3f' % np.mean(loss_ratio_list)} ",
                f"fairness: {'%.3f' % fairness} ",
                f"energy_consumption_ratio: {'%.3f' % np.mean(energy_consumption_ratio_list)} ",
            )

        else:
            writer.add_scalar(f'{tag}/episode_reward', episode_reward, total_steps)
            writer.add_scalar(f'{tag}/collect_data_ratio', np.sum(collect_data_ratio_list), total_steps)  # 0-1
            writer.add_scalar(f'{tag}/loss_ratio', np.mean(loss_ratio_list), total_steps)  # >0
            writer.add_scalar(f'{tag}/energy_consumption_ratio', np.mean(energy_consumption_ratio_list), total_steps)  # 0-1
            writer.add_scalar(f'{tag}/fairness', fairness, total_steps)
            writer.add_scalar(f'{tag}/have_nei_freq', np.mean(have_nei_freq_list), total_steps)  # 0-1
            # writer.add_scalar(f'{tag}/avail_act_prop', np.mean(avail_act_prop_list), total_steps)

        '''step2. save best trajs'''
        if test: return False

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

            np.savez(osp.join(save_traj_dir, f'eps{total_steps}.npz'), np.array(human_trajs), np.array(saved_uav_trajs), np.array(saved_car_trajs))
            np.savez(osp.join(save_traj_dir, 'eps_best.npz'), np.array(human_trajs), np.array(saved_uav_trajs), np.array(saved_car_trajs))

            with open(logging_path, 'a') as f:
                f.write('[' + datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S') + ']\n')
                f.write(f"best trajs have been changed in ts={total_steps}. "  
                        f"best_{tag}_reward: {'%.3f' % self.best_eval_reward if tag == 'eval' else '%.3f' % self.best_train_reward} "
                        f"efficiency: {'%.3f' % (np.sum(collect_data_ratio_list) * (1 - np.mean(loss_ratio_list)) * fairness / np.mean(energy_consumption_ratio_list))} "
                        f"collect_data_ratio: {'%.3f' % np.sum(collect_data_ratio_list)} "
                        f"loss_ratio: {'%.3f' % np.mean(loss_ratio_list)} "
                        f"fairness: {'%.3f' % fairness} "
                        f"energy_consumption_ratio: {'%.3f' % np.mean(energy_consumption_ratio_list)} "
                        + '\n'
                        )
                if svo_dict is not None:
                    if 'svo' in svo_dict.keys():  # copo1
                        svos = svo_dict['svo']
                        f.write(f'svo degree corresponds to best model: {np.round([svo.item() * 90 for svo in svos], 3)}' + '\n')
                    else:  # copo2
                        phis, thetas = svo_dict['phi'], svo_dict['theta']
                        if self.args.hcopo_shift:
                            f.write(f'phi degree corresponds to best model: {np.round([phi.item() * 180 for phi in phis], 3)}' + '\n')
                            f.write(f'theta degree corresponds to best model: {np.round([theta.item() * 180 - 45 for theta in thetas], 3)}' + '\n')
                        elif self.args.hcopo_shift_513:
                            f.write(f'phi degree corresponds to best model: {np.round([phi.item() * 180 - 90 for phi in phis], 3)}' + '\n')
                            f.write(f'theta degree corresponds to best model: {np.round([theta.item() * 180 - 45 for theta in thetas], 3)}' + '\n')
                        else:
                            f.write(f'phi degree corresponds to best model: {np.round([phi.item() * 90 for phi in phis], 3)}' + '\n')
                            f.write(f'theta degree corresponds to best model: {np.round([theta.item() * 360 for theta in thetas], 3)}' + '\n')

        return save

    def callback_MoreTestEpisodes(self, MoreTestEpisodesInfos):
        num_episode = len(MoreTestEpisodesInfos)

        '''test episodemetrics'''
        ratios = np.empty((num_episode,))  # for each episodes
        losses = np.empty((num_episode,))
        energies = np.empty((num_episode,))
        fairnesses = np.empty((num_episode,))
        UUFs = np.empty((num_episode,))

        for episode in range(num_episode):
            last_info = MoreTestEpisodesInfos[episode]
            collect_data_ratio_list = []
            loss_ratio_list = []  # for each agents
            energy_consumption_ratio_list = []
            fairness = None
            for k in self.agent_name_list:
                loss_ratio_list.append(last_info[k]['loss_ratio'])
                collect_data_ratio_list.append(last_info[k]['collect_data_ratio'])
                energy_consumption_ratio_list.append(last_info[k]['energy_consumption_ratio'])
                if fairness is None:
                    fairness = last_info[k]['fairness']
                else:
                    assert fairness == last_info[k]['fairness']

            ratios[episode] = np.sum(collect_data_ratio_list)
            losses[episode] = np.mean(loss_ratio_list)
            energies[episode] = np.mean(energy_consumption_ratio_list)
            fairnesses[episode] = fairness

        # test_episodeefficiency
        def get_effs():
            e1 = ratios * (1 - np.array(losses)) * UUFs / energies
            e2 = ratios * (1 - losses) * fairnesses / energies
            return e1, e2
        efficiency1s, efficiency2s = get_effs()

        '''episodemeanmaxepisodecsv'''
        # logging_path = os.path.join(self.output_dir, f'test_result/{num_episode}times_test_output.txt')
        result_dir = os.path.join(self.output_dir, 'test_result')
        if not os.path.exists(result_dir): os.makedirs(result_dir)
        path1 = result_dir + f'/{num_episode}times_statistical_metrics.csv'
        path2 = result_dir + f'/{num_episode}times_detail_metrics.csv'

        df1 = pd.DataFrame(index=('mean', 'max', 'std'), columns=METRICS_WITH_EFFICIENCY)
        df1.loc['mean'] = [np.mean(ratios), np.mean(losses), np.mean(energies), np.mean(fairnesses), np.mean(UUFs),
                           np.mean(efficiency1s), np.mean(efficiency2s)]
        df1.loc['max'] = [np.max(ratios), np.max(losses), np.max(energies), np.max(fairnesses), np.max(UUFs),
                           np.max(efficiency1s), np.max(efficiency2s)]
        df1.loc['std'] = [np.std(ratios), np.std(losses), np.std(energies), np.std(fairnesses), np.std(UUFs),
                           np.std(efficiency1s), np.std(efficiency2s)]
        df1.to_csv(path1)

        data = np.hstack((ratios.reshape(-1, 1),
                   losses.reshape(-1, 1),
                   energies.reshape(-1, 1),
                   fairnesses.reshape(-1, 1),
                   UUFs.reshape(-1, 1),
                   efficiency1s.reshape(-1, 1),
                   efficiency2s.reshape(-1, 1),
                   ))
        df2 = pd.DataFrame(data, columns=METRICS_WITH_EFFICIENCY)
        df2.to_csv(path2)

        print(f'successfully write {num_episode}times test result to storage')


    def render(self, mode='human'):
        pass









