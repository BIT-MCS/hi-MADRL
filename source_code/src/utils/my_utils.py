import os
import sys
import importlib
import argparse
import re

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_env_config(config_dir):
    spec = importlib.util.spec_from_file_location('config', config_dir)
    if spec is None:
        print('Config file not found.')
        exit(0)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    env_config = config.env_config
    return env_config

def fillin_lazy_args(args, dataset_str):
    if hasattr(args, 'config_dir'):
        args.config_dir = f'src/config/roadmap_config/{dataset_str}/env_config_' + args.config_dir + '.py'
    if hasattr(args, 'setting_dir'):
        args.setting_dir = f'src/envs/roadmap_env/setting/{dataset_str}/' + args.setting_dir
    if hasattr(args, 'roadmap_dir'):
        args.roadmap_dir = f'data/{dataset_str}/' + args.roadmap_dir
    return args

def env_config_wrapper(env_config, num_uv=None, sinr_demand=None, num_subchannel=None, uav_height=None):
    if num_uv is not None:
        assert env_config['num_uav'] == env_config['num_car']
        if env_config['num_uav'] != num_uv:
            env_config['name'] += f'_wrappedNU{num_uv}'
        env_config['num_uav'] = num_uv
        env_config['num_car'] = num_uv

    if sinr_demand is not None:
        if sinr_demand != env_config['sinr_demand']:
            env_config['name'] += f'_wrappedSD{sinr_demand}'
        env_config['sinr_demand'] = sinr_demand

    if num_subchannel is not None:
        if sinr_demand != env_config['num_subchannel']:
            env_config['name'] += f'_wrappedNS{num_subchannel}'
        env_config['num_subchannel'] = num_subchannel

    if uav_height is not None:
        if uav_height != env_config['uav_init_height']:
            env_config['name'] += f'_wrappedUH{uav_height}'
        env_config['uav_init_height'] = uav_height

    return env_config


def get_load_timestep(load_dir):

    pattern1 = r"ep\d+_"

    files = os.listdir(load_dir)
    for file in files:
        ans = re.search(pattern1, file)
        if ans is not None:
            load_ts = file[ans.start() + 2:ans.end() - 1]
            return int(load_ts)

    return -1



def li2di(a, agent_name_list):
    action_dict = {}
    for i, name in enumerate(agent_name_list):
        action_dict[name] = a[i]
    return action_dict

def li2di_vec(a, agent_name_list):
    n_rollout_threads = len(a)
    action_dict = []
    for e in range(n_rollout_threads):
        action_d = {}
        for i, name in enumerate(agent_name_list):
            action_d[name] = a[e][i]
        action_dict.append(action_d)
    return action_dict

def di2li(*args):
    out = [list(arg.values()) for arg in args]
    return out if len(out) >= 2 else out[0]

def di2li_vec(*args):
    # for arg in args:
    #     print(type(arg))  # np.array
    out = [[list(arg[i].values()) for i in range(len(arg))] for arg in args]
    return out if len(out) >= 2 else out[0]