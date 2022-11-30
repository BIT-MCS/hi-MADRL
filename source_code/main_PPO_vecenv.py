import time
import pandas as pd
import sys
import numpy as np
import torch
from tensorboardX import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
import json
import copy
from arguments import add_args
from src.utils.my_utils import *
from src.config.main_config import rllib_env_config
from src.config.main_copo_config import main_copo_config
from src.envs.roadmap_env.MultiAgentRoadmapEnv import MultiAgentRoadmapEnv
from src.envs.ccenv import get_ccenv
from src.envs.roadmap_env.DummyEnv import DummyEnv
from src.envs.wrappers.env_wrappers import SubprocVecEnv

def evaluate_policy(envs, dummy_env, agent, total_timesteps, writer, test, use_eoi, num_episode=1):
    ALLTIMESTART = time.time()
    n_rollout_threads = 1  #
    rscale = []
    ''' inspection rewardEOI num_episode 
    if test and use_eoi:
        eoi_predict_accuracy = [[] for _ in range(dummy_env.num_agent)]
    # ==
    '''
    MoreTestEpisodesInfos = []
    average_ep_r = []
    for ep in range(num_episode):
        seed = 2022 + 10 * ep  # episodeseed
        torch.manual_seed(seed)
        np.random.seed(seed)

        obs_dict, mask = envs.reset()
        ep_r = np.array([0.0 for _ in range(n_rollout_threads)])
        obs = np.array(di2li_vec(obs_dict))
        global_done = False
        while not global_done:
            if test:
                a_list, _ = agent.select_action_for_vec_2(obs, mask, mode="run")
            else:
                a_list, _ = agent.select_action_for_vec_2(obs, mask, mode="eval")  # Take deterministic actions at test time
            action_dict = li2di_vec(a_list, dummy_env.agent_name_list)
            obs_prime, mask_prime, r_dict, done_dict, info = envs.step(action_dict)
            done = [done_di["__all__"] for done_di in done_dict]
            global_done = done[0]
            obs_prime, rewards = di2li_vec(obs_prime, r_dict)
            obs_prime, rewards = np.array(obs_prime), np.array(rewards)
            rscale += list(rewards[0])
            '''
            if test and use_eoi:  # debug: eoi_net OK
                for i in range(dummy_env.num_agent):
                    to_pred = torch.tensor(obs_prime[0][i]).unsqueeze(0).to(device)
                    acc = agent.eoi_net(to_pred).squeeze(0)[i].item()
                    eoi_predict_accuracy[i].append(acc)
            '''
            obs = obs_prime
            mask = mask_prime
            ep_r += np.sum(rewards, axis=-1)
        MoreTestEpisodesInfos.append(info[0])
        average_ep_r.append(ep_r[0])

    if num_episode == 1:
        saved_uav_trajs, saved_car_trajs = envs.get_saved_trajs()
        if hasattr(agent, 'svo'):
            svo_dict = {'svo': agent.svo}
        elif hasattr(agent, 'phi'):
            svo_dict = {'phi': agent.phi, 'theta': agent.theta}
        else:
            svo_dict = None
        is_newbest = dummy_env.callback(total_timesteps, writer, ep_r[0],
                                        saved_uav_trajs[0], saved_car_trajs[0],
                                        last_info=info[0], svo_dict=svo_dict, is_evaluate=True, test=test)
        if is_newbest and not test:
            agent.save(total_timesteps, is_evaluate=True, is_newbest=True)
    else:
        if test:
            dummy_env.callback_MoreTestEpisodes(MoreTestEpisodesInfos)
        else:  # 4/15 average eval reward
            saved_uav_trajs, saved_car_trajs = envs.get_saved_trajs()  #
            svo_dict = None
            is_newbest = dummy_env.callback(total_timesteps, writer, np.mean(average_ep_r),
                                        saved_uav_trajs[0], saved_car_trajs[0],
                                        last_info=info[0], svo_dict=svo_dict, is_evaluate=True, test=test)  # info
            if is_newbest and not test:
                agent.save(total_timesteps, is_evaluate=True, is_newbest=True)

    # print('reward:', np.mean(rscale))
    '''
    if test and use_eoi:
        eoi_predict_accuracy = np.round(list(map(np.mean, eoi_predict_accuracy)), 2)
        print(f"episodeagent{eoi_predict_accuracy}")
        print(f"{dummy_env.num_agent}agent{'%.2f' % (1 / dummy_env.num_agent)}, "
              f"{['%.2f' % (acc - 1 / dummy_env.num_agent) for acc in eoi_predict_accuracy]}")
    '''
    ALLTIMEEND = time.time()
    print('ALLTIME:', ALLTIMEEND - ALLTIMESTART)
    return np.mean(average_ep_r)


def main(args):
    test, debug = args.test, args.debug

    if args.use_hcopo:
        args.use_copo = True
        args.copo_kind = 2

    if test:  # params.jsonargs
        tmp_seed = args.seed  # testseed
        tmp_output_dir = args.output_dir
        tmp_num_test_episode = args.num_test_episode
        json_file = os.path.join(args.output_dir, 'params.json')
        with open(json_file, 'r') as f:
            params = json.load(f)
        args = argparse.Namespace(**{**vars(args), **params['args']})
        # test
        args.test = True
        args.seed = tmp_seed
        args.output_dir = tmp_output_dir
        args.num_test_episode = tmp_num_test_episode

    # default
    if args.dataset == 'purdue':
        args.roadmap_dir = 'drive_service'
        args.config_dir = 'purdue100_2u2c_QoS1'
        args.setting_dir = 'purdue100cluster'
    elif args.dataset == 'NCSU':
        args.roadmap_dir = 'drive'
        args.config_dir = 'NCSU100_2u2c_QoS1'
        args.setting_dir = 'NCSU100cluster'

    if debug:
        args.output_dir = '../runs/debug'
        args.Max_train_steps = 100
        args.T_horizon = 10
        args.eval_interval = 50
        args.save_interval = 20
        args.n_rollout_threads = 2

    if test:
        args.n_rollout_threads = 1
        args.eval_interval = 1

    if args.gr == 50:
        args.type2_act_dim = 20
    elif args.gr == 100:
        args.type2_act_dim = 15
    elif args.gr == 200:
        args.type2_act_dim = 10

    use_eoi, use_copo, use_ccobs, eoi_kind, copo_kind, share_parameter = args.use_eoi, args.use_copo, args.use_ccobs, args.eoi_kind, args.copo_kind, args.share_parameter
    n_rollout_threads = args.n_rollout_threads

    if not test:
        timestr = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
        output_dir = args.output_dir + '/' + timestr
        # if args.n_rollout_threads != 32:
        #     output_dir += f'_threads={args.n_rollout_threads}'
        if use_eoi:
            output_dir += f'_UseEoi{args.eoi_kind}'
        if use_copo:
            output_dir += f'_UseCopo{args.copo_kind}'
        if use_copo and args.copo_kind == 2:
            if args.HID_phi != [0, 0]:
                output_dir += f'_HIDPhi={args.HID_phi}'
            if args.HID_theta != [45, 45]:
                output_dir += f'_HIDTheta={args.HID_theta}'
            if args.hcopo_shift:
                output_dir += f'_HShift'
            if args.hcopo_shift_513:
                output_dir += f'_HShift513'
        if share_parameter:
            output_dir += '_ShareParam'
        if args.share_layer:
            output_dir += '_ShareLayer'
        if args.use_ccobs:
            output_dir += '_CCobs'
        if args.dist != 'Beta':
            output_dir += f'_{args.dist}'
        if args.type2_act_dim != 10:
            output_dir += f'_Act{args.type2_act_dim}'
        if args.gr != 200:
            output_dir += f'_GR={args.gr}'
        if args.eoi_coef_decay != 1.0:  # 4/4 try2
            output_dir += f'_EoiCoefDecay={args.eoi_coef_decay}'
        if args.entropy_coef != 1e-3:
            output_dir += f'_EntropyCoef={args.entropy_coef}'
        if args.batch_size != 64:
            output_dir += f'_BatchSize={args.batch_size}'
        if args.net_width != 150:
            output_dir += f'_NetWidth={args.net_width}'
        if args.svo_lr != 1e-4:
            output_dir += f'_SVOLR={args.svo_lr}'
        if args.nei_dis_scale != 0.25:
            output_dir += f'_NeiDisScale={args.nei_dis_scale}'
        if not args.hcopo_sqrt2_scale:
            output_dir += f'_NotHcopoSqrt2Scale'
        if args.svo_frozen:
            output_dir += f'_SvoFrozen'


        if args.num_uv is not None:
            output_dir += f'_NU={args.num_uv}'
        if args.sinr_demand is not None:
            output_dir += f'_SD={args.sinr_demand}'
        if args.num_subchannel is not None:
            output_dir += f'_NS={args.num_subchannel}'
        if args.uav_height is not None:
            output_dir += f'_UH={args.uav_height}'

        if os.path.exists(output_dir): shutil.rmtree(output_dir)
        writer = SummaryWriter(log_dir=output_dir)
        args.output_dir = output_dir
    else:
        writer = None
    # == rllib_env_conf==
    args = fillin_lazy_args(args, dataset_str=args.dataset)
    my_env_config = import_env_config(args.config_dir)
    my_env_config = env_config_wrapper(my_env_config, args.num_uv, args.sinr_demand, args.num_subchannel, args.uav_height)
    rllib_env_config['my_env_config'] = my_env_config
    rllib_env_config['args'] = args

    if use_copo:
        rllib_env_config.update(main_copo_config)

    # rllibparams.jsonvis
    if not test:
        tmp_dict = copy.deepcopy(rllib_env_config)
        tmp_dict['args'] = vars(tmp_dict['args'])  # Namespaceargsdictparams.json
        tmp_dict['setting_dir'] = args.setting_dir
        with open(os.path.join(output_dir, 'params.json'), 'w') as f:
            f.write(json.dumps(tmp_dict))

    env_class = get_ccenv(MultiAgentRoadmapEnv)
    def make_vec_env(threads):
        return SubprocVecEnv([env_class(rllib_env_config) for _ in range(threads)])

    envs = make_vec_env(n_rollout_threads)
    eval_envs = make_vec_env(1)
    dummy_env = get_ccenv(DummyEnv)(rllib_env_config)  # 1.env（num_uav）2.callback
    n_uav = dummy_env.num_uav
    n_car = dummy_env.num_car
    max_steps = dummy_env.num_timestep
    T_horizon = args.T_horizon  # lenth of long trajectory
    print('max_steps', max_steps, 'T_horizon', T_horizon)
    Max_train_steps = args.Max_train_steps
    save_interval = args.save_interval  # in steps
    eval_interval = args.eval_interval  # in steps
    assert test or n_rollout_threads < T_horizon and n_rollout_threads < eval_interval and n_rollout_threads < save_interval  # ts//

    random_seed = args.seed
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    kwargs = {
        "output_dir": args.output_dir,
        "device": device,
        "writer": writer,
        "T_horizon": T_horizon,
        "n_rollout_threads": n_rollout_threads,
        "share_parameter": share_parameter,
        "share_layer": args.share_layer,
        "use_eoi": use_eoi,
        "use_copo": use_copo,
        "use_ccobs": use_ccobs,
        "eoi_kind": eoi_kind,
        "copo_kind": copo_kind,
        "svo_lr": args.svo_lr,
        "hcopo_shift": args.hcopo_shift,
        "hcopo_shift_513": args.hcopo_shift_513,
        "obs_dim": envs.observation_space.shape[0],
        "uav_continuous_action_dim": envs.action_space['uav'].shape[0],
        "car_discrete_action_dim": envs.action_space['car'].n,
        "n_agent": dummy_env.num_agent,
        "n_uav": n_uav,
        "n_car": n_car,
         "env_with_Dead": False,  # dummy
        "gamma": args.gamma,
        "lambd": args.lambd,  # For GAE
        "clip_rate": args.clip_rate,  # 0.2
        "K_epochs": args.K_epochs,
        "W_epochs": args.W_epochs,
        "net_width": args.net_width,
        "a_lr": args.a_lr,
        "c_lr": args.c_lr,
        "dist": args.dist,
        "l2_reg": args.l2_reg,  # L2 regulization for Critic
        "a_optim_batch_size": args.batch_size,
        "c_optim_batch_size": args.batch_size,
        "entropy_coef": args.entropy_coef,  # Entropy Loss for Actor: Large entropy_coef for large exploration, but is harm for convergence.
        "entropy_coef_decay": args.entropy_coef_decay,
        "vf_coef": args.vf_coef,  # TBD rllib
        "eoi3_coef": args.eoi3_coef,
        "eoi_coef_decay": args.eoi_coef_decay,
        # "use_o_prime_compute_ir": args.use_o_prime_compute_ir,
        "HID_phi": args.HID_phi,
        "HID_theta": args.HID_theta,
        "hcopo_sqrt2_scale": args.hcopo_sqrt2_scale,
        "svo_frozen": args.svo_frozen
    }
    # if Dist[distnum] == 'Beta' :
    #     kwargs["a_lr"] *= 2 #Beta dist need large lr|maybe
    #     kwargs["c_lr"] *= 4  # Beta dist need large lr|maybe

    from src.algo_roadmap.hybrid_ppo_vecenv import VecHybridPPO
    from src.algo_roadmap.hybrid_copo_sharelayer_vecenv import VecHybridShareLayerCopo
    if use_copo and args.share_layer:
        agent = VecHybridShareLayerCopo(kwargs)
    else:
        agent = VecHybridPPO(kwargs)

    if test:
        # test the memory usage
        from torchstat import stat
        model = agent.actor[0]
        stat(model, (100, 4, 316))  # 316 is the hard code of obs_dim  # 100 = timesteps, 4 = num_agents

        load_dir = os.path.join(args.output_dir, 'model/train/best_model')
        agent.load(load_dir=load_dir,
                   timestep=get_load_timestep(load_dir))

    train_count, eval_count, save_count, print_count = 0, 0, 0, 0
    total_steps = 0
    while total_steps < Max_train_steps:
        if print_count // (Max_train_steps // 20) >= 1:  # 20ts
            print(f'timestep = {total_steps}...')
            print_count -= (Max_train_steps // 20)

            print(f'timesteps {total_steps} ...')
        obs_dict, mask = envs.reset()
        obs = np.array(di2li_vec(obs_dict))  # shape = (n_rollout_threads, num_agent)
        ep_r = np.array([0.0 for _ in range(n_rollout_threads)])
        global_done = False
        my_start_1 = time.time()
        '''Interact & train'''
        while not global_done:
            my_start_5 = time.time()
            a_list, logprob_a_list = agent.select_action_for_vec_2(obs, mask, mode='run')

            my_end_5 = time.time()
            if writer:
                writer.add_scalar('time/select_action_time_sec', my_end_5 - my_start_5, total_steps)
            action_dict = li2di_vec(a_list, dummy_env.agent_name_list)
            obs_prime_dict, mask_prime, r_dict, done_dict, info = envs.step(action_dict)
            # shape = (n_rollout_threads, num_agent), mask_primeshape = (n_rollout_threads, num_car)
            assert pd.Series(done_dict).value_counts().size == 1  # # envagentdone
            done = [done_di["__all__"] for done_di in done_dict]
            global_done = done[0]
            obs_prime, rewards = di2li_vec(obs_prime_dict, r_dict)
            obs_prime, rewards = np.array(obs_prime), np.array(rewards)
            nei_r = np.array([[inf[key]['nei_rewards'] for key in inf.keys()] for inf in info])  # shape=(threads, n_agent) rewardsshape
            uav_r = np.array([[inf[key]['uav_rewards'] for key in inf.keys()] for inf in info])
            car_r = np.array([[inf[key]['car_rewards'] for key in inf.keys()] for inf in info])
            global_r = np.array([[inf[key]['global_rewards'] for key in inf.keys()] for inf in info])
            data = (obs, mask, a_list, rewards, obs_prime, logprob_a_list, done, nei_r, uav_r, car_r, global_r)
            agent.put_data(data)
            obs = obs_prime
            mask = mask_prime
            ep_r += np.sum(rewards, axis=-1)  # threadreward

            print_count += n_rollout_threads  # change
            train_count += n_rollout_threads  # change
            eval_count += n_rollout_threads  # change
            save_count += n_rollout_threads  # change
            total_steps += n_rollout_threads  # change
            '''train'''
            if not test:
                if train_count // T_horizon >= 1:
                    my_end_1 = my_start_2 = time.time()
                    train_count -= T_horizon

                    agent.train(total_steps)
                    if writer:
                        writer.add_scalar('time/rollout_time_sec', my_end_1 - my_start_1, total_steps)
                    my_start_1 = my_end_2 = time.time()
                    if writer:
                        writer.add_scalar('time/train_time_sec', my_end_2 - my_start_2, total_steps)

            '''evaluate & log'''
            if eval_count // eval_interval >= 1:
                my_start_4 = time.time()
                eval_count -= eval_interval

                eval_ep_r = evaluate_policy(eval_envs, dummy_env, agent, total_steps, writer, test, use_eoi, num_episode=args.num_test_episode)
                print('[' + datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S") + ']')
                print('seed:', random_seed, 'steps: {}k'.format(int(total_steps / 1000)), 'eval episode reward:', eval_ep_r)
                if test:
                    sys.exit(0)
                my_end_4 = time.time()
                if writer:
                    writer.add_scalar('time/eval_time_sec', my_end_4 - my_start_4, total_steps)

            '''routinely save model if train mode'''

            if not test and save_count // save_interval >= 1:
                my_start_3 = time.time()
                agent.save(total_steps, is_evaluate=False)
                save_count -= save_interval

                my_end_3 = time.time()
                if writer:
                    writer.add_scalar('time/save_time_sec', my_end_3 - my_start_3, total_steps)

        if not test:
            # reward
            saved_uav_trajs, saved_car_trajs = envs.get_saved_trajs()
            if hasattr(agent, 'svo'):
                svo_dict = {'svo': agent.svo}
            elif hasattr(agent, 'phi'):
                svo_dict = {'phi': agent.phi, 'theta': agent.theta}
            else:
                svo_dict = None
            # metricinfoep_rrllibmaxmeanminmax
            is_newbest = dummy_env.callback(total_steps, writer, np.max(ep_r),
                                            saved_uav_trajs[np.argmax(ep_r)], saved_car_trajs[np.argmax(ep_r)],
                                            last_info=info[np.argmax(ep_r)], svo_dict=svo_dict, is_evaluate=False)
            if is_newbest:
                agent.save(total_steps, is_evaluate=False, is_newbest=True)

    print('！')

    envs.close()  # pass

if __name__ == '__main__':
    '''Hyperparameter Setting'''
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    print(args)

    if not args.test and args.cuda:
        print(f'choose use gpu {args.gpu_id}...')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        print('choose use cpu...')
        device = torch.device("cpu")

    main(args)
