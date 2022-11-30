import copy
import os
import torch
import shutil
import math
import numpy as np
from torch.nn import Parameter
from src.algo_roadmap.net.network import EOI_Net
from src.algo_roadmap.utils import normalize


class Hybrid_PPO_base():
    def __init__(self,
                 output_dir,
                 device,
                 writer,
                 T_horizon,
                 n_rollout_threads,
                 share_parameter,
                 share_layer,
                 use_ccobs,
                 use_eoi,
                 use_copo,
                 eoi_kind,
                 copo_kind,
                 hcopo_shift,
                 hcopo_shift_513,
                 obs_dim,
                 uav_continuous_action_dim,
                 car_discrete_action_dim,
                 n_agent,
                 n_uav,
                 n_car,
                 env_with_Dead,
                 gamma=0.99,
                 lambd=0.95,
                 clip_rate=0.2,
                 K_epochs=10,
                 W_epochs=5,
                 net_width=256,
                 a_lr=3e-4,
                 c_lr=3e-4,  # mute
                 l2_reg=1e-3,
                 dist='GS_ms',
                 a_optim_batch_size=64,
                 c_optim_batch_size=64,  # mute
                 entropy_coef=0,  # 0.001
                 entropy_coef_decay=0.9998,
                 vf_coef=1.0,
                 # eoi
                 eoi3_coef=0.01,
                 eoi_coef_decay=1.0,
                 use_o_prime_compute_ir=False,
                 # svo
                 initial_svo_degree=0.0,
                 HID_phi=[90, 90],
                 HID_theta=[45, 45],
                 svo_lr=1e-4,
                 hcopo_sqrt2_scale=False,
                 svo_frozen=False,
                 ):

        assert eoi_kind in (1, 2, 3)

        self.output_dir = output_dir
        self.device = device
        self.writer = writer
        self.T_horizon = T_horizon
        self.n_rollout_threads = n_rollout_threads
        self.share_parameter = share_parameter
        self.share_layer = share_layer
        self.use_centralized_critc = use_ccobs
        self.use_eoi = use_eoi
        self.use_eoi3 = use_eoi and eoi_kind == 3
        self.use_copo = use_copo
        self.use_copo1 = use_copo and copo_kind == 1
        self.use_copo2 = use_copo and copo_kind == 2
        self.copo_kind = copo_kind
        self.hcopo_shift = hcopo_shift
        self.hcopo_shift_513 = hcopo_shift_513
        self.n_agent = n_agent
        self.n_uav = n_uav
        self.n_car = n_car
        self.eoi3_coef = eoi3_coef
        self.eoi_coef_decay = eoi_coef_decay
        self.use_o_prime_compute_ir = use_o_prime_compute_ir
        self.HID_phi = HID_phi
        self.HID_theta = HID_theta
        self.obs_dim = obs_dim
        self.uav_continuous_action_dim = uav_continuous_action_dim
        self.car_discrete_action_dim = car_discrete_action_dim
        self.net_width = net_width
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.dist = dist
        self.env_with_Dead = env_with_Dead
        self.clip_rate = clip_rate
        self.gamma = gamma
        self.lambd = lambd
        self.clip_rate = clip_rate
        self.K_epochs = K_epochs
        self.W_epochs = W_epochs
        self.l2_reg = l2_reg
        self.a_optim_batch_size = a_optim_batch_size
        self.c_optim_batch_size = c_optim_batch_size
        self.svo_optim_batch_size = a_optim_batch_size  #
        self.eoi_optim_batch_size = 256  # TBD
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.vf_coef = vf_coef
        self.svo_lr = svo_lr
        self.hcopo_sqrt2_scale = hcopo_sqrt2_scale
        self.initial_svo_degree = initial_svo_degree
        self.svo_frozen = svo_frozen
        self.timesteps = None
        assert not (self.hcopo_shift and self.hcopo_shift_513)

        '''EOI_net'''
        # agent
        if use_eoi:
            self.eoi_net = EOI_Net(obs_dim, n_agent).to(self.device)
            self.eoi_optimizer = torch.optim.Adam(self.eoi_net.parameters(), lr=a_lr)

        self.data = []

        '''initialize svo'''
        if self.use_copo1:
            deg2sp_tanh = {-90: -10.0, -60: -0.8045, -30: -0.3466, 0: 0.0,
                           90: 10.0, 60: 0.8045, 30: 0.3466}
            svo_init = deg2sp_tanh.get(initial_svo_degree, 0.0)
        if self.use_copo2:
            if self.hcopo_shift:
                # degree to phi_param, by sigmoid
                deg2pp_sgm = {90: 0.0, 60: -0.7}
                # degree to theta_param, by sigmoid
                deg2tp_sgm = {45: 0.0, 15: -0.7, 30: -0.35, 60: 0.35}
            elif self.hcopo_shift_513:
                deg2pp_sgm = {0: 0.0}
                deg2tp_sgm = {45: 0.0}
            else:
                deg2pp_sgm = {90: 8.0, 0: -8.0}
                deg2tp_sgm = {45: -1.9450, 135: -0.5108, 315: 1.9460}
            if self.share_parameter:
                pass
            else:
                initial_phi_degree = [self.HID_phi[0] for _ in range(self.n_uav)] + [self.HID_phi[1] for _ in range(self.n_car)]  # OK
                initial_theta_degree = [self.HID_theta[0] for _ in range(self.n_uav)] + [self.HID_theta[1] for _ in range(self.n_car)]  # OK
                phi_init = [deg2pp_sgm.get(initial_phi_degree[i]) for i in range(self.n_agent)]
                theta_init = [deg2tp_sgm.get(initial_theta_degree[i]) for i in range(self.n_agent)]
                if self.hcopo_shift:
                    print('initial value of phi(degree):', torch.sigmoid(torch.tensor(phi_init)) * 180)
                    print('initial value of theta(degree):', torch.sigmoid(torch.tensor(theta_init)) * 180 - 45)
                elif self.hcopo_shift_513:
                    print('initial value of phi(degree):', torch.sigmoid(torch.tensor(phi_init)) * 180 - 90)
                    print('initial value of theta(degree):', torch.sigmoid(torch.tensor(theta_init)) * 180 - 45)
                else:
                    print('initial value of phi(degree):', torch.sigmoid(torch.tensor(phi_init)) * 90)
                    print('initial value of theta(degree):', torch.sigmoid(torch.tensor(theta_init)) * 360)


        if self.share_parameter:
            raise ValueError()
        else:
            if self.use_copo1:
                self.svo_param = [Parameter(torch.tensor(svo_init, dtype=torch.float32).to(self.device)) for _ in range(self.n_agent)]
                self.svo_opt = [torch.optim.Adam([self.svo_param[i]], lr=self.svo_lr) for i in range(self.n_agent)]
            if self.use_copo2:
                self.phi_param = [Parameter(torch.tensor(phi_init[i], dtype=torch.float32).to(self.device)) for i in range(self.n_agent)]
                self.theta_param = [Parameter(torch.tensor(theta_init[i], dtype=torch.float32).to(self.device)) for i in range(self.n_agent)]
                self.phi_opt = [torch.optim.Adam([self.phi_param[i]], lr=self.svo_lr) for i in range(self.n_agent)]  # TBD tuningphithetalr~
                self.theta_opt = [torch.optim.Adam([self.theta_param[i]], lr=self.svo_lr) for i in range(self.n_agent)]

            # first assign
        if self.use_copo1:
            self.svo = [torch.clamp(torch.tanh(self.svo_param[i]), -1 + 1e-6, 1 - 1e-6) for i in range(self.n_agent)]  # tanh(-1, 1)
        if self.use_copo2:
            # phi:(0, 90) sigmoid
            # theta:(0, 360) sigmoid
            self.phi = [torch.clamp(torch.sigmoid(self.phi_param[i]), 0 + 1e-6, 1 - 1e-6) for i in range(self.n_agent)]
            self.theta = [torch.clamp(torch.sigmoid(self.theta_param[i]), 0 + 1e-6, 1 - 1e-6) for i in range(self.n_agent)]

    def is_uav(self, i):
        return True if i < self.n_uav else False

    def save(self, timestep, is_evaluate, is_newbest=False):
        save_dir = os.path.join(self.output_dir, 'model')
        save_dir += '/eval' if is_evaluate else '/train'
        if is_newbest:  # 
            save_dir += '/best_model'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            else:
                shutil.rmtree(save_dir)
                os.makedirs(save_dir)
        else:
            if not os.path.exists(save_dir): os.makedirs(save_dir)

        if self.use_eoi:
            torch.save(self.eoi_net.state_dict(), save_dir + f"/eoi_net_ep{timestep}.pth")

        if self.share_parameter:
            raise ValueError()
        else:
            if self.share_layer:
                for i in range(self.n_agent):
                    torch.save(self.model[i].state_dict(), save_dir + f"/share_layer_model_ep{timestep}_agent{i}.pth")
            else:
                for i in range(self.n_agent):
                    # torch.save(self.critic[i].state_dict(), save_dir + f"/ppo_critic_ep{timestep}_agent{i}.pth")
                    torch.save(self.actor[i].state_dict(), save_dir + f"/ppo_actor_ep{timestep}_agent{i}.pth")

    def load(self, load_dir, timestep):
        if self.use_eoi:
            self.eoi_net.load_state_dict(torch.load(load_dir + f"/eoi_net_ep{timestep}.pth"))

        if self.share_parameter:
            raise ValueError()
        else:
            if self.share_layer:
                for i in range(self.n_agent):
                    self.model[i].load_state_dict(torch.load(load_dir + f"/share_layer_model_ep{timestep}_agent{i}.pth"))

            else:
                for i in range(self.n_agent):
                    # self.critic[i].load_state_dict(torch.load(load_dir + f"/ppo_critic_ep{timestep}_agent{i}.pth"))
                    self.actor[i].load_state_dict(torch.load(load_dir + f"/ppo_actor_ep{timestep}_agent{i}.pth"))

    def gen_intrinsic_reward_by_eoinet(self, s):
        '''eoi_netintrinsic_reward'''
        T_horizon = s[0].shape[0]
        assert self.n_rollout_threads == s[0].shape[1]
        intrinsic_reward_list = []
        with torch.no_grad():
            for i in range(self.n_agent):
                # I = torch.tensor([[i] for _ in range(s[0].shape[0])]).to(self.device)
                I = (torch.ones((T_horizon, self.n_rollout_threads, 1)) * i).long().to(self.device)
                intrinsic_reward = self.eoi_net(s[i]).gather(dim=-1, index=I)
                intrinsic_reward_list.append(intrinsic_reward)
        return intrinsic_reward_list

    def EOI_update2(self, s_prime):
        s_p = torch.stack(s_prime)  # shape = (agent, T_horizon, threads, dim)
        assert self.n_agent == s_p.shape[0]
        TtimesT, obs_dim = s_p.shape[1] * s_p.shape[2], s_p.shape[3]
        s_p = s_p.reshape(self.n_agent, -1, obs_dim)  # shape = (agent, T_horizon*threads, dim)

        agent_ids = []  # shape = (agent, T_horizon*threads, agent)
        for i in range(self.n_agent):
            agent_id = torch.zeros(self.n_agent).to(self.device)
            agent_id[i] = 1
            agent_id = agent_id.repeat((TtimesT, 1))
            agent_ids.append(agent_id)
        agent_ids = torch.stack(agent_ids)

        s_p = s_p.reshape(-1, obs_dim)
        agent_ids = agent_ids.reshape(-1, self.n_agent)
        assert s_p.shape[0] == agent_ids.shape[0]
        total_num = s_p.shape[0]

        # ② train eoi_net
        els = []
        eoi_optim_iter_num = int(math.ceil(total_num / self.eoi_optim_batch_size))
        eoi_epoch = 1  # pymarl+EOI
        for k in range(eoi_epoch):
            perm = np.arange(total_num)
            np.random.shuffle(perm)
            s_p = s_p[perm].to(self.device)
            agent_ids = agent_ids[perm].to(self.device)
            for j in range(eoi_optim_iter_num):
                index = slice(j * self.eoi_optim_batch_size, min((j + 1) * self.eoi_optim_batch_size, total_num))
                X, Y = s_p[index], agent_ids[index]
                p = self.eoi_net(X)
                loss_1 = -(Y * (torch.log(p + 1e-8))).mean() - 0.1 * (p * (torch.log(p + 1e-8))).mean()
                # 
                #
                els.append(loss_1.item())
                self.eoi_optimizer.zero_grad()
                loss_1.backward()
                self.eoi_optimizer.step()

        self.writer.add_scalar('watch/EOI/eoi_loss', np.mean(els), self.timesteps)

    def merge(self, o, a, logprob_a, adv_list, r_target_list,
              state,
              state_prime,
              shaping_adv_list,
              shaping_target_list,
              global_adv_list,
              global_target_list,
              nei_adv_list,
              nei_target_list,
              uav_adv_list,
              uav_target_list,
              car_adv_list,
              car_target_list,
              ):
        for i in range(self.n_agent):
            o[i] = o[i].reshape(-1, o[i].shape[-1])
            a[i] = a[i].reshape(-1, a[i].shape[-1])
            logprob_a[i] = logprob_a[i].reshape(-1, logprob_a[i].shape[-1])
            adv_list[i] = adv_list[i].reshape(-1, 1)
            r_target_list[i] = r_target_list[i].reshape(-1, 1)

            if state is not None:
                state[i] = state[i].reshape(-1, state[i].shape[-1])
                state_prime[i] = state_prime[i].reshape(-1, state_prime[i].shape[-1])

            if shaping_adv_list is not None:
                shaping_adv_list[i] = shaping_adv_list[i].reshape(-1, 1)
                shaping_target_list[i] = shaping_target_list[i].reshape(-1, 1)
            if global_adv_list is not None:
                global_adv_list[i] = global_adv_list[i].reshape(-1, 1)
                global_target_list[i] = global_target_list[i].reshape(-1, 1)
            if nei_adv_list is not None:
                nei_adv_list[i] = nei_adv_list[i].reshape(-1, 1)
                nei_target_list[i] = nei_target_list[i].reshape(-1, 1)
            if uav_adv_list is not None:
                uav_adv_list[i] = uav_adv_list[i].reshape(-1, 1)
                uav_target_list[i] = uav_target_list[i].reshape(-1, 1)
                car_adv_list[i] = car_adv_list[i].reshape(-1, 1)
                car_target_list[i] = car_target_list[i].reshape(-1, 1)

    def svo_forward(self, adv_list, shaping_adv_list, nei_adv_list, uav_adv_list, car_adv_list, global_adv_list):
        '''svo'''
        if self.use_copo1:
            co_adv_list = []
            for i in range(self.n_agent):
                used_svo = self.svo[i] * np.pi / 2  # (-1,1)tensor(-pi/2, pi/2)
                used_svo = used_svo.cpu().detach().numpy()  # svoco_adv
                if self.use_eoi3:  # 4/5Our Solution
                    co_adv = np.cos(used_svo) * shaping_adv_list[i] + np.sin(used_svo) * nei_adv_list[i]
                else:
                    co_adv = np.cos(used_svo) * adv_list[i] + np.sin(used_svo) * nei_adv_list[i]
                co_adv_list.append(co_adv)
                # adv_listnei_adv_list！svo
                # normalize advs!
            global_adv_list, _, _ = normalize(global_adv_list)  #
            co_adv_list, raw_co_adv_mean, raw_co_adv_std = normalize(co_adv_list)  # actor, svo
            return co_adv_list, raw_co_adv_mean, raw_co_adv_std
        elif self.use_copo2:
            co_adv_list = []
            for i in range(self.n_agent):
                if self.hcopo_shift:
                    phi_rad = (self.phi[i] * np.pi).cpu().detach().numpy()  # from (0,1/2) to (0, pi/2)
                    theta_rad = (self.theta[i] * np.pi - np.pi / 4).cpu().detach().numpy()  # from (0,1) to (-pi/4, 3pi/4)
                elif self.hcopo_shift_513:
                    phi_rad = (self.phi[i] * np.pi - np.pi / 2).cpu().detach().numpy()  # from (1/2,1) to (0, pi/2)
                    theta_rad = (self.theta[i] * np.pi - np.pi / 4).cpu().detach().numpy()  # from (0,1) to (-pi/4, 3pi/4)
                else:
                    phi_rad = (self.phi[i] * np.pi / 2).cpu().detach().numpy()  # from (0,1) to (0, pi/2)
                    theta_rad = (self.theta[i] * np.pi * 2).cpu().detach().numpy()  # from (0,1) to (0, 2*pi)
                if self.hcopo_sqrt2_scale:
                    nei = np.sqrt(2) * np.cos(theta_rad) * uav_adv_list[i] + np.sin(theta_rad) * car_adv_list[i]
                else:
                    nei = np.cos(theta_rad) * uav_adv_list[i] + np.sin(theta_rad) * car_adv_list[i]
                if self.use_eoi3:  # 4/5Our Solution
                    co_adv = np.cos(phi_rad) * shaping_adv_list[i] + np.sin(phi_rad) * nei
                else:
                    co_adv = np.cos(phi_rad) * adv_list[i] + np.sin(phi_rad) * nei
                co_adv_list.append(co_adv)
            global_adv_list, _, _ = normalize(global_adv_list)  #
            co_adv_list, raw_co_adv_mean, raw_co_adv_std = normalize(co_adv_list)  # actor, svo
            return co_adv_list, raw_co_adv_mean, raw_co_adv_std
        else:
            adv_list, _, _ = normalize(adv_list)  # svo normalize original adv
            # (normalizedthread mappo)
            return None, None, None


    def make_batch(self):

        o_batch, mask_batch, r_batch, o_prime_batch, done_batch = [], [], [], [], []
        uav_a_batch, car_a_batch = [], []
        uav_logprob_a_batch, car_logprob_a_batch = [], []
        nei_r_batch, uav_r_batch, car_r_batch, global_r_batch = [], [], [], []
        for transition in self.data:  # T_horizon # self.data——
            s_list, mask_list, a_list, r_list, s_prime_list, logprob_a_list, done, nei_r_list, uav_r_list, car_r_list, global_r_list = transition
            o_batch.append(s_list)
            mask_batch.append(mask_list)
            # uav_a_batch.append(a_list[:self.n_uav])
            # car_a_batch.append(a_list[self.n_uav:])
            uav_a_batch.append([a_li[:self.n_uav] for a_li in a_list])  #
            car_a_batch.append([a_li[self.n_uav:] for a_li in a_list])
            uav_logprob_a_batch.append([logprob_a_li[:self.n_uav] for logprob_a_li in logprob_a_list])
            car_logprob_a_batch.append([logprob_a_li[self.n_uav:] for logprob_a_li in logprob_a_list])
            r_batch.append(
                np.expand_dims(np.array(r_list), -1)  # shape = (threads, agents, 1)
            )
            o_prime_batch.append(s_prime_list)
            done_batch.append(
                np.expand_dims(np.repeat(np.expand_dims(np.array(done), -1), self.n_agent, axis=-1), -1)  # shape = (threads, agents, 1)
            )

            nei_r_batch.append(np.expand_dims(np.array(nei_r_list), -1))
            uav_r_batch.append(np.expand_dims(np.array(uav_r_list), -1))
            car_r_batch.append(np.expand_dims(np.array(car_r_list), -1))
            global_r_batch.append(np.expand_dims(np.array(global_r_list), -1))
        self.data = []  # Clean history trajectory

        '''list to tensor'''
        with torch.no_grad():
            # arraylist->array->tensor  list->tensor
            o_batch, uav_a_batch, car_a_batch, mask_batch, r_batch, o_prime_batch, uav_logprob_a_batch, car_logprob_a_batch, done_batch, nei_r_batch, uav_r_batch, car_r_batch, global_r_batch = \
                torch.tensor(np.array(o_batch), dtype=torch.float).to(self.device), \
                torch.tensor(np.array(uav_a_batch), dtype=torch.float).to(self.device), \
                torch.tensor(np.array(car_a_batch), dtype=torch.float).to(self.device), \
                torch.tensor(np.array(mask_batch), dtype=torch.float).to(self.device), \
                torch.tensor(np.array(r_batch), dtype=torch.float).to(self.device), \
                torch.tensor(np.array(o_prime_batch), dtype=torch.float).to(self.device), \
                torch.tensor(np.array(uav_logprob_a_batch), dtype=torch.float).to(self.device), \
                torch.tensor(np.array(car_logprob_a_batch), dtype=torch.float).to(self.device), \
                torch.tensor(np.array(done_batch), dtype=torch.float).to(self.device), \
                torch.tensor(np.array(nei_r_batch), dtype=torch.float).to(self.device), \
                torch.tensor(np.array(uav_r_batch), dtype=torch.float).to(self.device), \
                torch.tensor(np.array(car_r_batch), dtype=torch.float).to(self.device), \
                torch.tensor(np.array(global_r_batch), dtype=torch.float).to(self.device),

        # agent——
        # MAPPOshape = (T_horizon, threads, agent, dim)
        # devide experience for each agent (permuteagent )
        s = [o_batch[:, :, i, :] for i in range(self.n_agent)]
        mask = [mask_batch[:, :, i, :] for i in range(self.n_car)]
        r = [r_batch[:, :, i, :] for i in range(self.n_agent)]
        s_prime = [o_prime_batch[:, :, i, :] for i in range(self.n_agent)]
        done = [done_batch[:, :, i, :] for i in range(self.n_agent)]
        # len(a) = 6, a[0].shape = (T_horizon, 2)a[4].shape = (T_horizon, 1)
        a = [uav_a_batch[:, :, i, :] for i in range(self.n_uav)] + [car_a_batch[:, :, i].unsqueeze(-1) for i in range(self.n_car)]
        logprob_a = [uav_logprob_a_batch[:, :, i, :] for i in range(self.n_uav)] + [car_logprob_a_batch[:, :, i].unsqueeze(-1) for i in range(self.n_car)]

        nei_r = [nei_r_batch[:, :, i, :] for i in range(self.n_agent)]
        uav_r = [uav_r_batch[:, :, i, :] for i in range(self.n_agent)]
        car_r = [car_r_batch[:, :, i, :] for i in range(self.n_agent)]
        global_r = [global_r_batch[:, :, i, :] for i in range(self.n_agent)]

        return s, mask, a, r, s_prime, logprob_a, done, nei_r, uav_r, car_r, global_r

    def put_data(self, transition):
        self.data.append(transition)


