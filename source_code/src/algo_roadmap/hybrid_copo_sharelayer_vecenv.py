from torch.distributions import Categorical
import copy
import numpy as np
import torch
import math
from src.algo_roadmap.hybrid_ppo_base import Hybrid_PPO_base
from src.algo_roadmap.net.hybrid_share_layer_network import ShareLayerCopoModel
from src.algo_roadmap.utils import normalize

class VecHybridShareLayerCopo(Hybrid_PPO_base):
    def __init__(self, kwargs):
        super().__init__(**kwargs)

        '''ShareLayerCopoModel'''
        if self.share_parameter:
            raise ValueError('')
        else:
            # 
            self.model = [ShareLayerCopoModel(self.obs_dim, self.uav_continuous_action_dim, self.net_width, copo_kind=self.copo_kind, is_uav=True).to(self.device)
                          if self.is_uav(i) else
                          ShareLayerCopoModel(self.obs_dim, self.car_discrete_action_dim, self.net_width, copo_kind=self.copo_kind, is_uav=False).to(self.device)
                          for i in range(self.n_agent)]
            self.optimizer = [torch.optim.Adam(self.model[i].parameters(), lr=self.a_lr) for i in range(self.n_agent)]

        if self.use_copo1:
            self.branches_uav = ('dist', 'value', 'nei', 'global')
            self.branches_car = ('act_prob', 'value', 'nei', 'global')
        if self.use_copo2:
            self.branches_uav = ('dist', 'value', 'uav', 'car', 'global')
            self.branches_car = ('act_prob', 'value', 'uav', 'car', 'global')


    def select_action_for_vec_2(self, obs, mask, mode):
        actions = []  # agentthread
        action_log_probs = []
        n_rollout_threads = obs.shape[0]  # train64eval1
        obs_for_agents = obs.transpose((1, 0, 2))
        mask_for_agents = mask.transpose((1, 0, 2))

        for i in range(self.n_agent):
            input = torch.FloatTensor(obs_for_agents[i]).to(self.device)
            if self.is_uav(i):  # for uav
                if mode == 'run':
                    dist = self.model[i].get_dist(input)  # value
                    a = dist.sample()  # sample
                    a = torch.clamp(a, 0, 1)
                    logprob_a = dist.log_prob(a)
                else:
                    a = self.model[i].evaluate_mode(input)
                    a = torch.clamp(a, 0, 1)
                    logprob_a = torch.tensor(-1)  # dummy
                actions.append(a.cpu().detach().numpy())
                action_log_probs.append(logprob_a.cpu().detach().numpy())
            else:  # for car
                actor_actprobs = self.model[i].get_dist(input)
                if mode == 'run':
                    c = Categorical(actor_actprobs * torch.tensor(mask_for_agents[i - self.n_uav]).to(self.device))  # mask01Categoricalsample()1
                    a = c.sample()
                    logprob_a = c.log_prob(a)
                else:
                    a = torch.argmax(actor_actprobs * torch.tensor(mask_for_agents[i - self.n_uav]).to(self.device), dim=-1)
                    logprob_a = torch.tensor(-1)  # dummy
                actions.append(a.cpu().detach().numpy())
                action_log_probs.append(logprob_a.cpu().detach().numpy())

        '''agents, threads, dim threads, agents, dim'''
        actions_out = []
        for e in range(n_rollout_threads):  # train128eval1self.n_rollout_threads
            act_for_one_thread = []
            for act_for_one_agent in actions:
                if act_for_one_agent.shape == ():  # scalar
                    act_for_one_thread.append(act_for_one_agent)
                else:
                    act_for_one_thread.append(act_for_one_agent[e])
            actions_out.append(act_for_one_thread)

        action_log_probs_out = []
        for e in range(n_rollout_threads):
            actprob_for_one_thread = []
            for actprob_for_one_agent in action_log_probs:
                if actprob_for_one_agent.shape == ():
                    actprob_for_one_thread.append(actprob_for_one_agent)
                else:
                    actprob_for_one_thread.append(actprob_for_one_agent[e])
            action_log_probs_out.append(actprob_for_one_thread)

        return actions_out, action_log_probs_out

    def train(self, timesteps):
        self.timesteps = timesteps
        self.entropy_coef *= self.entropy_coef_decay  # start with 0.001, decay by *= 0.99
        # == myadd ==
        self.eoi3_coef *= self.eoi_coef_decay
        # == myadd ==


        o, mask, a, r, o_prime, logprob_a, done, nei_r, uav_r, car_r, global_r = self.make_batch()  # maska

        # shape = (agent, T_horizon, threads, dim)
        # OK checkbuffer typeshape

        def gen_cc_obs(input):
            stack = torch.cat([input[i].unsqueeze(-1) for i in range(self.n_agent)], dim=-1)
            cc_obs = torch.max(stack, dim=-1)[0]
            output = [copy.deepcopy(cc_obs) for _ in range(self.n_agent)]
            return output

        if self.use_centralized_critc:  # centralized1agentmax 2
            state = gen_cc_obs(o)
            state_prime = gen_cc_obs(o_prime)

        '''train eoi_net'''
        if self.use_eoi:
            self.EOI_update2(o_prime)


        '''Use TD+GAE+LongTrajectory to compute Advantage and TD target'''

        def func(ref_r, branch):
            adv_list, target_list = [], []
            with torch.no_grad():
                for i in range(self.n_agent):
                    '''1. criticvsvs_'''
                    branches = self.branches_uav if self.is_uav(i) else self.branches_car
                    assert branch in branches
                    pos = branches.index(branch)
                    if self.use_centralized_critc:
                        vs, vs_ = self.model[i](state[i])[pos], self.model[i](state_prime[i])[pos]
                    else:
                        vs, vs_ = self.model[i](o[i])[pos], self.model[i](o_prime[i])[pos]

                    '''2. deltas'''
                    deltas = ref_r[i] + self.gamma * vs_ - vs
                    deltas = deltas.squeeze(-1).cpu().numpy()
                    '''3. done for GAE deltas adv_list, target_list'''
                    adv = [[0 for _ in range(self.n_rollout_threads)]]
                    for dlt, mask in zip(deltas[::-1], done[i].squeeze(-1).cpu().numpy()[::-1]):
                        advantage = dlt + self.gamma * self.lambd * np.array(adv[-1]) * (1 - mask)
                        adv.append(list(advantage))
                    adv.reverse()
                    adv = copy.deepcopy(adv[0:-1])
                    adv = torch.tensor(adv).unsqueeze(1).float().to(self.device)
                    target = adv + vs
                    target_list.append(target)
                    adv_list.append(adv)
            return adv_list, target_list

        '''advtd-target'''
        adv_list, r_target_list = func(r, branch='value')

        # Our Solutioncopoeoi
        if self.use_eoi:
            intrinsic_reward_list = self.gen_intrinsic_reward_by_eoinet(o)
            #  shape = (agent, T_horizon, 1) (agent, T_horizon, threads, 1)
            if self.use_eoi3:
                shaping_r_list = [r[i] + self.eoi3_coef * intrinsic_reward_list[i] for i in range(self.n_agent)]
                shaping_adv_list, shaping_target_list = func(shaping_r_list, branch='value')

        if self.use_copo:  # always True
            global_adv_list, global_target_list = func(global_r, branch='global')
            if self.use_copo1:
                nei_adv_list, nei_target_list = func(nei_r, branch='nei')
            if self.use_copo2:
                uav_adv_list, uav_target_list = func(uav_r, branch='uav')
                car_adv_list, car_target_list = func(car_r, branch='car')

        '''vecenv！(T_horizon, threads, dim)(T_horizon*threads, dim)'''
        self.merge(o, a, logprob_a, adv_list, r_target_list,
                   state if self.use_centralized_critc else None,
                   state_prime if self.use_centralized_critc else None,
                   shaping_adv_list if self.use_eoi3 else None,
                   shaping_target_list if self.use_eoi3 else None,
                   global_adv_list if self.use_copo else None,
                   global_target_list if self.use_copo else None,
                   nei_adv_list if self.use_copo1 else None,
                   nei_target_list if self.use_copo1 else None,
                   uav_adv_list if self.use_copo2 else None,
                   uav_target_list if self.use_copo2 else None,
                   car_adv_list if self.use_copo2 else None,
                   car_target_list if self.use_copo2 else None,
                   )

        '''svo'''
        co_adv_list, raw_co_adv_mean, raw_co_adv_std = self.svo_forward(
            adv_list,
            shaping_adv_list if self.use_eoi3 else None,
            nei_adv_list if self.use_copo1 else None,
            uav_adv_list if self.use_copo2 else None,
            car_adv_list if self.use_copo2 else None,
            global_adv_list if self.use_copo else None,
        )

        old_model = copy.deepcopy(self.model)  # θold,
        '''Kactorcritic'''
        # Slice long trajectopy into short trajectory and perform mini-batch PPO update
        als, c1ls, c2ls, c3ls, c4ls = [], [], [], [], []
        optim_iter_num = int(math.ceil(o[0].shape[0] / self.a_optim_batch_size))
        for k in range(self.K_epochs):
            # Shuffle the trajectory, Good for training
            perm = np.arange(o[0].shape[0])
            np.random.shuffle(perm)
            for i in range(self.n_agent):
                o[i], a[i], logprob_a[i], r_target_list[i] = o[i][perm], a[i][perm], logprob_a[i][perm], r_target_list[i][perm]
                if self.use_centralized_critc:
                    state[i] = state[i][perm]
                if self.use_eoi3:
                    shaping_adv_list[i] = shaping_adv_list[i][perm]
                    shaping_target_list[i] = shaping_target_list[i][perm]
                if self.use_copo1:
                    co_adv_list[i] = co_adv_list[i][perm]
                    nei_target_list[i] = nei_target_list[i][perm]
                    global_target_list[i] = global_target_list[i][perm]
                elif self.use_copo2:
                    co_adv_list[i] = co_adv_list[i][perm]
                    uav_target_list[i] = uav_target_list[i][perm]
                    car_target_list[i] = car_target_list[i][perm]
                    global_target_list[i] = global_target_list[i][perm]
                else:
                    adv_list[i] = adv_list[i][perm]

            # update the actor and critic for vanilla PPO
            if self.use_copo1 or self.use_copo2:  # Our Solutionco_advshaping_advadv
                adv_list_for_opt_actor = co_adv_list
            elif self.use_eoi3:  # always False
                adv_list_for_opt_actor = shaping_adv_list
            else:
                adv_list_for_opt_actor = adv_list
            if self.use_centralized_critc:
                input_state = state
            else:
                input_state = None
            if self.use_eoi3:
                target_list_for_opt_critic = shaping_target_list
            else:
                target_list_for_opt_critic = r_target_list
            if self.use_copo1:
                al, c1l, c2l, c4l = self.COPO_update_model(o, a, logprob_a,
                                                               adv_list_for_opt_actor,
                                                               target_list_for_opt_critic, nei_target_list, global_target_list,
                                                               optim_iter_num, state=input_state)
            if self.use_copo2:
                al, c1l, c2l, c3l, c4l = self.HCOPO_update_model(o, a, logprob_a,
                                                               adv_list_for_opt_actor,
                                                               target_list_for_opt_critic, uav_target_list, car_target_list, global_target_list,
                                                               optim_iter_num, state=input_state)


            als.append(al)
            c1ls.append(c1l)
            if self.use_copo1:
                c2ls.append(c2l)
                c4ls.append(c4l)
            if self.use_copo2:
                c2ls.append(c2l)
                c3ls.append(c3l)
                c4ls.append(c4l)

        self.writer.add_scalar('watch/PPO/actor_loss', np.mean(als), timesteps)
        self.writer.add_scalar('watch/PPO/critic_loss', np.mean(c1ls), timesteps)
        if self.use_copo1:
            self.writer.add_scalar('watch/PPO/nei_critic_loss', np.mean(c2ls), timesteps)
            self.writer.add_scalar('watch/PPO/global_critic_loss', np.mean(c4ls), timesteps)
        if self.use_copo2:
            self.writer.add_scalar('watch/PPO/nei_uav_critic_loss', np.mean(c2ls), timesteps)
            self.writer.add_scalar('watch/PPO/nei_car_critic_loss', np.mean(c3ls), timesteps)
            self.writer.add_scalar('watch/PPO/global_critic_loss', np.mean(c4ls), timesteps)

        '''wsvo'''
        if (self.use_copo1 or self.use_copo2) and not self.svo_frozen:
            l1s, l2s, l3s, l4s = [], [], [], []  # epochminibatchagentloss training_step
            svo_optim_iter_num = int(math.ceil(o[0].shape[0] / self.svo_optim_batch_size))
            for w in range(self.W_epochs):
                # Shuffle the trajectory, Good for training
                perm = np.arange(o[0].shape[0])
                np.random.shuffle(perm)
                for i in range(self.n_agent):
                    o[i], a[i], logprob_a[i] = o[i][perm], a[i][perm], logprob_a[i][perm]
                    if self.use_eoi3:
                        shaping_adv_list[i] = shaping_adv_list[i][perm]
                    else:
                        adv_list[i] = adv_list[i][perm]
                    global_adv_list[i] = global_adv_list[i][perm]
                    if self.use_copo1:
                        nei_adv_list[i] = nei_adv_list[i][perm]
                    if self.use_copo2:
                        uav_adv_list[i] = uav_adv_list[i][perm]
                        car_adv_list[i] = car_adv_list[i][perm]

                for j in range(svo_optim_iter_num):
                    index = slice(j * self.svo_optim_batch_size, min((j + 1) * self.svo_optim_batch_size, o[0].shape[0]))
                    for i in range(self.n_agent):  # update svo
                        input_dict = {}
                        input_dict['s'] = o[i][index]
                        input_dict['a'] = a[i][index]
                        input_dict['logprob_a'] = logprob_a[i][index]
                        if self.use_eoi3:
                            input_dict['advantage'] = shaping_adv_list[i][index]
                        else:
                            input_dict['advantage'] = adv_list[i][index]
                        if self.use_copo1:
                            input_dict['nei_advantage'] = nei_adv_list[i][index]
                        if self.use_copo2:
                            input_dict['uav_advantage'] = uav_adv_list[i][index]
                            input_dict['car_advantage'] = car_adv_list[i][index]
                        input_dict['global_advantage'] = global_adv_list[i][index]
                        input_dict['raw_co_adv_mean'] = raw_co_adv_mean[i]
                        input_dict['raw_co_adv_std'] = raw_co_adv_std[i]
                        l1, l2, l3, l4 = self.build_meta_gradient_and_update_svo(i, input_dict, self.model, old_model)
                        l1s.append(l1)
                        l2s.append(l2)
                        l3s.append(l3)
                        l4s.append(l4)

            # svoloss
            for i in range(self.n_agent):
                if self.use_copo1:
                    self.writer.add_scalar(f'agent{i}/svo', self.svo[i].item(), timesteps)
                    self.writer.add_scalar(f'agent{i}/svo_degree', self.svo[i].item() * 90, timesteps)
                if self.use_copo2:
                    self.writer.add_scalar(f'agent{i}/phi', self.phi[i].item(), timesteps)
                    self.writer.add_scalar(f'agent{i}/theta', self.theta[i].item(), timesteps)
                    if self.hcopo_shift:
                        self.writer.add_scalar(f'agent{i}/phi_degree', self.phi[i].item() * 180, timesteps)
                        self.writer.add_scalar(f'agent{i}/theta_degree', self.theta[i].item() * 180 - 45, timesteps)
                    else:
                        self.writer.add_scalar(f'agent{i}/phi_degree', self.phi[i].item() * 90, timesteps)
                        self.writer.add_scalar(f'agent{i}/theta_degree', self.theta[i].item() * 360, timesteps)

            self.writer.add_scalar('watch/meta/new_policy_ego_loss', np.mean(l1s), timesteps)
            self.writer.add_scalar('watch/meta/old_policy_ego_loss', np.mean(l2s), timesteps)
            self.writer.add_scalar('watch/meta/svo_svo_adv_loss', np.mean(l3s), timesteps)
            self.writer.add_scalar('watch/meta/final_loss', np.mean(l4s), timesteps)
        del old_model

    def COPO_update_model(self, s, a, logprob_a, adv,
                          r_target, nei_target, global_target,
                          optim_iter_num, state=None):
        als, c1ls, c2ls, c4ls = [], [], [], []
        for j in range(optim_iter_num):  #
            index = slice(j * self.a_optim_batch_size, min((j + 1) * self.a_optim_batch_size, s[0].shape[0]))
            for i in range(self.n_agent):
                if self.is_uav(i):
                    distribution, value_out, nei_out, global_out = self.model[i](s[i][index])  # branch -actorloss--valueloss”ok
                    logprob_a_now = distribution.log_prob(a[i][index])  # shape = (batch, act_dim)
                    dist_entropy = distribution.entropy().sum(-1, keepdim=True)
                else:
                    act_prob, value_out, nei_out, global_out = self.model[i](s[i][index])
                    distribution = Categorical(act_prob)
                    logprob_a_now = distribution.log_prob(a[i][index].squeeze(-1)).unsqueeze(-1)  # shape = (batch, )
                    dist_entropy = distribution.entropy().unsqueeze(-1)

                if self.is_uav(i):
                    ratio = torch.exp(logprob_a_now.sum(-1, keepdim=True) - logprob_a[i][index].sum(1, keepdim=True))  # a/b == exp(log(a)-log(b))
                else:
                    ratio = torch.exp(logprob_a_now - logprob_a[i][index])

                surr1 = ratio * adv[i][index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[i][index]
                surr_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                actor_loss = torch.mean(surr_loss)
                als.append(actor_loss.item())

                if self.use_centralized_critc:  # cc_obscriticoutput,o
                    assert state is not None
                    branches = self.branches_uav if self.is_uav(i) else self.branches_car
                    fetchlist = ['value', 'nei', 'global']
                    poslist = [branches.index(fetch) for fetch in fetchlist]
                    out = self.model[i](state[i][index])
                    value_out = out[poslist[0]]
                    nei_out = out[poslist[1]]
                    global_out = out[poslist[2]]

                value_loss = (value_out - r_target[i][index]).pow(2).mean()
                nei_loss = (nei_out - nei_target[i][index]).pow(2).mean()
                global_loss = (global_out - global_target[i][index]).pow(2).mean()
                for name, param in self.model[i].named_parameters():
                    if 'value' in name:
                        value_loss += param.pow(2).sum() * self.l2_reg
                    if 'nei' in name:
                        nei_loss += param.pow(2).sum() * self.l2_reg
                    if 'global' in name:
                        global_loss += param.pow(2).sum() * self.l2_reg
                c1ls.append(value_loss.item())
                c2ls.append(nei_loss.item())
                c4ls.append(global_loss.item())

                total_loss = actor_loss + \
                             self.vf_coef * (value_loss + nei_loss + global_loss)
                self.optimizer[i].zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model[i].parameters(), 60)  # TBD 60
                self.optimizer[i].step()
        return np.mean(als), np.mean(c1ls), np.mean(c2ls), np.mean(c4ls)

    def HCOPO_update_model(self, s, a, logprob_a, adv,
                          r_target, uav_target, car_target, global_target,
                          optim_iter_num, state):
        als, c1ls, c2ls, c3ls, c4ls = [], [], [], [], []
        for j in range(optim_iter_num):  #
            index = slice(j * self.a_optim_batch_size, min((j + 1) * self.a_optim_batch_size, s[0].shape[0]))
            for i in range(self.n_agent):
                if self.is_uav(i):
                    distribution, value_out, uav_out, car_out, global_out = self.model[i](s[i][index])  # branch Q -actorloss--valueloss" OK A OK
                    logprob_a_now = distribution.log_prob(a[i][index])
                    dist_entropy = distribution.entropy().sum(-1, keepdim=True)
                else:
                    act_prob, value_out, uav_out, car_out, global_out = self.model[i](s[i][index])
                    distribution = Categorical(act_prob)
                    logprob_a_now = distribution.log_prob(a[i][index].squeeze(-1)).unsqueeze(-1)  # shape = (batch, )
                    dist_entropy = distribution.entropy().unsqueeze(-1)

                if self.is_uav(i):
                    ratio = torch.exp(logprob_a_now.sum(-1, keepdim=True) - logprob_a[i][index].sum(-1, keepdim=True))  # a/b == exp(log(a)-log(b))
                    # logprobprob
                else:
                    ratio = torch.exp(logprob_a_now - logprob_a[i][index])

                surr1 = ratio * adv[i][index]  # ratiosurr2clamp
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[i][index]
                surr_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                actor_loss = torch.mean(surr_loss)
                als.append(actor_loss.item())

                if self.use_centralized_critc:  # cc_obscriticoutput,o
                    assert state is not None
                    branches = self.branches_uav if self.is_uav(i) else self.branches_car
                    fetchlist = ['value', 'uav', 'car', 'global']
                    poslist = [branches.index(fetch) for fetch in fetchlist]
                    out = self.model[i](state[i][index])
                    value_out = out[poslist[0]]
                    uav_out = out[poslist[1]]
                    car_out = out[poslist[2]]
                    global_out = out[poslist[3]]

                value_loss = (value_out - r_target[i][index]).pow(2).mean()
                uav_loss = (uav_out - uav_target[i][index]).pow(2).mean()
                car_loss = (car_out - car_target[i][index]).pow(2).mean()
                global_loss = (global_out - global_target[i][index]).pow(2).mean()
                for name, param in self.model[i].named_parameters():
                    if 'value' in name:
                        value_loss += param.pow(2).sum() * self.l2_reg
                    if 'uav' in name:
                        uav_loss += param.pow(2).sum() * self.l2_reg
                    if 'car' in name:
                        car_loss += param.pow(2).sum() * self.l2_reg
                    if 'global' in name:
                        global_loss += param.pow(2).sum() * self.l2_reg
                c1ls.append(value_loss.item())
                c2ls.append(uav_loss.item())
                c3ls.append(car_loss.item())
                c4ls.append(global_loss.item())

                total_loss = actor_loss + \
                             self.vf_coef * (value_loss + 0.5 * uav_loss + 0.5 * car_loss + global_loss)  # 0.5
                self.optimizer[i].zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model[i].parameters(), 60)  # TBD 60
                self.optimizer[i].step()
        return np.mean(als), np.mean(c1ls), np.mean(c2ls), np.mean(c3ls), np.mean(c4ls)


    def build_meta_gradient_and_update_svo(self, i, input_dict, new_model, old_model):
        def _flatten(tensor):
            assert tensor is not None
            flat = tensor.reshape(-1, )
            return flat, tensor.shape, flat.shape

        '''step1. Build the loss between new policy and global advantage.'''
        #  OK
        term1_opt = torch.optim.Adam(new_model[i].parameters(), lr=self.a_lr)
        if self.is_uav(i):
            curr_dist = new_model[i].get_dist(input_dict['s'])
            logprob_a_now = curr_dist.log_prob(input_dict['a'])
            ratio = torch.exp(logprob_a_now.sum(-1, keepdim=True) - input_dict['logprob_a'].sum(-1, keepdim=True))
        else:
            curr_dist = Categorical(new_model[i].get_dist(input_dict['s']))
            logprob_a_now = curr_dist.log_prob(input_dict['a'].squeeze(-1)).unsqueeze(-1)
            ratio = torch.exp(logprob_a_now - input_dict['logprob_a'])

        adv = input_dict['global_advantage']
        surrogate_loss = torch.min(adv * ratio, adv * torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate))
        new_policy_ego_loss = torch.mean(surrogate_loss)  # svo
        term1_opt.zero_grad(set_to_none=True)
        new_policy_ego_loss.backward()

        '''step2. Build the loss between old policy and old log prob.'''
        term2_opt = torch.optim.Adam(old_model[i].parameters(), lr=self.a_lr)
        if self.is_uav(i):
            old_dist = old_model[i].get_dist(input_dict['s'])
            old_logp = old_dist.log_prob(input_dict['a'])  # old_logp.shape = (batchsize, )
        else:
            old_dist = Categorical(old_model[i].get_dist(input_dict['s']))
            old_logp = old_dist.log_prob(input_dict['a'].squeeze(-1)).unsqueeze(-1)
        old_policy_logp_loss = torch.mean(old_logp)
        term2_opt.zero_grad(set_to_none=True)
        old_policy_logp_loss.backward()

        '''step3. Build the loss between SVO and SVO advantage'''
        if self.use_copo1:
            svo_rad = self.svo[i] * np.pi / 2  # OK self.svo[i]svo
            advantages = torch.cos(svo_rad) * input_dict['advantage'] + torch.sin(svo_rad) * input_dict['nei_advantage']
        if self.use_copo2:
            if self.hcopo_shift:
                phi_rad = self.phi[i] * np.pi
                theta_rad = self.theta[i] * np.pi - np.pi / 4
            else:
                phi_rad = self.phi[i] * np.pi / 2
                theta_rad = self.theta[i] * np.pi * 2
            if self.hcopo_sqrt2_scale:
                nei = torch.sqrt(torch.tensor(2.0)) * torch.cos(theta_rad) * input_dict['uav_advantage'] + torch.sin(theta_rad) * input_dict['car_advantage']
            else:
                nei = torch.cos(theta_rad) * input_dict['uav_advantage'] + torch.sin(theta_rad) * input_dict['car_advantage']
            advantages = torch.cos(phi_rad) * input_dict['advantage'] + torch.sin(phi_rad) * nei

        svo_advantages = (advantages - input_dict['raw_co_adv_mean']) / input_dict['raw_co_adv_std']
        svo_svo_adv_loss = torch.mean(svo_advantages)
        svo_svo_adv_loss.backward()  # OK svo_param.gradNone

        '''step4. Multiple gradients one by one'''
        new_policy_ego_grad_flatten = []
        shape_list = []  # For verification used.
        new_policy_ego_grad = [params.grad for name, params in new_model[i].named_parameters() if params.grad is not None]
        for g in new_policy_ego_grad:
            fg, s, _ = _flatten(g)
            shape_list.append(s)
            new_policy_ego_grad_flatten.append(fg)
        new_policy_ego_grad_flatten = torch.cat(new_policy_ego_grad_flatten, axis=0)  # shape = (90628, )
        new_policy_ego_grad_flatten = torch.reshape(new_policy_ego_grad_flatten, (1, -1))  #

        old_policy_logp_grad_flatten = []
        old_policy_logp_grad = [params.grad for name, params in old_model[i].named_parameters() if params.grad is not None]
        for g, verify_shape in zip(old_policy_logp_grad, shape_list):
            fg, s, _ = _flatten(g)
            assert verify_shape == s
            old_policy_logp_grad_flatten.append(fg)
        old_policy_logp_grad_flatten = torch.cat(old_policy_logp_grad_flatten, axis=0)
        old_policy_logp_grad_flatten = torch.reshape(old_policy_logp_grad_flatten, (-1, 1))  #

        grad_value = torch.matmul(new_policy_ego_grad_flatten, old_policy_logp_grad_flatten)  # scalar,  *
        final_loss = torch.reshape(grad_value, ()) * svo_svo_adv_loss  # Eqn. 11!

        '''step5. apply gradient!'''
        if self.use_copo1:
            single_grad = self.svo_param[i].grad.to(self.device)
            final_grad = torch.matmul(grad_value, torch.reshape(single_grad, (1, 1)))
            final_grad = torch.reshape(final_grad, ())
            self.svo_param[i].grad = -final_grad  #
            self.svo_opt[i].step()
            # assign
            self.svo[i] = torch.clamp(torch.tanh(self.svo_param[i]), -1 + 1e-6, 1 - 1e-6)
            # zero grad
            term1_opt.zero_grad(set_to_none=True)  # for new_policy_ego_grad
            term2_opt.zero_grad(set_to_none=True)  # for old_policy_logp_grad
            self.svo_param[i].grad.zero_()
        if self.use_copo2:
            phi_single_grad = self.phi_param[i].grad.to(self.device)
            theta_single_grad = self.theta_param[i].grad.to(self.device)
            phi_final_grad = torch.matmul(grad_value, torch.reshape(phi_single_grad, (1, 1)))
            theta_final_grad = torch.matmul(grad_value, torch.reshape(theta_single_grad, (1, 1)))
            phi_final_grad = torch.reshape(phi_final_grad, ())
            theta_final_grad = torch.reshape(theta_final_grad, ())
            self.phi_param[i].grad = -phi_final_grad
            self.theta_param[i].grad = -theta_final_grad  #

            self.phi_opt[i].step()
            self.theta_opt[i].step()
            # assign
            if self.hcopo_shift:  # clamp
                self.phi_param[i].data = torch.clamp(self.phi_param[i].data, float('-inf'), 0)
            self.phi[i] = torch.clamp(torch.sigmoid(self.phi_param[i]), 0 + 1e-6, 1 - 1e-6)
            self.theta[i] = torch.clamp(torch.sigmoid(self.theta_param[i]), 0 + 1e-6, 1 - 1e-6)
            # zero grad
            term1_opt.zero_grad(set_to_none=True)  # for new_policy_ego_grad
            term2_opt.zero_grad(set_to_none=True)  # for old_policy_logp_grad
            self.phi_param[i].grad.zero_()
            self.theta_param[i].grad.zero_()

        ret = [new_policy_ego_loss.item(), old_policy_logp_loss.item(), svo_svo_adv_loss.item(), final_loss.item()]
        return ret




