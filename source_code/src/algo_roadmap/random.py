import numpy as np
import torch
from torch.nn import Parameter




class Random():
    def __init__(
            self,
            output_dir,
            device,
            writer,
            share_parameter,
            use_eoi,
            use_copo,
            eoi_kind,
            copo_kind,
            state_dim,
            uav_continuous_action_dim,
            car_discrete_action_dim,
            n_agent,
            n_uav,
            env_with_Dead,
            gamma=0.99,
            lambd=0.95,
            clip_rate=0.2,
            K_epochs=10,
            W_epochs=5,
            net_width=256,
            a_lr=3e-4,
            c_lr=3e-4,
            l2_reg=1e-3,
            dist='GS_ms',  # 'Beta'
            a_optim_batch_size=64,
            c_optim_batch_size=64,
            entropy_coef=0,  # 0.001
            entropy_coef_decay=0.9998,
            # eoi
            eoi3_coef=0.2,  # TBD 0.2tuning
            # svo
            initial_svo_degree=0.0,
            svo_lr=1e-4,
    ):
        super().__init__()

        ''''''
        self.bad_low_ratio_num = 0
        self.bad_up_ratio_num = 0
        self.bad_ratio_lower_bound = 1 / 20.0  # TBD  here
        self.bad_ratio_upper_bound = 20.0

        self.output_dir = output_dir
        self.device = device
        self.writer = writer
        self.share_parameter = share_parameter
        self.use_eoi = use_eoi
        self.use_copo = use_copo
        self.n_agent = n_agent
        self.n_uav = n_uav


        '''initialize svo'''
        degree2svoparam = {-90: -10.0, -60: -0.8045, -30: -0.3466, 0: 0.0,
                           90: 10.0, 60: 0.8045, 30: 0.3466}
        param_init = degree2svoparam.get(initial_svo_degree, 0.0)
        if self.share_parameter:
            sp = Parameter(torch.tensor(param_init, dtype=torch.float32).to(self.device))
            opt_sp = torch.optim.Adam([sp], lr=svo_lr)
            self.svo_param = [sp for _ in range(self.n_agent)]
            self.svo_opt = [opt_sp for _ in range(self.n_agent)]
        else:
            self.svo_param = [Parameter(torch.tensor(param_init, dtype=torch.float32).to(self.device)) for _ in range(self.n_agent)]
            self.svo_opt = [torch.optim.Adam([self.svo_param[i]], lr=svo_lr) for i in range(self.n_agent)]
        self.svo = [torch.clamp(torch.tanh(self.svo_param[i]), -1 + 1e-6, 1 - 1e-6) for i in range(self.n_agent)]  # first assign

        self.dist = dist
        self.env_with_Dead = env_with_Dead
        self.uav_actdim = uav_continuous_action_dim
        self.car_actdim = car_discrete_action_dim
        self.clip_rate = clip_rate
        self.gamma = gamma
        self.lambd = lambd
        self.clip_rate = clip_rate
        self.K_epochs = K_epochs
        self.W_epochs = W_epochs
        self.data = []
        self.l2_reg = l2_reg
        self.a_optim_batch_size = a_optim_batch_size
        self.c_optim_batch_size = c_optim_batch_size
        self.svo_optim_batch_size = a_optim_batch_size  #
        self.eoi_optim_batch_size = 256  # TBD
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.a_lr = a_lr
        self.svo_lr = svo_lr
        self.timesteps = None

    def is_uav(self, i):
        return True if i < self.n_uav else False

    def select_action(self, mask, mode):  # only used when interact with the env
        '''
        mask: available action for each car
        mode: in ['run', 'eval']
        '''
        assert mode in ['run', 'eval']
        a_list = []
        for i in range(self.n_agent):
            if self.is_uav(i):
                a_list.append(np.random.uniform(0, 1, (self.uav_actdim,)))
            else:
                j = i - self.n_uav
                assert len(mask[j]) == self.car_actdim
                a_list.append(np.random.choice(len(mask[j]), p=mask[j]/sum(mask[j])))
        return a_list

    def save(self, episode, is_evaluate=False, is_newbest=False):
        pass


