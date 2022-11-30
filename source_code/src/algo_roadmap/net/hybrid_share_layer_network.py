import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
import math

class ShareLayerCopoModel(nn.Module):
    def __init__(self, obs_dim, action_dim, net_width, copo_kind, is_uav):
        '''hcopofunc()
        copo:
            uav: alpha, beta, value, nei, global
            car: actprobs, value, nei, global
        hcopo:
            uav: alpha, beta, value, uav, car, global
            car: actprobs, value, uav, car, global
        '''
        super().__init__()
        assert copo_kind in (1, 2)
        self.copo_kind = copo_kind
        self.is_uav = is_uav

        # share
        self.l1 = nn.Linear(obs_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)

        # actor head
        if is_uav:
            self.alpha_head = nn.Linear(net_width, action_dim)
            self.beta_head = nn.Linear(net_width, action_dim)
        else:
            self.logits = nn.Linear(net_width, action_dim)

        # value head
        self.value_head = nn.Linear(net_width, 1)
        self.global_head = nn.Linear(net_width, 1)
        if copo_kind == 1:
            self.nei_head = nn.Linear(net_width, 1)
        else:
            self.uav_head = nn.Linear(net_width, 1)
            self.car_head = nn.Linear(net_width, 1)

    def forward(self, obs):
        # shape
        x = torch.tanh(self.l1(obs))
        x = torch.tanh(self.l2(x))

        # actor head
        if self.is_uav:
            alpha = F.softplus(self.alpha_head(x)) + 1.0
            beta = F.softplus(self.beta_head(x)) + 1.0
        else:
            actprobs = torch.softmax(self.logits(x), dim=-1)

        # value head
        value = self.value_head(x)
        global_value = self.global_head(x)
        if self.copo_kind == 1:
            nei_value = self.nei_head(x)
            if self.is_uav:
                return Beta(alpha, beta), value, nei_value, global_value
            else:
                return actprobs, value, nei_value, global_value
        else:
            uav_value = self.uav_head(x)
            car_value = self.car_head(x)
            if self.is_uav:
                return Beta(alpha, beta), value, uav_value, car_value, global_value
            else:
                return actprobs, value, uav_value, car_value, global_value

    def get_dist(self, obs):
        out = self.forward(obs)
        dist = out[0]  # uavBetacaractprobs
        return dist

    def evaluate_mode(self, obs):
        out = self.forward(obs)
        dist = out[0]
        return dist.mean  #  alpha / (alpha + beta)







