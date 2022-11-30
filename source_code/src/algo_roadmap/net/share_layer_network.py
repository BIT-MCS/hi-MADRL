import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal
import math

class ShareLayerCopoModel(nn.Module):
    def __init__(self, obs_dim, action_dim, net_width, copo_kind):
        '''hcopofunc()
        copo: mu, sigma, value, nei, global
        hcopo: mu, sigma, value, uav, car, global
        '''
        super().__init__()
        assert copo_kind in (1, 2)
        self.copo_kind = copo_kind
        # share
        self.l1 = nn.Linear(obs_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        # unique
        self.mu_head = nn.Linear(net_width, action_dim)
        self.sigma_head = nn.Linear(net_width, action_dim)
        self.value_head = nn.Linear(net_width, 1)
        self.global_head = nn.Linear(net_width, 1)
        if copo_kind == 1:
            self.nei_head = nn.Linear(net_width, 1)
        else:
            self.uav_head = nn.Linear(net_width, 1)
            self.car_head = nn.Linear(net_width, 1)

    def forward(self, obs):
        # share
        x = torch.tanh(self.l1(obs))
        x = torch.tanh(self.l2(x))
        # unique
        mu = torch.sigmoid(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        sigma = torch.clamp(sigma, 1e-3, float('inf'))
        value = self.value_head(x)
        global_value = self.global_head(x)
        if self.copo_kind == 1:
            nei_value = self.nei_head(x)
            return mu, sigma, value, nei_value, global_value
        else:
            uav_value = self.uav_head(x)
            car_value = self.car_head(x)
            return mu, sigma, value, uav_value, car_value, global_value

    def get_dist(self, obs):
        out = self.forward(obs)
        mu, sigma = out[0], out[1]
        return Normal(mu, sigma)

    def evaluate_mode(self, obs):
        out = self.forward(obs)
        mu = out[0]
        return mu







