import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal
import math


class BetaActor(nn.Module):
    def __init__(self, obs_dim, action_dim, net_width):
        super(BetaActor, self).__init__()

        self.l1 = nn.Linear(obs_dim, net_width)  # (3, 150)
        self.l2 = nn.Linear(net_width, net_width)  # (150, 150)
        self.alpha_head = nn.Linear(net_width, action_dim)  # (150, 1)
        self.beta_head = nn.Linear(net_width, action_dim)  # (150, 1)

    def forward(self, obs):
        a = torch.tanh(self.l1(obs))
        a = torch.tanh(self.l2(a))

        alpha = F.softplus(self.alpha_head(a)) + 1.0
        beta = F.softplus(self.beta_head(a)) + 1.0

        return alpha, beta

    def get_dist(self, obs):
        alpha, beta = self.forward(obs)
        dist = Beta(alpha, beta)
        return dist

    def evaluate_mode(self, obs):
        alpha, beta = self.forward(obs)
        mode = (alpha) / (alpha + beta)
        return mode


class GaussianActor_musigma(nn.Module):
    def __init__(self, obs_dim, action_dim, net_width):
        super(GaussianActor_musigma, self).__init__()

        self.l1 = nn.Linear(obs_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.mu_head = nn.Linear(net_width, action_dim)
        self.sigma_head = nn.Linear(net_width, action_dim)

    def forward(self, obs):
        a = torch.tanh(self.l1(obs))
        a = torch.tanh(self.l2(a))
        mu = torch.sigmoid(self.mu_head(a))
        sigma = F.softplus(self.sigma_head(a))  # SoftPlus is a smooth approximation to the ReLU function and can be used to constrain the output to always be positive
        # if torch.any(sigma<1e-3):
        #     print('clip！')
        sigma = torch.clamp(sigma, 1e-3, float('inf'))  # sigma1e-3sigma headnan
        return mu, sigma

    def get_dist(self, obs):
        mu, sigma = self.forward(obs)
        dist = Normal(mu, sigma)
        return dist

    def evaluate_mode(self, obs):
        mu, sigma = self.forward(obs)
        return mu


class DiscreteActor(nn.Module):
    def __init__(self, obs_dim, action_dim, net_width):
        super(DiscreteActor, self).__init__()

        self.l1 = nn.Linear(obs_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.logits = nn.Linear(net_width, action_dim)

    def forward(self, obs):
        a = torch.tanh(self.l1(obs))
        a = torch.tanh(self.l2(a))
        actprobs = torch.softmax(self.logits(a), dim=-1)
        return actprobs

class Critic(nn.Module):
    def __init__(self, obs_dim, net_width):
        '''
        critic
        '''
        super(Critic, self).__init__()
        self.C1 = nn.Linear(obs_dim, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, obs):
        v = torch.tanh(self.C1(obs))
        v = torch.tanh(self.C2(v))
        v = self.C3(v)
        return v


class EOI_Net(nn.Module):
    ''''''

    def __init__(self, obs_dim, n_agent):
        super(EOI_Net, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)  # noma64128
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_agent)

    def forward(self, x):
        '''obsagent'''
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.softmax(self.fc3(y), dim=1)  # TBD softmaxdim
        return y





