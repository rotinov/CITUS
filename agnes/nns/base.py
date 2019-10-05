import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from agnes.common.init_weights import get_weights_init
from gym import spaces
import numpy


class BasePolicy:
    def __init__(self, input_shape, action_space):
        self.action_space = action_space

    @staticmethod
    def type_of_out():
        pass

    def forward(self, x):
        return None, None

    def get_action(self, x):
        if x.ndimension() < len(self.action_space.shape) + 1:
            x.unsqueeze_(0)

        dist, state_value = self.forward(x)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return (action.detach().squeeze(-1).cpu().numpy(),
                action.detach().squeeze(-1).cpu().numpy(),
                (log_prob.detach().squeeze(-1).cpu().numpy(),
                 state_value.detach().squeeze(-1).cpu().numpy()))

    def get_action_n_apply(self, x, func):
        if x.ndimension() < len(self.action_space.shape) + 1:
            x.unsqueeze_(0)

        dist, state_value = self.forward(x)

        smpled = dist.sample()

        action = func(smpled)

        log_prob = dist.log_prob(smpled)

        return (action.detach().squeeze(-1).cpu().numpy(),
                smpled.detach().squeeze(-1).cpu().numpy(),
                (log_prob.detach().squeeze(-1).cpu().numpy(),
                 state_value.detach().squeeze(-1).cpu().numpy()))
