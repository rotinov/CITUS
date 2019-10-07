from abc import ABC
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from agnes.common.init_weights import get_weights_init
from agnes.nns.base import BasePolicy
from gym import spaces
import numpy
import warnings
from agnes.common.make_nn import make_fc


class MlpFamily(BasePolicy, ABC):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space):
        super().__init__(observation_space, action_space)

        self.actor_head = make_fc(self.obs_space, self.actions_n, num_layers=3, hidden_size=64)
        self.critic_head = make_fc(self.obs_space, 1, num_layers=3, hidden_size=64)

        self.apply(get_weights_init('tanh'))


class MLPDiscrete(MlpFamily):
    def forward(self, x):
        if x.ndimension() > 2:
            x = x.view(tuple(x.shape[:-self.obs_space_n]) + (self.obs_space,))

        state_value = self.critic_head(x)

        probs = self.actor_head(x)

        dist = Categorical(logits=probs)

        return dist, state_value


class MLPContinuous(MlpFamily):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space):
        super().__init__(observation_space, action_space)
        logstd = 0.0
        self.log_std = nn.Parameter(torch.ones(self.actions_n) * logstd)

    def forward(self, x):
        if x.ndimension() > 2:
            x = x.view(tuple(x.shape[:-self.obs_space_n]) + (self.obs_space,))

        state_value = self.critic_head(x)

        mu = self.actor_head(x).squeeze(-1)

        state_value = state_value.view(-1)

        std = self.log_std.expand_as(mu).exp()
        dist = Normal(mu, std)

        return dist, state_value.squeeze(-1)


def MLP(observation_space: spaces.Space, action_space: spaces.Space):
    if len(observation_space.shape) == 3:
        warnings.warn("Looks like you're using MLP for images. CNN is recommended.")

    if isinstance(action_space, spaces.Box):
        return MLPContinuous(observation_space, action_space)
    else:
        return MLPDiscrete(observation_space, action_space)
