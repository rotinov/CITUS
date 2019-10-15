import abc
import torch
from torch.distributions import Categorical, Normal
from gym import spaces
import numpy
import warnings

from agnes.nns.base import _BasePolicy
from agnes.common import make_nn
from agnes.common.init_weights import get_weights_init


class _MlpFamily(_BasePolicy, abc.ABC):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space):
        super().__init__(observation_space, action_space)

        self.actor_head = make_nn.make_fc(self.obs_space, self.actions_n,
                                          num_layers=self.layers_num, hidden_size=self.hidden_size)
        self.critic_head = make_nn.make_fc(self.obs_space, 1,
                                           num_layers=self.layers_num, hidden_size=self.hidden_size)

        self.apply(get_weights_init('tanh'))


class MLPDiscrete(_MlpFamily):
    def forward(self, x):
        if x.ndimension() > 2:
            x = x.view(tuple(x.shape[:-self.obs_space_n]) + (self.obs_space,))

        state_value = self.critic_head(x)

        probs = self.actor_head(x)

        dist = Categorical(logits=probs)

        return dist, state_value


class MLPContinuous(_MlpFamily):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space):
        super().__init__(observation_space, action_space)
        logstd = 0.0
        self.log_std = torch.nn.Parameter(torch.ones(self.actions_n) * logstd)

    def forward(self, x):
        if x.ndimension() > 2:
            x = x.view(tuple(x.shape[:-self.obs_space_n]) + (self.obs_space,))

        state_value = self.critic_head(x)

        mu = self.actor_head(x).squeeze(-1)

        state_value = state_value.view(-1)

        std = self.log_std.expand_as(mu).exp()
        dist = Normal(mu, std)

        return dist, state_value.squeeze(-1)
