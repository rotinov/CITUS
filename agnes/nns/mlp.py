import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from agnes.common.init_weights import get_weights_init
from agnes.nns.base import BasePolicy
from gym import spaces
import numpy
import warnings
from agnes.common.make_fc import make_fc


class MLPDiscrete(nn.Module, BasePolicy):
    np_type = numpy.int16
    obs_space = 1

    def __init__(self,
                 observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
                 action_space=spaces.Discrete(5), mlp_fun=make_fc):
        super(MLPDiscrete, self).__init__()
        self.action_space = action_space

        self.obs_space_n = len(observation_space.shape)
        for item in observation_space.shape:
            self.obs_space *= item

        # actor's layer
        self.actor_head = make_fc(self.obs_space, action_space.n, num_layers=3, hidden_size=64)

        # critic's layer
        self.critic_head = make_fc(self.obs_space, 1, num_layers=3, hidden_size=64)

        self.apply(get_weights_init('tanh'))

    @staticmethod
    def type_of_out():
        return torch.int16

    def forward(self, x):
        if x.ndimension() > 2:
            x = x.view(tuple(x.shape[:-self.obs_space_n]) + (self.obs_space,))

        state_value = self.critic_head(x)

        probs = self.actor_head(x)

        dist = Categorical(logits=probs)

        return dist, state_value


class MLPContinuous(nn.Module, BasePolicy):
    obs_space = 1

    def __init__(self,
                 observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
                 action_space=spaces.Box(low=-10, high=10, shape=(1,)),
                 logstd=0.0, mlp_fun=make_fc):
        super(MLPContinuous, self).__init__()
        self.action_space = action_space

        self.obs_space_n = len(observation_space.shape)
        for item in observation_space.shape:
            self.obs_space *= item

        # actor's layer
        self.actor_head = make_fc(observation_space.shape[0], action_space.shape[0], num_layers=3, hidden_size=64)

        # critic's layer
        self.critic_head = make_fc(observation_space.shape[0], 1, num_layers=3, hidden_size=64)

        self.log_std = nn.Parameter(torch.ones(action_space.shape[0]) * logstd)

        self.apply(get_weights_init('tanh'))

    @staticmethod
    def type_of_out():
        return torch.float32

    def forward(self, x):
        if x.ndimension() > 2:
            x = x.view(tuple(x.shape[:-self.obs_space_n]) + (self.obs_space,))

        state_value = self.critic_head(x)

        mu = self.actor_head(x).squeeze(-1)

        state_value = state_value.view(-1)

        std = self.log_std.expand_as(mu).exp()
        dist = Normal(mu, std)

        return dist, state_value.squeeze(-1)

    def get_action(self, *args):
        return self.get_action_n_apply(*args,
                                       lambda z: torch.clamp(z, self.action_space.low[0], self.action_space.high[0])
                                       )


def MLP(observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
        action_space=spaces.Discrete(5),
        logstd=0.0):
    if len(observation_space.shape) == 3:
        warnings.warn("Looks like you're using MLP for images. CNN is recommended.")

    if isinstance(action_space, spaces.Box):
        return MLPContinuous(observation_space, action_space, logstd)
    else:
        return MLPDiscrete(observation_space, action_space)
