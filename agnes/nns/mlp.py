import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from agnes.common.init_weights import get_weights_init
from gym import spaces
import numpy
import warnings


def mlp2l(x, y):
    return nn.Sequential(nn.Linear(x, 64),
                         nn.Tanh(),
                         nn.Linear(64, 64),
                         nn.Tanh(),
                         nn.Linear(64, y))


def mlp1l(x, y):
    return nn.Sequential(nn.Linear(x, 128),
                         nn.Tanh(),
                         nn.Linear(128, y))


class MLPDiscrete(nn.Module):
    np_type = numpy.int16
    obs_space = 1

    def __init__(self,
                 observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
                 action_space=spaces.Discrete(5), mlp_fun=mlp1l):
        super(MLPDiscrete, self).__init__()
        self.action_space = action_space

        self.obs_space_n = len(observation_space.shape)
        for item in observation_space.shape:
            self.obs_space *= item

        # actor's layer
        self.actor_head = nn.Sequential(mlp_fun(self.obs_space, action_space.n),
                                        nn.Softmax(-1))

        # critic's layer
        self.critic_head = mlp_fun(self.obs_space, 1)

        self.apply(get_weights_init('tanh'))

    @staticmethod
    def type_of_out():
        return torch.int16

    def forward(self, x):
        if x.ndimension() > 2:
            x = x.view(tuple(x.shape[:-self.obs_space_n]) + (self.obs_space,))

        compressed = False
        if x.ndimension() == 3:
            compressed = True
            f = x.shape[0]
            s = x.shape[1]
            x = x.view(f * s, x.shape[2])

        state_value = self.critic_head(x)

        probs = self.actor_head(x)

        if compressed:
            state_value = state_value.view(f, s)
            probs = probs.view(f, s, -1)

        dist = Categorical(probs)

        return dist, state_value

    def get_action(self, x):
        if x.ndimension() < 2:
            x.unsqueeze_(0)
        dist, state_value = self.forward(x)
        action = dist.sample()

        return action.detach().squeeze(-1).cpu().numpy(), \
               action.detach().squeeze(-1).cpu().numpy(), \
               (dist.log_prob(action).detach().squeeze(-1).cpu().numpy(),
                state_value.detach().squeeze(-1).cpu().numpy())


class MLPContinuous(nn.Module):
    np_type = numpy.float32
    obs_space = 1

    def __init__(self,
                 observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
                 action_space=spaces.Box(low=-10, high=10, shape=(1,)),
                 logstd=0.0, mlp_fun=mlp2l):
        super(MLPContinuous, self).__init__()
        self.action_space = action_space

        self.obs_space_n = len(observation_space.shape)
        for item in observation_space.shape:
            self.obs_space *= item

        # actor's layer
        self.actor_head = mlp_fun(observation_space.shape[0], action_space.shape[0])

        # critic's layer
        self.critic_head = mlp_fun(observation_space.shape[0], 1)

        self.log_std = nn.Parameter(torch.ones(action_space.shape[0]) * logstd)

        self.apply(get_weights_init('tanh'))

    @staticmethod
    def type_of_out():
        return torch.float32

    def forward(self, x):
        if x.ndimension() > 2:
            x = x.view(tuple(x.shape[:-self.obs_space_n]) + (self.obs_space,))

        compressed = False
        if x.ndimension() == 3:
            compressed = True
            f = x.shape[0]
            s = x.shape[1]
            x = x.view(f * s, x.shape[2])

        state_value = self.critic_head(x)

        mu = self.actor_head(x)

        if compressed:
            state_value = state_value.view(f, s)
            mu = mu.view(f, s, -1)
        else:
            state_value = state_value.view(-1)

        std = self.log_std.expand_as(mu).exp()
        dist = Normal(mu, std)

        return dist, state_value.squeeze(0)

    def get_action(self, x):
        if x.ndimension() < 2:
            x.unsqueeze_(0)
        dist, state_value = self.forward(x)
        smpled = dist.sample()
        action = torch.clamp(smpled, self.action_space.low[0], self.action_space.high[0])

        return action.detach().cpu().numpy(), \
               smpled.detach().cpu().numpy(), \
               (dist.log_prob(smpled).detach().cpu().numpy(),
                state_value.detach().cpu().numpy())


def MLP(observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
        action_space=spaces.Discrete(5),
        logstd=0.0):
    if len(observation_space.shape) == 3:
        warnings.warn("Looks like you're using MLP for images. CNN is recommended.")


    if isinstance(action_space, spaces.Box):
        return MLPContinuous(observation_space, action_space, logstd)
    else:
        return MLPDiscrete(observation_space, action_space)
