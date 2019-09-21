import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from defaults import weights_init
from gym import spaces
import numpy


def policy(x, y):
    return nn.Sequential(nn.Linear(x, 256),
                         nn.Tanh(),
                         nn.Linear(256, 256),
                         nn.Tanh(),
                         nn.Linear(256, y))


def value(x):
    return nn.Sequential(nn.Linear(x, 256),
                         nn.Tanh(),
                         nn.Linear(256, 256),
                         nn.Tanh(),
                         nn.Linear(256, 1))


def classic_policy(x, y):
    return nn.Sequential(nn.Linear(x, 64),
                         nn.Tanh(),
                         nn.Linear(64, 64),
                         nn.Tanh(),
                         nn.Linear(64, y))


def classic_value(x):
    return nn.Sequential(nn.Linear(x, 64),
                         nn.Tanh(),
                         nn.Linear(64, 1))


class MLPDiscrete(nn.Module):
    np_type = numpy.int16

    def __init__(self,
                 observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
                 action_space=spaces.Discrete(5), policy_fun=policy, value_fun=value):
        super(MLPDiscrete, self).__init__()
        self.action_space = action_space

        # actor's layer
        self.actor_head = nn.Sequential(policy_fun(observation_space.shape[0], action_space.n),
                                        nn.Softmax(-1))

        # critic's layer
        self.critic_head = value_fun(observation_space.shape[0])

        self.apply(weights_init)

    def forward(self, x):
        state_value = self.critic_head(x)

        policy = self.actor_head(x)
        # print(policy)
        dist = Categorical(policy)

        return dist, state_value

    def get_action(self, x):
        dist, _ = self.forward(x)
        action = dist.sample()

        state_value = self.critic_head(x)
        return action.detach().cpu().numpy(), action.detach().cpu().numpy(), (dist.log_prob(action).detach().cpu().numpy(), state_value.detach().cpu().item())


class MLPContinuous(nn.Module):
    np_type = numpy.float32

    def __init__(self,
                 observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
                 action_space=spaces.Box(low=-10, high=10, shape=(1,)),
                 logstd=0.0, policy_fun=policy, value_fun=value):
        super(MLPContinuous, self).__init__()
        self.action_space = action_space

        # actor's layer
        self.actor_head = policy_fun(observation_space.shape[0], action_space.shape[0])

        # critic's layer
        self.critic_head = value_fun(observation_space.shape[0])

        self.log_std = nn.Parameter(torch.ones(action_space.shape[0]) * logstd)

        self.apply(weights_init)

    def forward(self, x):
        state_value = self.critic_head(x)

        mu = self.actor_head(x)
        std = self.log_std.expand_as(mu).exp()
        dist = Normal(mu, std)

        return dist, state_value

    def get_action(self, x):
        dist, _ = self.forward(x)
        smpled = dist.sample()
        action = torch.clamp(smpled, self.action_space.low[0], self.action_space.high[0])

        state_value = self.critic_head(x)
        return action.detach().cpu().numpy(), smpled.detach().cpu().numpy(), (dist.log_prob(smpled).detach().cpu().numpy(), state_value.detach().cpu().item())


class MLPCLSS:
    def __init__(self):
        pass

    def __call__(self,
                 observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
                 action_space=spaces.Discrete(5),
                 logstd=0.0, simple=False):
        if isinstance(action_space, spaces.Box):
            if simple:
                return MLPContinuous(observation_space, action_space, logstd, policy_fun=classic_policy, value_fun=classic_value)
            else:
                return MLPContinuous(observation_space, action_space, logstd)
        else:
            if simple:
                return MLPDiscrete(observation_space, action_space, policy_fun=classic_policy, value_fun=classic_value)
            else:
                return MLPDiscrete(observation_space, action_space)


MLP = MLPCLSS()
