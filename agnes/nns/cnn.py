from abc import ABC
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from agnes.common.init_weights import get_weights_init
from gym import spaces
import numpy
from agnes.nns.base import BasePolicy
from agnes.common.make_nn import Cnn, CnnSmallBody, make_fc


class CnnFamily(BasePolicy, ABC):
    pass


class CNNDiscreteCopy(CnnFamily):
    def __init__(self, observation_space, action_space: spaces.Space, policy_fn=None, value_fn=None):
        super(CNNDiscreteCopy, self).__init__(observation_space, action_space)

        input_shape = observation_space.shape

        # actor's layer
        if policy_fn is None:
            self.actor_head = Cnn(input_shape, self.actions_n, body=CnnSmallBody)
        else:
            self.actor_head = policy_fn(self.actions_n)

        # critic's layer
        if value_fn is None:
            self.critic_head = Cnn(input_shape, 1)
        else:
            self.critic_head = value_fn(1)

        self.actor_head.conv.apply(get_weights_init('relu'))

        self.actor_head.head.apply(get_weights_init(numpy.sqrt(0.01)))

        self.critic_head.conv.apply(get_weights_init('relu'))

        self.critic_head.head.apply(get_weights_init(numpy.sqrt(0.01)))

    def forward(self, x):
        state_value = self.critic_head(x)

        policy = self.actor_head(x)

        dist = Categorical(logits=policy)

        return dist, state_value


class CNNDiscreteShared(CnnFamily):
    hidden_size = 512

    def __init__(self, observation_space, action_space: spaces.Space):
        super(CNNDiscreteShared, self).__init__(observation_space, action_space)
        input_shape = observation_space.shape

        self.conv = Cnn(input_shape, self.hidden_size, num_layers=1, activate_last=True, body=CnnSmallBody)

        # actor's layer
        self.actor_head = make_fc(self.hidden_size, self.actions_n, num_layers=1)

        # critic's layer
        self.critic_head = make_fc(self.hidden_size, 1, num_layers=1)
        self.conv.apply(get_weights_init('relu'))

        self.actor_head.apply(get_weights_init(numpy.sqrt(0.01)))

        self.critic_head.apply(get_weights_init(numpy.sqrt(0.01)))

        self.apply(get_weights_init('relu'))

    def forward(self, x):
        both = self.conv(x)

        state_value = self.critic_head(both)
        policy = self.actor_head(both)

        dist = Categorical(logits=policy)

        return dist, state_value
