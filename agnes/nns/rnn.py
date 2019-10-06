import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from agnes.common.init_weights import get_weights_init
from agnes.nns.base import BasePolicy
from gym import spaces
import numpy
import warnings
from agnes.common.make_fc import make_fc


class RecurrentFamily:
    pass


class RNNDiscrete(nn.Module, BasePolicy, RecurrentFamily):
    np_type = numpy.int16
    obs_space = 1
    hidden_size = 64
    layers_num = 1

    def __init__(self,
                 observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
                 action_space=spaces.Discrete(5), mlp_fun=make_fc):
        super(RNNDiscrete, self).__init__()
        self.action_space = action_space

        self.obs_space_n = len(observation_space.shape)
        for item in observation_space.shape:
            self.obs_space *= item

        # actor's layer
        self.actor_body_fc = make_fc(self.obs_space, self.hidden_size, num_layers=1)
        self.actor_body_rnn = nn.RNN(self.hidden_size,
                                     self.hidden_size,
                                     self.layers_num,
                                     batch_first=True)
        self.actor_head = make_fc(self.hidden_size, action_space.n, num_layers=1)

        # critic's layer
        self.critic_head = make_fc(self.obs_space, 1, num_layers=4, hidden_size=128)

        self.apply(get_weights_init('tanh'))

        self.hs = None

    @staticmethod
    def type_of_out():
        return torch.int16

    def get_val(self, x):
        state_value = self.critic_head(x)
        return state_value

    def forward(self, x, hs):
        state_value = self.critic_head(x)

        shapes = None
        if len(x.shape) > 2:
            shapes = x.shape[:-1]
            x.view(-1, self.obs_space)

        fc_out = self.actor_body_fc(x)

        if shapes is not None:
            fc_out = fc_out.view(shapes + (-1,))
        else:
            fc_out = fc_out.unsqueeze(1)

        rnn_out, hs = self.actor_body_rnn(fc_out, hs)

        rnn_out = rnn_out.contiguous().view(-1, self.hidden_size)

        probs = self.actor_head(rnn_out)

        dist = Categorical(logits=probs)

        return dist, hs, state_value

    def get_action(self, x, dones):
        assert x.ndimension() == 1 + self.obs_space_n, "Only batches are supported"
        hs = self.hs
        if self.hs is not None:
            self.hs = self.hs * (1. - dones.unsqueeze(-1))
            hs = self.hs

        dist, self.hs, state_value = self.forward(x, self.hs)

        if hs is None:
            hs = torch.zeros_like(self.hs)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return (action.detach().squeeze(-1).cpu().numpy(),
                action.detach().squeeze(-1).cpu().numpy(),
                (hs.detach().squeeze(-1).cpu().numpy(),
                 log_prob.detach().squeeze(-1).cpu().numpy(),
                 state_value.detach().squeeze(-1).cpu().numpy()))


def RNN(observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
        action_space=spaces.Discrete(5),
        logstd=0.0):
    return RNNDiscrete(observation_space, action_space)
