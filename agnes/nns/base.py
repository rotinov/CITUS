import abc
import torch
from torch.distributions import Categorical, Normal
from gym import spaces
import numpy

from agnes.common.init_weights import get_weights_init


class _BasePolicy(torch.nn.Module, abc.ABC):
    action_space = None
    actions_n = None
    obs_space_n = None
    obs_space = 1
    is_discrete = True

    layers_num = 3
    hidden_size = 64

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space):
        super(_BasePolicy, self).__init__()
        self.action_space = action_space

        if isinstance(action_space, spaces.Discrete):
            self.is_discrete = True
            self.actions_n = action_space.n
        else:
            self.is_discrete = False
            self.actions_n = action_space.shape[0]

        self.obs_space_n = len(observation_space.shape)
        for item in observation_space.shape:
            self.obs_space *= item

    def type_of_out(self):
        return torch.int16 if self.is_discrete else torch.float32

    @abc.abstractmethod
    def forward(self, *args):
        pass

    def get_action(self, x, done):
        if self.is_discrete:
            return self._get_for_env(x, done)
        else:
            smpled, _, outs = self._get_for_env(x, done)
            action = numpy.clip(smpled, self.action_space.low[0], self.action_space.high[0])
            return action, smpled, outs

    def _get_for_env(self, x, done):
        if x.ndimension() < len(self.action_space.shape) + 1:
            x.unsqueeze_(0)

        dist, state_value = self.forward(x)

        smpled = dist.sample()

        log_prob = dist.log_prob(smpled)

        return (smpled.detach().squeeze(-1).cpu().numpy(),
                smpled.detach().squeeze(-1).cpu().numpy(),
                (log_prob.detach().squeeze(-1).cpu().numpy(),
                 state_value.detach().squeeze(-1).cpu().numpy()))

    def get_val(self, states):
        return self.forward(states)[1]

    def reset(self):
        pass
