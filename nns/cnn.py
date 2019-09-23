import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from common.init_weights import weights_init
from gym import spaces
import numpy


class CnnSmall(nn.Module):
    def __init__(self, output):
        super(CnnSmall, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU())

        self.fc = nn.Sequential(nn.Linear(5184, 128),
                                nn.ReLU(),
                                nn.Linear(128, output))

    def forward(self, x):
        x = x / 255.

        cv = self.conv(x)

        cv_f = cv.view(-1, 5184)

        return self.fc(cv_f)


class MLPDiscrete(nn.Module):
    np_type = numpy.int16

    def __init__(self,
                 observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
                 action_space=spaces.Discrete(5)):
        super(MLPDiscrete, self).__init__()
        self.action_space = action_space

        # actor's layer
        self.actor_head = nn.Sequential(CnnSmall(action_space.n),
                                        nn.Softmax(-1))

        # critic's layer
        self.critic_head = CnnSmall(1)

        self.apply(weights_init)

    def forward(self, x):
        xt = x.permute(0, 3, 1, 2)
        state_value = self.critic_head(xt)

        policy = self.actor_head(xt)
        # print(policy)
        dist = Categorical(policy)

        return dist, state_value

    def get_action(self, x):
        dist, state_value = self.forward(x)
        action = dist.sample()

        return action.detach().cpu().numpy(), action.detach().cpu().numpy(), (dist.log_prob(action).detach().cpu().numpy(), state_value.detach().cpu().item())


def CNN(observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
        action_space=spaces.Discrete(5),
        logstd=0.0, simple=False):

    return MLPDiscrete(observation_space, action_space)
