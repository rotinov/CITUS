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
        compressed = False
        if x.ndimension() == 5:
            compressed = True
            f = x.shape[0]
            s = x.shape[1]
            x = x.view(f*s, x.shape[2], x.shape[3], x.shape[4])

        xt = x.permute(0, 3, 1, 2)
        state_value = self.critic_head(xt)

        policy = self.actor_head(xt)

        if compressed:
            state_value = state_value.view(f, s)
            policy = policy.view(f, s, -1)

        dist = Categorical(policy)

        return dist, state_value

    def get_action(self, x: torch.FloatTensor):
        if x.ndimension() < 4:
            x.unsqueeze_(0)

        dist, state_value = self.forward(x)
        action = dist.sample()

        return action.detach().cpu().numpy(), \
               action.detach().cpu().numpy(), \
               (dist.log_prob(action).detach().cpu().numpy(),
                state_value.detach().squeeze(-1).cpu().numpy())


def CNN(observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
        action_space=spaces.Discrete(5),
        logstd=0.0, simple=False):

    return MLPDiscrete(observation_space, action_space)
