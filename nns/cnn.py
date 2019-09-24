import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from common.init_weights import get_weights_init
from gym import spaces
import numpy


class CnnSmallBody(nn.Module):
    cnn_out_size = 16 * 9 * 9

    def __init__(self):
        super(CnnSmallBody, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(4, 8, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(8, 16, 4, stride=2),
                                  nn.ReLU())

    def forward(self, x):
        # (4, 84, 84)
        x = x / 255.

        cv = self.conv(x)
        # (16, 9, 9)

        return cv


class CnnSmallHead(nn.Module):
    cnn_out_size = 16 * 9 * 9

    def __init__(self, output):
        super(CnnSmallHead, self).__init__()

        self.fc = nn.Sequential(nn.Linear(self.cnn_out_size, 128),
                                nn.ReLU(),
                                nn.Linear(128, output))

    def forward(self, cv):
        cv_f = cv.view(-1, self.cnn_out_size)

        return self.fc(cv_f)


class CnnSmall(nn.Module):
    cnn_out_size = 16 * 9 * 9

    def __init__(self, output):
        super(CnnSmall, self).__init__()

        self.conv = CnnSmallBody()

        self.head = CnnSmallHead(output)

    def forward(self, x):
        # (4, 84, 84)
        cv = self.conv(x)
        # (16, 9, 9)

        return self.head(cv)


class CNNDiscreteShared(nn.Module):
    np_type = numpy.int16

    def __init__(self,
                 observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
                 action_space=spaces.Discrete(5)):
        super(CNNDiscreteShared, self).__init__()
        self.action_space = action_space

        self.conv = CnnSmallBody()

        # actor's layer
        self.actor_head = nn.Sequential(CnnSmallHead(action_space.n),
                                        nn.Softmax(-1))

        # critic's layer
        self.critic_head = CnnSmallHead(1)

        self.apply(get_weights_init('relu'))

    def forward(self, x):
        compressed = False
        if x.ndimension() == 5:
            compressed = True
            f = x.shape[0]
            s = x.shape[1]
            x = x.view(f * s, x.shape[2], x.shape[3], x.shape[4])

        xt = x.permute(0, 3, 1, 2)

        both = self.conv(xt)

        state_value = self.critic_head(both)

        policy = self.actor_head(both)

        if compressed:
            state_value = state_value.view(f, s)
            policy = policy.view(f, s, -1)

        dist = Categorical(policy)

        return dist, state_value

    def get_value(self, x):
        compressed = False
        if x.ndimension() == 5:
            compressed = True
            f = x.shape[0]
            s = x.shape[1]
            x = x.view(f * s, x.shape[2], x.shape[3], x.shape[4])

        xt = x.permute(0, 3, 1, 2)
        both = self.conv(xt)

        state_value = self.critic_head(both).detach()

        if compressed:
            state_value = state_value.view(f, s)
        else:
            state_value.squeeze(-1)

        return state_value.cpu().numpy()

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
    return CNNDiscreteShared(observation_space, action_space)
