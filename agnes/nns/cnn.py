import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from agnes.common.init_weights import get_weights_init
from gym import spaces
import numpy


class CnnSmallBody(nn.Module):
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
    def __init__(self, output):
        super(CnnSmall, self).__init__()

        self.conv = CnnSmallBody()

        self.head = CnnSmallHead(output)

    def forward(self, x):
        # (4, 84, 84)
        cv = self.conv(x)
        # (16, 9, 9)

        return self.head(cv)


class CnnBody(nn.Module):
    def __init__(self):
        super(CnnBody, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1),
                                  nn.ReLU())

    def forward(self, x):
        # (4, 84, 84)
        x = x / 255.

        cv = self.conv(x)
        # (64, 7, 7)

        return cv


class CnnHead(nn.Module):
    cnn_out_size = 64 * 7 * 7

    def __init__(self, output):
        super(CnnHead, self).__init__()

        self.fc = nn.Sequential(nn.Linear(self.cnn_out_size, 512),
                                nn.ReLU(),
                                nn.Linear(512, output))

    def forward(self, cv):
        cv_f = cv.view(-1, self.cnn_out_size)

        return self.fc(cv_f)


class Cnn(nn.Module):
    def __init__(self, output):
        super(Cnn, self).__init__()

        self.conv = CnnBody()

        self.head = CnnHead(output)

    def forward(self, x):
        # (4, 84, 84)
        cv = self.conv(x)
        # (64, 7, 7)

        return self.head(cv)


class CNNDiscreteCopy(nn.Module):
    def __init__(self, action_space=spaces.Discrete(5), policy_fn=None, value_fn=None):
        super(CNNDiscreteCopy, self).__init__()
        self.action_space = action_space

        self.conv = CnnBody()

        # actor's layer
        if policy_fn is None:
            self.actor_head = nn.Sequential(CnnBody(),
                                            CnnHead(action_space.n),
                                            nn.Softmax(-1))
        else:
            self.actor_head = nn.Sequential(policy_fn(action_space.n),
                                            nn.Softmax(-1))

        # critic's layer
        if value_fn is None:
            self.critic_head = nn.Sequential(CnnBody(),
                                             CnnHead(1))
        else:
            self.critic_head = value_fn(1)

        self.apply(get_weights_init('relu'))

    @staticmethod
    def type_of_out():
        return torch.int16

    def forward(self, x):
        compressed = False
        if x.ndimension() == 5:
            compressed = True
            f = x.shape[0]
            s = x.shape[1]
            x = x.view(f * s, x.shape[2], x.shape[3], x.shape[4])

        xt = x.permute(0, 3, 1, 2)

        state_value = self.critic_head(xt)

        policy = self.actor_head(xt)

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

        state_value = self.critic_head(xt).detach()

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


class CNNDiscreteShared(nn.Module):

    def __init__(self, action_space=spaces.Discrete(5)):
        super(CNNDiscreteShared, self).__init__()
        self.action_space = action_space

        self.conv = CnnBody()

        # actor's layer
        self.actor_head = nn.Sequential(CnnHead(action_space.n),
                                        nn.Softmax(-1))

        # critic's layer
        self.critic_head = CnnHead(1)

        self.apply(get_weights_init('relu'))

    @staticmethod
    def type_of_out():
        return torch.int16

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


class CNNChooser:
    def __init__(self, shared=True, policy_nn=None, value_nn=None):
        if shared:
            if policy_nn is not None or value_nn is not None:
                raise NameError('Shared network with custom layers is not supported for now.')

            self.nn = CNNDiscreteShared
        else:
            self.nn = CNNDiscreteCopy
            self.policy_nn = policy_nn
            self.value_nn = value_nn

    def __call__(self, observation_space, action_space):
        if isinstance(action_space, spaces.Box):
            raise NameError('Continuous environments are not supported yet.')

        if self.nn == CNNDiscreteShared:
            return self.nn(action_space)
        else:
            return self.nn(action_space, self.policy_nn, self.value_nn)


CNN = CNNChooser()
