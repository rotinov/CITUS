import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from agnes.common.init_weights import get_weights_init
from gym import spaces
import numpy
from agnes.nns.base import BasePolicy


class ImagePreprocess(nn.Module):
    def __init__(self, normalize=True, swap_axis=True):
        super(ImagePreprocess, self).__init__()

        self.normalize = normalize
        self.swap_axis = swap_axis

    def forward(self, x):
        if self.normalize:
            x = x / 255.

        if self.swap_axis:
            x = x.permute(0, 3, 1, 2)

        return x


class CnnSmallBody(nn.Module):
    def __init__(self):
        super(CnnSmallBody, self).__init__()

        self.conv = nn.Sequential(ImagePreprocess(),
                                  nn.Conv2d(4, 8, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(8, 16, 4, stride=2),
                                  nn.ReLU())

    def forward(self, x):
        # (4, 84, 84)
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
    def __init__(self, input_shape=(4, 84, 84)):
        super(CnnBody, self).__init__()

        test_input = torch.rand(input_shape).unsqueeze(0)

        self.conv = nn.Sequential(ImagePreprocess(),
                                  nn.Conv2d(in_channels=4,
                                            out_channels=32,
                                            kernel_size=8,
                                            stride=4,
                                            padding=0),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=32,
                                            out_channels=64,
                                            kernel_size=4,
                                            stride=2,
                                            padding=0),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=64,
                                            out_channels=64,
                                            kernel_size=3,
                                            stride=1,
                                            padding=0),
                                  nn.ReLU())

        test_output = self.conv(test_input)
        self.test_output_shape = tuple(test_output.shape)

    @property
    def output_size(self):
        return self.test_output_shape

    def forward(self, x):
        # (4, 84, 84)
        cv = self.conv(x)
        # (64, 7, 7)

        return cv


class CnnHead(nn.Module):
    def __init__(self, cnn_out_size, output):
        super(CnnHead, self).__init__()

        self.cnn_out_size = int(numpy.prod(cnn_out_size))

        self.fc = nn.Sequential(nn.Linear(self.cnn_out_size, 512),
                                nn.ReLU(),
                                nn.Linear(512, output))

    def forward(self, cv):
        cv_f = cv.view(-1, self.cnn_out_size)

        return self.fc(cv_f)


class Cnn(nn.Module):
    def __init__(self, input_shape, output):
        super(Cnn, self).__init__()

        self.conv = CnnBody(input_shape=input_shape)

        self.head = CnnHead(self.conv.output_size, output)

    def forward(self, x):
        # (4, 84, 84) -> (64, 7, 7)
        cv = self.conv(x)

        return self.head(cv)


class CNNDiscreteCopy(nn.Module, BasePolicy):
    def __init__(self, input_shape, action_space=spaces.Discrete(5), policy_fn=None, value_fn=None):
        super(CNNDiscreteCopy, self).__init__()
        self.action_space = action_space

        # actor's layer
        if policy_fn is None:
            self.actor_head = Cnn(input_shape, action_space.n)
        else:
            self.actor_head = policy_fn(action_space.n)

        # critic's layer
        if value_fn is None:
            self.critic_head = Cnn(input_shape, 1)
        else:
            self.critic_head = value_fn(1)

        self.actor_head.conv.apply(get_weights_init('relu'))

        self.actor_head.head.apply(get_weights_init(numpy.sqrt(0.01)))

        self.critic_head.conv.apply(get_weights_init('relu'))

        self.critic_head.head.apply(get_weights_init(numpy.sqrt(0.01)))

        # self.apply(get_weights_init('relu'))

    @staticmethod
    def type_of_out():
        return torch.int16

    # noinspection PyUnboundLocalVariable
    def forward(self, x):
        state_value = self.critic_head(x)

        policy = self.actor_head(x)

        dist = Categorical(logits=policy)

        return dist, state_value


class CNNDiscreteShared(nn.Module, BasePolicy):
    def __init__(self, input_shape, action_space=spaces.Discrete(5)):
        super(CNNDiscreteShared, self).__init__()
        self.action_space = action_space

        self.conv = CnnBody(input_shape=input_shape)

        # actor's layer
        self.actor_head = CnnHead(self.conv.output_size, action_space.n)

        # critic's layer
        self.critic_head = CnnHead(self.conv.output_size, 1)
        self.conv.apply(get_weights_init('relu'))

        self.actor_head.apply(get_weights_init(numpy.sqrt(0.01)))

        self.critic_head.apply(get_weights_init(numpy.sqrt(0.01)))

        self.apply(get_weights_init('relu'))

    @staticmethod
    def type_of_out():
        return torch.int16

    def forward(self, x):
        both = self.conv(x)

        state_value = self.critic_head(both)

        policy = self.actor_head(both)

        dist = Categorical(logits=policy)

        return dist, state_value


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

        input_shape = observation_space.shape

        if self.nn == CNNDiscreteShared:
            return self.nn(input_shape, action_space)
        else:
            return self.nn(input_shape, action_space, self.policy_nn, self.value_nn)


CNN = CNNChooser()
