import torch
from abc import ABCMeta, abstractmethod


class BaseBuffer(object):
    def append(self, transition):
        pass

    def rollout(self):
        pass

    def learn(self, data, nminibatches):
        pass

    def __len__(self):
        pass


class BaseAlgo(metaclass=ABCMeta):
    _nn: torch.nn.Module

    @abstractmethod
    def __init__(self, nn,
                 observation_space,
                 action_space,
                 cnfg):
        pass

    def __call__(self, state):
        pass

    def experience(self, transition):
        pass

    def learn(self, data):
        pass

    def update(self, from_agent):
        pass

    def to(self, device):
        pass
