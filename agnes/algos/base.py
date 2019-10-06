import torch
from abc import ABCMeta, abstractmethod


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


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
    _nnet: torch.nn.Module

    @abstractmethod
    def __init__(self, *args):
        pass

    def __call__(self, state, done):
        pass

    def experience(self, transition):
        pass

    def learn(self, data):
        pass

    def update(self, from_agent):
        pass

    def to(self, device):
        pass
