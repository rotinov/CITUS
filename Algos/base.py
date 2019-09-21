import torch


class BaseBuffer(object):
    def append(self, transition):
        pass

    def rollout(self):
        pass

    def learn(self, data, nminibatches):
        pass

    def __len__(self):
        pass


class BaseAlgo:
    _nn: torch.nn.Module

    def __call__(self, state):
        pass

    def experience(self, transition):
        pass

    def learn(self, data):
        pass

    def update(self, from_agent):
        pass
