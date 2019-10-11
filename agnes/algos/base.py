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
        with torch.no_grad():
            if self._device == torch.device('cpu'):
                return self._nnet.get_action(torch.FloatTensor(state), torch.FloatTensor(done))
            else:
                return self._nnet.get_action(torch.cuda.FloatTensor(state), torch.cuda.FloatTensor(done))

    def reset(self):
        self._nnet.reset()

    def update(self, from_agent):
        assert not self._trainer

        self._nnet.load_state_dict(from_agent._nnet.state_dict())

        return True

    def get_state_dict(self):
        assert self._trainer
        return self._nnet.state_dict()

    def load_state_dict(self, state_dict):
        return self._nnet.load_state_dict(state_dict)

    def save(self, filename):
        torch.save(self._nnet.state_dict(), filename)

    def load(self, filename):
        self._nnet.load_state_dict(torch.load(filename))

    def get_nn_instance(self):
        assert self._trainer
        return self._nnet

    def experience(self, transition):
        pass

    def learn(self, data):
        pass

    def to(self, device):
        pass

    def device_info(self):
        if self._device.type == 'cuda':
            return torch.cuda.get_device_name(device=self._device)
        else:
            return 'CPU'
