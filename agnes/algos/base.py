import torch
import numpy
import abc

from typing import *

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


class _BaseBuffer(object):
    nsteps: int

    def __init__(self, nsteps: int):
        self.nsteps = nsteps

    def append(self, transition):
        pass

    def rollout(self):
        pass

    def learn(self, data, nminibatches):
        pass

    def __len__(self):
        pass


class LossArgs:
    __slots__ = ["new_log_probs", "old_log_probs", "advantages", "new_vals", "old_vals", "returns", "entropies"]

    new_log_probs: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    new_vals: torch.Tensor
    old_vals: torch.Tensor
    returns: torch.Tensor
    entropies: torch.Tensor


class _BaseAlgo(abc.ABC):
    _nnet: torch.nn.Module

    meta = "BASE"

    _trainer: bool = True
    _device: torch.device = torch.device("cpu")

    @abc.abstractmethod
    def __init__(self, *args):
        pass

    def __call__(self, state: numpy.ndarray, done: numpy.ndarray):
        with torch.no_grad():
            if self._device == torch.device('cpu'):
                return self._nnet.get_action(
                    torch.from_numpy(numpy.asarray(state, dtype=numpy.float32)),
                    torch.from_numpy(numpy.asarray(done))
                )
            else:
                return self._nnet.get_action(
                    torch.from_numpy(numpy.asarray(state, dtype=numpy.float32)).to(self._device),
                    torch.from_numpy(numpy.asarray(done)).to(self._device)
                )

    def reset(self):
        self._nnet.reset()

    def update(self, from_agent):
        assert not self._trainer

        self.load_state_dict(from_agent.get_state_dict())

        return True

    def is_trainer(self) -> bool:
        return self._trainer

    def get_state_dict(self) -> Dict[str, dict]:
        assert self._trainer
        return {
            "nnet": self._nnet.state_dict(),
            "optimizer": self._optimizer.state_dict()
        }

    def load_state_dict(self, state_dict: Dict[str, dict] or dict) -> tuple:
        if state_dict.get("optimizer") is None:
            return self._nnet.load_state_dict(state_dict)
        info = []
        if state_dict.get("optimizer") and hasattr(self, "_optimizer"):
            info.append(self._optimizer.load_state_dict(state_dict["optimizer"]))
        return (self._nnet.load_state_dict(state_dict["nnet"]),
                *info)

    def save(self, filename: str):
        torch.save(self.get_state_dict(), filename)

    def load(self, filename: str):
        self.load_state_dict(torch.load(filename))

    def get_nn_instance(self):
        assert self.is_trainer(), "Is not a trainer."
        return self._nnet

    def experience(self, transition: dict) -> list or None:
        pass

    def train(self, data: Dict[str, list] or None) -> list:
        pass

    def to(self, device):
        return self

    def device_info(self) -> str:
        if self._device.type == 'cuda':
            return torch.cuda.get_device_name(device=self._device)
        else:
            return 'CPU'
