from pprint import pprint
from typing import *

import numpy
import torch
from gym.spaces import Space

from agnes.algos.base import LossArgs
from agnes.algos.a2c import A2cClass, MyDataParallel
from agnes.algos.configs.ppo_config import get_config
from agnes.common import schedules, logger
from agnes.nns.rnn import _RecurrentFamily
from agnes.nns.base import _BasePolicy
from agnes.nns.initializer import _BaseChooser


class PPOLoss(torch.nn.Module):
    _CLIPRANGE = 0.0

    def __init__(self, vf_coef, ent_coef):
        super().__init__()
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    @property
    def CLIPRANGE(self):
        return self._CLIPRANGE

    @CLIPRANGE.setter
    def CLIPRANGE(self, VALUE):
        self._CLIPRANGE = VALUE

    def forward(self, tensors_storage: LossArgs) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Making critic losses
        t_state_vals_clipped = tensors_storage.old_vals + torch.clamp(
            tensors_storage.new_vals - tensors_storage.old_vals,
            - self._CLIPRANGE, + self._CLIPRANGE
        )

        # Making critic final loss
        t_critic_loss1 = (tensors_storage.new_vals - tensors_storage.returns).pow(2)
        t_critic_loss2 = (t_state_vals_clipped - tensors_storage.returns).pow(2)
        t_critic_loss = 0.5 * torch.mean(torch.max(t_critic_loss1, t_critic_loss2))

        # Calculating ratio
        t_ratio = torch.exp(tensors_storage.new_log_probs - tensors_storage.old_log_probs)

        ADVS = tensors_storage.advantages.view(t_ratio.shape[:-1] + (-1,))

        with torch.no_grad():
            approxkl = (.5 * torch.mean((tensors_storage.old_log_probs - tensors_storage.new_log_probs) ** 2))
            clipfrac = torch.mean((torch.abs(t_ratio - 1.0) > self._CLIPRANGE).float())

        # Calculating surrogates
        t_rt1 = ADVS * t_ratio
        t_rt2 = ADVS * torch.clamp(t_ratio,
                                   1 - self._CLIPRANGE,
                                   1 + self._CLIPRANGE)
        t_actor_loss = - torch.min(t_rt1, t_rt2).mean()

        t_entropy = tensors_storage.entropies.mean()

        # Making loss for Neural Network
        t_loss = t_actor_loss + self.vf_coef * t_critic_loss - self.ent_coef * t_entropy

        return t_loss, {"loss/policy_loss": t_actor_loss.item(),
                        "loss/value_loss": t_critic_loss.item(),
                        "loss/policy_entropy": t_entropy.item(),
                        "loss/approxkl": approxkl.item(),
                        "loss/clipfrac": clipfrac.item()}


class PpoClass(A2cClass):
    get_config = get_config

    meta: str = "PPO"

    multigpu = False

    _nnet: _BasePolicy
    _lossfun: PPOLoss

    def __init__(self, nn: _BaseChooser,
                 observation_space: Space,
                 action_space: Space,
                 cnfg: Dict = None,
                 workers: int = 1,
                 trainer: bool = True,
                 betas: Tuple[float, float] = (0.99, 0.999),
                 eps: float = 1e-5, **network_args):
        super().__init__(nn, observation_space, action_space, cnfg, workers, trainer, betas, eps, **network_args)

        self.CLIPRANGE = cnfg['cliprange']

        final_epoch = int(self.final_timestep / self.nsteps * self.nminibatches * self.noptepochs)  # 312500

        if trainer:
            self._lossfun = PPOLoss(self.vf_coef, self.ent_coef)

            if isinstance(self.CLIPRANGE, float):
                self._cr_schedule = schedules.LinearSchedule(lambda x: self.CLIPRANGE, eta_min=1.0,
                                                             to_epoch=final_epoch)
            else:
                self._cr_schedule = schedules.LinearSchedule(self.CLIPRANGE, eta_min=0.0, to_epoch=final_epoch)

    def _one_train(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # Tensors
        for key in data.keys():
            if self._device != torch.device('cpu'):
                data[key] = data[key].to(self._device)

        # Feedforward with building computation graph
        t_probs, t_state_vals_un = self._nnet(data["state"].float())
        t_distrib = self._nnet.wrap_dist(t_probs)

        t_state_vals = t_state_vals_un.squeeze(-1)

        # Calculating entropy
        t_entropies = t_distrib.entropy()

        t_state_vals = t_state_vals.view_as(data["old_vals"])

        self.CLIPRANGE = self._cr_schedule.get_v()
        self._lossfun._CLIPRANGE = self.CLIPRANGE

        # Getting log probs
        t_new_log_probs = t_distrib.log_prob(data["action"]).view_as(data["old_log_probs"])

        tensors_storage = LossArgs()

        tensors_storage.new_log_probs = t_new_log_probs
        tensors_storage.old_log_probs = data["old_log_probs"]
        tensors_storage.advantages = data["advantages"]
        tensors_storage.new_vals = t_state_vals
        tensors_storage.old_vals = data["old_vals"]
        tensors_storage.returns = data["returns"]
        tensors_storage.entropies = t_entropies

        t_loss, stat_from_lr = self._lossfun(tensors_storage)

        # Calculating gradients
        self._optimizer.zero_grad()
        t_loss.backward()

        # Normalizing gradients
        torch.nn.utils.clip_grad_norm_(self._nnet.parameters(), self.MAX_GRAD_NORM)

        # Optimizer step
        self._optimizer.step()
        self._lr_scheduler.step()
        self._cr_schedule.step()

        return self.format_explained_variance(t_state_vals.detach().cpu().numpy(),
                                              data["returns"].detach().cpu().numpy(),
                                              stat_from_lr)

    def _one_train_seq(self, data: Dict[str, torch.Tensor or List[torch.Tensor]]) -> Dict[str, float]:
        if data["additions"].shape[1] == 2:
            is_lstm = True
        else:
            is_lstm = False

        t_addition = data["additions"][0]

        # Tensors
        if self._device == torch.device('cpu'):
            if is_lstm:
                if t_addition[0].ndim == 2:
                    t_addition = t_addition.unsqueeze(1)
            else:
                if t_addition.ndim == 2:
                    t_addition = t_addition.unsqueeze(0)
        else:
            data["old_log_probs"] = data["old_log_probs"].to(self._device)
            data["old_vals"] = data["old_vals"].to(self._device)
            data["returns"] = data["returns"].to(self._device)
            data["advantages"] = data["advantages"].to(self._device)
            t_addition = t_addition.to(self._device)
            if is_lstm:
                if t_addition[0].ndim == 2:
                    t_addition = t_addition.unsqueeze(1)
            else:
                if t_addition.ndim == 2:
                    t_addition = t_addition[0].unsqueeze(0)
                else:
                    t_addition = t_addition[0]

        # Feedforward with building computation graph
        l_new_log_probs = []
        l_state_vals = []
        l_entropies = []

        for t_state, t_done, t_action in zip(data["state"], data["done"], data["action"]):
            if t_done.size == 0:
                break

            t_done = t_done[-1]
            if self._device != torch.device('cpu'):
                t_state = t_state.to(self._device)
                t_done = t_done.to(self._device)
                t_action = t_action.to(self._device)

            t_probs, t_addition, t_state_vals_un = self._nnet(t_state, t_addition)
            t_probs = t_probs.view(-1, t_probs.shape[-1])
            t_distrib = self._nnet.wrap_dist(t_probs)

            if t_done.ndimension() < 2:
                t_done = t_done.unsqueeze(-1)

            if is_lstm:
                t_addition = (t_addition[0].masked_fill(t_done.unsqueeze(0), 0.0),
                              t_addition[1].masked_fill(t_done.unsqueeze(0), 0.0))
            else:
                t_addition = t_addition.masked_fill(t_done.unsqueeze(0), 0.0)

            l_state_vals.append(t_state_vals_un)

            t_action = t_action.view(t_probs.shape[0], -1)

            l_new_log_probs.append(t_distrib.log_prob(t_action.squeeze(-1)))
            l_entropies.append(t_distrib.entropy())

        t_new_log_probs = torch.cat(l_new_log_probs, dim=0)
        t_state_vals = torch.cat(l_state_vals, dim=0)
        t_entropies = torch.cat(l_entropies, dim=0)

        t_new_log_probs = t_new_log_probs.view(-1, t_new_log_probs.shape[-1])

        OLDLOGPROBS = data["old_log_probs"].view_as(t_new_log_probs)
        OLDVALS = data["old_vals"].view_as(t_state_vals)
        RETURNS = data["returns"].view_as(t_state_vals)

        self.CLIPRANGE = self._cr_schedule.get_v()

        self._lossfun.CLIPRANGE = self.CLIPRANGE

        tensors_storage = LossArgs()

        tensors_storage.new_log_probs = t_new_log_probs
        tensors_storage.old_log_probs = OLDLOGPROBS
        tensors_storage.advantages = data["advantages"]
        tensors_storage.new_vals = t_state_vals
        tensors_storage.old_vals = OLDVALS
        tensors_storage.returns = RETURNS
        tensors_storage.entropies = t_entropies

        t_loss, stat_from_lr = self._lossfun(tensors_storage)

        # Calculating gradients
        self._optimizer.zero_grad()
        t_loss.backward()

        # Normalizing gradients
        torch.nn.utils.clip_grad_norm_(self._nnet.parameters(), self.MAX_GRAD_NORM)

        # Optimizer step
        self._optimizer.step()
        self._lr_scheduler.step()
        self._cr_schedule.step()

        return self.format_explained_variance(t_state_vals.detach().cpu().numpy(),
                                              data["returns"].detach().cpu().numpy(),
                                              stat_from_lr)


class PpoInitializer:
    betas = (0.99, 0.999)
    eps = 1e-5

    def __init__(self):
        pass

    def config(self, betas: Tuple, eps: float):
        self.betas = betas
        self.eps = eps
        return self

    def __call__(self, nn,
                 observation_space: Space,
                 action_space: Space,
                 cnfg: Dict = None,
                 workers=1,
                 trainer=True):
        return PpoClass(nn,
                        observation_space,
                        action_space,
                        cnfg,
                        workers,
                        trainer,
                        betas=self.betas,
                        eps=self.eps)

    @staticmethod
    def get_config(env_type: str):
        return get_config(env_type)

    @property
    def meta(self) -> str:
        return PpoClass.meta


PPO = PpoInitializer()
