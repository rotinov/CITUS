from pprint import pprint
from typing import *

import numpy
import torch
from gym.spaces import Space

from agnes.algos.base import _BaseAlgo, _BaseBuffer, LossArgs
from agnes.algos.configs.a2c_config import get_config
from agnes.common import schedules, logger
from agnes.nns.rnn import _RecurrentFamily
from agnes.nns.base import _BasePolicy
from agnes.nns.initializer import _BaseChooser


class Buffer(_BaseBuffer):
    def __init__(self, nsteps):
        super().__init__(nsteps)
        self.rollouts = dict()
        self.offset = 0
        self.first = True

    def append(self, transition: Dict[str, numpy.ndarray]):
        for key in transition.keys():
            lnk_to_arr = transition[key]
            if self.first:
                self.rollouts[key] = numpy.empty((self.nsteps,) + lnk_to_arr.shape, dtype=lnk_to_arr.dtype)
            self.rollouts[key][self.offset] = lnk_to_arr

        self.offset += 1
        self.first = False

    def rollout(self):
        transitions = [dict(zip(self.rollouts, t)) for t in zip(*self.rollouts.values())]
        self.offset = 0
        return list(transitions)

    def __len__(self):
        return self.offset


class A2CLoss(torch.nn.Module):
    def __init__(self, vf_coef, ent_coef):
        super().__init__()
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    def forward(self, tensors_storage: LossArgs) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Making critic final loss
        t_critic_loss = 0.5 * (tensors_storage.new_vals - tensors_storage.returns).pow(2).mean()

        ADVS = tensors_storage.advantages.view(tensors_storage.new_log_probs.shape[:-1] + (-1,))

        with torch.no_grad():
            approxkl = (.5 * torch.mean((tensors_storage.old_log_probs - tensors_storage.new_log_probs) ** 2))

        # Calculating surrogates
        t_rt1 = ADVS * tensors_storage.new_log_probs
        t_actor_loss = - t_rt1.mean()

        t_entropy = tensors_storage.entropies.mean()

        # Making loss for Neural Network
        t_loss = t_actor_loss + self.vf_coef * t_critic_loss - self.ent_coef * t_entropy

        return t_loss, {"loss/policy_loss": t_actor_loss.item(),
                        "loss/value_loss": t_critic_loss.item(),
                        "loss/policy_entropy": t_entropy.item(),
                        "loss/approxkl": approxkl.item()}


class MyDataParallel(torch.nn.DataParallel):
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class A2cClass(_BaseAlgo):
    get_config = get_config

    meta: str = "A2C"

    multigpu = False

    _nnet: _BasePolicy
    _lossfun: A2CLoss
    _buffer: Buffer

    def __init__(self, nn: _BaseChooser,
                 observation_space: Space,
                 action_space: Space,
                 cnfg: Dict = None,
                 workers: int = 1,
                 trainer: bool = True,
                 betas: Tuple[float, float] = (0.99, 0.999),
                 eps: float = 1e-5, **network_args):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        self.nn_type = nn

        if trainer:
            pprint(cnfg)

        self._nnet = nn(observation_space, action_space, **network_args)

        self.GAMMA = cnfg['gamma']
        self.learning_rate = cnfg['learning_rate']
        self.vf_coef = cnfg['vf_coef']
        self.ent_coef = cnfg['ent_coef']
        self.final_timestep = cnfg['timesteps']
        self.nsteps = cnfg['nsteps']
        self.nminibatches = cnfg['nminibatches']
        self.LAM = cnfg['lam']
        self.noptepochs = cnfg['noptepochs']
        self.MAX_GRAD_NORM = cnfg['max_grad_norm']
        self.bptt = cnfg['bptt'] if 'bptt' in cnfg.keys() else self.nsteps // self.nminibatches
        self.workers_num = workers

        self.nbatch = self.workers_num * self.nsteps
        self.nbatch_train = self.nbatch // self.nminibatches

        final_epoch = int(self.final_timestep / self.nsteps * self.nminibatches * self.noptepochs)  # 312500

        if trainer:
            self._lossfun = A2CLoss(self.vf_coef, self.ent_coef)
            if torch.cuda.device_count() > 1:
                self._nnet = MyDataParallel(self._nnet).cuda()
                print(self._nnet.device_ids)
                self.multigpu = True

            print(self._nnet)

            if isinstance(self.learning_rate, float):
                lrnow = self.learning_rate
                lr_final = self.learning_rate
            else:
                lrnow = self.learning_rate(1)
                lr_final = 0.0

            self._optimizer = torch.optim.Adam(self._nnet.parameters(), lr=lrnow, betas=betas, eps=eps,
                                               weight_decay=1e-6)

            self._lr_scheduler = schedules.LinearAnnealingLR(self._optimizer, eta_min=lr_final,
                                                             to_epoch=final_epoch)
        else:
            self._nnet.eval()

        self._buffer = Buffer(self.nsteps)

        self._trainer = trainer

        self.is_recurrent = isinstance(
            self._nnet, _RecurrentFamily
        ) or (self.multigpu and isinstance(self._nnet.module, _RecurrentFamily))

    def experience(self, transition):
        self._buffer.append(transition)
        if len(self._buffer) >= self.nsteps:
            data = self._buffer.rollout()

            data = self._calculate_advantages(data)

            return data
        return None

    def device_info(self) -> str:
        if self.multigpu:
            return 'Multi-GPU'
        elif self._device.type == 'cuda':
            return torch.cuda.get_device_name(device=self._device)
        else:
            return 'CPU'

    def to(self, device: str):
        device = torch.device(device)
        self._device = device

        self._nnet = self._nnet.to(device)
        if self._trainer:
            self._lossfun = self._lossfun.to(device)

        return self

    def _calculate_advantages(self, data: List[Dict[str, list]]):
        assert all(k in data[0] for k in ("state", "action", "new_state", "reward",
                                          "done", "old_log_probs", "old_vals")), "Necessary keys are not in the dict!"
        # transition = {
        #     "state": self.state,
        #     "action": pred_action,
        #     "new_state": nstate,
        #     "reward": reward,
        #     "done": done,
        #     "old_log_probs": old_log_probs,
        #     "old_vals": old_vals
        # }

        train_dict = {k: numpy.asarray([dic[k] for dic in data]) for k in data[0]}

        n_shape = train_dict["done"].shape

        with torch.no_grad():
            t_nstates = torch.from_numpy(numpy.asarray(train_dict["new_state"][-1], dtype=numpy.float32))
            if self._device != torch.device('cpu'):
                t_nstates = t_nstates.to(self._device)

            last_values = self._nnet.get_val(t_nstates).detach().squeeze(-1).cpu().numpy()

            train_dict["old_vals"] = train_dict["old_vals"].reshape(n_shape)

        # Making GAE from td residual
        n_returns = numpy.zeros_like(train_dict["old_vals"])
        lastgaelam = 0.
        nextvalues = last_values
        for t in reversed(range(n_returns.shape[0])):
            nextnonterminal = 1. - train_dict["done"][t]
            delta = train_dict["reward"][t] + self.GAMMA * nextnonterminal * nextvalues - train_dict["old_vals"][t]
            n_returns[t] = lastgaelam = delta + self.LAM * self.GAMMA * nextnonterminal * lastgaelam
            nextvalues = train_dict["old_vals"][t]

        n_returns += train_dict["old_vals"]

        transitions = {
            "state": train_dict["state"],
            "action": train_dict["action"],
            "returns": n_returns,
            "old_log_probs": train_dict["old_log_probs"],
            "old_vals": train_dict["old_vals"]
        }

        if train_dict.get("additions") is not None:
            transitions["additions"] = train_dict.get("additions")

        if train_dict["reward"].ndim == 1 or self.is_recurrent:
            if self.is_recurrent:
                transitions["done"] = train_dict["done"]
        else:
            for key in transitions.keys():
                transitions[key] = transitions[key].reshape((-1,) + transitions[key].shape[2:])

        return [dict(zip(transitions, t)) for t in zip(*transitions.values())]

    def train(self, data: List[Dict[str, numpy.ndarray] or List[Dict[str, numpy.ndarray]]]) -> List[dict] or None:
        assert data is not None, "No data is provided"

        if self.is_recurrent:
            return self.train_with_bptt(data)

        if isinstance(data[0], list):
            data = sum(data, [])

        assert all(k in data[0][0] if isinstance(data[0], list) else data[0]
                   for k in ("state", "action", "old_log_probs",
                             "old_vals", "returns", "done")), "Necessary keys are not in the dict!"

        train_dict: Dict[str, torch.Tensor] = {
            k: torch.from_numpy(numpy.asarray([dic[k] for dic in data])) for k in data[0]
        }

        with torch.no_grad():
            train_dict["advantages"] = train_dict["returns"] - train_dict["old_vals"]

            # Normalizing advantages over rollout
            train_dict["advantages"] = ((train_dict["advantages"] - train_dict["advantages"].mean())
                                        / (train_dict["advantages"].std() + 1e-8))

            train_dict["state"] = train_dict["state"].float()

            info = []
            for i in range(self.noptepochs):
                indexes = torch.randperm(len(data))

                batches_data = dict()
                for key in train_dict.keys():
                    train_dict[key] = train_dict[key][indexes]
                    batches_data[key] = torch.chunk(train_dict[key], self.nminibatches, dim=0)

                batches_data_list = [dict(zip(batches_data, t)) for t in zip(*batches_data.values())]

                with torch.enable_grad():
                    info.extend([self._one_train(minibatch) for minibatch in batches_data_list])

        return info

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

        # Getting log probs
        t_new_log_probs = t_distrib.log_prob(data["action"]).view_as(data["old_log_probs"])

        tensors_dict = {
            "new_log_probs": t_new_log_probs,
            "old_log_probs": data["old_log_probs"],
            "advantages": data["advantages"],
            "new_vals": t_state_vals,
            "returns": data["returns"],
            "entropies": t_entropies
        }

        tensors_storage = LossArgs()

        tensors_storage.new_log_probs = t_new_log_probs
        tensors_storage.old_log_probs = data["old_log_probs"]
        tensors_storage.advantages = data["advantages"]
        tensors_storage.new_vals = t_state_vals
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
        RETURNS = data["returns"].view_as(t_state_vals)

        tensors_storage = LossArgs()

        tensors_storage.new_log_probs = t_new_log_probs
        tensors_storage.old_log_probs = OLDLOGPROBS
        tensors_storage.advantages = data["advantages"]
        tensors_storage.new_vals = t_state_vals
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

        return self.format_explained_variance(t_state_vals.detach().cpu().numpy(),
                                              data["returns"].detach().cpu().numpy(),
                                              stat_from_lr)

    def train_with_bptt(self, data):
        assert all(k in data[0][0] if isinstance(data[0], list) else data[0]
                   for k in ("state", "action", "old_log_probs",
                             "old_vals", "returns", "additions",
                             "done")), "Necessary keys are not in the dict!"

        # transitions = {
        #     "state",
        #     "action",
        #     "old_log_probs",
        #     "old_vals",
        #     "returns",
        #     "additions",
        #     "done"
        # }

        with torch.no_grad():
            if isinstance(data[0], list):
                train_dict: Dict[str, torch.Tensor] = {
                    k: torch.from_numpy(numpy.asarray([[dic[k] for dic in LD] for LD in data])) for k in data[0][0]
                }

                for key in train_dict.keys():
                    if key == "additions":
                        train_dict[key] = train_dict[key].permute([1, 2, 3, 0, 4, 5])
                        train_dict["additions"] = train_dict["additions"].reshape(
                            train_dict["additions"].shape[:3] + (-1,) + train_dict["additions"].shape[5:]
                        )
                    else:
                        train_dict[key] = train_dict[key].transpose(1, 0)
                        train_dict[key] = train_dict[key].reshape(
                            (train_dict[key].shape[0], -1) + train_dict[key].shape[3:]
                        )
            else:
                train_dict: Dict[str, torch.Tensor] = {
                    k: torch.from_numpy(numpy.asarray([dic[k] for dic in data])) for k in data[0]
                }

            if train_dict["action"].ndim < 3:
                for key in ["action", "old_log_probs", "old_vals", "returns"]:
                    train_dict[key].unsqueeze_(-1)

            train_chankes: Dict[str, torch.Tensor] = dict()

            for key in train_dict.keys():
                train_chankes[key] = train_dict[key].reshape((self.nsteps // self.bptt, self.bptt)
                                                             + train_dict[key].shape[1:])

                if key == "additions":
                    train_chankes[key] = train_chankes[key].permute([1, 2, 3, 4, 0, 5])
                    train_chankes[key] = train_chankes[key].reshape(train_chankes[key].shape[:3]
                                                                    + (-1,) + train_chankes[key].shape[5:])
                else:
                    train_chankes[key] = train_chankes[key].transpose(1, 0).transpose(2, 1)
                    if key == "done":
                        train_chankes[key] = train_chankes[key].reshape((self.bptt, -1))
                    else:
                        train_chankes[key] = train_chankes[key].reshape((self.bptt, -1,) + train_chankes[key].shape[3:])

            info = []

            for i in range(self.noptepochs):
                indexes = torch.randperm(train_chankes["state"].shape[1])

                batches_dict: Dict[str, torch.Tensor] = dict()

                for key in train_chankes.keys():
                    if key == "additions":
                        train_chankes[key] = train_chankes[key][:, :, :, indexes]
                        batches_dict[key] = train_chankes[key].reshape(
                            train_chankes[key].shape[:3] + (self.nbatch_train // self.bptt, -1) +
                            train_chankes[key].shape[4:]
                        ).permute([4, 0, 1, 2, 3, 5])
                    else:
                        train_chankes[key] = train_chankes[key][:, indexes]
                        if key == "done":
                            batches_dict[key] = train_chankes[key].reshape(
                                (train_chankes[key].shape[0],) + (self.nbatch_train // self.bptt, -1)
                            ).transpose(2, 1).transpose(1, 0)
                        else:
                            batches_dict[key] = train_chankes[key].reshape(
                                (train_chankes[key].shape[0], self.nbatch_train // self.bptt, -1)
                                + train_chankes[key].shape[2:]
                            ).transpose(2, 1).transpose(1, 0)

                batches_dict["advantages"] = batches_dict["returns"] - batches_dict["old_vals"]

                # Normalizing advantages over rollout
                batches_dict["advantages"] = (
                        (batches_dict["advantages"] - torch.mean(batches_dict["advantages"]))
                        / (torch.std(batches_dict["advantages"]) + 1e-8)
                )

                batches_data_list: List[Dict[str, torch.Tensor]] = [
                    dict(zip(batches_dict, t)) for t in zip(*batches_dict.values())
                ]

                for minibatch in batches_data_list:
                    indexes = torch.unique(torch.nonzero(minibatch["done"].sum(dim=-1)), dim=0) + 1
                    sizes: list = indexes.squeeze(-1).tolist()
                    sizes.insert(0, 0)
                    sizes.append(minibatch["done"].shape[0])
                    sizes = [sizes[i + 1] - sizes[i] for i in range(len(sizes) - 1)]
                    if sizes[-1] == 0:
                        sizes.pop(-1)

                    for key in ["state", "action", "done"]:
                        minibatch[key] = torch.split(minibatch[key], sizes, dim=0)

                    with torch.enable_grad():
                        info.append(self._one_train_seq(minibatch))

        return info

    @staticmethod
    def format_explained_variance(t_state_vals: numpy.ndarray,
                                  returns: numpy.ndarray,
                                  stat_from_lr: Dict[str, float]) -> Dict[str, float]:
        if returns.ndim > 1:
            stat_from_lr["misc/explained_variance"] = logger.explained_variance(
                numpy.mean(t_state_vals, axis=-1),
                numpy.mean(returns, axis=-1))
        else:
            stat_from_lr["misc/explained_variance"] = logger.explained_variance(t_state_vals, returns)

        return stat_from_lr


class A2cInitializer:
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
        return A2cClass(nn,
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
        return A2cClass.meta


A2C = A2cInitializer()
