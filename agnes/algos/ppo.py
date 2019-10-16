import torch
import numpy
import random
from typing import Tuple
from gym.spaces import Space

from agnes.algos.base import _BaseAlgo, _BaseBuffer
from agnes.nns.initializer import _BaseChooser
from agnes.nns import rnn
from agnes.common import schedules, logger
from pprint import pprint
from agnes.algos.configs.ppo_config import get_config


class Buffer(_BaseBuffer):
    def __init__(self):
        self.rollouts = []

    def append(self, transition):
        self.rollouts.append(transition)

    def rollout(self):
        transitions = self.rollouts
        self.rollouts = []

        return list(transitions)

    def learn(self, data, minibatchsize):
        batches = []
        for i in range(0, len(data), minibatchsize):
            one_batch = data[i:min(i + minibatchsize, len(data))]

            batches.append(one_batch)

        return batches

    def __len__(self):
        return len(self.rollouts)


class PpoClass(_BaseAlgo):
    _device = torch.device('cpu')

    get_config = get_config

    meta = "PPO"

    def __init__(self, nn: _BaseChooser,
                 observation_space: Space,
                 action_space: Space,
                 cnfg=None,
                 workers=1,
                 trainer=True,
                 betas=(0.99, 0.999),
                 eps=1e-5):
        super().__init__()

        self.nn_type = nn

        if trainer:
            pprint(cnfg)

        self._nnet = nn(observation_space, action_space)

        if trainer:
            print(self._nnet)
        else:
            self._nnet.eval()

        self.GAMMA = cnfg['gamma']
        self.learning_rate = cnfg['learning_rate']
        self.CLIPRANGE = cnfg['cliprange']
        self.vf_coef = cnfg['vf_coef']
        self.ent_coef = cnfg['ent_coef']
        self.final_timestep = cnfg['timesteps']
        self.nsteps = cnfg['nsteps']
        self.nminibatches = cnfg['nminibatches']
        self.LAM = cnfg['lam']
        self.noptepochs = cnfg['noptepochs']
        self.MAX_GRAD_NORM = cnfg['max_grad_norm']
        self.workers_num = workers

        self.nbatch = self.workers_num * self.nsteps
        self.nbatch_train = self.nbatch // self.nminibatches

        final_epoch = int(self.final_timestep / self.nsteps * self.nminibatches * self.noptepochs)  # 312500

        if trainer:
            self._optimizer = torch.optim.Adam(self._nnet.parameters(), lr=self.learning_rate, betas=betas, eps=eps)

            self._lr_scheduler = schedules.LinearAnnealingLR(self._optimizer, eta_min=0.0,
                                                             to_epoch=final_epoch)

            if isinstance(self.CLIPRANGE, float):
                self._cr_schedule = schedules.LinearSchedule(lambda x: self.CLIPRANGE, eta_min=1.0,
                                                             to_epoch=final_epoch)
            else:
                self._cr_schedule = schedules.LinearSchedule(self.CLIPRANGE, eta_min=0.0, to_epoch=final_epoch)

        self._buffer = Buffer()

        self._trainer = trainer

    def experience(self, transition):
        self._buffer.append(transition)
        if len(self._buffer) >= self.nsteps:
            data = self._buffer.rollout()

            data = self._calculate_advantages(data)

            return data
        return None

    def train(self, data):
        if data is None:
            return None

        if isinstance(self._nnet, rnn._RecurrentFamily):
            return self.train_with_bptt(data)

        if isinstance(data[0], list):
            data_combined = []
            for data_per_worker in data:
                data_combined.extend(data_per_worker)

            data = data_combined

        # Unpack
        states, actions, old_log_probs, old_vals, returns = zip(*data)

        states = numpy.asarray(states)
        actions = numpy.asarray(actions)
        old_log_probs = numpy.asarray(old_log_probs)
        old_vals = numpy.asarray(old_vals)
        returns = numpy.asarray(returns)

        info = []
        for i in range(self.noptepochs):
            indexes = numpy.random.permutation(len(data))

            states = states.take(indexes, axis=0)
            actions = actions.take(indexes, axis=0)
            old_log_probs = old_log_probs.take(indexes, axis=0)
            old_vals = old_vals.take(indexes, axis=0)
            returns = returns.take(indexes, axis=0)

            states_batches = numpy.split(states, self.nminibatches, axis=0)
            actions_batchs = numpy.split(actions, self.nminibatches, axis=0)
            old_log_probs_batchs = numpy.split(old_log_probs, self.nminibatches, axis=0)
            old_vals_batchs = numpy.split(old_vals, self.nminibatches, axis=0)
            returns_batchs = numpy.split(returns, self.nminibatches, axis=0)

            for (
                    states_batch,
                    actions_batch,
                    old_log_probs_batch,
                    old_vals_batch,
                    returns_batch
            ) in zip(
                states_batches,
                actions_batchs,
                old_log_probs_batchs,
                old_vals_batchs,
                returns_batchs
            ):
                info.append(
                    self._one_train(states_batch, actions_batch, old_log_probs_batch, old_vals_batch, returns_batch)
                )

        return info

    def to(self, device: str):
        device = torch.device(device)
        self._device = device
        self._nnet = self._nnet.to(device)

        return self

    def _calculate_advantages(self, data):
        states, actions, nstates, rewards, dones, outs = zip(*data)
        if isinstance(self._nnet, rnn._RecurrentFamily):
            additions, old_log_probs, old_vals = zip(*outs)
        else:
            old_log_probs, old_vals = zip(*outs)

        n_rewards = numpy.asarray(rewards)
        n_dones = numpy.asarray(dones)
        n_shape = n_dones.shape

        n_state_vals = numpy.asarray(old_vals)

        with torch.no_grad():
            if self._device == torch.device('cpu'):
                t_nstates = torch.FloatTensor(nstates[-1])
            else:
                t_nstates = torch.cuda.FloatTensor(nstates[-1])

            last_values = self._nnet.get_val(t_nstates).detach().squeeze(-1).cpu().numpy()

            n_state_vals = n_state_vals.reshape(n_shape)

        # Making GAE from td residual
        n_returns = numpy.zeros_like(n_state_vals)
        lastgaelam = 0
        nextvalues = last_values
        for t in reversed(range(n_returns.shape[0])):
            nextnonterminal = 1. - n_dones[t]
            delta = n_rewards[t] + self.GAMMA * nextnonterminal * nextvalues - n_state_vals[t]
            n_returns[t] = lastgaelam = delta + self.LAM * self.GAMMA * nextnonterminal * lastgaelam
            nextvalues = n_state_vals[t]

        n_returns += n_state_vals

        if n_rewards.ndim == 1 or isinstance(self._nnet, rnn._RecurrentFamily):
            if isinstance(self._nnet, rnn._RecurrentFamily):
                transitions = (numpy.asarray(states), numpy.asarray(actions),
                               numpy.asarray(old_log_probs), numpy.asarray(old_vals), n_returns,
                               numpy.asarray(additions),
                               n_dones)
            else:
                transitions = (numpy.asarray(states), numpy.asarray(actions),
                               numpy.asarray(old_log_probs), numpy.asarray(old_vals), n_returns)
        else:
            li_states = numpy.asarray(states)
            li_states = li_states.reshape((-1,) + li_states.shape[2:])

            li_actions = numpy.asarray(actions)
            li_actions = li_actions.reshape((-1,) + li_actions.shape[2:])
            li_old_vals = n_state_vals.reshape((-1,) + n_state_vals.shape[2:])

            li_old_log_probs = numpy.asarray(old_log_probs)
            li_old_log_probs = li_old_log_probs.reshape((-1,) + li_old_log_probs.shape[2:])

            li_n_returns = n_returns.reshape((-1,) + n_returns.shape[2:])

            transitions = (li_states, li_actions, li_old_log_probs, li_old_vals, li_n_returns)

        return list(zip(*transitions))

    def _one_train(self,
                   STATES,
                   ACTIONS,
                   OLDLOGPROBS,
                   OLDVALS,
                   RETURNS):
        # Tensors
        if self._device == torch.device('cpu'):
            STATES = torch.FloatTensor(STATES)
            if self._nnet.type_of_out() == torch.int16:
                ACTIONS = torch.LongTensor(ACTIONS)
            else:
                ACTIONS = torch.FloatTensor(ACTIONS)
            OLDLOGPROBS = torch.FloatTensor(OLDLOGPROBS)
            OLDVALS = torch.FloatTensor(OLDVALS)
            RETURNS = torch.FloatTensor(RETURNS)
        else:
            STATES = torch.cuda.FloatTensor(STATES)
            if self._nnet.type_of_out() == torch.int16:
                ACTIONS = torch.cuda.LongTensor(ACTIONS)
            else:
                ACTIONS = torch.cuda.FloatTensor(ACTIONS)
            OLDLOGPROBS = torch.cuda.FloatTensor(OLDLOGPROBS)
            OLDVALS = torch.cuda.FloatTensor(OLDVALS)
            RETURNS = torch.cuda.FloatTensor(RETURNS)

        # Feedforward with building computation graph
        t_distrib, t_state_vals_un = self._nnet(STATES)
        t_state_vals = t_state_vals_un.squeeze(-1)

        OLDVALS = OLDVALS.view_as(t_state_vals)
        ADVANTAGES = RETURNS - OLDVALS

        self.CLIPRANGE = self._cr_schedule.get_v()

        # Normalizing advantages
        ADVS = ((ADVANTAGES - ADVANTAGES.mean()) / (ADVANTAGES.std() + 1e-8))

        if OLDLOGPROBS.ndimension() == 2:
            ADVS = ADVS.unsqueeze(-1)

        # Making critic losses
        t_state_vals_clipped = OLDVALS + torch.clamp(t_state_vals - OLDVALS,
                                                     - self.CLIPRANGE,
                                                     + self.CLIPRANGE)

        # Making critic final loss
        t_critic_loss1 = (t_state_vals - RETURNS).pow(2)
        t_critic_loss2 = (t_state_vals_clipped - RETURNS).pow(2)
        t_critic_loss = 0.5 * torch.mean(torch.max(t_critic_loss1, t_critic_loss2))

        # Getting log probs
        t_new_log_probs = t_distrib.log_prob(ACTIONS)

        # Calculating ratio
        t_ratio = torch.exp(t_new_log_probs - OLDLOGPROBS)

        with torch.no_grad():
            approxkl = (.5 * torch.mean((OLDLOGPROBS - t_new_log_probs) ** 2)).item()
            clipfrac = torch.mean((torch.abs(t_ratio - 1.0) > self.CLIPRANGE).float()).item()

        # Calculating surrogates
        t_rt1 = ADVS * t_ratio
        t_rt2 = ADVS * torch.clamp(t_ratio,
                                   1 - self.CLIPRANGE,
                                   1 + self.CLIPRANGE)
        t_actor_loss = - torch.min(t_rt1, t_rt2).mean()

        # Calculating entropy
        t_entropy = t_distrib.entropy().mean()

        # Making loss for Neural Network
        t_loss = t_actor_loss + self.vf_coef * t_critic_loss - self.ent_coef * t_entropy

        # Calculating gradients
        self._optimizer.zero_grad()
        t_loss.backward()

        # Normalizing gradients
        torch.nn.utils.clip_grad_norm_(self._nnet.parameters(), self.MAX_GRAD_NORM)

        # Optimizer step
        self._optimizer.step()
        self._lr_scheduler.step()
        self._cr_schedule.step()

        return (t_actor_loss.item(),
                t_critic_loss.item(),
                t_entropy.item(),
                approxkl,
                clipfrac,
                logger.explained_variance(t_state_vals.detach().cpu().numpy(), RETURNS.detach().cpu().numpy()),
                ()
                )

    def _one_train_seq(self,
                       STATES,
                       ACTIONS,
                       OLDLOGPROBS,
                       OLDVALS,
                       RETURNS,
                       ADDITIONS,
                       DONES):

        FIRST_ADDITION = ADDITIONS[0]
        if FIRST_ADDITION.ndim == 4:
            is_lstm = True
        else:
            is_lstm = False

        # Tensors
        if self._device == torch.device('cpu'):
            OLDLOGPROBS = torch.FloatTensor(OLDLOGPROBS)
            OLDVALS = torch.FloatTensor(OLDVALS)
            RETURNS = torch.FloatTensor(RETURNS)
            if is_lstm:
                t_addition = (torch.FloatTensor(FIRST_ADDITION[0]).requires_grad_(),
                              torch.FloatTensor(FIRST_ADDITION[1]).requires_grad_())
            else:
                t_addition = torch.FloatTensor(FIRST_ADDITION).requires_grad_()
        else:
            OLDLOGPROBS = torch.cuda.FloatTensor(OLDLOGPROBS)
            OLDVALS = torch.cuda.FloatTensor(OLDVALS)
            RETURNS = torch.cuda.FloatTensor(RETURNS)
            if is_lstm:
                t_addition = (torch.cuda.FloatTensor(FIRST_ADDITION[0]).requires_grad_(),
                              torch.cuda.FloatTensor(FIRST_ADDITION[1]).requires_grad_())
            else:
                t_addition = torch.cuda.FloatTensor(FIRST_ADDITION).requires_grad_()

        # Feedforward with building computation graph
        l_new_log_probs = []
        l_state_vals = []

        for n_state, n_done, n_action in zip(STATES, DONES, ACTIONS):
            if n_done.size == 0:
                break

            n_done = n_done[-1]
            if self._device == torch.device('cpu'):
                t_state = torch.FloatTensor(n_state)
                t_done = torch.BoolTensor(n_done)
                if self._nnet.type_of_out() == torch.int16:
                    t_action = torch.LongTensor(n_action)
                else:
                    t_action = torch.FloatTensor(n_action)
            else:
                t_state = torch.cuda.FloatTensor(n_state)
                t_done = torch.cuda.BoolTensor(n_done)
                if self._nnet.type_of_out() == torch.int16:
                    t_action = torch.cuda.LongTensor(n_action)
                else:
                    t_action = torch.cuda.FloatTensor(n_action)

            t_distrib, t_addition, t_state_vals_un = self._nnet(t_state, t_addition)
            if t_done.ndimension() < 2:
                t_done = t_done.unsqueeze(-1)

            if is_lstm:
                t_addition = (t_addition[0].masked_fill(t_done.unsqueeze(0), 0.0),
                              t_addition[1].masked_fill(t_done.unsqueeze(0), 0.0))
            else:
                t_addition = t_addition.masked_fill(t_done.unsqueeze(0), 0.0)

            l_state_vals.append(t_state_vals_un.squeeze(-1))
            if t_action.ndimension() == 1:
                t_action = t_action.unsqueeze(-1)
            l_new_log_probs.append(t_distrib.log_prob(t_action).squeeze(-1))

        t_new_log_probs = torch.cat(l_new_log_probs, dim=0)
        t_state_vals = torch.cat(l_state_vals, dim=0)

        OLDVALS = OLDVALS.view_as(t_state_vals)
        ADVANTAGES = RETURNS - OLDVALS

        self.CLIPRANGE = self._cr_schedule.get_v()

        # Normalizing advantages
        ADVS = ((ADVANTAGES - ADVANTAGES.mean()) / (ADVANTAGES.std() + 1e-8))

        if OLDLOGPROBS.ndimension() != 2:
            ADVS = ADVS.squeeze(-1)

        # Making critic losses
        t_state_vals_clipped = OLDVALS + torch.clamp(t_state_vals - OLDVALS,
                                                     - self.CLIPRANGE,
                                                     + self.CLIPRANGE)

        # Making critic final loss
        t_critic_loss1 = (t_state_vals - RETURNS).pow(2)
        t_critic_loss2 = (t_state_vals_clipped - RETURNS).pow(2)
        t_critic_loss = 0.5 * torch.mean(torch.max(t_critic_loss1, t_critic_loss2))

        # Calculating ratio
        t_ratio = torch.exp(t_new_log_probs - OLDLOGPROBS)

        with torch.no_grad():
            approxkl = (.5 * torch.mean((OLDLOGPROBS - t_new_log_probs) ** 2)).item()
            clipfrac = torch.mean((torch.abs(t_ratio - 1.0) > self.CLIPRANGE).float()).item()

        if ADVS.ndimension() < t_ratio.ndimension():
            ADVS = ADVS.unsqueeze(-1)

        # Calculating surrogates
        t_rt1 = ADVS * t_ratio
        t_rt2 = ADVS * torch.clamp(t_ratio,
                                   1 - self.CLIPRANGE,
                                   1 + self.CLIPRANGE)
        t_actor_loss = - torch.min(t_rt1, t_rt2).mean()

        # Calculating entropy
        t_entropy = t_distrib.entropy().mean()

        # Making loss for Neural Network
        t_loss = t_actor_loss + self.vf_coef * t_critic_loss - self.ent_coef * t_entropy

        # Calculating gradients
        self._optimizer.zero_grad()
        t_loss.backward()

        # Normalizing gradients
        torch.nn.utils.clip_grad_norm_(self._nnet.parameters(), self.MAX_GRAD_NORM)

        # Optimizer step
        self._optimizer.step()
        self._lr_scheduler.step()
        self._cr_schedule.step()

        return (t_actor_loss.item(),
                t_critic_loss.item(),
                t_entropy.item(),
                approxkl,
                clipfrac,
                logger.explained_variance(t_state_vals.detach().view(-1).cpu().numpy(),
                                          RETURNS.detach().view(-1).cpu().numpy()),
                ()
                )

    def train_with_bptt(self, data):
        # Unpack
        if isinstance(data[0], list):
            states_combined = []
            actions_combined = []
            old_log_probs_combined = []
            old_vals_combined = []
            returns_combined = []
            additions_combined = []
            dones_combined = []
            for data_per_worker in data:
                states_per_worker, actions_per_worker, old_log_probs_per_worker, old_vals_per_worker, \
                    returns_per_worker, additions_per_worker, dones_per_worker = zip(*data_per_worker)

                states_combined.append(numpy.asarray(states_per_worker))
                actions_combined.append(numpy.asarray(actions_per_worker))
                old_log_probs_combined.append(numpy.asarray(old_log_probs_per_worker))
                old_vals_combined.append(numpy.asarray(old_vals_per_worker))
                returns_combined.append(numpy.asarray(returns_per_worker))
                dones_combined.append(numpy.asarray(dones_per_worker))

                hs = list(zip(*additions_per_worker))
                #  axis = 3

                additions_combined.append(numpy.asarray(hs))

            states = numpy.concatenate(states_combined, axis=1)
            actions = numpy.concatenate(actions_combined, axis=1)
            old_log_probs = numpy.concatenate(old_log_probs_combined, axis=1)
            old_vals = numpy.concatenate(old_vals_combined, axis=1)
            returns = numpy.concatenate(returns_combined, axis=1)
            additions = numpy.concatenate(additions_combined, axis=3).swapaxes(0, 1)
            dones = numpy.concatenate(dones_combined, axis=1)
        else:
            states, actions, old_log_probs, old_vals, returns, additions, dones = zip(*data)

            states = numpy.asarray(states)
            actions = numpy.asarray(actions)
            old_log_probs = numpy.asarray(old_log_probs)
            old_vals = numpy.asarray(old_vals)
            returns = numpy.asarray(returns)
            additions = numpy.asarray(additions)
            dones = numpy.asarray(dones)

        states_batches = numpy.asarray(numpy.split(states, self.nminibatches, axis=0))
        actions_batchs = numpy.asarray(numpy.split(actions, self.nminibatches, axis=0))
        old_log_probs_batchs = numpy.asarray(numpy.split(old_log_probs, self.nminibatches, axis=0))
        old_vals_batchs = numpy.asarray(numpy.split(old_vals, self.nminibatches, axis=0))
        returns_batchs = numpy.asarray(numpy.split(returns, self.nminibatches, axis=0))
        additions_batchs = numpy.asarray(numpy.split(additions, self.nminibatches, axis=0))
        dones_batchs = numpy.asarray(numpy.split(dones, self.nminibatches, axis=0))

        info = []
        for i in range(self.noptepochs):
            indexes = numpy.random.permutation(len(states_batches))

            states_batches = states_batches.take(indexes, axis=0)
            actions_batchs = actions_batchs.take(indexes, axis=0)
            old_log_probs_batchs = old_log_probs_batchs.take(indexes, axis=0)
            old_vals_batchs = old_vals_batchs.take(indexes, axis=0)
            returns_batchs = returns_batchs.take(indexes, axis=0)
            additions_batchs = additions_batchs.take(indexes, axis=0)
            dones_batchs = dones_batchs.take(indexes, axis=0)

            for (states_batch,
                 actions_batch,
                 old_log_probs_batch,
                 old_vals_batch,
                 returns_batch,
                 additions_batch,
                 dones_batch
                 ) in zip(states_batches,
                          actions_batchs,
                          old_log_probs_batchs,
                          old_vals_batchs,
                          returns_batchs,
                          additions_batchs,
                          dones_batchs
                          ):
                indexes = 1 + numpy.argwhere(numpy.max(dones_batch, axis=-1) == 1).reshape(-1)
                states_batch_prep = numpy.split(states_batch, indexes, 0)
                actions_batch_prep = numpy.split(actions_batch, indexes, 0)
                dones_batch_prep = numpy.split(dones_batch, indexes, 0)
                info.append(
                    self._one_train_seq(states_batch_prep,
                                        actions_batch_prep,
                                        old_log_probs_batch,
                                        old_vals_batch,
                                        returns_batch,
                                        additions_batch,
                                        dones_batch_prep)
                )

        # return info
        return info


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
                 cnfg=None,
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
    def get_config(env_type):
        return get_config(env_type)


PPO = PpoInitializer()
