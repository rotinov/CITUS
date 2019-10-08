import torch
import numpy
import random
import gym
from agnes.algos import base
from agnes.nns import rnn
from agnes.common import schedules, logger
from pprint import pprint
from agnes.algos.configs.ppo_config import get_config


class Buffer(base.BaseBuffer):
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


class PpoClass(base.BaseAlgo):
    _device = torch.device('cpu')

    get_config = get_config

    def __init__(self, nn,
                 observation_space=gym.spaces.Discrete(5),
                 action_space=gym.spaces.Discrete(5),
                 cnfg=None,
                 workers=1,
                 trainer=True):
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
            self._optimizer = torch.optim.Adam(self._nnet.parameters(), lr=self.learning_rate,
                                               betas=(0.99, 0.999),
                                               eps=1e-5)

            self.lr_scheduler = schedules.LinearAnnealingLR(self._optimizer, eta_min=0.0,
                                                            to_epoch=final_epoch)

            if isinstance(self.CLIPRANGE, float):
                self.cr_schedule = schedules.LinearSchedule(lambda x: self.CLIPRANGE * x, eta_min=1.0, to_epoch=final_epoch)
            else:
                self.cr_schedule = schedules.LinearSchedule(self.CLIPRANGE, eta_min=0.0, to_epoch=final_epoch)

        self.buffer = Buffer()

        self._trainer = trainer
        PPO.FIRST = False

    def __call__(self, state, done):
        with torch.no_grad():
            if self._device == torch.device('cpu'):
                return self._nnet.get_action(torch.FloatTensor(state), torch.FloatTensor(done))
            else:
                return self._nnet.get_action(torch.cuda.FloatTensor(state), torch.cuda.FloatTensor(done))

    def experience(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) >= self.nsteps:
            data = self.buffer.rollout()

            data = self._calculate_advantages(data)

            return data
        return None

    def train(self, data):
        if data is None:
            return None

        if isinstance(self._nnet, rnn.RecurrentFamily):
            return self.train_with_bptt(data)

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

    def update(self, from_agent: base.BaseAlgo):
        assert not self._trainer

        self._nnet.load_state_dict(from_agent._nnet.state_dict())

        return True

    def get_state_dict(self):
        assert self._trainer
        return self._nnet.state_dict()

    def get_nn_instance(self):
        assert self._trainer
        return self._nnet

    def load_state_dict(self, state_dict):
        return self._nnet.load_state_dict(state_dict)

    def to(self, device: str):
        device = torch.device(device)
        self._device = device
        self._nnet = self._nnet.to(device)

        return self

    def device_info(self):
        if self._device.type == 'cuda':
            return torch.cuda.get_device_name(device=self._device)
        else:
            return 'CPU'

    def _calculate_advantages(self, data):
        states, actions, nstates, rewards, dones, outs = zip(*data)
        if isinstance(self._nnet, rnn.RecurrentFamily):
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

        if n_rewards.ndim == 1 or isinstance(self._nnet, rnn.RecurrentFamily):
            if isinstance(self._nnet, rnn.RecurrentFamily):
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

        self.CLIPRANGE = self.cr_schedule.get_v()

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
        self.lr_scheduler.step()
        self.cr_schedule.step()

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
            ADDITIONS = torch.FloatTensor(ADDITIONS[0]).requires_grad_()
            DONES = torch.FloatTensor(DONES)
        else:
            STATES = torch.cuda.FloatTensor(STATES)
            if self._nnet.type_of_out() == torch.int16:
                ACTIONS = torch.cuda.LongTensor(ACTIONS)
            else:
                ACTIONS = torch.cuda.FloatTensor(ACTIONS)
            OLDLOGPROBS = torch.cuda.FloatTensor(OLDLOGPROBS)
            OLDVALS = torch.cuda.FloatTensor(OLDVALS)
            RETURNS = torch.cuda.FloatTensor(RETURNS)
            ADDITIONS = torch.cuda.FloatTensor(ADDITIONS[0]).requires_grad_()
            DONES = torch.cuda.FloatTensor(DONES)

        # Feedforward with building computation graph
        l_new_log_probs = []
        l_state_vals = []
        t_addition = ADDITIONS

        for t_state, t_done, t_action in zip(STATES, DONES, ACTIONS):
            t_distrib, t_addition, t_state_vals_un = self._nnet(t_state, t_addition)
            if t_done.ndimension() < 2:
                t_done = t_done.unsqueeze(-1)

            t_addition = (1. - t_done) * t_addition

            l_state_vals.append(t_state_vals_un.squeeze(-1))
            l_new_log_probs.append(t_distrib.log_prob(t_action).squeeze(-1))

        t_new_log_probs = torch.stack(l_new_log_probs, dim=0)
        t_state_vals = torch.stack(l_state_vals, dim=0)

        OLDVALS = OLDVALS.view_as(t_state_vals)
        ADVANTAGES = RETURNS - OLDVALS

        self.CLIPRANGE = self.cr_schedule.get_v()

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
        self.lr_scheduler.step()
        self.cr_schedule.step()

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
                info.append(
                    self._one_train_seq(states_batch,
                                        actions_batch,
                                        old_log_probs_batch,
                                        old_vals_batch,
                                        returns_batch,
                                        additions_batch,
                                        dones_batch)
                )

        # return info
        return info


class PpoInitializer:
    def __init__(self):
        pass

    def __call__(self, nn,
                 observation_space=gym.spaces.Discrete(5),
                 action_space=gym.spaces.Discrete(5),
                 cnfg=None,
                 workers=1,
                 trainer=True):
        return PpoClass(nn,
                        observation_space,
                        action_space,
                        cnfg,
                        workers,
                        trainer)

    @staticmethod
    def get_config(env_type):
        return get_config(env_type)


PPO = PpoInitializer()
