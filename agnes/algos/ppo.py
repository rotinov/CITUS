import torch
import numpy
import random
import gym
from agnes.algos import base
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
        self.lam = cnfg['lam']
        self.noptepochs = cnfg['noptepochs']
        self.MAX_GRAD_NORM = cnfg['max_grad_norm']
        self.workers_num = workers

        self.nbatch = self.workers_num * self.nsteps
        self.nbatch_train = self.nbatch // self.nminibatches

        final_epoch = int(self.final_timestep / self.nsteps * self.nminibatches * self.noptepochs)  # 312500

        if trainer:
            self._optimizer = torch.optim.Adam(self._nnet.parameters(), lr=self.learning_rate,
                                               betas=(0.99, 0.999),
                                               eps=1e-3)

            self.lr_scheduler = schedules.LinearAnnealingLR(self._optimizer, eta_min=0.0,  # 1e-6
                                                            to_epoch=final_epoch)

        self.buffer = Buffer()

        self._trainer = trainer
        PPO.FIRST = False

    def __call__(self, state):
        with torch.no_grad():
            if self._device == torch.device('cpu'):
                return self._nnet.get_action(torch.FloatTensor(state))
            else:
                return self._nnet.get_action(torch.cuda.FloatTensor(state))

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
            last_values = self._nnet(t_nstates)[1].detach().squeeze(-1).cpu().numpy()

            n_state_vals = n_state_vals.reshape(n_shape)

        # Making GAE from td residual
        n_advs = numpy.zeros_like(n_state_vals)
        lastgaelam = 0
        for t in reversed(range(n_advs.shape[0])):
            if t == n_advs.shape[0] - 1:
                nextvalues = last_values
            else:
                nextvalues = n_state_vals[t+1]

            nextnonterminal = 1. - n_dones[t]
            delta = n_rewards[t] + self.GAMMA * nextnonterminal * nextvalues - n_state_vals[t]
            n_advs[t] = lastgaelam = delta + self.lam * self.GAMMA * nextnonterminal * lastgaelam

        n_returns = n_advs + n_state_vals

        if n_rewards.ndim == 1:
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
            if self._nnet.type_of_out == torch.int16:
                ACTIONS = torch.LongTensor(ACTIONS)
            else:
                ACTIONS = torch.FloatTensor(ACTIONS)
            OLDLOGPROBS = torch.FloatTensor(OLDLOGPROBS)
            OLDVALS = torch.FloatTensor(OLDVALS)
            RETURNS = torch.FloatTensor(RETURNS)
        else:
            STATES = torch.cuda.FloatTensor(STATES)
            if self._nnet.type_of_out == torch.int16:
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
        ADVANTAGES = RETURNS - t_state_vals.detach()

        # Normalizing advantages
        ADVS = ((ADVANTAGES - ADVANTAGES.mean()) / (ADVANTAGES.std() + 1e-8)).unsqueeze(-1)

        # Making critic losses
        t_state_vals_clipped = OLDVALS + torch.clamp(t_state_vals - OLDVALS,
                                                     - self.CLIPRANGE,
                                                     self.CLIPRANGE)

        # Making critic final loss
        t_critic_loss1 = (t_state_vals - RETURNS).pow(2)
        t_critic_loss2 = (t_state_vals_clipped - RETURNS).pow(2)
        t_critic_loss = 0.5 * torch.mean(torch.max(t_critic_loss1, t_critic_loss2))

        # Getting log probs
        t_new_log_probs = t_distrib.log_prob(ACTIONS)

        # Calculating ratio
        t_ratio = torch.exp(t_new_log_probs - OLDLOGPROBS)

        approxkl = (.5 * torch.mean((OLDLOGPROBS - t_new_log_probs) ** 2)).item()
        clipfrac = torch.mean((torch.abs(t_ratio - 1.0) > self.CLIPRANGE).float()).item()

        # Calculating surrogates
        t_rt1 = torch.mul(ADVS, t_ratio)
        t_rt2 = torch.mul(ADVS, torch.clamp(t_ratio, 1 - self.CLIPRANGE, 1 + self.CLIPRANGE))
        t_actor_loss = torch.min(t_rt1, t_rt2).mean()

        # Calculating entropy
        t_entropy = t_distrib.entropy().mean()

        # Making surrogate loss
        t_surrogate = t_actor_loss - self.vf_coef * t_critic_loss + self.ent_coef * t_entropy

        # Making loss for Neural network
        t_loss = - t_surrogate

        # Calculating gradients
        self._optimizer.zero_grad()
        t_loss.backward()

        # Normalizing gradients
        torch.nn.utils.clip_grad_norm_(self._nnet.parameters(), self.MAX_GRAD_NORM)

        # Optimizer step
        self._optimizer.step()
        self.lr_scheduler.step()

        return (- t_actor_loss.item(),
                t_critic_loss.item(),
                t_entropy.item(),
                approxkl,
                clipfrac,
                logger.explained_variance(t_state_vals.detach().cpu().numpy(), RETURNS.detach().cpu().numpy()),
                ()
                )


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
