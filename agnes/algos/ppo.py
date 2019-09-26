import torch
import numpy
import random
import gym
from agnes.algos import base
from agnes.common import schedules
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


class PPO(base.BaseAlgo):
    _device = torch.device('cpu')
    lossfun = torch.nn.MSELoss(reduction='none').to(_device)

    get_config = get_config

    def __init__(self, nn, observation_space=gym.spaces.Discrete(5), action_space=gym.spaces.Discrete(5), cnfg=None,
                 workers=1, trainer=True):
        super().__init__()

        self.nn_type = nn

        if trainer:
            pprint(cnfg)

        self._nnet = nn(observation_space, action_space)

        if trainer:
            print(self._nnet)
        else:
            self._nnet.eval()

        self.gamma = cnfg['gamma']
        self.learning_rate = cnfg['learning_rate']
        self.cliprange = cnfg['cliprange']
        self.vf_coef = cnfg['vf_coef']
        self.ent_coef = cnfg['ent_coef']
        self.final_timestep = cnfg['timesteps']
        self.nsteps = cnfg['nsteps']
        self.nminibatches = cnfg['nminibatches']
        self.lam = cnfg['lam']
        self.noptepochs = cnfg['noptepochs']
        self.max_grad_norm = cnfg['max_grad_norm']
        self.workers_num = workers

        self.nbatch = self.workers_num * self.nsteps
        self.nbatch_train = self.nbatch // self.nminibatches

        final_epoch = int(self.final_timestep / self.nsteps * self.nminibatches * self.noptepochs)  # 312500

        if trainer:
            self._optimizer = torch.optim.Adam(self._nnet.parameters(), lr=self.learning_rate, betas=(0.99, 0.999),
                                               eps=1e-5)

            self.lr_scheduler = schedules.LinearAnnealingLR(self._optimizer, eta_min=0.0,  # 1e-6
                                                            to_epoch=final_epoch)

        self.buffer = Buffer()

        self._trainer = trainer
        PPO.FIRST = False

    def __call__(self, state):
        with torch.no_grad():
            return self._nnet.get_action(torch.from_numpy(numpy.array(state, dtype=numpy.float32))
                                         .to(self._device))

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

        info = []

        # Unpack
        states, actions, new_state_old_vals, rewards, dones, old_log_probs, old_vals, advs = zip(*data)

        # Tensors
        if self._device == torch.device('cpu'):
            t_states = torch.FloatTensor(states)
            t_actions = torch.from_numpy(numpy.array(actions, dtype=self._nnet.np_type))
            t_rewards = torch.FloatTensor(rewards)
            t_new_state_old_vals = torch.FloatTensor(new_state_old_vals).reshape(t_rewards.shape)
            t_dones = torch.FloatTensor(dones)
            t_old_log_probs = torch.from_numpy(numpy.array(old_log_probs, dtype=numpy.float32))
            t_state_old_vals = torch.FloatTensor(numpy.array(old_vals, dtype=numpy.float32))
            t_advs = torch.FloatTensor(advs)
        else:
            t_states = torch.cuda.FloatTensor(states)
            t_actions = torch.from_numpy(numpy.array(actions, dtype=self._nnet.np_type)).to(self._device)
            t_rewards = torch.cuda.FloatTensor(rewards)
            t_new_state_old_vals = torch.cuda.FloatTensor(new_state_old_vals).reshape(t_rewards.shape)
            t_dones = torch.cuda.FloatTensor(dones)
            t_old_log_probs = torch.from_numpy(numpy.array(old_log_probs, dtype=numpy.float32)).to(self._device)
            t_state_old_vals = torch.cuda.FloatTensor(numpy.array(old_vals, dtype=numpy.float32))
            t_advs = torch.cuda.FloatTensor(advs)

        indexes = [i for i in range(len(data))]

        for i in range(self.noptepochs):
            random.shuffle(indexes)
            ind_batches = self.buffer.learn(indexes, self.nbatch_train)
            for ind_minibatch in ind_batches:
                minibatch = (t_states[ind_minibatch], t_actions[ind_minibatch], t_new_state_old_vals[ind_minibatch],
                             t_rewards[ind_minibatch], t_dones[ind_minibatch], t_old_log_probs[ind_minibatch],
                             t_state_old_vals[ind_minibatch], t_advs[ind_minibatch])
                info.append(self._one_train(minibatch))

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
        self.lossfun = self.lossfun.to(device)

        return self

    def _calculate_advantages(self, data):
        states, actions, nstates, rewards, dones, outs = zip(*data)
        old_log_probs, old_vals = zip(*outs)

        n_rewards = numpy.array(rewards)
        n_dones = numpy.array(dones)
        n_shape = n_dones.shape

        n_state_vals = numpy.array(old_vals)
        n_new_state_vals = numpy.array(old_vals)
        n_new_state_vals[:-1] = n_new_state_vals[1:]

        with torch.no_grad():
            t_nstates = torch.FloatTensor(nstates[-1]).to(self._device)
            n_new_state_vals[-1] = self._nnet(t_nstates)[1].detach().squeeze(-1).cpu().numpy()
            n_new_state_vals = n_new_state_vals.reshape(n_shape)
            n_state_vals = n_state_vals.reshape(n_shape)

        # Making td residual
        td_residual = - n_state_vals.reshape(n_shape) + n_rewards + self.gamma * (1. - n_dones) * n_new_state_vals

        # Making GAE from td residual
        n_advs = list(self._gae(td_residual, n_dones))

        l_new_state_vals = list(n_new_state_vals)

        if n_rewards.ndim == 1:
            transitions = (states, actions, l_new_state_vals, rewards, dones, old_log_probs, old_vals, n_advs)
        else:
            li_states = []
            li_actions = []
            li_new_state_vals = []
            li_rewards = []
            li_dones = []
            li_old_log_probs = []
            li_old_vals = []
            li_n_advs = []
            for i in range(len(states)):
                li_states.extend(list(states[i]))
                li_actions.extend(list(actions[i]))
                li_new_state_vals.extend(list(l_new_state_vals[i]))
                li_rewards.extend(list(rewards[i]))
                li_dones.extend(list(dones[i]))
                li_old_log_probs.extend(list(old_log_probs[i]))
                li_old_vals.extend(list(old_vals[i]))
                li_n_advs.extend(list(n_advs[i]))
            transitions = (li_states, li_actions, li_new_state_vals, li_rewards, li_dones, li_old_log_probs, li_old_vals, li_n_advs)

        return list(zip(*transitions))

    def _one_train(self, data):
        t_states, t_actions, t_new_state_old_vals, t_rewards, t_dones, t_old_log_probs, t_state_old_vals, t_advs = data

        # Feedforward with building computation graph
        t_distrib, t_state_vals_un = self._nnet(t_states)
        t_state_vals = t_state_vals_un.squeeze(-1)

        # Making target for value update and for td residual
        t_target_state_vals = t_rewards + self.gamma * (1. - t_dones) * t_new_state_old_vals

        # Making critic losses
        t_state_vals_clipped = t_state_old_vals.view_as(t_state_vals) + \
                               torch.clamp(t_state_vals - t_state_old_vals.view_as(t_state_vals), - self.cliprange, self.cliprange)
        t_critic_loss1 = self.lossfun(t_state_vals, t_target_state_vals)

        # Making critic final loss
        t_critic_loss2 = self.lossfun(t_state_vals_clipped, t_target_state_vals)
        t_critic_loss = .5 * torch.max(t_critic_loss1, t_critic_loss2).mean()

        # Normalizing advantages
        # t_advantages = t_advs
        t_advantages = ((t_advs - t_advs.mean()) / (t_advs.std() + 1e-8)).unsqueeze(-1)

        # Getting log probs
        t_new_log_probs = t_distrib.log_prob(t_actions)

        # Calculating ratio
        t_ratio = torch.exp(t_new_log_probs - t_old_log_probs)

        approxkl = (.5 * torch.mean((t_old_log_probs - t_new_log_probs).pow(2))).item()
        clipfrac = torch.mean((torch.abs(t_ratio - 1.0) > self.cliprange).float()).item()

        # Calculating surrogates
        t_rt1 = torch.mul(t_advantages, t_ratio)
        t_rt2 = torch.mul(t_advantages, torch.clamp(t_ratio, 1 - self.cliprange, 1 + self.cliprange))
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
        torch.nn.utils.clip_grad_norm_(self._nnet.parameters(), self.max_grad_norm)

        # Optimizer step
        self._optimizer.step()
        self.lr_scheduler.step()

        return t_actor_loss.item(), t_critic_loss.item(), t_entropy.item(), approxkl, clipfrac, \
               t_distrib.variance.mean().item(), (self.lr_scheduler.get_lr()[0], self.lr_scheduler.get_count())

    def _gae(self, td_residual, dones):
        for i in reversed(range(td_residual.shape[0] - 1)):
            td_residual[i] += self.lam * self.gamma * (1. - dones[i]) * td_residual[i + 1]

        return td_residual
