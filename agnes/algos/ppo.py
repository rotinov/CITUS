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
            self._optimizer = torch.optim.Adam(self._nnet.parameters(), lr=self.learning_rate, betas=(0.99, 0.999),
                                               eps=1e-5)

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

        info = []

        # Unpack
        states, actions, old_log_probs, old_vals, advs = zip(*data)

        # Tensors
        if self._device == torch.device('cpu'):
            t_states = torch.FloatTensor(states)
            if self._nnet.type_of_out == torch.int16:
                t_actions = torch.LongTensor(actions)
            else:
                t_actions = torch.FloatTensor(actions)
            t_old_log_probs = torch.FloatTensor(old_log_probs)
            t_state_old_vals = torch.FloatTensor(old_vals)
            t_advs = torch.FloatTensor(advs)
        else:
            t_states = torch.cuda.FloatTensor(states)
            if self._nnet.type_of_out == torch.int16:
                t_actions = torch.cuda.LongTensor(actions)
            else:
                t_actions = torch.cuda.FloatTensor(actions)
            t_old_log_probs = torch.cuda.FloatTensor(old_log_probs)
            t_state_old_vals = torch.cuda.FloatTensor(old_vals)
            t_advs = torch.cuda.FloatTensor(advs)

        indexes = [i for i in range(len(data))]

        for i in range(self.noptepochs):
            random.shuffle(indexes)
            ind_batches = self.buffer.learn(indexes, self.nbatch_train)
            for ind_minibatch in ind_batches:
                minibatch = (t_states[ind_minibatch], t_actions[ind_minibatch], t_old_log_probs[ind_minibatch],
                             t_state_old_vals[ind_minibatch], t_advs[ind_minibatch])
                info.append(self._one_train(minibatch))  # STATES, ACTIONS, OLDLOGPROBS, OLDVALS, RETURNS

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

    def device_info(self):
        if self._device.type == 'cuda':
            return torch.cuda.get_device_name(device=self._device)
        else:
            return 'CPU'

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
            if self._device == torch.device('cpu'):
                t_nstates = torch.FloatTensor(nstates[-1])
            else:
                t_nstates = torch.cuda.FloatTensor(nstates[-1])
            n_new_state_vals[-1] = self._nnet(t_nstates)[1].detach().squeeze(-1).cpu().numpy()

            n_new_state_vals = n_new_state_vals.reshape(n_shape)
            n_state_vals = n_state_vals.reshape(n_shape)

        # Making td residual
        td_residual = - n_state_vals.reshape(n_shape) + n_rewards + self.GAMMA * (1. - n_dones) * n_new_state_vals

        # Making GAE from td residual
        n_advs = td_residual
        gaelam = 0
        for i in reversed(range(n_advs.shape[0])):
            n_advs[i] = gaelam = n_advs[i] + self.lam * self.GAMMA * (1. - n_dones[i]) * gaelam

        # n_advs = self._gae(td_residual, n_dones)
        n_returns = n_advs + n_state_vals

        if n_rewards.ndim == 1:
            transitions = (numpy.array(states), numpy.array(actions),
                           numpy.array(old_log_probs), numpy.array(old_vals), n_returns)
        else:
            li_states = numpy.array(states)
            li_states = li_states.reshape((-1,) + li_states.shape[2:])

            li_actions = numpy.array(actions)
            li_actions = li_actions.reshape((-1,) + li_actions.shape[2:])
            li_old_vals = n_state_vals.reshape((-1,) + n_state_vals.shape[2:])

            li_old_log_probs = numpy.array(old_log_probs)
            li_old_log_probs = li_old_log_probs.reshape((-1,) + li_old_log_probs.shape[2:])

            li_n_returns = n_returns
            li_n_returns = li_n_returns.reshape((-1,) + li_n_returns.shape[2:])

            transitions = (li_states, li_actions, li_old_log_probs, li_old_vals, li_n_returns)

        return list(zip(*transitions))

    def _one_train(self, DATA):
        STATES, ACTIONS, OLDLOGPROBS, OLDVALS, RETURNS = DATA

        # Feedforward with building computation graph
        t_distrib, t_state_vals_un = self._nnet(STATES)
        t_state_vals = t_state_vals_un.squeeze(-1)

        STATEVALSNEW = t_state_vals.detach()
        OLDVALS = OLDVALS.view_as(t_state_vals)
        ADVS = RETURNS - STATEVALSNEW

        # Normalizing advantages
        ADVS = ((ADVS - ADVS.mean()) / (ADVS.std() + 1e-8)).unsqueeze(-1)
        TARGETVAL = ADVS.squeeze(-1) + STATEVALSNEW

        # Making critic losses
        t_state_vals_clipped = OLDVALS + torch.clamp(t_state_vals - OLDVALS, - self.CLIPRANGE, self.CLIPRANGE)

        # print('Value:', STATEVALSNEW.mean())
        # print('Target:', RETURNS.mean())
        # print('Unsquared loss1:', (STATEVALSNEW - RETURNS).mean())
        # print('Unsquared loss2:', (t_state_vals_clipped.detach() - RETURNS).mean())
        # print('-'*15)

        # Making critic final loss
        t_critic_loss1 = self.lossfun(t_state_vals, TARGETVAL)
        t_critic_loss2 = self.lossfun(t_state_vals_clipped, TARGETVAL)
        t_critic_loss = .5 * torch.max(t_critic_loss1, t_critic_loss2).mean()

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

        return - t_actor_loss.item(), t_critic_loss.item(), t_entropy.item(), approxkl, clipfrac, \
               logger.explained_variance(t_state_vals.detach().cpu().numpy(), RETURNS.detach().cpu().numpy()), \
               (self.lr_scheduler.get_lr()[0], self.lr_scheduler.get_count())
