import torch
import numpy
import random
import gym


class Buffer(object):
    def __init__(self):
        self.rollouts = []

    def append(self, transition):
        self.rollouts.append(transition)

    def rollout(self):
        transitions = self.rollouts
        self.rollouts = []

        return list(transitions)

    def learn(self, data, nminibatches):

        batches = []
        for i in range(0, len(data), nminibatches):
            last_ind = (i + nminibatches) if (len(data) - i - nminibatches) >= nminibatches else len(data)
            one_batch = data[i:min(last_ind, len(data))]

            states, actions, nstates, rewards, dones, old_log_probs, vals, returns = zip(*one_batch)

            transition = (states, actions, nstates, rewards, dones, old_log_probs, vals, returns)

            batches.append(transition)
            if not last_ind == len(data):
                break

        return batches

    def __len__(self):
        return len(self.rollouts)


class PPO:
    lossfun = torch.nn.SmoothL1Loss()

    def __init__(self, nn,
                 observation_space=gym.spaces.Discrete(5),
                 action_space=gym.spaces.Discrete(5),
                 cnfg=None):

        self.nn_type = nn

        self.nn = nn(observation_space, action_space, simple=cnfg['simple_nn'])

        print(self.nn)

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

        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.learning_rate, betas=(0.99, 0.999), eps=1e-08)

        self.buffer = Buffer()

    def __call__(self, state):
        return self.nn.get_action(torch.Tensor(state))

    def experience(self, transition, timestep_now):
        self.buffer.append(transition)
        if len(self.buffer) >= self.nsteps:
            data = self.buffer.rollout()

            data = self.calculate_advantages(data)

            batches = self.buffer.learn(data, self.nminibatches)

            for i in range(self.noptepochs):
                random.shuffle(batches)
                for minibatch in batches:
                    ans = self.learn(minibatch, timestep_now)
            return ans

        return None

    def calculate_advantages(self, data):
        states, actions, nstates, rewards, dones, outs = zip(*data)
        old_log_probs, old_vals = zip(*outs)

        n_state_old_vals = numpy.array(old_vals)
        t_nstates = torch.FloatTensor(nstates)
        n_rewards = numpy.array(rewards)
        n_dones = numpy.array(dones)

        n_new_state_vals = self.nn(t_nstates)[1].detach().squeeze(-1).cpu().numpy()

        # Making td residual
        td_residual = - n_state_old_vals + n_rewards + self.gamma * (1. - n_dones) * n_new_state_vals

        # Making GAE from td residual
        n_advs = list(self._gae(td_residual, n_dones))

        transitions = (states, actions, nstates, rewards, dones, old_log_probs, old_vals, n_advs)

        return list(zip(*transitions))

    def learn(self, input, timestep_now):

        # Unpack
        states, actions, nstates, rewards, dones, old_log_probs, old_vals, advs = input

        # Tensors
        t_states = torch.FloatTensor(states)
        t_actions = torch.from_numpy(numpy.array(actions, dtype=self.nn.np_type))
        t_nstates = torch.FloatTensor(nstates)
        t_rewards = torch.FloatTensor(rewards)
        t_dones = torch.FloatTensor(dones)
        t_old_log_probs = torch.from_numpy(numpy.array(old_log_probs, dtype=numpy.float32))
        t_state_old_vals = torch.FloatTensor(old_vals)
        t_advs = torch.FloatTensor(advs)

        # Feedforward with building computation graph
        t_distrib, t_state_vals_un = self.nn(t_states)
        t_state_vals = t_state_vals_un.squeeze(-1)
        t_new_state_vals = self.nn(t_nstates)[1].detach().squeeze(-1)

        # Making target for value update and for td residual
        t_target_state_vals = t_rewards + self.gamma * (1. - t_dones) * t_new_state_vals

        # Making critic losses
        t_state_vals_clipped = t_state_old_vals + torch.clamp(t_state_vals - t_state_old_vals, - self.cliprange, self.cliprange)
        t_critic_loss1 = self.lossfun(t_state_vals, t_target_state_vals)

        # Making critic final loss
        clip_value = True
        if clip_value:
            t_critic_loss2 = self.lossfun(t_state_vals_clipped, t_target_state_vals)
            t_critic_loss = .5 * torch.max(t_critic_loss1, t_critic_loss2)
        else:
            t_critic_loss = .5 * t_critic_loss1

        # Normalizing advantages
        # t_advantages = (t_advs - t_advs.mean()) / (t_advs.std() + 1e-8)
        t_advantages = (t_advs - t_advs.mean()) / (t_advs.std() + 1e-8)

        # Getting log probs
        t_new_log_probs = t_distrib.log_prob(t_actions)

        # Calculating ratio
        t_ratio = torch.exp(t_new_log_probs - t_old_log_probs)

        # Calculating surrogates
        t_rt1 = torch.mul(t_advantages.unsqueeze(-1), t_ratio)
        t_rt2 = torch.mul(t_advantages.unsqueeze(-1), torch.clamp(t_ratio, 1 - self.cliprange, 1 + self.cliprange))
        t_actor_loss = torch.min(t_rt1, t_rt2).mean()

        # Calculating entropy
        t_entropy = t_distrib.entropy().mean()

        # Making surrogate loss
        t_surrogate = t_actor_loss - self.vf_coef * t_critic_loss + self.ent_coef * t_entropy

        # Making loss for Neural network
        t_loss = - t_surrogate

        # Calculating gradients
        self.optimizer.zero_grad()
        t_loss.backward()

        # Normalizing gradients
        for param in self.nn.parameters():
            param.grad.data.clamp_(-self.max_grad_norm, self.max_grad_norm)

        # Adjusting learning rate
        lr_now = self.learning_rate * (1 - timestep_now / self.final_timestep)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_now

        # Optimizer step
        self.optimizer.step()

        return t_actor_loss.item(), t_critic_loss.item(), t_entropy.item(), t_distrib.variance.min().item()

    def _gae(self, td_residual, dones):
        for i in reversed(range(td_residual.shape[0] - 1)):
            td_residual[i] += self.lam * self.gamma * (1. - dones[i]) * td_residual[i+1]

        return td_residual
