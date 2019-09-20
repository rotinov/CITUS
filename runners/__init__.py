import time
from collections import deque
import re
import nns
import PPO
from defaults import *


def safemean(x):
    return sum(x) / max(1, len(x))


def log(eplenmean, rewardarr, entropy, actor_loss, critic_loss, nupdates, frames, beg, *debug):
    print('-' * 38)
    print('| eplenmean: {:.2f}'.format(safemean(eplenmean)),
          '\n| eprewmean: {:.2f}'.format(safemean(rewardarr)),
          '\n| loss/policy_entropy: {:.2f}'.format(safemean(entropy)),
          '\n| loss/policy_loss: {:.2f}'.format(safemean(actor_loss)),
          '\n| loss/value_loss: {:.2f}'.format(safemean(critic_loss)),
          '\n| misc/nupdates: {:.2e}'.format(nupdates),
          '\n| misc/serial_timesteps: {:.2e}'.format(frames),
          '\n| misc/time_elapsed: {:.2e}'.format(int(time.time() - beg)))

    i = 1
    for item in debug:
        print('| misc/debug {}: {:.2e}'.format(i, safemean(item)))
        i += 1

    print('-' * 38)


class Single:
    def __init__(self, env, algo=PPO.PPO, nn=nns.MLPDiscrete):
        self.env = env
        env_type = str(env.unwrapped.__class__)
        env_type2 = re.split('[, \']', env_type)
        self.env_type = env_type2[2].split('.')[2]
        print(self.env_type)

        if self.env_type == 'classic_control':
            self.cnfg = classic_config()
        elif self.env_type == 'mujoco':
            self.cnfg = mujoco_config()
        elif self.env_type == 'box2d':
            self.cnfg = box2d_config()
        else:
            self.cnfg = default_config()

        self.actor = algo(nn, env.observation_space, env.action_space, self.cnfg)

    def run(self, log_interval=500):
        timesteps = self.cnfg['timesteps']

        frames = 0
        nupdates = 0
        eplenmean = deque(maxlen=log_interval)
        rewardarr = deque(maxlen=log_interval)
        lr_things = []
        beg = time.time()
        print("Stepping environment...")
        while frames < timesteps:
            state = self.env.reset()
            frames_beg = frames
            frames += 1
            rewardsum = 0

            while True:
                action, pred_action, out = self.actor(state)

                nstate, reward, done, _ = self.env.step(action)
                rewardsum += reward

                transition = (state, pred_action, nstate, reward, done, out)
                lr_thing = self.actor.experience(transition, frames)
                if lr_thing is not None:
                    lr_things.append(lr_thing)

                state = nstate
                frames += 1

                if frames % log_interval == 0 and len(lr_things) > 0:
                    actor_loss, critic_loss, entropy, debug = zip(*lr_things)
                    log(eplenmean, rewardarr, entropy, actor_loss, critic_loss, nupdates, frames, beg, debug)
                    lr_things = []

                if done:
                    eplenmean.append(frames - frames_beg)
                    rewardarr.append(rewardsum)
                    break
