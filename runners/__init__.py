from collections import deque
import re
import nns
import Algos
from defaults import *
from common.logger import log


class Single:
    def __init__(self, env, algo=Algos.PPO, nn=nns.MLPDiscrete):
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

        self.trainer = algo(nn, env.observation_space, env.action_space, self.cnfg)

        self.actor.update(self.trainer)

    def run(self, log_interval=1):
        timesteps = self.cnfg['timesteps']

        frames = 0
        nupdates = 0
        eplenmean = deque(maxlen=log_interval)
        rewardarr = deque(maxlen=log_interval)
        lr_things = []
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
                data = self.actor.experience(transition)

                if data:
                    print("Done.")
                    lr_thing = self.trainer.train(data)
                    lr_things.extend(lr_thing)
                    nupdates += 1

                    self.actor.update(self.trainer)

                    if nupdates % log_interval == 0:
                        actor_loss, critic_loss, entropy, debug = zip(*lr_things)
                        log(eplenmean, rewardarr, entropy, actor_loss, critic_loss, nupdates, frames, *zip(*debug))
                        lr_things = []

                    print("Stepping environment...")

                state = nstate
                frames += 1

                if done:
                    eplenmean.append(frames - frames_beg)
                    rewardarr.append(rewardsum)
                    break

        actor_loss, critic_loss, entropy, debug = zip(*lr_things)
        log(eplenmean, rewardarr, entropy, actor_loss, critic_loss, nupdates, frames, *zip(*debug))
