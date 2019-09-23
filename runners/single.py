from collections import deque
import nns
import algos
from common import logger
import time
from torch import cuda


class Single:
    def __init__(self, env, algo: algos.base.BaseAlgo.__class__ = algos.PPO, nn=nns.MLP):
        self.env = env
        self.cnfg, self.env_type = algo.get_config(env)
        print(self.env_type)

        self.worker = algo(nn, env.observation_space, env.action_space, self.cnfg, trainer=False)

        self.trainer = algo(nn, env.observation_space, env.action_space, self.cnfg)
        if cuda.is_available():
            self.trainer = self.trainer.to('cuda:0')

        self.worker.update(self.trainer)

        self.logger = logger.TensorboardLogger(".logs/"+str(time.time()))

    def run(self, log_interval=1):
        timesteps = self.cnfg['timesteps']

        frames = 0
        nupdates = 0
        eplenmean = deque(maxlen=5*log_interval)
        rewardarr = deque(maxlen=5*log_interval)
        lr_things = []
        print("Stepping environment...")
        while frames < timesteps:
            state = self.env.reset()
            frames_beg = frames
            frames += 1
            rewardsum = 0

            while True:
                action, pred_action, out = self.worker(state)

                nstate, reward, done, _ = self.env.step(action)
                rewardsum += reward

                transition = (state, pred_action, nstate, reward, done, out)
                data = self.worker.experience(transition)

                if data:
                    print("Done.")
                    lr_thing = self.trainer.train(data)
                    lr_things.extend(lr_thing)
                    nupdates += 1

                    self.worker.update(self.trainer)

                    if nupdates % log_interval == 0 and lr_things:
                        actor_loss, critic_loss, entropy, approxkl, clipfrac, variance, debug = zip(*lr_things)
                        self.logger(eplenmean, rewardarr, entropy,
                                    actor_loss, critic_loss, nupdates,
                                    frames, approxkl, clipfrac, variance, zip(*debug))
                        lr_things = []

                    print("Stepping environment...")

                state = nstate
                frames += 1

                if done:
                    eplenmean.append(frames - frames_beg)
                    rewardarr.append(rewardsum)
                    break

        if lr_things:
            actor_loss, critic_loss, entropy, approxkl, clipfrac, variance, debug = zip(*lr_things)
            self.logger(eplenmean, rewardarr, entropy,
                        actor_loss, critic_loss, nupdates,
                        frames, approxkl, clipfrac, variance, zip(*debug))
