from collections import deque
import agnes
from agnes.common import logger
from torch import cuda
import numpy


class Single:
    """"Single" runner releases learning with a single worker that is also a trainer.
    "Single" runner is compatible with vector environments(config or env_type should be specified manually).
    """

    def __init__(self, env,
                 algo: agnes.algos.base.BaseAlgo.__class__ = agnes.algos.PPO,
                 nn=agnes.nns.MLP, config=None):
        env, env_type, vec_num = env
        self.env = env

        self.cnfg, self.env_type = algo.get_config(env_type)
        if config is not None:
            self.cnfg = config

        self.vec_num = vec_num

        print('Env type: ', self.env_type, 'Envs num:', vec_num)

        self.trainer = algo(nn, env.observation_space, env.action_space, self.cnfg, workers=vec_num)
        if cuda.is_available():
            self.trainer = self.trainer.to('cuda:0')

        self.logger = logger.ListLogger()

    def log(self, *args):
        self.logger = logger.ListLogger(args)

    def run(self, log_interval=1):
        print(self.trainer.device_info(), 'will be used.')
        timesteps = self.cnfg['timesteps']

        frames = 0
        nupdates = 0
        eplenmean = [deque(maxlen=5*log_interval)]*self.vec_num
        rewardarr = [deque(maxlen=5*log_interval)]*self.vec_num
        lr_things = []
        print("Stepping environment...")

        state = self.env.reset()

        rewardsum = numpy.zeros(self.vec_num)
        beg = numpy.zeros(self.vec_num)

        while frames < timesteps:

            action, pred_action, out = self.trainer(state)

            nstate, reward, done, _ = self.env.step(action)
            rewardsum += numpy.array(reward)

            transition = (state, pred_action, nstate, reward, done, out)
            data = self.trainer.experience(transition)

            if data:
                if self.logger.is_active():
                    print("Done.")
                lr_thing = self.trainer.train(data)
                lr_things.extend(lr_thing)
                nupdates += 1

                if nupdates % log_interval == 0 and lr_things:
                    actor_loss, critic_loss, entropy, approxkl, clipfrac, variance, debug = zip(*lr_things)
                    self.logger(numpy.array(eplenmean).reshape(-1), numpy.array(rewardarr).reshape(-1), entropy,
                                actor_loss, critic_loss, nupdates,
                                frames, approxkl, clipfrac, variance, zip(*debug))
                    lr_things = []

                if self.logger.is_active():
                    print("Stepping environment...")

            state = nstate
            frames += 1

            for i in range(self.vec_num):
                if done[i]:
                    rewardarr[i].append(rewardsum[i])
                    eplenmean[i].append(frames - beg[i])
                    rewardsum[i] = 0
                    beg[i] = frames

        print("Done.")

        self.env.close()

        if lr_things:
            actor_loss, critic_loss, entropy, approxkl, clipfrac, variance, debug = zip(*lr_things)
            self.logger(numpy.array(eplenmean).reshape(-1), numpy.array(rewardarr).reshape(-1), entropy,
                        actor_loss, critic_loss, nupdates,
                        frames, approxkl, clipfrac, variance, zip(*debug))

    def __del__(self):
        self.env.close()

        del self.env
        del self.logger
        del self.trainer
