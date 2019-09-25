from collections import deque
import agnes.nns
import agnes.algos
from agnes.common import logger
from torch import cuda
import numpy


class Single:
    def __init__(self, env,
                 algo: agnes.algos.base.BaseAlgo.__class__ = agnes.algos.PPO,
                 nn=agnes.nns.MLP, env_type=None, cnfg=None, workers_num=1, all_cuda=False):
        self.env = env
        if env_type is None:
            env_type = env
        self.cnfg, self.env_type = algo.get_config(env_type)

        if cnfg is not None:
            self.cnfg = cnfg
        print(self.env_type)
        self.workers_num = workers_num

        self.worker = algo(nn, env.observation_space, env.action_space, self.cnfg, trainer=False, workers=1)

        self.trainer = algo(nn, env.observation_space, env.action_space, self.cnfg)
        if cuda.is_available():
            self.trainer = self.trainer.to('cuda:0')
            if all_cuda:
                self.trainer.to('cuda:0')

        self.worker.update(self.trainer)

        self.logger = logger.ListLogger()

    def log(self, *args):
        self.logger = logger.ListLogger(args)

    def run(self, log_interval=1):
        timesteps = self.cnfg['timesteps']

        frames = 0
        nupdates = 0
        eplenmean = [deque(maxlen=5*log_interval)]*self.workers_num
        rewardarr = [deque(maxlen=5*log_interval)]*self.workers_num
        lr_things = []
        print("Stepping environment...")

        state = self.env.reset()

        rewardsum = numpy.zeros(self.workers_num)
        beg = numpy.zeros(self.workers_num)

        while frames < timesteps:

            action, pred_action, out = self.worker(state)

            nstate, reward, done, _ = self.env.step(action)
            rewardsum += numpy.array(reward)

            transition = (state, pred_action, nstate, reward, done, out)
            data = self.worker.experience(transition)

            if data:
                if self.logger.is_active():
                    print("Done.")
                lr_thing = self.trainer.train(data)
                lr_things.extend(lr_thing)
                nupdates += 1

                self.worker.update(self.trainer)

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

            for i in range(self.workers_num):
                if (isinstance(done, bool) and done) or (not isinstance(done, bool) and done[i]):
                    rewardarr[i].append(rewardsum[i])
                    eplenmean[i].append(frames - beg[i])
                    rewardsum[i] = 0
                    beg[i] = frames
                    if isinstance(done, bool):
                        self.env.reset()

        print("Done.")

        self.env.close()

        if lr_things:
            actor_loss, critic_loss, entropy, approxkl, clipfrac, variance, debug = zip(*lr_things)
            self.logger(numpy.array(eplenmean).reshape(-1), numpy.array(rewardarr).reshape(-1), entropy,
                        actor_loss, critic_loss, nupdates,
                        frames, approxkl, clipfrac, variance, zip(*debug))

        del self.env
        del self.logger
        del self.worker
