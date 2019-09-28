from mpi4py import MPI

import agnes
from agnes.common import logger
import time
from torch import cuda
from collections import deque
import numpy


class Distributed:
    logger = logger.ListLogger()

    def __init__(self, env,
                 algo: agnes.algos.base.BaseAlgo.__class__ = agnes.PPO,
                 nn=agnes.MLP, config=None):
        env, env_type, vec_num = env
        self.env = env

        self.cnfg, self.env_type = algo.get_config(env_type)
        if config is not None:
            self.cnfg = config

        # self.communication = Communications()
        self.communication = MPI.COMM_WORLD

        self.workers_num = (self.communication.Get_size() - 1)
        self.vec_num = vec_num

        if self.communication.Get_rank() == 0:
            print(self.env_type)
            self.trainer = algo(nn, env.observation_space, env.action_space, self.cnfg,
                                workers=self.workers_num*self.vec_num, trainer=True)
            if cuda.is_available():
                self.trainer = self.trainer.to('cuda:0')
        else:
            self.worker = algo(nn, env.observation_space, env.action_space, self.cnfg, trainer=False)

    def log(self, *args):
        if self.is_trainer():
            self.logger = logger.ListLogger(args)

    def run(self, log_interval=1):
        if self.communication.Get_rank() == 0:
            self._train(log_interval)
        else:
            self._work()

    def is_trainer(self):
        return self.communication.Get_rank() == 0

    def _train(self, log_interval=1):
        print(self.trainer.device_info(), 'will be used.')
        lr_things = []
        nupdates = 0
        print("Stepping environment...")

        finish = False

        while True:
            # Get rollout
            data = self.communication.gather((), root=0)[1:]

            if data:
                if self.logger.is_active():
                    print("Done.")
                batch = []
                info_arr = []
                for item in data:
                    if isinstance(item, bool):
                        finish = True
                        break
                    info, for_batch = item
                    batch.extend(for_batch)
                    info_arr.append(info)

                if finish:
                    break

                eplenmean, rewardarr, frames = zip(*info_arr)

                len_arr = []
                rew_arr = []
                for l_item, r_item in zip(eplenmean, rewardarr):
                    len_arr.extend(l_item)
                    rew_arr.extend(r_item)

                lr_thing = self.trainer.train(batch)
                lr_things.extend(lr_thing)
                nupdates += 1

                self.communication.bcast(self.trainer.get_state_dict(), root=0)

                if nupdates % log_interval == 0:
                    actor_loss, critic_loss, entropy, approxkl, clipfrac, variance, debug = zip(*lr_things)

                    self.logger(len_arr, rew_arr, entropy,
                                actor_loss, critic_loss, nupdates,
                                logger.safemean(frames), approxkl, clipfrac, variance, zip(*debug))

                    lr_things = []

                if self.logger.is_active():
                    print("Stepping environment...")

        MPI.Finalize()

        print("Training finished.")

    def _work(self):
        timesteps = self.cnfg['timesteps']

        frames = 0
        eplenmean = [deque(maxlen=5)]*self.vec_num
        rewardarr = [deque(maxlen=5)]*self.vec_num
        rewardsum = numpy.zeros(self.vec_num)
        beg = numpy.zeros(self.vec_num)
        state = self.env.reset()

        while frames < timesteps:
            frames += 1

            action, pred_action, out = self.worker(state)

            nstate, reward, done, _ = self.env.step(action)
            rewardsum += numpy.array(reward)

            transition = (state, pred_action, nstate, reward, done, out)
            data = self.worker.experience(transition)

            if frames >= timesteps:
                break

            if data:
                self.communication.gather(((eplenmean, rewardarr, frames), data), root=0)

                self.worker.load_state_dict(self.communication.bcast(None, root=0))

            state = nstate
            frames += 1

            for i in range(self.vec_num):
                if done[i]:
                    rewardarr[i].append(rewardsum[i])
                    eplenmean[i].append(frames - beg[i])
                    rewardsum[i] = 0
                    beg[i] = frames

        if self.communication.Get_size() == self.workers_num + 1:
            print("Worker", self.communication.Get_rank(), "finished.")
            self.communication.gather(True, root=0)

        MPI.Finalize()

    def __del__(self):
        self.env.close()

        del self.env
        if self.logger.is_active():
            del self.logger
