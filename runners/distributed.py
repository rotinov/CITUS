from mpi4py import MPI

from collections import deque
import re
import nns
import algos
from algos.configs.ppo_config import *
from common import logger
import time
from torch import cuda


class Communications:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        print(self.size)

    def trainer(self):
        return self.rank == 0

    def gather_trainer(self):
        data = ()
        stat = self.comm.gather(data, root=0)
        print(stat)
        if len(data) == 1:
            return None

        puredata = data[1:]
        return data  # zip(*puredata)

    def gather_worker(self, data):
        self.comm.gather(data, root=0)
        print(data)

    def close(self):
        del self.comm


class Distributed:
    def __init__(self, env, algo: algos.base.BaseAlgo.__class__ = algos.PPO, nn=nns.MLP):
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

        # self.communication = Communications()
        self.communication = MPI.COMM_WORLD

        if self.communication.Get_rank() == 0:
            self.trainer = algo(nn, env.observation_space, env.action_space, self.cnfg)
            if cuda.is_available():
                self.trainer = self.trainer.to('cuda:0')
        else:
            self.worker = algo(nn, env.observation_space, env.action_space, self.cnfg)

    def run(self, log_interval=1):
        if self.communication.Get_rank() == 0:
            self._train(log_interval)
        else:
            self._work(log_interval)

    def _train(self, log_interval=1):
        lr_things = []

        while True:
            data = self.communication.gather((), root=0)[1:]
            if data:
                batch = []
                for item in data:
                    batch.extend(item)

                lr_thing = self.trainer.train(batch)
                lr_things.extend(lr_thing)
                print(lr_things)

    def _work(self, log_interval=1):
        timesteps = self.cnfg['timesteps']

        frames = 0
        eplenmean = deque(maxlen=log_interval)
        rewardarr = deque(maxlen=log_interval)
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
                    self.communication.gather(data, root=0)

                    # self.worker.update(self.trainer)

                    print("Stepping environment...")

                state = nstate
                frames += 1

                if done:
                    eplenmean.append(frames - frames_beg)
                    rewardarr.append(rewardsum)
                    break
