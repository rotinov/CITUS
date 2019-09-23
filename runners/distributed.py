from mpi4py import MPI

import re
import nns
import algos
from algos.configs.ppo_config import *
from common import logger
import time
from torch import cuda


class Distributed:
    onexit = None

    def __init__(self, env, algo: algos.base.BaseAlgo.__class__ = algos.PPO, nn=nns.MLP):
        self.env = env
        env_type = str(env.unwrapped.__class__)
        env_type2 = re.split('[, \']', env_type)
        self.env_type = env_type2[2].split('.')[2]

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

        self.workers_num = (self.communication.Get_size() - 1)

        if self.communication.Get_rank() == 0:
            print(self.env_type)
            self.trainer = algo(nn, env.observation_space, env.action_space, self.cnfg, workers=self.workers_num,
                                trainer=True)
            if cuda.is_available():
                self.trainer = self.trainer.to('cuda:0')
        else:
            self.worker = algo(nn, env.observation_space, env.action_space, self.cnfg, trainer=False)

    def run(self, log_interval=1):
        if self.communication.Get_rank() == 0:
            self._train(log_interval)
        else:
            self._work()

        del self

    def apply_onexit(self, func):
        self.onexit = func

    def _train(self, log_interval=1):
        lr_things = []
        nupdates = 0
        self.logger = logger.TensorboardLogger(".logs/"+str(time.time()))
        print("Stepping environment...")

        finish = False

        while True:
            # Get rollout
            data = self.communication.gather((), root=0)[1:]

            if data:
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

                print("Stepping environment...")

        MPI.Finalize()

        print("Training finished.")

        if self.onexit is not None:
            self.onexit(self.trainer.get_nn_instance())
            print("Trainer: External job done.")

        print("Trainer finished.")

    def _work(self):
        timesteps = self.cnfg['timesteps']

        frames = 0
        eplenmean = []
        rewardarr = []
        finish = False
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

                if frames >= timesteps:
                    finish = True
                    break

                if data:
                    self.communication.gather(((eplenmean, rewardarr, frames), data), root=0)

                    self.worker.load_state_dict(self.communication.bcast(None, root=0))

                    eplenmean = []
                    rewardarr = []

                state = nstate
                frames += 1

                if done:
                    eplenmean.append(frames - frames_beg)
                    rewardarr.append(rewardsum)
                    break
            if finish:
                break

        if self.communication.Get_size() == self.workers_num + 1:
            print("Worker", self.communication.Get_rank(), "finished.")
            self.communication.gather(True, root=0)

        MPI.Finalize()
