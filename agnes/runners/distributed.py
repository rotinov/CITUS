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

        self.timesteps = self.cnfg['timesteps']
        self.nsteps = self.cnfg['nsteps']
        self.run_times = int(numpy.ceil(self.timesteps / self.nsteps))

        self.communication = MPI.COMM_WORLD
        self.rank = self.communication.Get_rank()

        self.workers_num = (self.communication.Get_size() - 1)
        self.vec_num = vec_num

        if self.rank == 0:
            print(self.env_type)
            self.trainer = algo(nn, env.observation_space, env.action_space, self.cnfg,
                                workers=self.workers_num*self.vec_num, trainer=True)
            if cuda.is_available():
                self.trainer = self.trainer.to('cuda:0')

            self.env.close()
        else:
            self.worker = algo(nn, env.observation_space, env.action_space, self.cnfg, trainer=False)

    def log(self, *args):
        if self.is_trainer():
            self.logger = logger.ListLogger(args)

    def run(self, log_interval=1):
        if self.rank == 0:
            self._train(log_interval)
        else:
            self._work()

    def is_trainer(self):
        return self.rank == 0

    def _train(self, log_interval=1):
        print(self.trainer.device_info(), 'will be used.')
        lr_things = []

        b_time = time.time()

        self.logger.stepping_environment()

        for nupdates in range(self.run_times):
            # Get rollout
            data = self.communication.gather((), root=0)[1:]

            self.logger.done()

            batch = []
            info_arr = []
            for item in data:
                info, for_batch = item
                batch.extend(for_batch)
                info_arr.append(info)

            eplenmean, rewardarr, frames = zip(*info_arr)

            lr_thing = self.trainer.train(batch)
            lr_things.extend(lr_thing)

            self.communication.bcast(self.trainer.get_state_dict(), root=0)

            if nupdates % log_interval == 0:
                actor_loss, critic_loss, entropy, approxkl, clipfrac, variance, debug = zip(*lr_things)

                time_now = time.time()
                kvpairs = {
                    "eplenmean": logger.safemean(numpy.asarray(eplenmean).reshape(-1)),
                    "eprewmean": logger.safemean(numpy.asarray(rewardarr).reshape(-1)),
                    "fps": logger.safemean(frames) / max(1e-8, float(time_now - b_time)),
                    "loss/approxkl": logger.safemean(approxkl),
                    "loss/clipfrac": logger.safemean(clipfrac),
                    "loss/policy_entropy": logger.safemean(entropy),
                    "loss/policy_loss": logger.safemean(actor_loss),
                    "loss/value_loss": logger.safemean(critic_loss),
                    "misc/explained_variance": logger.safemean(variance),
                    "misc/nupdates": nupdates,
                    "misc/serial_timesteps": logger.safemean(frames),
                    "misc/time_elapsed": int(time_now - b_time),
                    "misc/total_timesteps": logger.safemean(frames)
                }

                self.logger(kvpairs, nupdates)

                lr_things = []

            self.logger.stepping_environment()

        MPI.Finalize()

        print("Training finished.")

    def _work(self):
        self.state = self.env.reset()
        self.done = numpy.zeros(self.env.num_envs, dtype=numpy.bool)

        epinfobuf = deque(maxlen=100)

        for nupdates in range(self.run_times):
            data, epinfos = self._one_run()
            epinfobuf.extend(epinfos)
            self.communication.gather(((numpy.asarray([epinfo['l'] for epinfo in epinfobuf]).reshape(-1),
                                        numpy.asarray([epinfo['r'] for epinfo in epinfobuf]).reshape(-1),
                                        self.nsteps*nupdates), data), root=0)

            self.worker.load_state_dict(self.communication.bcast(None, root=0))

        self.env.close()
        time.sleep(0.1)

        # MPI.Finalize()

    def _one_run(self):
        data = None
        epinfos = []
        for step in range(self.nsteps):
            action, pred_action, out = self.worker(self.state, self.done)
            nstate, reward, done, infos = self.env.step(action)
            self.done = done
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

            transition = (self.state, pred_action, nstate, reward, done, out)
            data = self.worker.experience(transition)

            self.state = nstate

        return data, epinfos

    def __del__(self):
        pass
