try:
    from mpi4py import MPI
except ImportError:
    raise ImportError("Distributed runner is not available due to the lack of mpi4py package. "
                      "You can install it by executing "
                      "'pip install mpi4py' or '"
                      "pip install agnes[distributed]'") from None

import time
from torch import cuda
from collections import deque
import numpy

from agnes.common.schedules import Saver
from agnes.runners.base_runner import BaseRunner
from agnes.nns.initializer import _BaseChooser
from agnes.algos.base import _BaseAlgo


class Distributed(BaseRunner):
    trainer: _BaseAlgo
    worker: _BaseAlgo

    def __init__(self, env, algo, nn: _BaseChooser, config=None):
        super().__init__(env, algo, nn, config)

        self.run_times = int(numpy.ceil(self.timesteps / self.nsteps))

        self.communication = MPI.COMM_WORLD
        self.rank = self.communication.Get_rank()

        self.workers_num = (self.communication.Get_size() - 1)

        if self.rank == 0:
            print(self.env_type)
            self.trainer = algo(nn, self.env.observation_space, self.env.action_space, self.cnfg,
                                workers=self.workers_num*self.vec_num, trainer=True)
            if cuda.is_available():
                self.trainer = self.trainer.to('cuda:0')

            self.env.close()
        else:
            self.worker = algo(nn, self.env.observation_space, self.env.action_space, self.cnfg, trainer=False)

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

        tfirststart = time.perf_counter()
        nbatch = self.nsteps * self.env.num_envs * self.workers_num
        epinfobuf = deque(maxlen=100 * self.workers_num * log_interval)
        stepping_to_learning = deque(maxlen=10)

        self.communication.bcast(self.trainer.get_state_dict(), root=0)

        for nupdates in range(self.run_times):
            tstart = time.perf_counter()
            # Get rollout
            data = self.communication.gather((), root=0)[1:]
            self.communication.bcast(self.trainer.get_state_dict(), root=0)

            s_learning_time = time.perf_counter()

            batch = []
            stepping_time = []
            for item in data:
                epinfos, for_batch = item
                batch.append(for_batch)
                stepping_time.append(epinfos[1])
                epinfobuf.extend(epinfos[0])

            lr_thing = self.trainer.train(batch)
            lr_things.extend(lr_thing)

            tnow = time.perf_counter()
            lr_time = tnow - s_learning_time
            stepping_to_learning.append(numpy.mean(stepping_time) / lr_time)

            self.saver.save(self.trainer, self.nsteps*nupdates)

            if nupdates % log_interval == 0:
                self._one_log(lr_things, epinfobuf, nbatch, tfirststart, tstart, tnow, nupdates, numpy.mean(stepping_to_learning))

                lr_things = []

        MPI.Finalize()

        print("Training finished.")

    def _work(self):
        self.state = self.env.reset()
        self.done = numpy.zeros(self.env.num_envs, dtype=numpy.bool)

        epinfobuf = deque(maxlen=100)

        self.worker.load_state_dict(self.communication.bcast(None, root=0))

        for nupdates in range(self.run_times):
            s_stepping_time = time.perf_counter()
            data, epinfos = self._one_run()
            epinfobuf.extend(epinfos)

            f_stepping_time = time.perf_counter()

            self.communication.gather(((epinfobuf, f_stepping_time - s_stepping_time), data), root=0)
            self.worker.load_state_dict(self.communication.bcast(None, root=0))

        self.env.close()
        time.sleep(0.1)

        # MPI.Finalize()

    def __del__(self):
        pass
