from collections import deque
from agnes.runners.base_runner import BaseRunner
from agnes.algos.base import _BaseAlgo
from agnes.nns.initializer import _BaseChooser
from torch import cuda
import numpy
import time


class Single(BaseRunner):
    """"Single" runner releases learning with a single worker that is also a trainer.
    "Single" runner is compatible with vector environments(config or env_type should be specified manually).
    """

    def __init__(self, env, algo, nn: _BaseChooser, config=None):
        super().__init__(env, algo, nn, config)

        print('Env type: ', self.env_type, 'Envs num:', self.vec_num)

        self.trainer: _BaseAlgo = algo(nn, self.env.observation_space, self.env.action_space,
                                       self.cnfg, workers=self.vec_num)

        self.worker: _BaseAlgo = algo(nn, self.env.observation_space, self.env.action_space,
                                      self.cnfg, workers=self.vec_num, trainer=False)
        if cuda.is_available():
            try:
                self.trainer = self.trainer.to('cuda:0')
            except RuntimeError:
                self.trainer = self.trainer.to('cpu')

    def run(self, log_interval=1):
        print(self.trainer.device_info(), 'will be used.')
        nbatch = self.nsteps * self.env.num_envs

        lr_things = []

        self.state = self.env.reset()
        self.done = numpy.zeros(self.env.num_envs, dtype=numpy.bool)

        run_times = int(self.timesteps // self.nsteps)
        epinfobuf = deque(maxlen=100 * log_interval)
        tfirststart = time.perf_counter()

        for nupdates in range(1, run_times+1):
            self.logger.stepping_environment()
            tstart = time.perf_counter()

            data, epinfos = self._one_run()

            self.logger.done()
            lr_thing = self.trainer.train(data)
            lr_things.extend(lr_thing)
            epinfobuf.extend(epinfos)
            
            self.worker.update(self.trainer)

            tnow = time.perf_counter()

            if nupdates % log_interval == 0:
                self._one_log(lr_things, epinfobuf, nbatch, tfirststart, tstart, tnow, nupdates)

                lr_things = []

        self.env.close()

    def __del__(self):
        self.env.close()

        del self.env
        del self.trainer
