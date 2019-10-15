import abc
from typing import Dict

from agnes.common import logger
from agnes.common.schedules import Saver
from agnes.nns.initializer import _BaseChooser
from agnes.algos.base import _BaseAlgo

import numpy
import re


class BaseRunner(abc.ABC):
    logger = logger.ListLogger()
    saver = Saver()

    def __init__(self, env, algo, nn: _BaseChooser, config: Dict):
        env, env_type, vec_num = env
        self.env = env
        self.nn_name = nn.meta

        self.cnfg, self.env_type = algo.get_config(env_type)
        if config is not None:
            self.cnfg = config

        self.timesteps = self.cnfg['timesteps']
        self.nsteps = self.cnfg['nsteps']

        self.vec_num = vec_num

    def log(self, *args):
        if self.is_trainer():
            self.logger = logger.ListLogger(args)
            self.logger.info({
                "envs_num": self.vec_num,
                "device": self.trainer.device_info(),
                "env_type": self.env_type,
                "NN type": self.nn_name,
                "algo": self.trainer.meta
            })

    def run(self, log_interval=1):
        pass

    def is_trainer(self):
        return True

    def save_every(self, filename, frames_period):
        if self.is_trainer():
            self.saver = Saver(filename, frames_period)

    def _one_log(self, lr_things, epinfobuf, nbatch, tfirststart, tstart, tnow, nupdates, stepping_to_learning=None):
        actor_loss, critic_loss, entropy, approxkl, clipfrac, variance, debug = zip(*lr_things)

        kvpairs = {
            "eplenmean": logger.safemean(numpy.asarray([epinfo['l'] for epinfo in epinfobuf]).reshape(-1)),
            "eprewmean": logger.safemean(numpy.asarray([epinfo['r'] for epinfo in epinfobuf]).reshape(-1)),
            "fps": int(nbatch / (tnow - tstart)),
            "loss/approxkl": logger.safemean(approxkl),
            "loss/clipfrac": logger.safemean(clipfrac),
            "loss/policy_entropy": logger.safemean(entropy),
            "loss/policy_loss": logger.safemean(actor_loss),
            "loss/value_loss": logger.safemean(critic_loss),
            "misc/explained_variance": logger.safemean(variance),
            "misc/nupdates": nupdates,
            "misc/serial_timesteps": self.nsteps * nupdates,
            "misc/time_elapsed": (tnow - tfirststart),
            "misc/total_timesteps": self.nsteps * nupdates
        }

        if stepping_to_learning is not None:
            kvpairs['misc/stepping_to_learning'] = stepping_to_learning

        self.logger(kvpairs, nupdates)

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
