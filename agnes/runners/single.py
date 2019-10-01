from collections import deque
import agnes
from agnes.common import logger
from torch import cuda
import numpy
import time


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
            try:
                self.trainer = self.trainer.to('cuda:0')
            except RuntimeError:
                self.trainer = self.trainer.to('cpu')

        self.logger = logger.ListLogger()

    def log(self, *args):
        self.logger = logger.ListLogger(args)

    def run(self, log_interval=1):
        print(self.trainer.device_info(), 'will be used.')
        timesteps = self.cnfg['timesteps']
        self.nsteps = self.cnfg['nsteps']
        b_time = time.time()

        lr_things = []

        self.state = self.env.reset()

        run_times = int(numpy.ceil(timesteps / self.nsteps))
        epinfobuf = deque(maxlen=100)

        for nupdates in range(run_times):
            self.logger.stepping_environment()

            data, epinfos = self._one_run()

            self.logger.done()
            lr_thing = self.trainer.train(data)
            lr_things.extend(lr_thing)
            epinfobuf.extend(epinfos)

            if nupdates % log_interval == 0 or (lr_things and nupdates == run_times - 1):
                actor_loss, critic_loss, entropy, approxkl, clipfrac, variance, debug = zip(*lr_things)

                time_now = time.time()
                kvpairs = {
                    "eplenmean": logger.safemean(numpy.asarray([epinfo['l'] for epinfo in epinfobuf]).reshape(-1)),
                    "eprewmean": logger.safemean(numpy.asarray([epinfo['r'] for epinfo in epinfobuf]).reshape(-1)),
                    "fps": self.nsteps*nupdates / max(1e-8, float(time_now - b_time)),
                    "loss/approxkl": logger.safemean(approxkl),
                    "loss/clipfrac": logger.safemean(clipfrac),
                    "loss/policy_entropy": logger.safemean(entropy),
                    "loss/policy_loss": logger.safemean(actor_loss),
                    "loss/value_loss": logger.safemean(critic_loss),
                    "misc/explained_variance": logger.safemean(variance),
                    "misc/nupdates": nupdates,
                    "misc/serial_timesteps": self.nsteps*nupdates,
                    "misc/time_elapsed": int(time_now - b_time),
                    "misc/total_timesteps": self.nsteps*nupdates
                }

                self.logger(kvpairs, nupdates)
                lr_things = []

        self.env.close()

    def _one_run(self):
        data = None
        epinfos = []
        for step in range(self.nsteps):
            action, pred_action, out = self.trainer(self.state)
            nstate, reward, done, infos = self.env.step(action)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

            transition = (self.state, pred_action, nstate, reward, done, out)
            data = self.trainer.experience(transition)

            self.state = nstate

        return data, epinfos

    def __del__(self):
        self.env.close()

        del self.env
        del self.trainer
