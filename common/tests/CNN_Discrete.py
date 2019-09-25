import algos
import nns
import runners
from common.envs_prep import *
from common import logger


def test_config():
    return dict(
        timesteps=128*10,
        nsteps=128,
        nminibatches=4,
        gamma=1.0,
        lam=0.95,
        noptepochs=4,
        max_grad_norm=0.5,
        learning_rate=2.5e-4,
        cliprange=0.1,
        vf_coef=0.5,
        ent_coef=.01
    )


env_name = "PongNoFrameskip-v4"

envs, env, num_envs = wrap_vec_atari(env_name, envs_num=2)

runner = runners.Single(envs, algos.PPO, nns.CNN, workers_num=num_envs, all_cuda=False, cnfg=test_config())
runner.log(logger.TensorboardLogger(), logger.log)
runner.run()
