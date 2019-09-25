from agnes import algos, nns, runners
from agnes.common.envs_prep import *
from agnes.common import logger
import time


env_name = "PongNoFrameskip-v4"


if __name__ == '__main__':
    # envs, env, num_envs = wrap_vec_atari(env_name, envs_num=2)

    envs, env, num_envs = wrap_vec_gym("InvertedDoublePendulum-v2")

    runner = runners.Single(envs, algos.PPO, nns.CNN, env_type='mujoco', workers_num=num_envs, all_cuda=False)
    runner.log(logger.TensorboardLogger(".logs/"+str(time.time())), logger.log)
    runner.run()
