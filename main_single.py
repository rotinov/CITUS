import algos
import nns
import runners
from common.envs_prep import *


env_name = "PongNoFrameskip-v4"


if __name__ == '__main__':
    # envs, env, num_envs = wrap_vec_atari(env_name)

    envs, env, num_envs = wrap_vec_gym("InvertedDoublePendulum-v2")

    print(num_envs)

    runner = runners.SingleVec(envs, env, algos.PPO, nns.MLP, workers_num=num_envs, all_cuda=False)
    runner.run()
