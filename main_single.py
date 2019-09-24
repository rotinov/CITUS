import algos
import nns
import runners
from common.envs_prep import *


env_name = "BeamRiderNoFrameskip-v4"


if __name__ == '__main__':
    # env = gym.make("InvertedDoublePendulum-v2")

    envs, env, num_envs = wrap_vec_atari(env_name)

    # envs, env, num_envs = wrap_vec_gym("InvertedDoublePendulum-v2")

    print(num_envs)

    runner = runners.SingleVec(envs, env, algos.PPO, nns.CNN, workers_num=num_envs, all_cuda=False)
    runner.run()
