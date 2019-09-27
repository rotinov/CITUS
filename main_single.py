import agnes
from agnes.common.envs_prep import *
import time


env_name = "BreakoutNoFrameskip-v4"


if __name__ == '__main__':
    envs, env, num_envs = wrap_vec_atari(env_name, envs_num=6)

    # envs, env, num_envs = wrap_vec_gym("InvertedDoublePendulum-v2")

    runner = agnes.runners.Single(envs, agnes.algos.PPO, agnes.nns.CNN,
                                  env_type='atari', workers_num=num_envs)
    runner.log(agnes.log, agnes.TensorboardLogger(".logs/"+str(time.time())))
    runner.run()
