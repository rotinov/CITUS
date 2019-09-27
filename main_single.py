import algos
import nns
import runners
from common.envs_prep import *


env_name = "PongNoFrameskip-v4"


if __name__ == '__main__':
<<<<<<< Updated upstream
    envs, env, num_envs = wrap_vec_atari(env_name)

    # envs, env, num_envs = wrap_vec_gym("InvertedDoublePendulum-v2")

    print(num_envs)

    runner = runners.SingleVec(envs, env, algos.PPO, nns.CNN, workers_num=num_envs, all_cuda=False)
=======
    envs, env, num_envs = wrap_vec_atari(env_name, envs_num=6)

    # envs, env, num_envs = wrap_vec_gym("InvertedDoublePendulum-v2")

    runner = agnes.runners.Single(envs, agnes.algos.PPO, agnes.nns.CNN,
                                  env_type='atari', workers_num=num_envs)
    runner.log(agnes.log, agnes.TensorboardLogger(".logs/"+str(time.time())))
>>>>>>> Stashed changes
    runner.run()
