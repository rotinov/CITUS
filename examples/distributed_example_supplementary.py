import agnes
import torch
import time


env_name3 = "Walker2d-v2"

if __name__ == '__main__':
    envs = agnes.make_vec_env(env_name3, envs_num=8)

    runner = agnes.Distributed(envs, agnes.PPO, agnes.MLP, config=config)
    runner.log(agnes.log, agnes.TensorboardLogger(".logs/"+str(time.time())))
    runner.run()

    del runner
