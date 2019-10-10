import agnes
import multiprocessing
import torch
import time


env_name = "BreakoutNoFrameskip-v4"

if __name__ == '__main__':
    envs = agnes.make_vec_env(env_name, envs_num=multiprocessing.cpu_count() // 2, config={"frame_stack": True})

    runner = agnes.Distributed(envs, agnes.PPO, agnes.LSTMCNN)
    runner.log(agnes.log, agnes.TensorboardLogger(".logs/"+str(time.time())))
    runner.run()

    del runner
