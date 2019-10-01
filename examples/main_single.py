import agnes
import time


env_name1 = "BreakoutNoFrameskip-v4"
env_name2 = "InvertedDoublePendulum-v2"
env_name3 = "Swimmer-v2"


if __name__ == '__main__':
    envs = agnes.make_vec_env(env_name1, envs_num=16)

    runner = agnes.Single(envs, agnes.PPO, agnes.CNN)
    runner.log(agnes.TensorboardLogger(".logs/"+str(time.time())), agnes.log)
    runner.run()
