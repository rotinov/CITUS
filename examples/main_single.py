import agnes
import time


env_name1 = "BreakoutNoFrameskip-v4"
env_name2 = "InvertedDoublePendulum-v2"
env_name3 = "Swimmer-v2"


if __name__ == '__main__':
    envs = agnes.make_vec_env(env_name3, envs_num=32)

    runner = agnes.Single(envs, agnes.PPO, agnes.MLP)
    runner.log(agnes.TensorboardLogger(".logs/"+str(time.time())), agnes.log)
    runner.run()
