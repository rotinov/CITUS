import agnes
import time


env_name1 = "BreakoutNoFrameskip-v4"
env_name2 = "Ant-v2"
env_name3 = "InvertedDoublePendulum-v2"


if __name__ == '__main__':
    envs = agnes.make_vec_env(env_name3)

    runner = agnes.Single(envs, agnes.PPO, agnes.MLP)
    runner.log(agnes.log, agnes.TensorboardLogger(".logs/"+str(time.time())))
    runner.run()
