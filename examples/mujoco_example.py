import agnes
import time


env_name = "CartPole-v0"  # "InvertedDoublePendulum-v2"  # "Swimmer-v2"

if __name__ == '__main__':
    envs = agnes.make_vec_env(env_name, envs_num=16)

    runner = agnes.Single(envs, agnes.PPO, agnes.MLP)
    runner.log(agnes.TensorboardLogger(".logs/"+str(time.time())), agnes.log)
    runner.run()
