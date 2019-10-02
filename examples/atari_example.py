import agnes
import time


env_name = "BreakoutNoFrameskip-v4"

if __name__ == '__main__':
    envs = agnes.make_vec_env(env_name, envs_num=16)

    runner = agnes.Single(envs, agnes.PPO, agnes.CNN)
    runner.log(agnes.TensorboardLogger(".logs/"+str(time.time())), agnes.log)
    runner.run()
