import agnes
import time


env_name = "BreakoutNoFrameskip-v4"

if __name__ == '__main__':
    envs = agnes.make_vec_env(env_name, envs_num=4, config={"frame_stack": False})
    config, _ = agnes.PPO.get_config(envs[1])
    config['vf_coef'] = 0.5  # Increase vf_coef
    # config['learning_rate'] = 0.00005
    # print(config)

    runner = agnes.Single(envs, agnes.PPO, agnes.RNNCNN, config=config)
    runner.log(agnes.TensorboardLogger(".logs/"+str(time.time())), agnes.log)
    runner.run()
