import agnes
import time
import multiprocessing


env_name = "BreakoutNoFrameskip-v4"

if __name__ == '__main__':
    envs = agnes.make_vec_env(env_name, envs_num=multiprocessing.cpu_count() // 2, config={"frame_stack": True})
    config, _ = agnes.PPO.get_config(envs[1])
    # config['vf_coef'] = 1.0  # Increased vf_coef
    # config['learning_rate'] = 0.00005
    # print(config)

    runner = agnes.Single(envs, agnes.PPO, agnes.LSTMCNN, config=config)
    runner.log(agnes.TensorboardLogger(".logs/"+str(time.time())), agnes.log)
    runner.run()

    env = agnes.make_env(env_name, config={"frame_stack": True})

    agnes.common.Visualize(runner.worker, env).run()
