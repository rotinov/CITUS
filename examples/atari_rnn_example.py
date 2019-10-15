import agnes
import time
import multiprocessing


env_name = "BreakoutNoFrameskip-v4"

if __name__ == '__main__':
    env = agnes.make_env(env_name, config={"frame_stack": True})
    config, _ = agnes.PPO.get_config(env[1])

    runner = agnes.Single(env, agnes.PPO, agnes.LSTMCNN, config=config)
    # runner.log(agnes.TensorboardLogger(".logs/"+str(time.time())), agnes.log)
    # runner.run()

    runner.worker.load("examples/distributed_rnn/Temporary_Breakout.pth")

    agnes.common.Visualize(runner.worker, env).run()
