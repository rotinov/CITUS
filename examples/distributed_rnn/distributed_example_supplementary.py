import agnes
import multiprocessing


env_name = "BreakoutNoFrameskip-v4"

if __name__ == '__main__':
    envs = agnes.make_vec_env(env_name, envs_num=multiprocessing.cpu_count() // 2, config={"frame_stack": True})

    runner = agnes.Distributed(envs, agnes.PPO, agnes.LSTMCNN)
    runner.save_every("Temporary_Breakout.pth", 1e5)
    runner.log(agnes.log, agnes.TensorboardLogger())
    runner.run()

    if runner.is_trainer():
        runner.trainer.save("Breakout.pth")

    del runner
