import agnes
import torch


# env = gym.make("InvertedDoublePendulum-v2")
# env = gym.make("CartPole-v1")
if __name__ == '__main__':
    envs = agnes.make_vec_env("BreakoutNoFrameskip-v4", envs_num=6)

    runner = agnes.Distributed(envs, agnes.PPO, agnes.CNN)
    runner.log(agnes.log)
    runner.run()

    if runner.is_trainer():
        nnet = runner.trainer.get_nn_instance()
        torch.save(nnet, "IDP-v2.pth")
        print("wawfaf")

    del runner
