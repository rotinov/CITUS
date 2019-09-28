import agnes
import torch
import time


# env = gym.make("InvertedDoublePendulum-v2")
# env = gym.make("CartPole-v1")
if __name__ == '__main__':
    envs = agnes.make_vec_env("InvertedDoublePendulum-v2", envs_num=8)

    runner = agnes.Distributed(envs, agnes.PPO, agnes.MLP)
    runner.log(agnes.log, agnes.TensorboardLogger(".logs/"+str(time.time())))
    runner.run()

    if runner.is_trainer():
        state_dict = runner.trainer.get_state_dict()
        torch.save(state_dict, "IDP-v2.pth")
        print("wawfaf")

    del runner
