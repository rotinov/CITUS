import agnes
import torch
import time


env_name3 = "Walker2d-v2"
if __name__ == '__main__':
    envs = agnes.make_vec_env(env_name3)

    runner = agnes.Distributed(envs, agnes.PPO, agnes.MLP)
    runner.log(agnes.log, agnes.TensorboardLogger(".logs/"+str(time.time())))
    runner.run()

    if runner.is_trainer():
        state_dict = runner.trainer.get_state_dict()
        torch.save(state_dict, "IDP-v2.pth")
        print("wawfaf")

    del runner
