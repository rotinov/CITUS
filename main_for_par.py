import algos
import nns
import runners
import gym
import torch


def check(x):
    torch.save(x.state_dict(), "IDP-v2.pth")


env = gym.make("InvertedDoublePendulum-v2")
runner = runners.Distributed(env, algos.PPO, nns.MLP)
runner.apply_onexit(check)
runner.run()

del runner
