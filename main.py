import Algos
import nns
import runners
import gym

env = gym.make("InvertedDoublePendulum-v2")
# env = gym.make("CartPole-v0")

runner = runners.Single(env, Algos.PPO, nns.MLP)
runner.run(1)
