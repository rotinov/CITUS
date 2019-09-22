import algos
import nns
import runners
import gym

env = gym.make("InvertedDoublePendulum-v2")

runner = runners.Single(env, algos.PPO, nns.MLP)
runner.run()
