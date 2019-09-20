import PPO
import nns
import runners
import gym

env = gym.make("Swimmer-v2")
# env = gym.make("CartPole-v0")

runner = runners.Single(env, PPO.PPO, nns.MLP)
runner.run(5000)
