import algos
import nns
import runners
import gym
from common.atari_wrappers import wrap_deepmind, make_atari


# env = gym.make("InvertedDoublePendulum-v2")
env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, clip_rewards=False)


runner = runners.Single(env, algos.PPO, nns.CNN)
runner.run()
