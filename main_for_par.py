import algos
import nns
import runners
import torch
from common.envs_prep.atari_wrappers import wrap_deepmind, make_atari


# env = gym.make("InvertedDoublePendulum-v2")
# env = gym.make("CartPole-v1")

env = make_atari("EnduroNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, clip_rewards=False)

runner = runners.Distributed(env, algos.PPO, nns.CNN)
runner.run()

if runner.is_trainer():
    nnet = runner.trainer.get_nn_instance()
    torch.save(nnet, "IDP-v2.pth")
    print("wawfaf")

del runner
