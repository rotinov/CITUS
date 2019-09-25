import agnes.algos
import agnes.nns
import agnes.runners
import gym
from agnes.common import logger


def test_config():
    return dict(
        timesteps=30000,
        nsteps=128,
        nminibatches=4,
        gamma=1.0,
        lam=0.95,
        noptepochs=4,
        max_grad_norm=0.5,
        learning_rate=2.5e-4,
        cliprange=0.1,
        vf_coef=0.5,
        ent_coef=.01
    )


env = gym.make('CartPole-v0')

runner = agnes.runners.Single(env, agnes.algos.PPO, agnes.nns.MLP, cnfg=test_config())
runner.run()
