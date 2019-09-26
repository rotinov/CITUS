import agnes
import gym
from agnes.common.envs_prep import wrap_vec_gym


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


def test_single():
    env = gym.make('Pendulum-v0')

    runner = agnes.runners.Single(env, agnes.algos.PPO, agnes.nns.MLP, cnfg=test_config())
    runner.run()


def test_vec():
    envs, env, workers_num = wrap_vec_gym('Pendulum-v0')

    runner = agnes.runners.Single(envs, agnes.algos.PPO, agnes.nns.MLP, cnfg=test_config(), workers_num=workers_num)
    runner.log(agnes.log)
    runner.run()
