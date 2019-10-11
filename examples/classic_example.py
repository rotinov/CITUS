import agnes
import time


def test_config():
    return dict(
        timesteps=30000,
        nsteps=128,
        nminibatches=4,
        gamma=1.0,
        lam=1.0,
        noptepochs=4,
        max_grad_norm=40.0,
        learning_rate=1e-3,
        cliprange=lambda x: 0.3*x,
        vf_coef=1.0,
        ent_coef=.005
    )


env_name = "CartPole-v0"


if __name__ == '__main__':
    envs = agnes.make_vec_env(env_name, envs_num=32)

    runner = agnes.Single(envs, agnes.PPO, agnes.RNN, config=test_config())
    runner.log(agnes.log)
    runner.run()
