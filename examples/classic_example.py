import agnes
import time


env_name = "CartPole-v1"


def test_config():
    return dict(
        timesteps=15000,
        nsteps=128,
        nminibatches=4,
        gamma=1.0,
        lam=1.0,
        noptepochs=4,
        max_grad_norm=5,
        learning_rate=0.002,
        cliprange=0.1,
        vf_coef=0.5,
        ent_coef=0.01
    )


if __name__ == '__main__':
    envs = agnes.make_vec_env(env_name)

    runner = agnes.Single(envs, agnes.PPO, agnes.MLP, config=test_config())
    runner.log(agnes.log)
    runner.run()
