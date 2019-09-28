import agnes


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


if __name__ == '__main__':
    env = agnes.make_env('CartPole-v0')

    runner = agnes.Distributed(env, agnes.PPO, agnes.MLP, config=test_config())
    runner.log(agnes.log)
    runner.run()
