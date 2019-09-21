def default_config():
    return dict(
        timesteps=5e5,
        nsteps=128,
        nminibatches=2,
        gamma=0.99,
        lam=0.95,
        trunc=10,
        learning_rate=3e-4,
        cliprange=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        lr_decay=1e-6,
        simple_nn=False
    )


def mujoco_config():
    return dict(
        timesteps=1e6,
        nsteps=2048,
        nminibatches=32,
        gamma=0.99,
        lam=0.95,
        noptepochs=10,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        cliprange=0.2,
        vf_coef=1.0,
        ent_coef=0.0,
        simple_nn=True
    )


def classic_config():
    return dict(
        timesteps=10e5,
        nsteps=2048,
        nminibatches=32,
        noptepochs=10,
        gamma=1,
        lam=0.95,
        learning_rate=3e-4,
        max_grad_norm=0.5,
        cliprange=0.2,
        vf_coef=1,
        ent_coef=0.0,
        simple_nn=True
    )


def box2d_config():
    return dict(
        timesteps=10e5,
        nsteps=256,
        nminibatches=10,
        gamma=1,
        lam=0.95,
        trunc=10,
        learning_rate=1e-3,
        cliprange=0.2,
        vf_coef=1,
        ent_coef=0.1,
        simple_nn=False
    )


def test_config():
    return dict(
        timesteps=1.5e4,
        nsteps=50,
        nminibatches=1,
        gamma=0.5,
        lam=0.95,
        trunc=3,
        learning_rate=2e-4,
        cliprange=0.1,
        vf_coef=1,
        ent_coef=0.01,
        lr_decay=1e-7,
        simple_nn=True
    )
