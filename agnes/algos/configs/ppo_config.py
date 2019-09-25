import re


def atari_config():
    return dict(
        timesteps=10e6,
        nsteps=128,
        nminibatches=4,
        gamma=0.99,
        lam=0.95,
        noptepochs=4,
        max_grad_norm=0.5,
        learning_rate=2.5e-4,
        cliprange=0.1,
        vf_coef=0.5,
        ent_coef=.01
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
        vf_coef=0.5,
        ent_coef=0.0
    )


def classic_config():
    return dict(
        timesteps=10e5,
        nsteps=1024,
        nminibatches=32,
        noptepochs=2,
        gamma=0.99,
        lam=0.95,
        learning_rate=3e-5,
        max_grad_norm=0.5,
        cliprange=0.2,
        vf_coef=0.1,
        ent_coef=0.0001
    )


def default_config():
    return dict(
        timesteps=10e6,
        nsteps=128,
        nminibatches=4,
        gamma=0.99,
        lam=0.95,
        noptepochs=4,
        max_grad_norm=0.5,
        learning_rate=2.5e-4,
        cliprange=0.1,
        vf_coef=0.5,
        ent_coef=.01
    )


def get_config(env):
    if not isinstance(env, str):
        env_type = str(env.unwrapped.__class__)
        env_type2 = re.split('[, \']', env_type)
        env_type = env_type2[2].split('.')[2]
    else:
        env_type = env

    if env_type == 'classic_control':
        cnfg = classic_config()
    elif env_type == 'mujoco':
        cnfg = mujoco_config()
    elif env_type == 'atari':
        cnfg = atari_config()
    else:
        cnfg = default_config()

    return cnfg, env_type
