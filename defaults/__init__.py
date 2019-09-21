from torch import nn


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
        vf_coef=0.5,
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
        vf_coef=0.5,
        ent_coef=0.01,
        lr_decay=1e-7,
        simple_nn=True
    )


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, nn.init.calculate_gain('tanh'))
        # nn.init.normal_(m.bias.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)