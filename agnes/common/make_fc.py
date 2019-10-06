import torch.nn as nn


def make_fc(x, y, num_layers=1, hidden_size=64):
    if num_layers == 1:
        return nn.Sequential(nn.Linear(x, y))

    modules = [nn.Linear(x, hidden_size)]
    for i in range(1, num_layers-1):
        modules.append(nn.Tanh())
        modules.append(nn.Linear(hidden_size, hidden_size))

    modules.append(nn.Tanh())
    modules.append(nn.Linear(hidden_size, y))

    return nn.Sequential(*modules)