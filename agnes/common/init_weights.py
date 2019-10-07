from torch import nn


def get_weights_init(activation='tanh'):
    if isinstance(activation, str):
        gain = nn.init.calculate_gain(activation)
    else:
        gain = activation

    recurrent_gain = nn.init.calculate_gain('tanh')

    def weights_init(m):
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            # nn.init.xavier_normal_(m.weight.data)
            nn.init.orthogonal_(m.weight.data, gain)
            if m.bias is not None:
                # nn.init.normal_(m.bias.data)
                nn.init.zeros_(m.bias.data)
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
            nn.init.orthogonal_(m.weight.data, gain)
            # nn.init.normal_(m.bias.data)
            nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data, recurrent_gain)
                else:
                    nn.init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data, recurrent_gain)
                else:
                    # nn.init.normal_(param.data)
                    nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data, recurrent_gain)
                else:
                    # nn.init.normal_(param.data)
                    nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data, recurrent_gain)
                else:
                    nn.init.zeros_(m.bias.data)
                    # nn.init.normal_(param.data)

    return weights_init
