import torch.nn as nn
import torch
import numpy


def make_fc(x, y, num_layers=1, hidden_size=64, activation=nn.Tanh, activate_last=False):
    if num_layers == 1:
        modules = [nn.Linear(x, y)]
    else:
        modules = [nn.Linear(x, hidden_size)]
        for i in range(1, num_layers-1):
            modules.append(activation())
            modules.append(nn.Linear(hidden_size, hidden_size))

        modules.append(activation())
        modules.append(nn.Linear(hidden_size, y))

    if activate_last:
        modules.append(activation())

    return nn.Sequential(*modules)


class ImagePreprocess(nn.Module):
    def __init__(self, normalize=True, swap_axis=True):
        super(ImagePreprocess, self).__init__()

        self.normalize = normalize
        self.swap_axis = swap_axis

    def forward(self, x):
        if self.normalize:
            x = x / 255.

        if self.swap_axis:
            x = x.permute(0, 3, 1, 2)

        return x


class CnnBody(nn.Module):
    def __init__(self, input_shape=(84, 84, 4)):
        super(CnnBody, self).__init__()

        test_input = torch.rand(input_shape).unsqueeze(0)

        self.conv = nn.Sequential(ImagePreprocess(),
                                  nn.Conv2d(in_channels=input_shape[-1],
                                            out_channels=32,
                                            kernel_size=8,
                                            stride=4,
                                            padding=0),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=32,
                                            out_channels=64,
                                            kernel_size=4,
                                            stride=2,
                                            padding=0),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=64,
                                            out_channels=64,
                                            kernel_size=3,
                                            stride=1,
                                            padding=0),
                                  nn.ReLU())

        test_output = self.conv(test_input)
        self.test_output_shape = tuple(test_output.shape)

    @property
    def output_size(self):
        return self.test_output_shape

    def forward(self, x):
        return self.conv(x)


class CnnSmallBody(nn.Module):
    def __init__(self, input_shape=(84, 84, 4)):
        super(CnnSmallBody, self).__init__()

        test_input = torch.rand(input_shape).unsqueeze(0)

        self.conv = nn.Sequential(ImagePreprocess(),
                                  nn.Conv2d(in_channels=input_shape[-1],
                                            out_channels=32,
                                            kernel_size=3,
                                            stride=2,
                                            padding=0),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=32,
                                            out_channels=32,
                                            kernel_size=3,
                                            stride=2,
                                            padding=0),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=32,
                                            out_channels=32,
                                            kernel_size=3,
                                            stride=2,
                                            padding=0),
                                  nn.ReLU())

        test_output = self.conv(test_input)
        self.test_output_shape = tuple(test_output.shape)

    @property
    def output_size(self):
        return self.test_output_shape

    def forward(self, x):
        return self.conv(x)


class CnnImpalaShallowBody(nn.Module):
    def __init__(self, input_shape=(84, 84, 4)):
        super(CnnImpalaShallowBody, self).__init__()

        test_input = torch.rand(input_shape).unsqueeze(0)

        self.conv = nn.Sequential(ImagePreprocess(),
                                  nn.Conv2d(in_channels=input_shape[-1],
                                            out_channels=16,
                                            kernel_size=8,
                                            stride=4,
                                            padding=0),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=16,
                                            out_channels=32,
                                            kernel_size=4,
                                            stride=2,
                                            padding=0),
                                  nn.ReLU())

        test_output = self.conv(test_input)
        self.test_output_shape = tuple(test_output.shape)

    @property
    def output_size(self):
        return self.test_output_shape

    def forward(self, x):
        return self.conv(x)


class CnnAttentionBody(nn.Module):
    def __init__(self, input_shape=(84, 84, 4)):
        super(CnnAttentionBody, self).__init__()

        test_input = torch.rand(input_shape).unsqueeze(0)

        self.conv1 = nn.Sequential(ImagePreprocess(),
                                   nn.Conv2d(in_channels=input_shape[-1],
                                             out_channels=16,
                                             kernel_size=8,
                                             stride=4,
                                             padding=0),
                                   nn.ReLU())

        self.queries_nn = nn.Conv2d(in_channels=16,
                                    out_channels=32,
                                    kernel_size=4,
                                    stride=2,
                                    padding=0)

        self.keys_nn = nn.Conv2d(in_channels=16,
                                 out_channels=32,
                                 kernel_size=4,
                                 stride=2,
                                 padding=0)

        self.values_nn = nn.Conv2d(in_channels=16,
                                   out_channels=32,
                                   kernel_size=4,
                                   stride=2,
                                   padding=0)

        self.sub_nn = nn.Conv2d(in_channels=16,
                                out_channels=32,
                                kernel_size=4,
                                stride=2,
                                padding=0)

        self.softmax = nn.Softmax(dim=-3)

        test_output = self.forward(test_input)
        self.test_output_shape = tuple(test_output.shape)

    @property
    def output_size(self):
        return self.test_output_shape

    def forward(self, x):
        conv1_out = self.conv1(x)

        return self.att_forward(conv1_out) + self.sub_nn(conv1_out)

    def att_forward(self, x):
        queries = self.queries_nn(x)
        keys = self.keys_nn(x)
        values = self.values_nn(x)
        return self.softmax(queries * keys.transpose(-1, -2)) * values * 2


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(channels, channels, 3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(channels, channels, 3, stride=1, padding=1))

    def forward(self, inp):
        return self.block(inp) + inp


def make_one_block(channels_in, channels_out):
    return nn.Sequential(nn.Conv2d(in_channels=channels_in,
                                   out_channels=channels_out,
                                   kernel_size=3,
                                   stride=1),
                         nn.MaxPool2d(3, 2),
                         ResidualBlock(channels_out),
                         ResidualBlock(channels_out))


class CnnImpalaBody(nn.Module):
    def __init__(self, input_shape=(84, 84, 4)):
        super(CnnImpalaBody, self).__init__()

        test_input = torch.rand(input_shape).unsqueeze(0)

        self.conv = nn.Sequential(ImagePreprocess(),
                                  make_one_block(input_shape[-1], 16),
                                  make_one_block(16, 32),
                                  make_one_block(32, 32),
                                  nn.ReLU())

        test_output = self.conv(test_input)
        self.test_output_shape = tuple(test_output.shape)

    @property
    def output_size(self):
        return self.test_output_shape

    def forward(self, x):
        return self.conv(x)


class CnnHead(nn.Module):
    def __init__(self, cnn_out_size, output, num_layers=2, hidden_size=512, activate_last=False):
        super(CnnHead, self).__init__()

        self.cnn_out_size = int(numpy.prod(cnn_out_size))

        self.fc = make_fc(self.cnn_out_size, output, num_layers=num_layers, hidden_size=hidden_size,
                          activate_last=activate_last, activation=nn.ReLU)

    def forward(self, cv):
        return self.fc(cv.view(-1, self.cnn_out_size))


class Cnn(nn.Module):
    def __init__(self, input_shape, output, num_layers=2, hidden_size=512, activate_last=False,
                 body=CnnBody, head=CnnHead):
        super(Cnn, self).__init__()

        self.conv = body(input_shape=input_shape)

        self.head = head(self.conv.output_size, output, num_layers=num_layers,
                         hidden_size=hidden_size, activate_last=activate_last)

    def forward(self, x):
        return self.head(self.conv(x))
