import torch.nn as nn
import torch
import numpy


def make_fc(x, y, num_layers=1, hidden_size=64, activation=nn.Tanh):
    if num_layers == 1:
        return nn.Sequential(nn.Linear(x, y))

    modules = [nn.Linear(x, hidden_size)]
    for i in range(1, num_layers-1):
        modules.append(activation())
        modules.append(nn.Linear(hidden_size, hidden_size))

    modules.append(activation())
    modules.append(nn.Linear(hidden_size, y))

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
        # (4, 84, 84)
        cv = self.conv(x)
        # (64, 7, 7)

        return cv


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
        # (4, 84, 84)
        cv = self.conv(x)
        # (64, 7, 7)

        return cv


class CnnHead(nn.Module):
    def __init__(self, cnn_out_size, output, num_layers=2, hidden_size=512):
        super(CnnHead, self).__init__()

        self.cnn_out_size = int(numpy.prod(cnn_out_size))

        self.fc = make_fc(self.cnn_out_size, output, num_layers=num_layers, hidden_size=hidden_size, activation=nn.ReLU)

    def forward(self, cv):
        cv_f = cv.view(-1, self.cnn_out_size)

        return self.fc(cv_f)


class Cnn(nn.Module):
    def __init__(self, input_shape, output, num_layers=2, hidden_size=512, body=CnnBody, head=CnnHead):
        super(Cnn, self).__init__()

        self.conv = body(input_shape=input_shape)

        self.head = head(self.conv.output_size, output, num_layers=num_layers, hidden_size=hidden_size)

    def forward(self, x):
        # (4, 84, 84) -> (64, 7, 7)
        cv = self.conv(x)

        return self.head(cv)