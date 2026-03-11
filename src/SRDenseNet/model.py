import torch
import torch.nn as nn
import torch.nn.functional as function


class SingleLayer(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=31):
        super(SingleLayer, self).__init__()
        self.conv = nn.Conv1d(input_channel, output_channel, kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, x):
        out = function.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class SingleBlock(nn.Module):
    def __init__(self, input_channel, output_channel, n_denselayer):
        super(SingleBlock, self).__init__()
        self.block = self._make_dense(input_channel, output_channel, n_denselayer)

    @staticmethod
    def _make_dense(input_channel, output_channel, n_denselayer):
        layers = []
        for i in range(n_denselayer):
            layers.append(SingleLayer(input_channel, output_channel))
            input_channel += output_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out


class Net(nn.Module):
    def __init__(self, output_channel=16, n_denselayer=4, n_block=2):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(1, output_channel, 127, padding=127 // 2, bias=True)
        input_channel = output_channel
        self.denseblock = self._make_block(input_channel, output_channel, n_denselayer, n_block)
        input_channel += output_channel * n_denselayer * n_block
        self.Bottleneck = nn.Conv1d(input_channel, out_channels=32, kernel_size=1, padding=0, bias=True)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=31, padding=31 // 2,
                               bias=True)

    @staticmethod
    def _make_block(input_channel, output_channel, n_denselayer, n_block):
        blocks = []
        for i in range(n_block):
            blocks.append(SingleBlock(input_channel, output_channel, n_denselayer))
            input_channel += output_channel * n_denselayer
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = function.relu(self.conv1(x))
        out = self.denseblock(out)
        out = self.Bottleneck(out)
        hr = self.conv2(out)
        return hr
