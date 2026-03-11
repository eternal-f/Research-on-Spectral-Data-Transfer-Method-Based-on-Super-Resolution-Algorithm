import torch
from torch import nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv1d(in_channels, out_channels, kernel_size,
                     padding=kernel_size // 2, bias=bias)


class MulBlock(nn.Module):
    def __init__(self, input_channel, output_channel,
                 kernel_size_1, kernel_size_2, kernel_size_3, kernel_size_4):
        super(MulBlock, self).__init__()

        self.conv1 = default_conv(input_channel, output_channel, kernel_size_1)
        self.conv2 = default_conv(input_channel, output_channel, kernel_size_2)
        self.conv3 = default_conv(input_channel, output_channel, kernel_size_3)
        self.conv4 = default_conv(input_channel, output_channel, kernel_size_4)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        out = torch.cat((x1, x2, x3, x4), dim=1)
        return out


class MSRN(nn.Module):             # 用了conv1d，原算法是conv2d，跑不通的话改一下然后用数据折叠
    def __init__(self, num_channels=1):
        super(MSRN, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=127, padding=(127 // 2))
        self.mul_block = MulBlock(16, 16, 1, 15, 31, 63)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=31, padding=(31 // 2))
        self.conv3 = nn.Conv1d(32, num_channels, kernel_size=31, padding=(31 // 2))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.mul_block(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
