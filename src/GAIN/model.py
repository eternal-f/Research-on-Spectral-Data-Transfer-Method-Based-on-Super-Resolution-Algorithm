import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=31, padding=31 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x


class GuidedAttention(nn.Module):
    def __init__(self):
        super(GuidedAttention, self).__init__()
        head = [nn.Conv1d(in_channels=1, out_channels=64, kernel_size=127, padding=127 // 2, bias=False)]
        body = [SpatialAttention()]
        tail = [nn.Conv1d(in_channels=64, out_channels=32, kernel_size=63, padding=63 // 2, bias=False),
                nn.Conv1d(in_channels=32, out_channels=1, kernel_size=31, padding=31 // 2, bias=False)]
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.input = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=127, stride=1, padding=127 // 2, bias=False)
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=31, stride=1, padding=31 // 2, bias=False)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=31, stride=1, padding=31 // 2, bias=False)
        self.output = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=31, stride=1, padding=31 // 2, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.head(x)
        attention = self.body(x)
        x = x * attention
        x = self.tail(x)
        x = x + res
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs
        for _ in range(4):
            out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(out, inputs)
        out = self.output(self.relu(out))
        out = torch.add(out, residual)
        return out
