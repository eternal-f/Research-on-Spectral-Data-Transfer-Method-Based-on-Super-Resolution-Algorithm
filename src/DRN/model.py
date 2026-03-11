import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv1d(in_channels, out_channels, kernel_size,
                     padding=kernel_size // 2, bias=bias)


class DownBlock(nn.Module):
    def __init__(self, n_feats=64, in_channels=64, out_channels=64):
        super(DownBlock, self).__init__()
        dual_block = [
            nn.Sequential(
                nn.Conv1d(in_channels, n_feats, kernel_size=31, stride=1, padding=31 // 2, bias=False),
                nn.LeakyReLU(inplace=True)
            )
        ]
        for _ in range(1, 2):
            dual_block.append(
                nn.Sequential(
                    nn.Conv1d(n_feats, n_feats, kernel_size=31, stride=1, padding=31 // 2, bias=False),
                    nn.LeakyReLU(inplace=True)
                )
            )
        dual_block.append(nn.Conv1d(n_feats, out_channels, kernel_size=31, stride=1, padding=31 // 2, bias=False))
        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_du = nn.Sequential(
                nn.Conv1d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class DRN(nn.Module):
    def __init__(self, conv=default_conv, act=nn.ReLU(True),
                 n_feats=64):
        super(DRN, self).__init__()

        self.head = conv(1, n_feats, 127)
        self.down = [
            DownBlock(n_feats * pow(2, p), n_feats, n_feats * pow(2, p + 1))
            for p in range(1)
        ]
        self.down = nn.ModuleList(self.down)

        up_body_blocks = [[
            RCAB(
                conv, n_feats * pow(2, p), 31, act=act
            ) for _ in range(1)
        ] for p in range(1, 1, -1)
        ]

        up_body_blocks.insert(0, [
            RCAB(
                conv, n_feats * pow(2, 1), 31, act=act
            ) for _ in range(1)
        ])

        up = [[
            conv(n_feats * pow(2, 1), n_feats * pow(2, 0), kernel_size=1)
        ]]

        for p in range(0, 0, -1):
            up.append([
                conv(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)
            ])

        self.up_blocks = nn.ModuleList()
        for idx in range(1):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        tail = [conv(n_feats * pow(2, 1), 1, 31)]
        for p in range(1, 0, -1):
            tail.append(
                conv(n_feats * pow(2, p), 1, 31)
            )
        self.tail = nn.ModuleList(tail)

    def forward(self, x):

        x = self.head(x)
        copies = []
        for idx in range(1):
            copies.append(x)
            x = self.down[idx](x)

        sr = self.tail[0](x)
        results = [sr]
        for idx in range(1):
            x = self.up_blocks[idx](x)
            x = torch.cat((x, copies[1 - idx - 1]), 1)
            sr = self.tail[idx + 1](x)
            results.append(sr)

        return results[0]
