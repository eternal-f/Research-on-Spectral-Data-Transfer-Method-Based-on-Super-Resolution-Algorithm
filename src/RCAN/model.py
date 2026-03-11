import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv1d(in_channels, out_channels, kernel_size,
                     padding=kernel_size // 2, bias=bias)


class CALayer(nn.Module):                           # 通道注意力模块
    def __init__(self, channel, reduction=16):      # reduction是中间层缩小倍率，用于提炼注意力
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


class RCAB(nn.Module):                              # 残差加注意力块
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm1d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResidualGroup(nn.Module):                     # 残差注意力群
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RCAN(nn.Module):                              # RCAN
    def __init__(self, n_resgroups=2, n_resblocks=2, n_feats=64, kernel_size=31, reduction=16, conv=default_conv):
        super(RCAN, self).__init__()
        modules_head = [conv(1, n_feats, 127)]
        modules_body = [
            ResidualGroup(conv, n_feats, kernel_size, reduction, n_resblocks=n_resblocks) for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        modules_tail = [conv(n_feats, 1, 15)]
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x
