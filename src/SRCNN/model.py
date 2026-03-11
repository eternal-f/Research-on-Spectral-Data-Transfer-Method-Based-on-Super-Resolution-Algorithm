from torch import nn


class SRCNN(nn.Module):             # 用了conv1d，原算法是conv2d，跑不通的话改一下然后用数据折叠
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=127, padding=(127 // 2))
        self.conv2 = nn.Conv1d(64, 32, kernel_size=31, padding=(31 // 2))
        self.conv3 = nn.Conv1d(32, num_channels, kernel_size=31, padding=(31 // 2))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
