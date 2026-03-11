import csv
import math
import matplotlib.pyplot as plt
import torch
import os
import torch.optim as optim             # torch的优化器
import torch.optim.lr_scheduler as lrs  # torch的学习率调节器


def calc_psnr(data1, data2):  # 算psnr
    return 10. * torch.log10(1. / torch.mean((data1 - data2) ** 2))


class AverageMeter(object):
    def __init__(self):
        self.val = None         # 这四个None源代码中只定义在reset中，pycharm会警告，故在init中事先声明，若出现问题请更改
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def read_data(path):
    """
    读取csv文件
    :param path:csv文件路径
    :return: data，list格式，每个数据float
    """
    data = []
    with open(path) as csvfile:                 # 读取csv文件并转为float形式
        csv_reader = csv.reader(csvfile)
        for single_data in csv_reader:
            temp = []
            for single in single_data:
                temp.append(float(single))
            data.append(temp)
    return data


def bicubic(size=7933, data=None):
    """
    双三次插值对一维光谱数据，只考虑一维方向上4个相邻坐标，不考虑图像算法的16个坐标
    :param size: 目标大小
    :param data: 原始数据
    :return: result 插值后的结果
    """
    a = -0.5                                # 双三次插值的参数取值
    if data is None:
        data = [0] * 6143
    result = [0.0] * size
    length = len(data)
    scale = size/length                     # 放大倍数
    for idx, value in enumerate(result):
        data_idx = idx/scale                # 扩大后（X，Y）点对应原数据位置（带小数）
        x_0 = math.floor(data_idx)          # 向上下取整确定两个坐标点位置
        x_1 = math.ceil(data_idx)
        x_2 = x_0 - 1                       # 确定另外两坐标位置
        x_3 = x_1 + 1
        decimal = data_idx - x_0            # 确定小数部分
        w_x0 = (a+2)*(decimal**3)-(a+3)*(decimal**2)+1  # bicubic算法公式
        w_x1 = (a+2)*((1-decimal)**3)-(a+3)*((1-decimal)**2)+1
        w_x2 = a*((1+decimal)**3)-5*a*((1+decimal)**2)+8*a*(1+decimal)-4*a
        w_x3 = a*((2-decimal)**3)-5*a*((2-decimal)**2)+8*a*(2-decimal)-4*a
        if x_0 == 0:                        # 溢出部分
            w_x2 = 0
            x_2 = 0
        if x_1 == length-1:
            w_x3 = 0
            x_3 = length-1
        if x_1 == length:
            w_x1 = 0
            w_x3 = 0
            x_1 = length-1
            x_3 = length-1
        pixel_value = data[x_0]*w_x0+data[x_1]*w_x1+data[x_2]*w_x2+data[x_3]*w_x3
        result[idx] = pixel_value
    return result


def drawline(data):
    plt.figure(figsize=(20, 20), dpi=100)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams.update({'font.size': 40})
    length = len(data)
    x_axis = list(range(1, length+1))
    y_axis = data
    plt.plot(x_axis, y_axis)
    plt.show()


def draw2line(data_1, data_2):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 20), dpi=100)
    x_axis_1 = list(range(1, len(data_1)+1))
    x_axis_2 = list(range(1, len(data_2)+1))
    axs[0].plot(x_axis_1, data_1, c='red')
    axs[1].plot(x_axis_2, data_2, c='blue')
    fig.autofmt_xdate()
    plt.show()


def padding(data, size=8000):
    length = len(data)
    pad = size - length
    for i in range(0, pad):
        data.append(0)
    return data


def up_vector(data, size=100):
    length = len(data)
    if length % size != 0:
        raise
    col = size
    row = length//size
    result = []
    for i in range(0, row):
        temp = []
        for j in range(0, col):
            temp.append(data[100*i+j])
        result.append(temp)
    return result


def down_vector(data):
    row = len(data)
    col = len(data[0])
    result = []
    for i in range(0, row):
        for j in range(0, col):
            result.append(data[i][j])
    return result


def make_optimizer(target, lr=1e-4, weight_decay=0, optimizer='ADAM',
                   momentum=0.9, betas=(0.9, 0.999), epsilon=1e-8,
                   decay=200, gamma=0.5):
    """
        make optimizer and scheduler together
    """
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())  # filter返回一个可迭代对象，requires_grad计算梯度
    kwargs_optimizer = {'lr': lr, 'weight_decay': weight_decay}
    optimizer_class = None

    if optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = momentum
    elif optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = betas
        kwargs_optimizer['eps'] = epsilon
    elif optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = epsilon

    # scheduler
    temp = [str(decay)]
    milestones = list(map(lambda x: int(x), temp))
    kwargs_scheduler = {'milestones': milestones, 'gamma': gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *a_rgs, **kwargs):  # 这里加了一个下划线
            super(CustomOptimizer, self).__init__(*a_rgs, **kwargs)  # 同上

        def _register_scheduler(self, scheduler__class, **kwargs):  # 加了俩
            self.scheduler = scheduler__class(self, **kwargs)  # 同上

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch):
                    self.scheduler.step()

        @staticmethod
        def get_dir(dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer


def normalization(data):
    result = data
    sum_data = sum(result)
    for idx, value in enumerate(result):
        result[idx] = value / sum_data
    return result


def drawline_2(data1, data2):
    plt.figure(figsize=(20, 10), dpi=100)
    x_axis = [i for i in range(len(data1))]
    y_axis1 = data1
    y_axis2 = data2
    plt.plot(x_axis, y_axis1, c='red', label='Original')
    plt.plot(x_axis, y_axis2, c='green', linestyle='--', label='Super-Resolution')
    plt.legend(loc='best')
    plt.show()


def reduce_noise_1(data, noise):
    result = data
    for idx, value in enumerate(result):
        if value <= noise:
            result[idx] = 0
    return result
