import utils
import numpy as np
import torch

lr = utils.normalization(utils.read_data('F:/大学/1 大四所有学期/1 毕业设计/5 数据/大样本/chemcam测试集90.csv')[0])
lr_bicubic = utils.bicubic(7933, lr)
lr_bicubic = utils.normalization(lr_bicubic)
hr = utils.normalization(utils.read_data('F:/大学/1 大四所有学期/1 毕业设计/5 数据/大样本/supercam测试集90.csv')[0])
# utils.drawline(lr_bicubic)
utils.draw_zoom(hr, lr_bicubic)
