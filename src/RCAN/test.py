import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from model import RCAN
import utils
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', default='F:/Result/444/x3/best.pth', type=str)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = RCAN().to(device)
    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    model.eval()
    test_lr = utils.read_data('F:/大学/1 大四所有学期/1 毕业设计/5 数据/大样本/chemcam测试集90.csv')
    test_hr = utils.read_data('F:/大学/1 大四所有学期/1 毕业设计/5 数据/大样本/supercam测试集90.csv')

    data_select = 0                     # 选用哪一个数据进行测试
    normalization = True                # 用比例归一化还是最大值归一化,True是比例归一化

    if normalization:
        nor_hr = utils.normalization(test_hr[data_select])
        test = utils.normalization(utils.bicubic(size=7933, data=test_lr[data_select]))
        test = np.array(test).astype(np.float32)
        test_data = torch.from_numpy(test).to(device)
        test_data = test_data.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            start = time.time()
            preds = model(test_data).clamp(0.0, 1.0)
            end = time.time()
        preds = preds.squeeze(0).squeeze(0)
        preds = preds.cpu().numpy()

        # preds = utils.normalization(preds)
        # preds = utils.reduce_noise_1(preds, 0.00016)

        preds = utils.normalization(preds)
        preds_data = torch.from_numpy(preds)
        nor_hr_torch = np.array(nor_hr)
        nor_hr_torch = torch.from_numpy(nor_hr_torch)
        psnr = utils.calc_psnr(nor_hr_torch, preds_data)
        spend_time = end - start
        print('PSNR:{:.2f}'.format(psnr))
        print('耗时： ' + str(spend_time))
        # utils.draw2line(preds, nor_hr)
        utils.drawline_2(nor_hr, preds)

    if not normalization:
        test = utils.bicubic(size=7933, data=test_lr[data_select])
        test = np.array(test).astype(np.float32) / 75800000000000.
        test_data = torch.from_numpy(test).to(device)
        test_data = test_data.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            start = time.time()
            preds = model(test_data).clamp(0.0, 1.0)
            end = time.time()
        preds = preds.squeeze(0).squeeze(0)
        preds = preds.mul(75800000000000.).cpu().numpy()

        # preds = utils.normalization(preds)
        # preds = utils.reduce_noise_1(preds, 0.00016)

        preds = utils.normalization(preds)
        preds_data = torch.from_numpy(preds)
        test_hr_data = utils.normalization(test_hr[data_select])
        test_hr_torch = np.array(test_hr_data)
        test_hr_torch = torch.from_numpy(test_hr_torch)
        psnr = utils.calc_psnr(test_hr_torch, preds_data)
        spend_time = end - start
        print('PSNR:{:.2f}'.format(psnr))
        print('耗时： ' + str(spend_time))
        # utils.draw2line(preds, test_hr_data)
        # utils.drawline_2(test_hr_data, preds)
        utils.draw_zoom(test_hr_data, preds)
