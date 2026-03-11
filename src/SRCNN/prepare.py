import h5py
import numpy as np
import utils


def train(args, nor):
    h5_file = h5py.File(args, 'w')
    lr_patches = []
    hr_patches = []
    train_lr = utils.read_data('F:/大学/1 大四所有学期/1 毕业设计/5 数据/大样本/chemcam训练集1560.csv')
    train_hr = utils.read_data('F:/大学/1 大四所有学期/1 毕业设计/5 数据/大样本/supercam训练集1560.csv')

    if nor:
        for data in train_lr:
            temp = utils.normalization(utils.bicubic(size=7933, data=data))
            lr_patches.append(temp)
        for data in train_hr:
            temp = utils.normalization(data)
            hr_patches.append(temp)
        lr_patches = np.array(lr_patches)
        hr_patches = np.array(hr_patches)
        h5_file.create_dataset('lr', data=lr_patches)
        h5_file.create_dataset('hr', data=hr_patches)
        h5_file.close()

    if not nor:
        for data in train_lr:
            temp = utils.bicubic(size=7933, data=data)
            lr_patches.append(temp)
        lr_patches = np.array(lr_patches)
        hr_patches = np.array(train_hr)
        h5_file.create_dataset('lr', data=lr_patches)
        h5_file.create_dataset('hr', data=hr_patches)
        h5_file.close()


def eval_test(args, nor):
    h5_file = h5py.File(args, 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    test_lr = utils.read_data('F:/大学/1 大四所有学期/1 毕业设计/5 数据/大样本/chemcam测试集90.csv')
    test_hr = utils.read_data('F:/大学/1 大四所有学期/1 毕业设计/5 数据/大样本/supercam测试集90.csv')
    if nor:
        for i, data in enumerate(test_lr):
            temp = utils.normalization(utils.bicubic(size=7933, data=data))
            lr = temp
            hr = utils.normalization(test_hr[i])
            lr_group.create_dataset(str(i), data=lr)
            hr_group.create_dataset(str(i), data=hr)

    if not nor:
        for i, data in enumerate(test_lr):
            temp = utils.bicubic(size=7933, data=data)
            lr = temp
            hr = test_hr[i]
            lr_group.create_dataset(str(i), data=lr)
            hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


def main():
    flag = True
    normalization = True

    if normalization:
        path = 'F:/大学/Data/Normalization/Test/test'  # 最后创建的是ces.h5py，记得新建一个文件夹
        if not flag:
            train(path, normalization)
        else:
            eval_test(path, normalization)
    if not normalization:
        path = 'F:/大学/Data/Normal-data/Train/train'  # 最后创建的是ces.h5py，记得新建一个文件夹
        if not flag:
            train(path, normalization)
        else:
            eval_test(path, normalization)


if __name__ == '__main__':
    main()
