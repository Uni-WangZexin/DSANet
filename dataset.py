
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.utils.data

import logging
import numpy as np
import pandas as pd

class MTSFDataset(torch.utils.data.Dataset):
    """Multi-variate Time-Series Dataset for *.txt file

    Returns:
        [sample, label]
    """

    def __init__(self,
                window,
                horizon,
                data_name='electricity',
                set_type='train',    # 'train'/'validation'/'test'
                data_dir='./data/multivariate-time-series-data/'):
        assert type(set_type) == type('str')
        self.window = window
        self.horizon = horizon
        self.data_dir = data_dir
        self.set_type = set_type

        file_path = os.path.join(data_dir, data_name, '{}_{}.txt'.format(data_name, set_type))


        rawdata = np.loadtxt(open(file_path), delimiter=',')
        self.len, self.var_num = rawdata.shape
        self.sample_num = max(self.len - self.window - self.horizon + 1, 0)
        print("init")
        self.samples, self.labels = self.__getsamples(rawdata)
        print("init")
    def __getsamples(self, data):
        print("sample")
        print(self.sample_num, self.window, self.var_num)
        X = torch.zeros([self.sample_num, self.window, self.var_num])
        print("sample")
        Y = torch.zeros([self.sample_num, 1, self.var_num])
        print("sample")
        for i in range(self.sample_num):
            print(i)
            start = i
            end = i + self.window
            X[i, :, :] = torch.from_numpy(data[start:end, :])
            print(i)
            Y[i, :, :] = torch.from_numpy(data[end+self.horizon-1, :])
            print(i)
        return (X, Y)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx, :, :]]
        return sample


# test
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    logging.debug('Test: data from .txt file')
    dataset = MTSFDataset(
        window=16,
        horizon=3,
        data_name='wecar',
        set_type='train',
        data_dir='./data'
    )
    i = 0
    sample = dataset[i]
    logging.debug('Sample #{}, label: {}'.format(i, sample[1]))
    logging.debug('data: {}'.format(sample[0]))
