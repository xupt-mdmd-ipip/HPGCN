import torch
import numpy as np
from torch.utils.data import Dataset


def data_normal(data):
    data_range = np.max(data) - np.min(data)
    data_nor = (data - np.min(data)) / data_range
    return data_nor


def data_normal2(data):
    data_range = torch.max(data) - torch.min(data)
    data_nor = (data - torch.min(data)) / data_range
    return data_nor


class data_set(Dataset):
    def __init__(self, data, label):
        super(data_set, self).__init__()
        data = data_normal(data)
        self.data = torch.FloatTensor(data)
        self.label = torch.LongTensor(label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class data_set2(Dataset):
    def __init__(self, data, label):
        super(data_set2, self).__init__()
        data = data_normal2(data)
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
