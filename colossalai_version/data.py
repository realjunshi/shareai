import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np


def get_x_and_y(data):
    x, y = [], []
    for i in data:
        x.append(i[:-2])
        y.append(i[1:])
    return x, y


def load(file_name):
    data_dir = os.path.join('data')
    train_data = np.load(os.path.join(data_dir, file_name), allow_pickle=True)
    print(len(train_data))
    return train_data


# 重写collate_fn函数，其输入为一个batch的sample数据
def collate_fn(batch):
    # 因为token_list是一个变长的数据，所以需要用一个list来装这个batch的token_list
    # token_lists = [(x, y) for (x, y) in enumerate(batch)]
    #
    x_val = []
    y_val = []
    for i, (x, y) in enumerate(batch):
        x_val.append(x)
        y_val.append(y)

    return torch.as_tensor(x_val), torch.as_tensor(y_val)


class SongDataset(Dataset):
    """
    1. 加载编码好的数据
    2. 构建训练数据的X,Y
    """

    def __init__(self, file_name):
        self.data = load(file_name)
        self.length = len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        x = torch.tensor(data[:-1], dtype=torch.long)
        y = torch.tensor(data[1:], dtype=torch.long)

        return x, y

    def __len__(self):
        return self.length


if __name__ == '__main__':

    # x, y = load("title.train.npy")
    # print(len(x))
    # train_dataset = SongDataset('val_max.npy')
    train_dataset = SongDataset('train_max.npy')
    loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

    for i, (x, y) in enumerate(loader):
        # print("x_batch:", x.shape)
        # print("y_batch", y.shape)
        # break
        print(x)
        print(y)
        break