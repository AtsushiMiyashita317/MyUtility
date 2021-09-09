import os
import pickle
import sys

import torch
from torch.utils.data import Dataset, DataLoader

def save(dataset, dataset_path, data_dir, batchsize):
    # make directory
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    data_path = os.path.join(dataset_path,data_dir)

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # save to files
    dataloader = DataLoader(dataset, batch_size=batchsize)

    print("path = ", data_path)

    for batch, (X,y) in enumerate(dataloader):
        print(f"processing... batch = {batch}\r",end='')
        torch.save(X, os.path.join(data_path,f"data{batch}.pt"))
        torch.save(y, os.path.join(data_path,f"label{batch}.pt"))

    # make annotations
    with open(os.path.join(dataset_path, f"{data_dir}.pickle"),'wb') as f:
        pickle.dump((batchsize, len(dataset), '1.0.0'),f)


class MyDataset(Dataset):
    def __init__(self, dataset_path, data_dir, transform=None, target_transform=None, length=None):
        self.dataset_path = dataset_path
        self.data_path = os.path.join(dataset_path, data_dir)

        with open(os.path.join(dataset_path, f"{data_dir}.pickle"), 'rb') as f:
            self.batchsize,self.length, self.version = pickle.load(f)

        if length:
            if length < self.length:
                self.length = length

        self.cache_idx = -1
        self.cache_data = None
        self.cache_label = None

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx//self.batchsize != self.cache_idx:
            self.cache_idx = idx//self.batchsize
            self.cache_data = torch.load(os.path.join(self.data_path, f"data{self.cache_idx}.pt"))
            self.cache_label = torch.load(os.path.join(self.data_path, f"label{self.cache_idx}.pt"))

        local_idx = idx - self.cache_idx*self.batchsize

        data = self.cache_data[local_idx]
        label = self.cache_label[local_idx]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)

        return data, label
