import os
import pickle
import sys

import torch
from torch.utils.data import Dataset, DataLoader

def save(dataset, dataset_path, data_dir, maxdatasize):
    # make directory
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    data_path = os.path.join(dataset_path,data_dir)

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # save to files
    one = dataset[0]
    datasize = sys.getsizeof(one)
    batchsize = maxdatasize//datasize

    dataloader = DataLoader(dataset, batch_size=batchsize)

    for batch, (X,y) in enumerate(dataloader):
        torch.save(X, os.path.join(data_path,f"data{batch}.pt"))
        torch.save(y, os.path.join(data_path,f"label{batch}.pt"))

    # make annotations
    with open(f"{data_dir}.pickle",'wb') as f:
        pickle.dump((batchsize, len(dataset), '1.0.0'),f)


class MyDataset(Dataset):
    def __init__(self, dataset_path, data_dir, transform=None, target_transform=None):
        self.dataset_path = dataset_path
        self.data_path = os.path.join(dataset_path, data_dir)

        with open(f"{data_dir}.pickle", 'rb') as f:
            self.batchsize,self.length, self.version = pickle.load(f)

        self.cache_idx = -1
        self.cache_data = None
        self.cache_label = None

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.batchsize

    def __getitem__(self, idx):
        if idx//self.batchsize != self.cache_idx:
            self.cache_idx = idx//self.batchsize
            self.cache_data = torch.load(f"data{self.cache_idx}.pt")
            self.cache_label = torch.load(f"label{self.cache_idx}.pt")

        local_idx = idx - self.cache_idx

        data = self.cache_data[local_idx]
        label = self.cache_label[local_idx]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)

        return data, label
