import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

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

class MyDatasetFramed(Dataset):
    def __init__(self, root, data_dir, 
                 transform=None, frame_transform=None, target_transform=None, 
                 nframe=15, datasize=None, cachesize = 100):

        self.nframe = nframe

        self.data_dir = os.path.join(root, data_dir)

        annotations_path = os.path.join(root, data_dir, "annotations.csv")
        self.annotations = pd.read_csv(annotations_path)

        spec_length = self.annotations['length'].values
        item_length = spec_length - self.nframe + 1
        self.maxidx = np.cumsum(item_length)
        
        self.transform = transform
        self.frame_transform = frame_transform
        self.target_transform = target_transform

        self.cachesize = cachesize
        self.cache_data = [np.empty(0)] * self.cachesize
        self.cache_label = [np.empty(0)] * self.cachesize
        self.cache_lower = np.full(self.cachesize+1,self.maxidx.max()+1,dtype=np.int64)
        self.cache_upper = np.full(self.cachesize+1,self.maxidx.max()+1,dtype=np.int64)
        self.cache_last = np.full(cachesize,-1,dtype=np.int64)
        self.cache_sorter = np.arange(cachesize+1)
        self.cache_time = 0

        self.datasize = datasize


    def __len__(self):
        if self.datasize:
            return self.datasize
        else:
            return self.maxidx.max()

    def __getitem__(self, idx):
        insert_idx = np.searchsorted(self.cache_upper,idx,sorter=self.cache_sorter,side='right')
        hit = self.cache_sorter[insert_idx]
        
        if not (self.cache_lower[hit]<=idx and idx<self.cache_upper[hit]):
            oldest = np.argmin(self.cache_last)

            id = np.searchsorted(self.maxidx,idx,side='right')
            
            data_path = os.path.join(self.data_dir, self.annotations.iat[id, 1])
            self.cache_data[oldest] = torch.load(data_path)

            if self.transform:
                self.cache_data[oldest] = self.transform(self.cache_data[oldest])    

            label_path = os.path.join(self.data_dir, self.annotations.iat[id, 2])
            self.cache_label[oldest] = torch.load(label_path)

            if id == 0:
                self.cache_lower[oldest] = 0
                self.cache_upper[oldest] = self.maxidx[0]
            else:
                self.cache_lower[oldest] = self.maxidx[id-1]
                self.cache_upper[oldest] = self.maxidx[id]

            hit = oldest
            self.cache_last[oldest] = self.cache_time

            # get rank of oldest with respect to upper
            delete_idx = np.where(self.cache_sorter==oldest)[0][0]
            # update sorter
            self.cache_sorter = np.delete(self.cache_sorter,delete_idx)
            if delete_idx<insert_idx:
                insert_idx -= 1
            self.cache_sorter = np.insert(self.cache_sorter,insert_idx,oldest)

        self.cache_time += 1

        local_idx = idx - self.cache_lower[hit]
        frames = self.cache_data[hit][local_idx:local_idx + self.nframe]
        label = self.cache_label[hit][local_idx + self.nframe//2]


        if self.frame_transform:
            frames = self.frame_transform(frames)
        if self.target_transform:
            label = self.target_transform(label)

        return frames, label
      
    
    def save(dataset, root, data_dir):
        # make directory
        if not os.path.exists(root):
            os.mkdir(root)

        data_path = os.path.join(root,data_dir)

        if not os.path.exists(data_path):
            os.mkdir(data_path)

        # save to files
        annotations = pd.DataFrame(index=[], 
                                   columns=['data_path',
                                            'label_path',
                                            'length'])
        annotations = annotations.astype({'length':np.int64})

        dataloader = DataLoader(dataset, batch_size=1)

        print("path = ", data_path)

        for batch, (X,y) in enumerate(dataloader):
            print(f"processing... batch = {batch}\r",end='')
            torch.save(X[0], os.path.join(data_path,f"data{batch}.pt"))
            torch.save(y[0], os.path.join(data_path,f"label{batch}.pt"))
            annotations = annotations.append({'data_path':f"data{batch}.pt",
                                              'label_path':f"label{batch}.pt",
                                              'length':y[0].shape[0]},
                                              ignore_index=True)

        print(f"processing... Done")

        # save annotations
        annotations.to_csv(os.path.join(data_path,"annotations.csv"))
