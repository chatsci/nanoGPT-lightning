import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


class LanguageModelDataset(Dataset):
    def __init__(self, data_dir, split, block_size):
        self.data = np.memmap(os.path.join(data_dir, f'{split}.bin'),
                              dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, index):
        x = self.data[index:index+self.block_size].astype(np.int32)
        y = self.data[index+1:index+1+self.block_size].astype(np.int32)
        return x, y


class LanguageModelDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, block_size, num_workers=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download dataset if necessary
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = LanguageModelDataset(
                self.data_dir, 'train', self.block_size)
            self.val_dataset = LanguageModelDataset(
                self.data_dir, 'val', self.block_size)
        if stage == 'test' or stage is None:
            self.test_dataset = LanguageModelDataset(
                self.data_dir, 'test', self.block_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
