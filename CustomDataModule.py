import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pickle
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split

from CustomDataset import CustomDataset


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, train_val_test_split=[0.7, 0.15, 0.15]):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split

    def prepare_data(self):
        self.data_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.pickle')]

    def setup(self, stage=None):
        dataset = CustomDataset(self.data_files)
        train_size = int(self.train_val_test_split[0] * len(dataset))
        val_size = int(self.train_val_test_split[1] * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_data, self.val_data, self.test_data = random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
