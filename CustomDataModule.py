import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pickle
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from CustomDataset import CustomDataset


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, dataframe, device, batch_size):
        super().__init__()
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.device = device

    def prepare_data(self):
        # No data downloading needed
        pass

    def setup(self, stage=None):
        # Split dataset into train, validation, and test sets
        # train_df, val_test_df = train_test_split(self.dataframe, test_size=0.0, random_state=42, stratify=self.dataframe['object_id'])
        # # val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42, stratify=val_test_df['object_id'])

        self.train_dataset = CustomDataset(self.dataframe, device=self.device)
        # self.val_dataset = CustomDataset(val_df, device=self.device)
        # self.test_dataset = CustomDataset(test_df, device=self.device)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size)
    #
    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size)
