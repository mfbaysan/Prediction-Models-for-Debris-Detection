import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pickle
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder


class CustomDataset(Dataset):
    def __init__(self, dataframe, device):
        self.dataframe = dataframe
        self.device = device

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        radar_return = self.dataframe.iloc[idx]['radar_return']
        label = self.dataframe.iloc[idx]['object_id']
        return torch.tensor(radar_return).to(self.device), torch.tensor(label).to(self.device)



