import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pickle
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder


class CustomDataset(Dataset):
    def __init__(self, data_files, max_length=512):
        self.samples = []
        self.label_encoder = LabelEncoder()  # Initialize label encoder
        for file in data_files:
            with open(file, 'rb') as f:
                sample = pickle.load(f)
                signal = np.squeeze(np.array(sample['radar_return']))
                class_label = self.label_encoder.fit_transform([sample['object_id']])[0]  # Encode object_id
                self.samples.append({'signal': signal, 'class': class_label})

        self.max_length = max_length

    def pad_or_truncate(self, signal):
        signal_length = signal.shape[1]
        if signal_length < self.max_length:
            pad_width = self.max_length - signal_length
            signal = np.pad(signal, ((0, 0), (0, pad_width)), mode='constant')
        elif signal_length > self.max_length:
            start_col = np.random.randint(0, signal_length - self.max_length)
            signal = signal[:, start_col:start_col+self.max_length]
        return signal

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        signal = self.pad_or_truncate(sample['signal'])
        signal = signal.flatten()
        signal = torch.tensor(signal, dtype=torch.float32)
        class_label = torch.tensor(sample['class'], dtype=torch.long)
        return signal, class_label
