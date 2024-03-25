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
    def __init__(self, data_files):
        self.samples = []
        self.label_encoder = LabelEncoder()  # Initialize label encoder
        for file in data_files:
            with open(file, 'rb') as f:
                sample = pickle.load(f)
                signal = np.squeeze(np.array(sample['signal']))
                class_label = self.label_encoder.fit_transform([sample['object_id']])[0]  # Encode object_id
                self.samples.append({'signal': signal, 'class': class_label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        signal = sample['signal']
        signal_length = signal.shape[1]
        if signal_length < 512:
            pad_width = 512 - signal_length
            signal = np.pad(signal, ((0, 0), (0, pad_width)), mode='constant')
        elif signal_length > 512:
            start_col = np.random.randint(0, signal_length - 512)
            signal = signal[:, start_col:start_col+512]
        # Flatten signal
        signal = signal.flatten()
        signal = torch.tensor(signal, dtype=torch.float32)
        class_label = torch.tensor(sample['object_id'], dtype=torch.long)
        return signal, class_label
