# This is a sample Python script.
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pickle
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
from CustomDataModule import CustomDataModule
from LSTMModel import LSTMModel


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    data_dir = 'First_data'  # Change this to your data folder path
    data_module = CustomDataModule(data_dir)
    model = LSTMModel(input_size=512, hidden_size=64, num_classes=11)

    trainer = pl.Trainer(max_epochs=10)  # Set max_epochs and gpus accordingly , gpus=1
    trainer.fit(model, data_module)

