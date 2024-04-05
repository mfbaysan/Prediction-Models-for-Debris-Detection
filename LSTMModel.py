import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import pickle
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from lightning.pytorch.loggers import WandbLogger


class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=10, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.save_hyperparameters()
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.validation_step_outputs = []

    def forward(self, x):
        x = x[..., None]
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1])  # Take the last time step's output
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        train_acc = self.accuracy(F.softmax(logits), y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', train_acc, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        val_acc = self.accuracy(F.softmax(logits), y)

        self.log('valid_acc', val_acc, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        test_acc = self.accuracy(F.softmax(logits), y)

        self.log('test_loss', loss)
        self.log('test_acc', test_acc, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
