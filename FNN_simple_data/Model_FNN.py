import math
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import random_split, Subset
import random
from torch.nn import functional as F
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from tqdm import tqdm
from pytorch_lightning.loggers import WandbLogger
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from LSTM import LSTMClassifier
from Transformer_conv import TimeSeriesTransformer
from resnet import ResNet, ResNetWrapper, _resnet, BasicBlock
from residual_network import ResidualNetwork
from timm.models.vision_transformer import Block
import torchmetrics

def positionalencoding1d(d_model, sample):
  """
  :param d_model: dimension of the model
  :param length: length of positions
  :return: length*d_model position matrix
  """
  if d_model % 2 != 0:
    raise ValueError("Cannot use sin/cos positional encoding with "
                     "odd dim (got dim={:d})".format(d_model))
  pe = torch.zeros(sample.shape[0], d_model)
  position = sample.unsqueeze(1)
  div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                        -(math.log(10000.0) / d_model)))
  pe[:, 0::2] = torch.sin(position.float() * div_term)
  pe[:, 1::2] = torch.cos(position.float() * div_term)

  return pe


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.data = []
        self.labels = []

        for class_idx, class_name in enumerate(tqdm(self.classes, desc="Loading data")):
            class_dir = os.path.join(root_dir, class_name)
            class_data = []
            file_list = sorted(os.listdir(class_dir))
            for file in file_list:
                file_path = os.path.join(class_dir, file)
                mat_file = loadmat(file_path)
                voltage_data = mat_file['objects_echo']
                db_data = 20 * np.log10(voltage_data ** 2)
                class_data.append(db_data)

            class_data = np.concatenate(class_data, axis=0)

            class_labels = np.full((class_data.shape[0], 1), class_idx)
            #print(class_data.shape)
            self.data.append(class_data)
            self.labels.append(class_labels)

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = torch.FloatTensor(self.data[idx])
        # convert data_sample to encoded
        #data_sample = positionalencoding1d(8, data_sample)
        label = torch.LongTensor(self.labels[idx])

        return data_sample, label



class RadarDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, val_split=0.2):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_split = val_split

    def setup(self, stage=None):
        # Split dataset into train and val sets
        if stage == 'fit' or stage is None:
            dataset_size = len(self.dataset)
            val_size = int(self.val_split * dataset_size)
            train_size = dataset_size - val_size
            self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


class TransformerLightning(LightningModule):
    def __init__(self, learning_rate=1e-3): # lr both higher and lower , bypass positional encoding
        super().__init__()
        self.model = torch.nn.Sequential(
            *[Block(8, 4) for _ in range(12)] #deeper
        )

        self.lr1 = nn.Linear(8, 5)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)
        # Loss for multi-class classification

    def forward(self, x):
        print(f"Input shape received by model: {x.shape}")


        x = self.model(x)

        # get the means over dim 512 then apply linear layer
        x = torch.mean(x, 1)
        print(f"Input shape after transformer encoding: {x.shape}")
        x = self.lr1(x)

        return x  # Forward pass through the ResNet model

    def training_step(self, batch, batch_idx):
        # Get data and targets
        data, target = batch

        # Forward pass
        logits = self(data)
        target_sq = target.squeeze()

        # Calculate loss
        loss = F.cross_entropy(logits, target_sq)
        acc = self.train_accuracy(logits, target_sq)

        # Logging
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log('train_accuracy', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        # Same logic as training_step but with validation loss
        data, target = batch

        logits = self(data) # 32, 512, 8 => 32,num classes
        target_sq = target.squeeze()

        loss = F.cross_entropy(logits, target_sq)
        acc = self.val_accuracy(logits, target_sq)

        # Logging
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log('val_accuracy', acc)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming your data directory is "data"
    root_dir = "../data_5cm"
    batch_size = 32

    custom_dataset = CustomDataset(root_dir)
    #custom_dataset = Subset(custom_dataset, range(32))
    data_module = RadarDataModule(custom_dataset, batch_size=batch_size, val_split=0.2)

    dataset_length = len(custom_dataset)
    print("Length of the dataset:", dataset_length)

    transformer = TransformerLightning()
    #Lstm = LSTMClassifier(input_size=1, hidden_size=64, num_layers=12, num_classes=5)

    # Define block type (BasicBlock or Bottleneck)
    # model = ResidualNetwork([2, 2, 2, 2, 2], num_classes=5, activation='silu', use_noise=False, noise_stddev=0.0,
    #                         fc_units=64, in_out_channels=[(32, 32), (32, 64), (64, 128), (128, 256), (256, 512)])
    #print(model)
    transformer_embed = TimeSeriesTransformer(num_features=1, num_classes=5, sequence_length=128, embedding_option='conv1d')
    wandb_logger = WandbLogger(project='Transformer_NewData')
    wandb_logger.watch(transformer_embed, log="all")

    trainer = Trainer(max_epochs=20, accelerator="auto", logger=wandb_logger)

    trainer.fit(transformer_embed, datamodule=data_module)

    # Print the model architecture (optional)



