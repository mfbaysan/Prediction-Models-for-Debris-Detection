from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import random_split
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
from resnet import ResNet, ResNetWrapper, _resnet, BasicBlock
from residual_network import ResidualNetwork


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
        data_sample = torch.FloatTensor(self.data[idx]).unsqueeze(0)
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


class ResNetLightning(LightningModule):
    def __init__(self, block_type, layers, num_classes=11, out_dim=None, learning_rate=1e-3):
        super().__init__()
        self.model = ResNetWrapper(block_type, layers, num_classes=num_classes, out_dim=out_dim)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()  # Loss for multi-class classification

    def forward(self, x):
        x = self.model(x)
        print(f"Input shape received by model: {x.shape}")

        return x  # Forward pass through the ResNet model

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming your data directory is "data"
    root_dir = "../New_data"
    batch_size = 32

    custom_dataset = CustomDataset(root_dir)
    data_module = RadarDataModule(custom_dataset, batch_size=batch_size, val_split=0.2)

    dataset_length = len(custom_dataset)
    print("Length of the dataset:", dataset_length)


    # Define block type (BasicBlock or Bottleneck)
    model = ResidualNetwork([2, 2, 2, 2, 2], num_classes=11, activation='silu', use_noise=False, noise_stddev=0.1,
                            fc_units=64, in_out_channels=[(32, 32), (32, 64), (64, 128), (128, 256), (256, 512)])
    #print(model)

    wandb_logger = WandbLogger(project='Resnet_NewData')
    wandb_logger.watch(model, log="all")

    trainer = Trainer(max_epochs=10, accelerator="auto", logger=wandb_logger)

    trainer.fit(model, datamodule=data_module)

    # Print the model architecture (optional)



