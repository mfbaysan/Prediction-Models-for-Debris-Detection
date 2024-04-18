from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl
import torch.nn as nn
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

            class_data = self.normalize_samples(class_data)
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

    def normalize_samples(self, data):
        # Calculate mean and standard deviation for each sample
        sample_means = np.mean(data, axis=1, keepdims=True)
        sample_stds = np.std(data, axis=1, keepdims=True)

        # Normalize each sample
        normalized_data = (data - sample_means) / sample_stds

        return normalized_data


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



class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet18(pl.LightningModule):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(16, 2)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.layer4 = self._make_layer(128, 2, stride=2)
        self.layer5 = self._make_layer(256, 2, stride=2)
        self.layer6 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

        self.lr_scheduler = ReduceLROnPlateau(optimizer=self.configure_optimizers()['optimizer'], mode='min',
                                              factor=0.1, patience=3)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        train_y = y.squeeze(1)
        #print("sqeueezed train y", train_y)
        loss = F.cross_entropy(logits, train_y)
        accuracy = (logits.argmax(-1) == y).float().mean().item()

        self.log('train/loss', loss)
        self.log('train/accuracy', accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        val_y = y.squeeze(1)
        #print("squeezed y", val_y)
        # Calculate loss using cross-entropy function
        loss = F.cross_entropy(logits, val_y)
        accuracy = (logits.argmax(-1) == y).float().mean().item()

        self.log('val/loss', loss)
        self.log('val/accuracy', accuracy)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.03)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming your data directory is "data"
    root_dir = "../New_data"
    batch_size = 32

    custom_dataset = CustomDataset(root_dir)
    data_module = RadarDataModule(custom_dataset, batch_size=batch_size, val_split=0.2)

    dataset_length = len(custom_dataset)
    print("Length of the dataset:", dataset_length)


    # Print out the random samples
    random_indices = random.sample(range(dataset_length), 5)

    # Print out the random samples
    for idx in random_indices:
        data_sample, label = custom_dataset.__getitem__(idx)
        print("\nSample Index:", idx)
        print("Data sample from the .mat file:", data_sample.shape)
        print("Label for the data sample:", label)

    num_classes = 11  # Update with the number of classes in your dataset
    model = ResNet18(num_classes)

    wandb_logger = WandbLogger(project='Resnet_NewData')
    wandb_logger.watch(model, log="all")
    # Training
    trainer = Trainer(max_epochs=50, accelerator="auto", log_every_n_steps=10, logger=wandb_logger)  # Set gpus=0 for CPU training
    trainer.fit(model, datamodule=data_module)

