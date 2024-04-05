import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models

from ResNet.ResNetGemini import combine_dataframes, get_tensor_data


class RadarResnet18(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        # Pre-trained ResNet18 model loaded with weights
        self.resnet18 = models.resnet18(pretrained=True)

        # Modify the first convolution layer to handle grayscale images
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the final fully-connected layer with a new one for your number of classes
        num_ftrs = 1000  # Number of features from the pre-trained model
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.resnet18(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        accuracy = (logits.argmax(-1) == y).float().mean().item()

        self.log('train/loss', loss)
        self.log('train/accuracy', accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        accuracy = (logits.argmax(-1) == y).float().mean().item()

        self.log('val/loss', loss)
        self.log('val/accuracy', accuracy)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


class PLDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, num_workers=8):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32, shuffle=True, num_workers=self.num_workers)


if __name__ == '__main__':
    data_dir = "../First_data"
    combined_df = combine_dataframes(data_dir)
    # Create custom dataset and data loaders

    pl.seed_everything(42)
    image_data, labels = get_tensor_data(combined_df)  # Your logic to load your dataframe
    X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2,
                                                        random_state=42, stratify=labels)

    X_train = (X_train - X_train.mean(dim=0)) / X_train.std(dim=0)
    X_test = (X_test - X_test.mean(dim=0)) / X_test.std(dim=0)

    num_classes = 10  # Your number of classes

    # Create PyTorch Dataset and DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    pl_datamodule = PLDataModule(train_dataset=train_dataset, val_dataset=val_dataset)

    model = RadarResnet18(num_classes)

    print("training the model...")
    wandb_logger = WandbLogger(project='radar_classification')
    wandb_logger.watch(model, log="all")

    trainer = pl.Trainer(max_epochs=100,
                         accelerator='mps',
                         logger=wandb_logger)

    trainer.fit(model, pl_datamodule)
