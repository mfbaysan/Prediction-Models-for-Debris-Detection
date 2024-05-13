import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics


class LSTMClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

        if not bidirectional:
            self.fc = nn.Linear(hidden_size, num_classes)
        else:
            print("b i d i r e c t  i o n a l !!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(bidirectional)
            self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, return_features: bool = False):
        # Determine the device of the input tensor
        device = x.device

        # Assuming input x is of shape [batch_size, channels, sequence_length]
        # channels=1

        print(x.shape)
        x = torch.unsqueeze(x, 1)
        print(f"shape after x unsqueezed: {x.shape}")
        # Reshape input to [batch_size, sequence_length, features]
        x = x.transpose(1, 2)
        print(f"Shape of x after transpose: {x.shape}")  # Add this line to check the shape

        # Initialize hidden state and cell state with consideration for bidirectionality
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        features = out[:, -1, :]
        logits = self.fc(features)

        if return_features:
            return logits, features
        else:
            return logits

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

        logits = self(data)
        target_sq = target.squeeze()

        loss = F.cross_entropy(logits, target_sq)
        acc = self.val_accuracy(logits, target_sq)

        # Logging
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log('val_accuracy', acc)

        return loss

    def configure_optimizers(self):
        # Define optimizer here (e.g., Adam)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer
