import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchmetrics
import pytorch_lightning as pl

class TimeSeriesTransformer(pl.LightningModule):
    def __init__(self, num_features, num_classes, sequence_length, embedding_option='conv1d', num_layers=1, nhead=1,
                 dim_feedforward=2048, embed_dim=256):
        super(TimeSeriesTransformer, self).__init__()

        self.sequence_length = sequence_length

        self.embed_dim = embed_dim
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)

        # Options for embedding layer
        if embedding_option == 'dense':
            self.embedding = nn.Linear(num_features, num_features)
        elif embedding_option == 'conv1d':
            # self.embedding = nn.Sequential(
            #     nn.Conv1d(in_channels=num_features, out_channels=128, kernel_size=3, padding=1),
            #     nn.BatchNorm1d(128),
            #     nn.ReLU(),
            #     nn.Conv1d(in_channels=128, out_channels=self.embed_dim, kernel_size=3, padding=1),
            #     nn.BatchNorm1d(self.embed_dim),
            #     nn.ReLU()
            # )
            self.embedding = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.SiLU(),
                nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Conv1d(in_channels=256, out_channels=self.embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(self.embed_dim),
                nn.SiLU()
            )
        else:
            raise ValueError('Invalid embedding option')

        # Transformer
        transformer_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=nhead,
                                                       dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(num_features, sequence_length)

        # Classifier head
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, num_features, sequence_length)

        # Convert time series data to embeddings
        if isinstance(self.embedding, nn.Linear):
            # Linear embedding, reshape is needed to match dimensions
            x = x.permute(0, 2, 1)  # (batch_size, sequence_length, num_features)
            x = self.embedding(x)  # (batch_size, sequence_length, num_features)
        elif isinstance(self.embedding, nn.Sequential):
            # Conv1D embedding
            #print(x.shape)
            x = x.unsqueeze(1)
            #print(x.shape)
            x = self.embedding(x)  # (batch_size, num_features, sequence_length)
            x = x.permute(0, 2, 1)  # (batch_size, sequence_length, num_features)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through Transformer
        x = self.transformer_encoder(x)

        # Pool the outputs to a single vector per sample
        x = x.mean(dim=1)

        # Classifier
        logits = self.classifier(x)

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

        logits = self(data) # 32, 512, 8 => 32,num classes
        target_sq = target.squeeze()

        loss = F.cross_entropy(logits, target_sq)
        acc = self.val_accuracy(logits, target_sq)

        # Logging
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log('val_accuracy', acc)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Usage:
# model = TimeSeriesTransformer(num_features=1, num_classes=10, sequence_length=128, embedding_option='conv1d')
# x = torch.randn(32, 1, 128) # example input batch
# out = model(x)
