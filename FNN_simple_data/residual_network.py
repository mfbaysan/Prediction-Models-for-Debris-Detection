import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import torchmetrics



def nan_hook(module, input, output):
    if torch.isnan(output).any():
        raise RuntimeError(f"NaN detected in {module.__class__.__name__} with input {input}")


def xavier_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def he_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class NoiseLayer(nn.Module):
    def __init__(self, noise_stddev=0.1):
        super(NoiseLayer, self).__init__()
        self.noise_stddev = noise_stddev

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.noise_stddev
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_noise=False, noise_stddev=0.1,
                 activation='relu'):
        super(ResidualBlock, self).__init__()
        self.use_noise = use_noise
        self.activation_func = nn.SiLU() if activation == 'silu' else nn.ReLU(inplace=True)

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.noise = NoiseLayer(noise_stddev) if use_noise else nn.Identity()

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_func(out)
        out = self.noise(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.activation_func(out)
        return out


class ResidualNetwork(pl.LightningModule):
    def __init__(self, res_layers_per_block, num_classes=5, activation='relu', use_noise=False, noise_stddev=0.1,
                 apply_he=False, fc_units=32, in_out_channels=None):
        super(ResidualNetwork, self).__init__()

        self.in_channels = 1
        self.activation = activation
        self.use_noise = use_noise
        self.noise_stddev = noise_stddev
        self.fc_units = fc_units
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        # Set default in_out_channels if not specified
        if in_out_channels is None:
            in_out_channels = [[32, 64], [64, 128], [128, 256], [256, 512]]

        self.pre_layers = nn.Sequential(
            nn.Conv1d(self.in_channels, in_out_channels[0][0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(in_out_channels[0][0]),
            nn.SiLU() if activation == 'silu' else nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Validate that in_out_channels and res_layers_per_block are of the same length
        assert len(in_out_channels) == len(
            res_layers_per_block), "in_out_channels and res_layers_per_block must be of the same length"

        # Dynamically create layers
        self.res_layers = nn.ModuleList()
        for i, (in_channels, out_channels) in enumerate(in_out_channels):
            stride = 2 if i > 0 else 1
            self.res_layers.append(self._make_res_layer(in_channels, out_channels, kernel_size=3, stride=stride,
                                                        num_blocks=res_layers_per_block[i]))

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.feature_extractor = nn.Sequential(
            nn.Linear(in_out_channels[-1][-1], self.fc_units),
            nn.BatchNorm1d(self.fc_units),
            nn.SiLU() if activation == 'silu' else nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(self.fc_units, num_classes)

        if apply_he:
            self.apply(he_init)

    def _make_res_layer(self, in_channels, out_channels, kernel_size, stride, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, kernel_size, stride, padding=1, use_noise=self.use_noise,
                                    noise_stddev=self.noise_stddev, activation=self.activation))
        for _ in range(1, num_blocks):
            layers.append(
                ResidualBlock(out_channels, out_channels, kernel_size, stride=1, padding=1, use_noise=self.use_noise,
                              noise_stddev=self.noise_stddev, activation=self.activation))
        return nn.Sequential(*layers)

    def forward(self, x, return_features: bool = False):
        out = self.pre_layers(x)

        for layer in self.res_layers:
            out = layer(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        features = self.feature_extractor(out)

        """
        fetures dim - batch x fc_units
        """
        logits = self.classifier(features)
        """
        logits dim - batch x class_num
        """

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

# Example instantiation (for example, 5 layers):
# model = ResidualNetwork([2, 2, 2, 2, 2], num_classes=4, activation='silu', use_noise=True, noise_stddev=0.1, fc_neurons=64, in_out_channels=[(32, 32), (32, 64), (64, 128), (128, 256), (256, 512)])
