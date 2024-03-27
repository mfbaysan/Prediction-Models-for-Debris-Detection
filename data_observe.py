import torch
import matplotlib.pyplot as plt

from CustomDataModule import CustomDataModule

# Assuming data_loader is your DataLoader instance
data_dir = 'First_data'  # Change this to your data folder path
data_module = CustomDataModule(data_dir=data_dir)

data_module.prepare_data()
data_module.setup()

# Get the data loader
data_loader = data_module.train_dataloader()

# Iterate over batches in the data loader
for batch in data_loader:
    inputs, targets = batch

    # Print or visualize the inputs and targets
    input_data, target_data = inputs[0], targets[0]

    # Print shapes
    print("Input data shape:", input_data.shape)
    print("Target data shape:", target_data.shape)


