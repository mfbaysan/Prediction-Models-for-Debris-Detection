# This is a sample Python script.
import torch.nn.functional as F
import pytorch_lightning as pl
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, random_split
from CustomDataModule import CustomDataModule
from LSTMModel import LSTMModel
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import summarize
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim





def process_radar_return(radar_return):
    # Ensure the radar return has shape (num_rows, num_cols)
    assert len(radar_return.shape) == 2

    # Define the target length of 512
    target_length = 512

    # Randomly select the starting index for the sequence
    start_index = np.random.randint(0, radar_return.shape[0] - 4)

    # Select the subsequent 4 indices to form a sequence of 5 adjacent pulses
    selected_pulses = np.arange(start_index, start_index + 5)

    # Concatenate selected pulses along rows
    concatenated_pulses = np.concatenate([radar_return[pulse, :] for pulse in selected_pulses], axis=0)

    # Take np.abs to convert complex numbers to real numbers
    epsilon = 1e-10
    processed_radar_return = 10 * np.log10(((np.abs(concatenated_pulses))**2)+epsilon)

    # Ensure the processed radar return has length 512
    if processed_radar_return.shape[0] > target_length:
        # If length is greater than 512, truncate the vector
        processed_radar_return = processed_radar_return[:target_length]
    elif processed_radar_return.shape[0] < target_length:
        # If length is less than 512, pad with zeros
        min_value = np.min(processed_radar_return)
        pad_value = -200
        processed_radar_return = np.pad(processed_radar_return,
                                        (0, target_length - processed_radar_return.shape[0]),
                                        mode='constant', constant_values=pad_value)

    return processed_radar_return


def normalize_radar_return_column(combined_df):

    # Extract radar_return column from combined_df
    radar_data = combined_df['radar_return']
    print(combined_df['radar_return'].shape)

    min_before = np.min([np.min(row) for row in radar_data])
    max_before = np.max([np.max(row) for row in radar_data])
    print("Min value before normalization:", min_before)
    print("Max value before normalization:", max_before)

    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = [scaler.fit_transform(np.array(row).reshape(-1, 1)).flatten().tolist() for row in radar_data]

    min_after = np.min([np.min(row) for row in normalized_data])
    max_after = np.max([np.max(row) for row in normalized_data])
    print("Min value after normalization:", min_after)
    print("Max value after normalization:", max_after)

    # Reshape the normalized data back to its original shape
    combined_df['radar_return'] = normalized_data
    print(combined_df['radar_return'].shape)
    return combined_df





if __name__ == '__main__':

    data_dir = 'Overfit_data'  # Change this to your data folder path
    dataframes = []

    # set device!
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iterate over each pickle file
    print("creating the dataframe...")
    for filename in os.listdir(data_dir):
        if filename.endswith('.pickle'):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'rb') as f:
                # Load data from pickle file
                data = pickle.load(f)
                # Extract radar_return and object_id
                radar_return = data['radar_return']
                object_id = data['object_id']
                # Concatenate radar_return along the columns
                concatenated_radar = process_radar_return(radar_return).astype('float32')
                # Create a DataFrame with concatenated radar and object_id
                df = pd.DataFrame({'radar_return': [concatenated_radar], 'object_id': [object_id]})
                # Append the DataFrame to the list
                dataframes.append(df)

    # Concatenate all DataFrames into one
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = normalize_radar_return_column(combined_df)


    # Encode object_id using LabelEncoder
    label_encoder = LabelEncoder()
    combined_df['object_id'] = label_encoder.fit_transform(combined_df['object_id']).astype('float32')
    print(combined_df.dtypes)
    print(combined_df.shape)
    # Example usage:
    data_module = CustomDataModule(combined_df, device=device, batch_size=32)

    input_size = 1  #512  # Size of concatenated radar_return
    hidden_size = 64
    num_classes = len(label_encoder.classes_)
    model = LSTMModel(input_size, hidden_size, num_classes).to(device)
    summarize(model, max_depth=-1)
    print(model)

    # Train the model
    print("training the model...")
    wandb_logger = WandbLogger(project='radar_classification')
    wandb_logger.watch(model, log="all")
    trainer = pl.Trainer(max_epochs=100,
                         accelerator='gpu' if device == torch.device('cuda') else "cpu",
                         devices=1 if device == torch.device('cuda') else "cpu",
                         logger=wandb_logger)
    trainer.fit(model, data_module)

