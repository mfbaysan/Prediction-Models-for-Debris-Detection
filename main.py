# This is a sample Python script.
import torch
import torch.nn as nn
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


def process_radar_return(radar_return):
    # Ensure the radar return has shape (num_rows, num_cols)
    assert len(radar_return.shape) == 2

    # Define the target length of 512
    target_length = 512

    # Concatenate around 5 pulses to obtain a sequence of approximately 512 sequential readings
    selected_pulses = np.random.choice(radar_return.shape[0],
                                       size=min(target_length // 100 + 1, radar_return.shape[0]), replace=False)

    # Sort selected pulses
    selected_pulses.sort()

    # Concatenate selected pulses along rows
    concatenated_pulses = np.concatenate([radar_return[pulse, :] for pulse in selected_pulses], axis=0)

    # Take np.abs to convert complex numbers to real numbers
    processed_radar_return = np.abs(concatenated_pulses)

    # Ensure the processed radar return has length 512
    if processed_radar_return.shape[0] > target_length:
        # If length is greater than 512, truncate the vector
        processed_radar_return = processed_radar_return[:target_length]
    elif processed_radar_return.shape[0] < target_length:
        # If length is less than 512, pad with zeros
        processed_radar_return = np.pad(processed_radar_return,
                                        (0, target_length - processed_radar_return.shape[0]), mode='constant')

    return processed_radar_return


if __name__ == '__main__':

    data_dir = 'First_data'  # Change this to your data folder path
    dataframes = []

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

    # Encode object_id using LabelEncoder
    label_encoder = LabelEncoder()
    combined_df['object_id'] = label_encoder.fit_transform(combined_df['object_id']).astype('float32')
    print(combined_df.dtypes)
    # Example usage:
    data_module = CustomDataModule(combined_df)

    input_size = 512  # Size of concatenated radar_return
    hidden_size = 64
    num_classes = len(label_encoder.classes_)
    model = LSTMModel(input_size, hidden_size, num_classes)

    # Train the model
    print("training the model...")
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, data_module)
