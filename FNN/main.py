import torch
import torch.nn as nn
import pickle
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from sklearn.model_selection import train_test_split
from fnn_architecture import FNN
from RadarDataset import RadarDataset
from torch.utils.data import Dataset, DataLoader


def process_radar_return(radar_return):
    # Ensure the radar return has shape (num_rows, num_cols)
    assert len(radar_return.shape) == 2

    # Define the target length of 512
    target_length = 512

    # Randomly select the starting index for the sequence
    start_index = np.random.randint(0, radar_return.shape[1] - 5)

    # Select the subsequent 4 indices to form a sequence of 5 adjacent pulses
    selected_pulses = np.arange(start_index, start_index + 6)

    # Concatenate selected pulses along rows
    concatenated_pulses = np.concatenate([radar_return[:, pulse] for pulse in selected_pulses], axis=0)

    # Take np.abs to convert complex numbers to real numbers
    epsilon = 1e-10
    processed_radar_return = 10 * np.log10((concatenated_pulses**2)+epsilon)

    # Ensure the processed radar return has length 512
    if processed_radar_return.shape[0] > target_length:
        # If length is greater than 512, truncate the vector
        processed_radar_return = processed_radar_return[:target_length]
    elif processed_radar_return.shape[0] < target_length:
        # If length is less than 512, pad with zeros
        print("error: default length is shorter than min length")
        # min_value = np.min(processed_radar_return)
        # pad_value = -200
        # processed_radar_return = np.pad(processed_radar_return,
        #                                 (0, target_length - processed_radar_return.shape[0]),
        #                                 mode='constant', constant_values=pad_value)

    return processed_radar_return

def combine_dataframes(data_dir: str) -> pd.DataFrame:
    dataframes = []
    print("creating the dataframe...")
    for filename in tqdm(os.listdir(data_dir)):
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
    combined_df = pd.concat(dataframes[::50], ignore_index=True)
    # combined_df = normalize_radar_return_column(combined_df)

    return combined_df

def get_tensor_data(combined_df: pd.DataFrame):
    print(combined_df['radar_return'].shape)
    radar_data = torch.tensor(np.stack(combined_df['radar_return'].values), dtype=torch.float32)
    radar_data = radar_data.unsqueeze(1)
    print(radar_data.shape)
    object_ids = combined_df['object_id'].values
    # Use LabelEncoder to convert object_ids to numerical labels
    label_encoder = LabelEncoder()
    object_ids_encoded = label_encoder.fit_transform(object_ids)

    object_ids_encoded = torch.tensor(object_ids_encoded, dtype=torch.uint8)

    return radar_data, object_ids_encoded


if __name__ == "__main__":
    data_dir = "../First_data"
    combined_df = combine_dataframes(data_dir)

    X, y = get_tensor_data(combined_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)


    fnn = FNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(fnn.parameters(), lr=0.0001, weight_decay=0.01)
    num_epochs = 50
    pbar = tqdm(total=50, desc='Epoch:')
    fnn.train()
    for epoch in range(num_epochs):
        # Training phase
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = fnn(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.detach().cpu(), 1)
        accuracy = (predicted == y_train).sum().item() / y_train.shape[0]

        pbar.set_description(f'Epoch: {epoch + 1}')
        pbar.set_postfix(acc=accuracy, loss=loss.item())
        pbar.update(1)

    fnn.eval()
    with torch.no_grad():
        y_pred = fnn(X_test)
        _, y_pred = torch.max(y_pred.detach().cpu(), 1)
        accuracy = (y_pred == y_test).sum().item() / y_test.shape[0]
    print("Test acc:", accuracy)
