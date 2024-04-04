import torch.nn as nn
import pickle
import os
import pandas as pd
from cnn_architecture import Net
import torch
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from sklearn.model_selection import train_test_split


def convert_to_3d_tensors(radar_return, target_size=(100, 50)):

    img = Image.fromarray(radar_return)
    # Resize the image while preserving aspect ratio
    img = img.resize(target_size, resample=Image.BILINEAR)
    # Convert the PIL Image to NumPy array
    resized_img = np.array(img)
    return resized_img


def combine_dataframes(data_dir: str) -> pd.DataFrame:
    dataframes = []
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
                concatenated_radar = convert_to_3d_tensors(radar_return).astype('float32')
                # Create a DataFrame with concatenated radar and object_id
                df = pd.DataFrame({'radar_return': [concatenated_radar], 'object_id': [object_id]})
                # Append the DataFrame to the list
                dataframes.append(df)

    # Concatenate all DataFrames into one
    combined_df = pd.concat(dataframes[::50], ignore_index=True)
    #combined_df = normalize_radar_return_column(combined_df)
    # print(normalize_radar_return_column(combined_df))

    return combined_df

def get_tensor_data(combined_df: pd.DataFrame):
    radar_data = torch.tensor(np.stack(combined_df['radar_return'].values), dtype=torch.float32)
    radar_data = radar_data.unsqueeze(1)
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

    cnn = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cnn.parameters(), lr=0.0001, weight_decay=0.01)

    pbar = tqdm(total=50, desc='Epoch:')
    cnn.train()
    for epoch in range(50):  # loop over the dataset multiple times

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnn(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # print statistics
        # print(loss.item())

        _, predicted = torch.max(outputs.detach().cpu(), 1)
        accuracy = (predicted == y_train).sum().item() / y_train.shape[0]

        pbar.set_description(f'Epoch: {epoch+1}')
        pbar.set_postfix(acc=accuracy, loss=loss.item())
        pbar.update(1)

    cnn.eval()
    with torch.no_grad():
        y_pred = cnn(X_test)
        _, y_pred = torch.max(y_pred.detach().cpu(), 1)
        accuracy = (y_pred == y_test).sum().item() / y_test.shape[0]
    print("Test acc:", accuracy)



