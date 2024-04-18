import torch.nn as nn
import pickle
import os
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm


class GrayScaleResnet18(nn.Module):
    def __init__(self, num_classes):
        super(GrayScaleResnet18, self).__init__()

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
    combined_df = pd.concat(dataframes, ignore_index=True) #[::50]
    # combined_df = normalize_radar_return_column(combined_df)
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

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # Training loop
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        for data, labels in tqdm(train_loader):
            data = data.to(device)
            labels = labels.to(device)

            # Clear gradients for optimizer
            optimizer.zero_grad()

            # Forward pass, calculate loss
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Backward pass and update weights
            loss.backward()
            optimizer.step()

            # Track training statistics
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        # Calculate training accuracy and loss
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient calculation for validation
            for data, labels in val_loader:
                data = data.to(device)
                labels = labels.to(device)

                # Forward pass only
                outputs = model(data)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_loss /= len(val_loader)

        # Print and track best validation accuracy
        print(f'Training Accuracy: {train_acc:.4f}, Training Loss: {train_loss:.4f}')
        print(f'Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'New best validation accuracy: {best_val_acc:.4f}')


if __name__ == "__main__":
    data_dir = "../First_data"
    combined_df = combine_dataframes(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create custom dataset and data loaders

    image_data, labels = get_tensor_data(combined_df)  # Your logic to load your dataframe

    X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2,
                                                        random_state=42, stratify=labels)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    num_classes = 10  # Your number of classes

    # Create PyTorch Dataset and DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # Don't shuffle validation data

    # Define model, loss function, and optimizer
    model = GrayScaleResnet18(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Adjust learning rate as needed

    # Train the model
    train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100)
