import plotly.graph_objs as go
import numpy as np
import random
import os

from Model_FNN import CustomDataset

def plot_samples_plotly(dataset, num_classes=5, num_samples_per_class=10):
    # Create a Plotly figure
    fig = go.Figure()

    # Get unique class labels from the dataset
    unique_labels = np.unique(dataset.labels)

    # Select random classes from unique_labels
    selected_classes = random.sample(unique_labels.tolist(), num_classes)

    # Define colors for each class
    colors = ['blue', 'green', 'red', 'orange', 'purple']

    # Plot samples for each selected class
    for i, class_label in enumerate(selected_classes):
        # Filter dataset for samples belonging to the selected class
        class_indices = np.where(dataset.labels == class_label)[0]  # Get indices of samples for this class
        selected_samples = random.sample(list(class_indices), min(num_samples_per_class, len(class_indices)))

        # Assign color for the class
        color = colors[i % len(colors)]  # Use modulo to ensure the colors repeat if there are more classes than colors

        # Plot each selected sample
        for sample_idx in selected_samples:
            sample = dataset.data[sample_idx]
            sample_dBsm = 10 * np.log10((sample.squeeze())**2)  # Convert to dBsm
            fig.add_trace(go.Scatter(y=sample_dBsm, mode='lines', name=f'Class {int(class_label)}', line=dict(color=color)))

    # Update layout
    fig.update_layout(title='Samples from Selected Classes',
                      xaxis_title='Time',
                      yaxis_title='Amplitude (dBsm)',
                      legend_title='Class')

    # Show plot
    fig.show()

# Define root directory
root_dir = "../data_5cm"

# Assuming you have instantiated your CustomDataset
dataset = CustomDataset(root_dir=root_dir)

# Visualize plots for 5 different classes with 10 samples each
plot_samples_plotly(dataset, num_classes=5, num_samples_per_class=2)
