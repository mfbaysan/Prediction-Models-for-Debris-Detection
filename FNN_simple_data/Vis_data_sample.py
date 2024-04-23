import os
import numpy as np
import matplotlib.pyplot as plt
import random

from Model_FNN import CustomDataset


def visualize_samples(dataset, output_dir, num_samples=20):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    for i in range(num_samples):
        idx = random.randint(0, len(dataset)-1)
        sample, label = dataset[idx]
        save_sample_as_png(sample, label, output_dir, i)

def save_sample_as_png(sample, label, output_dir, index):
    plt.figure(figsize=(8, 4))
    sample_dBsm = 10 * np.log10((sample.numpy().squeeze() + 1e-10)**2) # Add a small offset before taking the logarithm
    plt.plot(sample_dBsm)
    plt.title(f'Class: {label.item()}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude (dBsm)')
    class_id = label.item()
    filename = f"sample_{index}_class_{class_id}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()

root_dir = "../New_data"

# Assuming you have instantiated your CustomDataset
dataset = CustomDataset(root_dir=root_dir)

# Define output directory
output_dir = "D:\model_implementation\FNN_simple_data\sc_sample"

# Visualize and save 20 random samples from the dataset
visualize_samples(dataset, output_dir, num_samples=100)
