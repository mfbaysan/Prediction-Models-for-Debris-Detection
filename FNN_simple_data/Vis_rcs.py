import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Define the directory containing the .mat files
folder_path = "C:/Users/fatih/Desktop/rcs/Plane"

# List all .mat files in the directory
mat_files = [file for file in os.listdir(folder_path) if file.endswith('.mat')]

# Define the minimum and maximum values for the colormap
vmin = -100  # Set to None to auto-scale
vmax = 5  # Set to None to auto-scale

# Iterate through each .mat file
total_matrix = None
average_matrix_sum = 0

# Iterate through each .mat file
for file in mat_files:
    # Load the .mat file
    mat_data = loadmat(os.path.join(folder_path, file))

    # Extract the matrix named 'RCS_timetrial'
    matrix = mat_data['RCS_timetrial']

    # Add the matrix to the accumulator
    if total_matrix is None:
        total_matrix = matrix
    else:
        total_matrix += matrix
    average_matrix_sum += np.mean(matrix)

# Calculate the average matrix
average_matrix = total_matrix / len(mat_files)

average_of_average_matrix = average_matrix_sum / len(mat_files)

# Plot the average matrix
plt.figure(figsize=(8, 6))
plt.imshow(average_matrix, cmap='viridis', vmin=vmin, vmax=vmax)  # You can change the colormap as needed
plt.title(f"Rocket Average Matrix (Average Value: {average_of_average_matrix:.2f})")
plt.colorbar()
# Print or display the average value of the average matrix (optional)
print(f"Average value of the average matrix: {average_of_average_matrix}")

plt.show()

