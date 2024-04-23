import os
from scipy.io import loadmat

root_dir = "../New_data"  # Update with your dataset directory

files_with_too_many_rows = []

for class_name in os.listdir(root_dir):
    class_dir = os.path.join(root_dir, class_name)
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        mat_file = loadmat(file_path)
        objects_echo = mat_file['objects_echo']
        num_rows = objects_echo.shape[0]
        if num_rows > 2000:
            files_with_too_many_rows.append((file_path, num_rows))

# Print files with more than 2000 rows
for file_path, num_rows in files_with_too_many_rows:
    print(f"File: {file_path}, Number of rows: {num_rows}")
