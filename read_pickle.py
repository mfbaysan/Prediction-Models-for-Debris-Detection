import pickle
import os

def get_pickle_files(folder_path):
    """
    Get all the .pickle files from a folder.

    Args:
    - folder_path (str): Path to the folder containing .pickle files

    Returns:
    - List of .pickle files
    """
    pickle_files = [file for file in os.listdir(folder_path) if file.endswith('.pickle')]
    return pickle_files


def read_pickle_files(folder_path):
    """
    Process all .pickle files in a folder.

    Args:
    - folder_path (str): Path to the folder containing .pickle files

    Returns:
    - DataFrame containing all the samples
    """
    # Get .pickle files
    pickle_files = get_pickle_files(folder_path)
    dataframes = []
    if pickle_files:
        # Process only the first file
        first_file_path = os.path.join(folder_path, pickle_files[0])
        with open(first_file_path, 'rb') as f:
            file_data = pickle.load(f)
            radar_return = file_data['radar_return']
            # Check the data type of the first element
            print(type(radar_return))
            print(radar_return.shape)
            print(radar_return)
    else:
        print("No .pickle files found in the folder.")

folder_path = 'Overfit_data'
read_pickle_files(folder_path)
