# File: src/io_utils.py
import os
import requests # You might need to install this: pip install requests
from scipy.io import loadmat # You might need to install this: pip install scipy
import numpy as np

def ensure_dir(directory):
    """
    Ensures that a directory exists; if not, it creates it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_face_data(data_dir='../data', file_name='face.mat'):
    """
    Loads the face.mat dataset. Downloads it if it doesn't exist.

    Args:
        data_dir (str): The directory where the data file is stored or should be stored.
                        Relative to the location of this script (src/).
        file_name (str): The name of the data file.

    Returns:
        numpy.ndarray or None: A numpy array of shape (n_samples, n_features),
                                or None if loading fails.
    """
    # Construct the full path to the data file
    # os.path.dirname(__file__) gives the directory of the current script (src/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_dir = os.path.join(base_dir, data_dir)
    mat_file_path = os.path.join(full_data_dir, file_name)

    # Ensure the data directory exists
    ensure_dir(full_data_dir)

    # Check if the file exists, if not, download it
    if not os.path.exists(mat_file_path):
        print(f"{file_name} not found. Downloading...")
        # URL provided by the user
        url = 'https://github.com/yao-lab/yao-lab.github.io/raw/master/data/face.mat'
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes
            with open(mat_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {file_name} to {full_data_dir}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {file_name}: {e}")
            return None
        except IOError as e:
            print(f"Error saving {file_name}: {e}")
            return None

    # Load the .mat file
    try:
        # loadmat returns a dictionary
        mat_data = loadmat(mat_file_path)
        # We need to know the variable name inside the .mat file.
        # Common names might be 'X', 'Y', 'data', or based on the filename 'face_data'.
        # Let's inspect the keys to find the data array.
        print(f"Keys in {file_name}: {mat_data.keys()}")

        # --- !!! Assumption Alert !!! ---
        # Assume the data is stored under the key 'X'.
        # You might need to adjust 'X' based on the printed keys!
        data_key = 'Y'
        if data_key in mat_data:
            face_data = mat_data[data_key]
             # Data might be (features, samples), transpose if necessary
            if face_data.shape[0] != 33: # Assuming 33 samples
                face_data = face_data.T
            # Ensure data is float for consistency
            return face_data.astype(np.float64)
        else:
             # If 'X' is not found, try common alternatives or raise an error
             print(f"Error: Could not find the data array key '{data_key}' in {file_name}.")
             # Check other keys like 'Y' or others printed above
             # Example check:
             # potential_keys = [k for k in mat_data if not k.startswith('__')]
             # if len(potential_keys) == 1:
             #    face_data = mat_data[potential_keys[0]]
             #    if face_data.shape[0] != 33: face_data = face_data.T
             #    return face_data.astype(np.float64)

             print("Please inspect the keys printed above and adjust 'data_key' in load_face_data function.")
             return None

    except Exception as e:
        print(f"Error loading {mat_file_path}: {e}")
        return None