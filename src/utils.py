import pandas as pd
from pathlib import Path

import pandas as pd
from pathlib import Path
import os # We'll use os to check if the file exists

def save_features_to_csv(features_dict: dict, output_filepath: str):
    """
    Appends a dictionary of features for a single image to a CSV file.

    If the CSV file does not exist, it will be created with headers.
    If it exists, a new row will be appended without headers.

    Args:
        features_dict (dict): A dictionary of features for a single image.
        output_filepath (str): The full path where the CSV file will be saved.
    """
    if not features_dict:
        print("Warning: Feature dictionary is empty. Nothing to save.")
        return

    # Convert the single dictionary to a pandas DataFrame with one row
    features_df = pd.DataFrame([features_dict])
    
    output_path = Path(output_filepath)
    
    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if the file already exists to decide whether to write the header
    write_header = not output_path.is_file()

    # Append to the CSV file
    # mode='a' means append. header=write_header writes headers only if the file is new.
    features_df.to_csv(output_path, mode='a', header=write_header, index=False)