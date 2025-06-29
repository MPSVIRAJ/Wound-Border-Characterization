import pandas as pd
from pathlib import Path
import os 

def save_features_to_csv(ImageID: str, features_dict: dict, output_filepath: str):
    """
    Appends a dictionary of features for a single image to a CSV file,
    ensuring the 'id' column is first.

    If the CSV file does not exist, it will be created with headers.
    If it exists, a new row will be appended without headers.
    """
    if not features_dict:
        print("Warning: Feature dictionary is empty. Nothing to save.")
        return
    # Add the ImageID to the features dictionary
    features_dict['id'] = ImageID  

    # Create the desired column order with 'id' first
    column_order = ['id'] + [key for key in features_dict if key != 'id']
    
    # Create the DataFrame with the specified column order
    features_df = pd.DataFrame([features_dict], columns=column_order)
    
    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if the file exists to determine if we need to write the header
    write_header = not output_path.is_file()

    # Append to the CSV file
    features_df.to_csv(output_path, mode='a', header=write_header, index=False)