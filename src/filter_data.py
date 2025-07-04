import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def filter_masks_by_area_and_component_count(mask_directory: str, area_threshold: int, max_components: int) -> list:
    """
    Scans a directory of wound mask images and returns a list of image names
    that meet specific quality criteria.

    Args:
        mask_directory (str): The path to the folder containing the wound mask images.
        area_threshold (int): The minimum pixel area for the largest component.
        max_components (int): The maximum number of allowed components (e.g., 1 for a single wound).

    Returns:
        list: A list of ImageIDs (filenames without extension) that meet the criteria.
    """
    
    if not os.path.isdir(mask_directory):
        print(f"Error: Directory not found at '{mask_directory}'")
        return []

    print(f"Scanning directory: {mask_directory}")
    
    try:
        image_files = [f for f in os.listdir(mask_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(image_files)} total masks to check.")
    except Exception as e:
        print(f"Error reading directory: {e}")
        return []

    valid_image_ids = []

    for filename in tqdm(image_files, desc="Processing Masks"):
        try:
            image_path = os.path.join(mask_directory, filename)
            mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                continue

            # Find all connected components in the mask.
            # The first label (0) is always the background.
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            # The number of actual wound components is num_labels - 1
            num_wound_components = num_labels - 1
            
            # --- NEW FILTERING LOGIC ---
            # Condition 1: Check if the number of components is what we want (e.g., exactly 1).
            if num_wound_components != max_components:
                continue # Skip this image if it has more or less than the desired number of wounds.

            # If we pass the first check, we now check the area of that single component.
            # We get the area of the first (and only) component, ignoring the background.
            component_area = stats[1, cv2.CC_STAT_AREA]

            # Condition 2: Check if the component's area exceeds our threshold.
            if component_area > area_threshold:
                image_id = os.path.splitext(filename)[0]
                valid_image_ids.append(image_id)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
            
    return valid_image_ids

# --- Main execution block ---
if __name__ == '__main__':
    # --- Configuration ---
    WOUND_MASK_PATH = './data/wound_masks' 
    PIXEL_AREA_THRESHOLD = 15000
    # We want to keep only masks that have EXACTLY ONE wound component.
    MAX_WOUND_COMPONENTS = 1
    
    MANIFEST_PATH = './metadata/image_index.csv'
    FILTERED_MANIFEST_PATH = './metadata/image_index_filtered.csv'

    # --- Step 1: Run the filtering function ---
    filtered_ids = filter_masks_by_area_and_component_count(
        WOUND_MASK_PATH, 
        PIXEL_AREA_THRESHOLD,
        MAX_WOUND_COMPONENTS
    )

    print("\n--- Filtering Complete ---")
    print(f"Found {len(filtered_ids)} images with exactly {MAX_WOUND_COMPONENTS} wound component AND area > {PIXEL_AREA_THRESHOLD} pixels.")
    
    # --- Step 2: Update the manifest file ---
    if filtered_ids:
        try:
            print(f"\nReading original manifest from: {MANIFEST_PATH}")
            manifest_df = pd.read_csv(MANIFEST_PATH)
            
            id_column_name = manifest_df.columns[0]
            
            filtered_df = manifest_df[manifest_df[id_column_name].isin(filtered_ids)]
            
            filtered_df.to_csv(FILTERED_MANIFEST_PATH, index=False)
            
            print(f"\nSuccessfully created filtered manifest with {len(filtered_df)} entries.")
            print(f"New manifest saved to: {FILTERED_MANIFEST_PATH}")

        except FileNotFoundError:
            print(f"\nError: Original manifest file not found at '{MANIFEST_PATH}'.")
        except Exception as e:
            print(f"\nAn error occurred while updating the manifest: {e}")

