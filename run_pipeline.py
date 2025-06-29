import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse

# --- Import project's custom functions ---

from data_loader import data_loader
from preprocessing import (
    depth_corrction_for_body_curvature, 
    unroll_periwound_to_image
)
from feature_extraction import (
    calculate_depth_profiles,
    extract_features_from_profile
)
from utils import save_features_to_csv


def main(args):
    """
    The main function to run the entire feature extraction pipeline.
    """
    print("Starting batch feature extraction pipeline...")
    
    image_ids_to_process = []
    if args.image_id:
        image_ids_to_process = [args.image_id]
        print(f"Running in single-image mode for ID: {args.image_id}")
    else:
        print("Running in batch mode for all images in the manifest.")
        
        # --- Load the list of images to process ---
        try:
            # Read the manifest file to get the list of image IDs
            image_ids_to_process = pd.read_csv(args.manifest_path).iloc[:, 0].tolist()
            print(f"Found {len(image_ids_to_process )} images to process in {args.manifest_path}")
        
        except FileNotFoundError:
            print(f"Error: Manifest file not found at '{args.manifest_path}'. Please check the path.")
            return

    output_filepath = args.output_filepath
    
    # --- Clean up old results to ensure a fresh run ---
    if os.path.exists(output_filepath):
        print(f"Removing old results file: {output_filepath}")
        os.remove(output_filepath)
    
    # --- Loop through all images and process them one by one ---
    for image_id in tqdm(image_ids_to_process, desc="Processing Images"):
        try:
            # ---  Data Loading ---
            data = data_loader(image_id, args.data_path)
            
            # --- Preprocessing ---
            # Correct for body curvature
            _, _, depth_corrected = depth_corrction_for_body_curvature(
                data['wound'], data['body'], data['depth']
            )

            # Convert zeros to NaN before unrolling for accurate profiling
            depth_corrected_nan = np.where(depth_corrected != 0, depth_corrected, np.nan)
            
            # Rectify (unroll) the peri-wound area
            # Note: The 'd1' value is the number of eroded strips, which marks the border
            rect_depth, (d1, _) = unroll_periwound_to_image(
                depth_corrected_nan, data['wound'], iterations=args.unroll_iterations
            )

            # --- Profile Calculation ---
            mean_profile, _ = calculate_depth_profiles(rect_depth)
            
            # Check if the mean profile is empty. if so, skip this image
            if mean_profile.size == 0:
                print(f"Skipping image {image_id}: could not generate a valid profile.")
                continue

            # --- Feature Extraction ---
            features = extract_features_from_profile(mean_profile, d1, transition_width=args.transision_width)
            
            # --- Save Results for This Image ---
            save_features_to_csv(image_id, features, output_filepath)
            
        except Exception as e:
            # Report errors if any but continue
            print(f"\\n---! FAILED to process image {image_id}. Error: {e} !---")

    print(f"\\nPipeline complete. All features saved to {output_filepath}")


if __name__ == '__main__':
    #--- Command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Run the wound border feature extraction pipeline.")
    
    # --- Define command-line arguments ---
    #'-LcjI5OGPRxeAj_JwEb0.-LcjISTwgjaAZF_xYXxT.-LjFh4DROm9CaNMN6ic9'
    parser.add_argument('--image_id', type=str, default=None, 
                        help='Process a single ImageID instead of the full manifest.')
    
    parser.add_argument('--data_path', type=str, default='./data', 
                        help='Path to the root data folder.')
    
    parser.add_argument('--manifest_path', type=str, default='./metadata/image_index.csv', 
                        help='Path to the CSV file containing ImageIDs.')
                        
    parser.add_argument('--output_filepath', type=str, default='./outputs/comprehensive_features.csv', 
                        help='Filepath to save the final CSV of features.')
                        
    parser.add_argument('--unroll_iterations', type=int, default=100, 
                        help='Number of erosion/dilation steps for peri-wound rectification.')
    
    parser.add_argument('--transision_width', type=int, default=50, 
                    help='How wide the transition region is for feature extraction.')
    
    args = parser.parse_args()
    main(args)