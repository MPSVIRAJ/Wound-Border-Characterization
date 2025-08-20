"""
Main execution script for the wound border characterization pipeline.

This file serves as the command-line interface for the entire machine learning pipeline.
It orchestrates the flow of data through a series of configurable stages: `extract`,
`cluster`, `train`, and `predict`. The script is responsible for loading the
configuration, managing file paths, and calling the appropriate functions for each
pipeline step.

Functions:
    - `perform_cleanup`: A utility function to remove old output files before a fresh run.
    - `run_feature_extraction_step`: Executes the feature extraction stage of the pipeline.
    - `run_clustering_step`: Executes the dimensionality reduction and clustering stage.
    - `run_training_step`: Executes the model training and evaluation stage.
    - `run_prediction_step`: Executes the prediction stage using a pre-trained model.
    - `main`: The main entry point for the command-line tool.

Usage:
    To run a specific stage of the pipeline from the command line, use the following syntax:

    `python run_pipeline.py [stage] --optional_arguments`

    - `extract`: Runs the mask filtering and feature extraction process.
        - `--image_id [ID]`: (Optional) Process a single image and display plots for visualization.
    - `cluster`: Runs the dimensionality reduction and clustering process on existing features.
    - `train`: Trains a classifier on the clustered data and evaluates its performance.
    - `predict`: Uses a trained model to classify new features.
        - `--input_features [PATH]`: (Optional) Path to a CSV of new features. If omitted, `extract` is run first.
    - `all`: Runs the entire pipeline from `extract` to `train`.
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import argparse
import shutil

# --- Project Imports ---
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config_manager import Config
from src.data_loader import data_loader, load_and_clean_features, load_cluster_groups
from src.plotting import (
    plot_embedding, plot_clusters, plot_feature_distributions_by_cluster,
    plot_cluster_image_grid, plot_confusion_matrix, plot_feature_importances,
    plot_initial_data, show_unrolled_strip, plot_depth_profile, plot_profiles_and_fits
)
from src.logging_setup import setup_logging
from src.utils import (
    save_dataframe_to_csv, save_features_to_csv, save_cluster_assignments,
    generate_cluster_summary, generate_cluster_profiles
)
from src.preprocessing import validate_wound_masks, depth_corrction_for_body_curvature, unroll_periwound_to_image
from src.feature_extraction import calculate_depth_profiles, extract_features_from_profile
from src.clustering import apply_pacmap, perform_hdbscan_clustering
from src.classification import (
    label_clustered_data, prepare_training_data_splits, train_classifier,
    evaluate_classifier, get_feature_importances, predict_wound_border_type
)


setup_logging(log_level="INFO", log_file_path=Path("./outputs/run_pipeline.log"))
logger = logging.getLogger(__name__)

# --- Custom Cleanup Function ---

def perform_cleanup(paths: Dict[str, Path]):
    """
    Removes all existing output files, except for the trained model.

    Args:
        paths (Dict[str, Path]): 
            A dictionary of all relevant file and directory paths.

    Returns:
        None
    
    Relationships:
        - Dependencies:
            - `os`: For removing files.
            - `pathlib`: For path handling.
            - `shutil`: For removing directories.
            - `logging`: For outputting messages.
        - Used by:
            `main` before starting a fresh pipeline run.
    """
    logger.warning("Performing a fresh run. Removing all files from the output directory.")
    # Ensure the base directories exist before attempting to clean them
    paths['base_output_dir'].mkdir(exist_ok=True)
    paths['base_metadata_dir'].mkdir(exist_ok=True)

    if paths['base_output_dir'].exists():
        files_to_remove = [f for f in paths['base_output_dir'].iterdir() if f.is_file()]
        model_path = paths['random_forest_model_path']
        log_file_path = paths['base_output_dir'] / "run_pipeline.log"
        
        if model_path.exists() and model_path in files_to_remove:
            files_to_remove.remove(model_path)
            logger.info(f"Preserving trained model: {model_path}")
            
        if log_file_path.exists() and log_file_path in files_to_remove:
            files_to_remove.remove(log_file_path)
            logger.info(f"Preserving log file: {log_file_path}")

        for file in files_to_remove:
            try:
                os.remove(file)
                logger.debug(f"Removed old output file: {file}")
            except OSError as e:
                logger.error(f"Error removing file {file}: {e}")
        logger.info("All output files, except the trained model and log, have been removed.")

    if paths['filtered_manifest_path'].exists():
        try:
            os.remove(paths['filtered_manifest_path'])
            logger.info(f"Removed old manifest file: {paths['filtered_manifest_path']}")
        except OSError as e:
            logger.error(f"Error removing manifest file {paths['filtered_manifest_path']}: {e}")

# --- Pipeline Stage Functions ---

def run_feature_extraction_step(config: Config, paths: Dict[str, Path], image_ids_to_process: List[str], single_image_mode: bool):
    """
    Executes the feature extraction pipeline stage.

    This function iterates through a list of image IDs, loads the corresponding raw data,
    applies preprocessing steps (depth correction, unrolling), extracts features, and saves
    them to a CSV file. It also supports a single-image visualization mode for debugging.

    Args:
        config (Config): 
            The Config manager instance.
        paths (Dict[str, Path]): 
            A dictionary of all relevant file and directory paths.
        image_ids_to_process (List[str]): 
            A list of image IDs to process.
        single_image_mode (bool): 
            If True, enables visualization of intermediate steps for a single image.

    Returns:
        None

    Relationships:
        - Dependencies:
            - `tqdm`: For displaying a progress bar.
            - `config_manager.Config`: To access configuration parameters.
            - `data_loader.data_loader`: To load raw image data.
            - `preprocessing`: To perform `depth_corrction_for_body_curvature` and `unroll_periwound_to_image`.
            - `feature_extraction`: To perform `calculate_depth_profiles` and `extract_features_from_profile`.
            - `utils.save_features_to_csv`: To save the extracted features.
            - `plotting`: To display plots in single-image mode.
        - Used by:
            `main` to execute the 'extract' and 'all' stages.
    """
    feature_params = config.get_feature_extraction_params()
    comprehensive_features_csv = paths['comprehensive_features_csv']

    logger.info("Starting feature extraction.")
    for image_id in tqdm(image_ids_to_process, desc="Processing Images"):
        try:
            loaded_data = data_loader(image_id, paths['base_data_dir'], config.get_subdirs_params())
            if loaded_data is None: continue
            if single_image_mode: plot_initial_data(loaded_data['image'], loaded_data['wound'], loaded_data['body'], loaded_data['depth'])
            
            _, _, depth_corrected = depth_corrction_for_body_curvature(loaded_data['wound'], loaded_data['body'], loaded_data['depth'])
            depth_corrected_nan = np.where(depth_corrected != 0, depth_corrected, np.nan)
            
            rect_depth, (d1, _) = unroll_periwound_to_image(depth_corrected_nan, loaded_data['wound'], iterations=feature_params['unroll_iterations'])
            
            if single_image_mode:
                rect_rgb, (_, p1) = unroll_periwound_to_image(loaded_data['image'], loaded_data['wound'], iterations=feature_params['unroll_iterations'])
                show_unrolled_strip(rect_depth, rect_rgb, d1, p1, feature_params['unroll_iterations'])
            
            mean_prof, std_profile = calculate_depth_profiles(rect_depth)
            if single_image_mode: plot_depth_profile(mean_prof, std_profile, d1)
            
            if mean_prof.size == 0:
                logger.warning(f"Skipping image {image_id}: could not generate a valid profile.")
                continue
            
            features, smoothed_profile, success = extract_features_from_profile(mean_prof, d1, feature_params)
            
            if single_image_mode and success:
                plot_profiles_and_fits(mean_prof, std_profile, smoothed_profile, features, d1, d1, feature_params['transition_width'])
            
            if success:
                save_features_to_csv(image_id, features, comprehensive_features_csv)
            else:
                logger.warning(f"Feature extraction failed for {image_id}.")
        except Exception as e:
            logger.exception(f"Failed to process image {image_id}.")
    logger.info("Feature extraction complete.")

# --- Unsupervised Clustering Step ---

def run_clustering_step(config: Config, paths: Dict[str, Path]):
    """
    Executes the clustering pipeline stage.

    This function loads the comprehensive features, applies dimensionality reduction
    with PaCMAP, performs HDBSCAN clustering, saves the cluster assignments, and
    generates various plots to visualize the results.

    Args:
        config (Config): 
            The Config manager instance.
        paths (Dict[str, Path]): 
            A dictionary of all relevant file and directory paths.

    Returns:
        None

    Relationships:
        - Dependencies:
            - `config_manager.Config`: To access clustering parameters.
            - `data_loader`: To load and clean feature data.
            - `clustering`: To apply `apply_pacmap` and `perform_hdbscan_clustering`.
            - `utils`: To save cluster assignments, summary, and profiles.
            - `plotting`: To visualize the clustering results.
        - Used by:
            `main` to execute the 'cluster' and 'all' stages.
    """
    logger.info("Starting clustering step.")
    clustering_params = config.get_clustering_params()
    
    if not paths['comprehensive_features_csv'].exists():
        logger.error(f"Feature file not found at {paths['comprehensive_features_csv']}. Please run the 'extract' stage first.")
        return

    df_clean, _, _ = load_and_clean_features(paths['comprehensive_features_csv'])
    if df_clean is None or df_clean.empty:
        logger.error("No valid features found. Aborting clustering.")
        return
    
    if len(df_clean) < 50:
        logger.warning("Small dataset detected. Using lenient clustering parameters for demonstration.")
        clustering_params['hdbscan_min_cluster_size'] = 2
        clustering_params['hdbscan_min_samples'] = 1
        clustering_params['pacmap_mn_ratio'] = 0.5
        clustering_params['pacmap_fp_ratio'] = 2.0

    features_df = df_clean.drop(columns=['image_id'])
    embedding = apply_pacmap(features_df, clustering_params)
    if embedding is None:
        logger.error("PaCMAP failed. Aborting clustering.")
        return

    df_with_labels, _, _, cluster_labels = perform_hdbscan_clustering(embedding, df_clean, clustering_params)
    
    plot_embedding(embedding, paths['pacmap_graph'])
    plot_clusters(embedding, cluster_labels, paths['hdbscan_graph'])
    
    save_cluster_assignments(df_with_labels, paths['image_cluster_map_csv'])
    summary_df = generate_cluster_summary(df_with_labels, paths['cluster_summary_csv'])
    profiles_df = generate_cluster_profiles(df_with_labels, paths['cluster_profiles_csv'])
    
    features_to_plot = ['bed_mean', 'bed_std', 'edge_steepness', 'edge_amplitude', 'edge_std', 'skin_std']
    plot_feature_distributions_by_cluster(df_with_labels, features_to_plot, paths['feature_distribution_graph'])
    
    cluster_groups_for_plotting = df_with_labels.groupby('cluster_label')['image_id'].apply(list)
    
    if -1 in cluster_groups_for_plotting.index and len(cluster_groups_for_plotting) == 1:
        logger.warning("Only noise points were found. Skipping the cluster image grid plot.")
    else:
        plot_cluster_image_grid(cluster_groups=cluster_groups_for_plotting, 
                                image_dir=paths['images_dir'], 
                                save_path=paths['samples_for_clusters'])
    logger.info("Clustering step complete.")

# --- Random Forest Classification ---

def run_training_step(config: Config, paths: Dict[str, Path]):
    """
    Executes the model training and evaluation stage.

    This function prepares the data by merging features with cluster labels, trains a
    Random Forest classifier, evaluates its performance, and visualizes the results,
    including a confusion matrix and feature importances.

    Args:
        config (Config): 
            The Config manager instance.
        paths (Dict[str, Path]): 
            A dictionary of all relevant file and directory paths.

    Returns:
        None

    Relationships:
        - Dependencies:
            - `config_manager.Config`: To access classification parameters and labels.
            - `data_loader`: To load features and cluster map data.
            - `classification`: To `label_clustered_data`, `prepare_training_data_splits`,
              `train_classifier`, `evaluate_classifier`, and `get_feature_importances`.
            - `utils.save_dataframe_to_csv`: To save the labeled feature data.
            - `plotting`: To visualize the results.
        - Used by:
            `main` to execute the 'train' and 'all' stages.
    """
    logger.info("Starting model training step.")
    classification_params = config.get_classification_params()
    label_map = config.get_descriptive_labels()

    if not paths['comprehensive_features_csv'].exists() or not paths['image_cluster_map_csv'].exists():
        logger.error("Missing feature or cluster map files. Please run 'extract' and 'cluster' stages first.")
        return

    df, _, _ = load_and_clean_features(paths['comprehensive_features_csv'])
    _, cluster_map = load_cluster_groups(paths['image_cluster_map_csv'])
    
    df_merged = label_clustered_data(df, cluster_map, label_map)
    save_dataframe_to_csv(df_merged, paths['features_with_labels_csv'])

    X_train, X_test, y_train, y_test, X_train_df= prepare_training_data_splits(df_merged, classification_params)
    
    model = train_classifier(X_train, y_train, classification_params, paths['random_forest_model_path'])
    
    results = evaluate_classifier(model, X_test, y_test)
    logger.info(f"Trained model accuracy: {results['accuracy']:.4f}")

    plot_confusion_matrix(model, X_test, y_test, paths['confusion_matrix_plot'])
    
    feature_importance = get_feature_importances(model, X_train_df)
    plot_feature_importances(feature_importance, paths['feature_importance_graph'])
    logger.info("Model training step complete.")

# --- Prediction with trained model ---

def run_prediction_step(config: Config, paths: Dict[str, Path], features_path: Path):
    """
    Executes the prediction stage using a pre-trained model.

    This function loads a pre-trained model, loads a new set of features, and uses the
    model to predict the wound border types for the new data. The results are logged
    to the console.

    Args:
        config (Config):
            The Config manager instance.
        paths (Dict[str, Path]):
            A dictionary of all relevant file and directory paths.
        features_path (Path):
            The path to the CSV file containing new features for prediction.

    Returns:
        None
    
    Relationships:
        - Dependencies:
            - `config_manager.Config`: To access configuration parameters.
            - `data_loader.load_and_clean_features`: To load new feature data.
            - `classification.predict_wound_border_type`: To perform the prediction.
            - `logging`: For outputting messages.
        - Used by:
            `main` to execute the 'predict' stage.
    """
    logger.info(f"Starting prediction for features at: {features_path}")

    if not paths['random_forest_model_path'].exists():
        logger.error(f"Trained model not found at {paths['random_forest_model_path']}. Please run the 'train' stage first.")
        return
    if not features_path.exists():
        logger.error(f"Input features file not found at {features_path}.")
        return

    try:
        new_features_df, _, _ = load_and_clean_features(features_path)
        if new_features_df is None or new_features_df.empty:
            logger.error("No valid data in the provided features file for prediction.")
            return
    except Exception as e:
        logger.exception(f"Failed to load or clean features from {features_path}")
        return

    predict_results = predict_wound_border_type(paths['random_forest_model_path'], new_features_df)
    if predict_results is not None:
        logger.info("Prediction results:\n%s", predict_results.to_string())
    else:
        logger.error("Prediction failed.")

# --- Main Execution Logic ---

def main():
    """
    The main entry point for the wound classification pipeline.

    This function parses command-line arguments to determine the pipeline stage to run.
    It loads the configuration, sets up the logging, and then calls the appropriate
    stage specific function (`run_feature_extraction_step`, etc.) to execute the
    requested tasks. It also handles prerequisite checks and cleanup operations.
    """
    parser = argparse.ArgumentParser(description="A machine learning pipeline for wound border characterization.")
    parser.add_argument('stage', choices=['extract', 'cluster', 'train', 'predict'],
                        help="The pipeline stage to execute.")
    parser.add_argument('--image_id', type=str,
                        help="For 'extract' stage: process a single ImageID to visualize intermediate steps.")
    parser.add_argument('--input_features', type=Path,
                        help="For 'predict' stage: optionally specify a path to a feature CSV file. If omitted, feature extraction will run first.")
    args = parser.parse_args()
    
    if args.stage != 'extract' and args.image_id:
        parser.error("--image_id can only be used with the 'extract' stage.")
    if args.stage != 'predict' and args.input_features:
        parser.error("--input_features can only be used with the 'predict' stage.")
    
    logger.info(f"Pipeline starting for stage: '{args.stage}'.")
    try:
        config = Config("config.json")
        paths = config.get_paths()
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.critical(f"Failed to load configuration: {e}")
        return
    
    # --- Stage Execution ---
    
    if args.stage == 'extract':
        perform_cleanup(paths)
        filtered_df_path = paths['filtered_manifest_path']

        if not filtered_df_path.exists():
            logger.info("Filtered manifest not found. Running mask validation automatically.")
            filtering_params = config.get_filtering_params()
            filtered_df = validate_wound_masks(paths['wound_masks_dir'], filtering_params)
            save_dataframe_to_csv(filtered_df, filtered_df_path)
        else:
            filtered_df = pd.read_csv(filtered_df_path)
            logger.info("Using existing filtered manifest file.")

        if args.image_id:
            image_ids_to_process = [args.image_id]
        elif not filtered_df.empty:
            image_ids_to_process = filtered_df['image_id'].tolist()
        else:
            logger.error("Filtered manifest is empty. No images to process.")
            return
        
        run_feature_extraction_step(config, paths, image_ids_to_process, single_image_mode=(args.image_id is not None))
        if args.image_id:
            logger.info("Single image processing complete. Displaying plots.")
            plt.show()

    if args.stage == 'cluster':
        run_clustering_step(config, paths)

    if args.stage == 'train':
        run_training_step(config, paths)
    
    if args.stage == 'predict':
        if args.input_features:
            run_prediction_step(config, paths, args.input_features)
        else:
            logger.info("No input feature file specified. Running feature extraction to generate features for prediction.")
            
            perform_cleanup(paths)
            filtered_df_path = paths['filtered_manifest_path']
            if not filtered_df_path.exists():
                filtering_params = config.get_filtering_params()
                filtered_df = validate_wound_masks(paths['wound_masks_dir'], filtering_params)
                save_dataframe_to_csv(filtered_df, filtered_df_path)
            else:
                filtered_df = pd.read_csv(filtered_df_path)
            
            if filtered_df.empty:
                logger.error("Filtered manifest is empty. Cannot extract features for prediction.")
                return
            
            image_ids_to_process = filtered_df['image_id'].tolist()
            run_feature_extraction_step(config, paths, image_ids_to_process, single_image_mode=False)
            
            generated_features_path = paths['comprehensive_features_csv']
            run_prediction_step(config, paths, generated_features_path)

    logger.info("Pipeline execution finished.")

if __name__ == '__main__':
    main()