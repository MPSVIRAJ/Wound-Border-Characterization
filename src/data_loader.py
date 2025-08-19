"""
This module provides robust functions for loading various data types critical to the
wound analysis pipeline, including images, masks, depth maps, feature vectors,
and cluster assignments.

It centralizes data loading operations, ensuring consistency in file handling,
path resolution, and initial data validation (e.g., checking for missing files,
data integrity, and applying masks).

Functions:
    - `data_loader`: Loads all image, mask, and depth map files for a single ImageID.
    - `load_and_clean_features`: Loads a CSV of feature vectors, handles missing data, and prepares it for ML tasks.
    - `load_cluster_groups`: Loads image-to-cluster assignment data from a CSV and groups images by their assigned cluster.

Typical use:
    This module is primarily used by the main application entry point (`run_pipeline.py`)
    and other pipeline stages to retrieve and prepare necessary data for subsequent
    processing, such as feature extraction, clustering, and classification.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging # NEW: Import logging
from typing import Optional, Dict, Any, Tuple # Ensure all types are imported

# NEW: Get a logger instance for this module.
logger = logging.getLogger(__name__)

def data_loader(ImageID: str, data_root_path: Path, subdirs_config: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Loads various image and mask files associated with a given ImageID from specified paths.

    This function is responsible for retrieving all necessary image, mask, and depth map
    files for a single wound. It supports loading optional marker images and applies
    initial masking operations to the depth map. The function enforces strict checks for
    critical files (image, wound mask, depth map, body mask) and raises an error if any
    are missing or unreadable. Warnings are logged for optional files.

    Args:
        ImageID (str):
            The unique identifier for the image set (e.g., filename without extension).
            All corresponding files (body mask, wound mask, depths) are expected
            to share this name within their respective subdirectories.
        data_root_path (Path):
            The base directory where the 'images', 'wound_masks', etc.,
            subdirectories are located. This should be a resolved Path object.
        subdirs_config (Dict[str, str]):
            A dictionary containing subdirectory names
            (e.g.,'images_subdir', 'wound_masks_subdir', 'body_mask_subdir', 'depth_maps_subdir', 'marker_mask_subdir').

    Returns:
        Optional[Dict[str, Any]]:
            A dictionary containing the loaded image and mask data:
            - 'image' (np.ndarray): The loaded main RGB image. Its shape is (height, width, 3).
            - 'wound' (np.ndarray): The loaded grayscale wound mask. Its shape is (height, width).
            - 'body' (np.ndarray): The loaded grayscale body mask. Its shape is (height, width).
            - 'depth' (np.ndarray): The loaded grayscale depth map (e.g., 16-bit). Its shape is (height, width),
            and it will have been masked by the body mask and marker mask (if applied).
            Returns `None` if `ImageID` is invalid or a critical file is missing/unreadable.

    Raises:
        FileNotFoundError:
            If a critical file (image, wound, depth, body mask) is not found.
        IOError:
            If a critical file is found but cannot be read (e.g., corrupted, permission issues).
        ValueError:
            If `subdirs_config` is missing expected keys.

    Output:
        - Log:
            Informational messages on successful loads. Warnings for optional file issues
            or shape mismatches. Errors for critical file loading failures.
        - Return Value:
            A dictionary of NumPy arrays for `image`, `wound`, `body`, `depth`.

    Example:
        >>> import numpy as np
        >>> import cv2
        >>> from pathlib import Path
        >>> # Assume logging is set up
        >>> # Assume data root path, subdirectory configuration and Image ID are as follows:
        >>> temp_data_root = Path("./temp_data_loader_example")
        >>> subdirs = {
        ...     "images_subdir": "images", "wound_masks_subdir": "wound_masks",
        ...     "body_mask_subdir": "body_mask", "depth_maps_subdir": "depth_maps",
        ...     "marker_mask_subdir": "marker_mask"
        ... }
        >>> img_id = "sample_001"
        >>> loaded_data = data_loader(img_id, temp_data_root, subdirs)

    Relationships:
        - Dependencies:
            - `cv2`: For image I/O (`cv2.imread`) and masking operations (`cv2.bitwise_and`).
            - `numpy`: For array operations (`np.zeros`, `np.any`).
            - `pathlib`: For robust file path handling (`Path`).
            - `logging`: For outputting informational messages, warnings, and errors.
        - Used by:
            The main application entry point (e.g.,`run_pipeline.py`)
            to load raw image data for individual image processing.
    """
    file_name = f"{ImageID}.png" # Assuming all images and masks are .png

    # Construct full paths using data_root_path and subdirectory names from subdirs_config
    try:
        image_path = data_root_path / subdirs_config['images_subdir'] / file_name
        wound_mask_path = data_root_path / subdirs_config['wound_masks_subdir'] / file_name
        body_mask_path = data_root_path / subdirs_config['body_mask_subdir'] / file_name
        marker_mask_path = data_root_path / subdirs_config['marker_mask_subdir'] / file_name
        depth_map_path = data_root_path / subdirs_config['depth_maps_subdir'] / file_name
    except KeyError as e:
        logger.error(f"Missing subdirectory configuration key: {e}. Please check your subdirs_config.")
        raise ValueError(f"Missing subdirectory configuration key in subdirs_config: {e}") from e

    image = None
    wound = None
    body = None
    depth = None
    marker = None

    # --- Critical files: If any of these are missing or unreadable, raise an error ---

    # Image
    if image_path.exists():
        image = cv2.imread(str(image_path))
        if image is None: # Check if imread actually loaded something (e.g., not corrupted)
            logger.error(f"Image file at {image_path} could not be read. Terminating processing for this ID.")
            raise IOError(f"Failed to read image file: {image_path}")
        image = image[..., ::-1] # Convert BGR to RGB
    else:
        logger.error(f"Image file not found at {image_path}. Terminating processing for this ID.")
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Wound Mask
    if wound_mask_path.exists():
        wound = cv2.imread(str(wound_mask_path), cv2.IMREAD_GRAYSCALE)
        if wound is None:
            logger.error(f"Wound mask file at {wound_mask_path} could not be read. Terminating processing for this ID.")
            raise IOError(f"Failed to read wound mask file: {wound_mask_path}")
    else:
        logger.error(f"Wound mask file not found at {wound_mask_path}. Terminating processing for this ID.")
        raise FileNotFoundError(f"Wound mask file not found: {wound_mask_path}")
    
    # Depth Map
    if depth_map_path.exists():
        depth = cv2.imread(str(depth_map_path), cv2.IMREAD_ANYDEPTH)
        if depth is None:
            logger.error(f"Depth map file at {depth_map_path} could not be read. Terminating processing for this ID.")
            raise IOError(f"Failed to read depth map file: {depth_map_path}")
    else:
        logger.error(f"Depth map file not found at {depth_map_path}. Terminating processing for this ID.")
        raise FileNotFoundError(f"Depth map file not found: {depth_map_path}")
    
    # Body Mask (Compulsory)
    if body_mask_path.exists():
        body = cv2.imread(str(body_mask_path), cv2.IMREAD_GRAYSCALE)
        if body is None:
            logger.error(f"Body mask file at {body_mask_path} could not be read. Terminating processing for this ID (body mask compulsory).")
            raise IOError(f"Failed to read body mask file: {body_mask_path}")
    else:
        logger.error(f"Body mask file not found at {body_mask_path}. Terminating processing for this ID (body mask compulsory).")
        raise FileNotFoundError(f"Body mask file not found: {body_mask_path}")
    
    # --- Optional files: Log warnings if missing/unreadable, but continue processing ---

    # Marker Mask
    if marker_mask_path.exists():
        marker = cv2.imread(str(marker_mask_path), cv2.IMREAD_GRAYSCALE)
        if marker is None:
            logger.warning(f"Marker mask file at {marker_mask_path} could not be read. Defaulting marker mask to all black (no masking effect).")
            marker = np.zeros(depth.shape, dtype=np.uint8) # Default to all black mask of depth's shape
    else:
        logger.warning(f"Marker mask file not found at {marker_mask_path}. Defaulting marker mask to all black (no masking effect).")
        marker = np.zeros(depth.shape, dtype=np.uint8) # Default to all black mask of depth's shape

    logger.debug(f"Successfully loaded all critical files for {ImageID}.")

    # --- Apply masks (ensure shapes match before applying) ---

    # Apply body mask to depth map
    if body.shape == depth.shape:
        depth = cv2.bitwise_and(depth, depth, mask=body)
        logger.debug(f"Body mask applied to depth map for {ImageID}.")
    else:
        logger.warning(f"Body mask shape {body.shape} does not match depth map shape {depth.shape}. Skipping body masking for {ImageID}.")

    # Apply marker mask (inverted) to depth map and body mask
    # Only apply if marker mask has correct shape AND contains actual non-zero content
    if marker.shape == depth.shape and np.any(marker):
        depth = cv2.bitwise_and(depth, depth, mask=~marker) # Invert marker mask to remove the marker region
        if body.shape == depth.shape:
            body = cv2.bitwise_and(body, body, mask=~marker) # Apply to body mask as well
        logger.debug(f"Marker mask applied to depth map and body mask for {ImageID}.")
    elif marker.shape != depth.shape:
        logger.warning(f"Marker mask shape {marker.shape} does not match depth map shape {depth.shape}. Skipping marker masking for {ImageID}.")
    else: # Marker is all zeros, np.any(marker) is False
        logger.info(f"Marker mask for {ImageID} is all zeros/empty. No marker masking applied.")
        
    # Put all the loaded data into a dictionary
    loaded_data = {
        'image': image,
        'wound': wound,
        'body': body,
        'depth': depth
    }
    return loaded_data


def load_and_clean_features(file_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray] | Tuple[None, None, None]:
    """
    Loads a CSV file containing feature vectors, cleans the data by dropping
    rows with NaN values, and separates image IDs from feature vectors.

    This function acts as a robust loader for the comprehensive feature set,
    ensuring data integrity before further processing.

    Args:
        file_path (Path):
            The Path to the comprehensive_features.csv file.

    Returns:
        Tuple[pd.DataFrame, np.ndarray, np.ndarray]: A tuple containing:
            - df_clean (pd.DataFrame): Cleaned DataFrame without NaNs (includes 'image_id').
            - image_ids (np.ndarray): A NumPy array of image IDs from the cleaned DataFrame.
            - features (np.ndarray): A NumPy array of feature vectors (without 'image_id'). If the CSV is empty, returns an empty DataFrame and empty NumPy arrays.

    Raises:
        FileNotFoundError:
            If the CSV file specified by `file_path` is not found.
        IOError:
            If there's an issue reading the CSV file (e.g., permissions, corrupted, or other unexpected errors).
        KeyError:
            If the 'image_id' column is missing after loading.

    Output:
        - Log:
            Informational messages about loading progress and NaN rows. Error messages for critical failures like file not found or read errors.
        - Return Value:
            A tuple containing the cleaned DataFrame, image IDs array, and features array.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pathlib import Path
        >>> # Assume logging is set up
        >>> # Create a dummy CSV for the example
        >>> temp_csv_path = Path("./temp_features_data.csv")
        >>> df_clean, ids, feats = load_and_clean_features(temp_csv_path)

    Relationships:
        - Dependencies:
            - `pandas`: For DataFrame operations (`pd.read_csv`, `pd.DataFrame`, `.dropna()`, column selection).
            - `numpy`: For array manipulation (`np.ndarray`, `np.array`).
            - `pathlib`: For robust file path handling (`Path`).
            - `logging`: For outputting informational messages and errors.

        - Used by:
            The clustering and classification pipeline sections (e.g., in `run_pipeline.py`) to load the prepared feature set.

    """
    try:
        logger.info(f"Loading feature data from '{file_path}'.")
        df = pd.read_csv(str(file_path))
        logger.info(f"Loaded {len(df)} feature rows from '{file_path}'.")

        # Clean the dataset by dropping rows with NaN values
        df_clean = df.dropna()
        logger.info(f"{len(df_clean)} rows remaining after dropping NaNs.")

        # Ensure 'image_id' column exists before trying to extract it.
        if 'image_id' not in df_clean.columns:
            logger.error(f"Required 'image_id' column not found in loaded CSV: '{file_path}'.")
            raise KeyError(f"Required 'image_id' column not found in '{file_path}'.")

        # Extract image IDs and feature vectors
        image_ids = df_clean["image_id"].values
        features = df_clean.drop(columns=["image_id"]).values
        
        logger.info("Features loaded and cleaned successfully.")
        return df_clean, image_ids, features

    except FileNotFoundError:
        logger.error(f"CSV file not found at '{file_path}'. Please ensure the path is correct or generate the file first.")
        raise FileNotFoundError(f"Feature CSV file not found: {file_path}")
    except pd.errors.EmptyDataError:
        logger.warning(f"CSV file at '{file_path}' is empty. Returning empty data structures.")
        return pd.DataFrame(columns=['image_id']), np.array([]), np.array([]).reshape(0,0)
    except Exception as e:
        logger.exception(f"An unexpected error occurred while loading or cleaning features from '{file_path}'.")
        raise IOError(f"Error loading/cleaning features from '{file_path}': {e}") from e


def load_cluster_groups(file_path: Path) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Loads a CSV containing image IDs and cluster labels, and groups the image IDs by cluster.

    This function reads the manifest that maps image identifiers to their assigned
    cluster labels, providing a convenient structure for accessing cluster-specific lists of images.

    Args:
        file_path (Path):
            Path to the image_cluster_map.csv file.

    Returns:
        Tuple[pd.Series, pd.DataFrame]: A tuple containing:
            - pd.Series: A Series mapping each cluster label to a list of image IDs, sorted by cluster label.
            - pd.DataFrame: The original DataFrame loaded from the CSV (includes 'image_id' and 'cluster_label').
                If the CSV is empty, returns an empty Series and an empty DataFrame with expected columns.

    Raises:
        FileNotFoundError:
            If the CSV file specified by `file_path` is not found.
        IOError:
            If there's an issue reading the CSV file (e.g., permissions, corrupted, or other unexpected errors).
        KeyError:
            If required columns ('image_id', 'cluster_label') are missing.

    Output:
        - Log:
            Informational messages about loading. Error messages for critical failures.
        - Return Value:
            A tuple containing a Series of grouped image IDs and the original DataFrame.

    Examples:
        >>> import pandas as pd
        >>> from pathlib import Path
        >>> # Assume logging is set up
        >>> temp_csv_path = Path("./temp_cluster_map.csv")
        >>> cluster_groups_series, cluster_map_df = load_cluster_groups(temp_csv_path)
    
    Relationships:
        - Dependencies:
            - `pandas`: For DataFrame operations (`pd.read_csv`, `pd.DataFrame`, `pd.Series`, `.groupby()`, `.apply()`).
            - `pathlib`: For robust file path handling (`Path`).
            - `logging`: For outputting informational messages and errors.

        - Used by:
            The clustering and classification pipelines (e.g., in `run_pipeline.py`)
            to retrieve cluster assignment information.
    """
    try:
        logger.info(f"Loading cluster map from '{file_path}'.")
        cluster_map = pd.read_csv(str(file_path))
        
        # Ensure required columns exist
        if 'image_id' not in cluster_map.columns or 'cluster_label' not in cluster_map.columns:
            logger.error(f"Required 'image_id' or 'cluster_label' column not found in loaded CSV: '{file_path}'.")
            raise KeyError(f"Required 'image_id' or 'cluster_label' column not found in '{file_path}'.")

        # Group image IDs by cluster and sort by cluster label
        cluster_groups = cluster_map.groupby("cluster_label")["image_id"].apply(list)
        cluster_groups = cluster_groups.sort_index()
        logger.info(f"Successfully loaded cluster groups for {len(cluster_groups)} clusters.")

        return cluster_groups, cluster_map

    except FileNotFoundError:
        logger.error(f"Cluster map CSV file not found at '{file_path}'. Cannot load cluster groups.")
        raise FileNotFoundError(f"Cluster map CSV file not found: {file_path}")
    except pd.errors.EmptyDataError:
        logger.warning(f"CSV file at '{file_path}' is empty. Returning empty data structures for cluster groups.")
        return pd.Series(dtype=object), pd.DataFrame(columns=['image_id', 'cluster_label'])
    except Exception as e:
        logger.exception(f"An unexpected error occurred while loading cluster groups from '{file_path}'.")
        raise IOError(f"Error loading cluster groups from '{file_path}': {e}") from e