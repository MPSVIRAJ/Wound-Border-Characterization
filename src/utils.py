"""
This module provides a collection of utility functions for common data manipulation,
analysis, and file I/O operations within the project's data processing pipeline.

It centralizes functions for saving Pandas DataFrames to CSV, handling feature
saving incrementally, and generating various statistical summaries and profiles
from clustered data.

Functions:
    - `save_dataframe_to_csv`: A general-purpose function for saving DataFrames to CSV,
                                with flexible options for appending and header control.
    - `save_features_to_csv`: Appends individual image features to a CSV file.
    - `save_cluster_assignments`: Calculates and saves image-to-cluster assignments.
    - `generate_cluster_summary`: Generates and saves comprehensive statistics for each cluster.
    - `generate_cluster_profiles`: Calculates and saves mean feature values (profiles) for each cluster.

Typical use:
    This module is primarily used by various stages of the data processing and machine learning
    pipeline to perform standardized saving operations, data aggregation, and reporting of results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os 
import logging
from typing import List, Dict, Any, Tuple, Optional

# Get a logger instance for this module
logger = logging.getLogger(__name__)

def save_dataframe_to_csv(df: pd.DataFrame, output_filepath: Path, append_mode: bool = False, include_header: Optional[bool] = None, index: bool = False): # Added 'index: bool = False'
    """
    Saves a Pandas DataFrame to a CSV file, handling directory creation and header logic.

    This function provides flexible options for saving DataFrames, including appending to
    existing files and intelligent control over header writing to avoid duplicate headers
    when appending. It ensures the output directory exists before attempting to write.

    Args:
        df (pd.DataFrame):
            The DataFrame to save. It is expected to contain the data intended for the CSV.
        output_filepath (Path):
            The full path, including filename, where the CSV will be saved. This should be a
            resolved `Path` object.
        append_mode (bool, optional):
            If `True`, the DataFrame will be appended to the file if it exists. If `False`,
            the file will be overwritten. Defaults to `False`.
        include_header (Optional[bool], optional):
            Controls when the header is written:
            - If `True`, the header will always be written.
            - If `False`, the header will never be written.
            - If `None` (default), the header is written only if `append_mode` is `True`
              and the file does not already exist.
            When `append_mode` is `False` (overwrite mode), `pd.to_csv` generally writes a
            header by default, unless `include_header` is explicitly set to `False`.
        index (bool, optional):
            Whether to write the DataFrame index as a column in the CSV. Defaults to `False`.

    Returns:
        None:
            The function does not return any value. It performs a file-saving operation.

    Raises:
        IOError:
            If there's an issue creating the directory or writing the DataFrame to the file.
            This could be due to permission issues, disk full, or an invalid path.
        TypeError:
            If `df` is not a Pandas DataFrame or `output_filepath` is not a `Path` object.

    Example:
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> # Assume 'config' provides paths like paths['filtered_manifest_csv']
        >>> valid_ids_df = pd.DataFrame({'image_id': ['img_001', 'img_002']}) # Assume this is the DataFrame
        >>> temp_csv_path = Path("./temp_data.csv")
        >>> save_dataframe_to_csv(valid_ids_df, temp_csv_path, append_mode=False)

    Relationships:
        - Dependencies:
            - `pandas`: For DataFrame operations (`pd.DataFrame`, `.to_csv()`).
            - `pathlib`: For path manipulation (`Path`, `.parent`, `.mkdir()`, `.is_file()`).
            - `logging`: For logging informational and error messages.
            - `typing.Optional`: For type hinting.
        Used by:
            This function is typically called by various parts of the main application
            (`run_pipeline.py` or other utility functions) to persist data to CSV files.

    Notes:
        - The function uses `pandas.DataFrame.to_csv()` internally.
        - Logging messages indicate the success or failure of the file saving operation.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Invalid type for df: Expected pd.DataFrame, got {type(df)}")
        raise TypeError(f"df must be a pandas.DataFrame, got {type(df)}")
    if not isinstance(output_filepath, Path):
        logger.error(f"Invalid type for output_filepath: Expected Path, got {type(output_filepath)}")
        raise TypeError(f"output_filepath must be a pathlib.Path object, got {type(output_filepath)}")
    try:
        # Ensure the parent directory exists
        output_filepath.parent.mkdir(parents=True, exist_ok=True)

        mode = 'a' if append_mode else 'w'
        header = True       # Default for overwrite mode
        if append_mode:
            if include_header is None:
                header = not output_filepath.is_file()
            else:
                header = include_header 
        else: # Overwrite mode ('w')
            # If not appending, header should always be true unless explicitly set to False
            if include_header is False:
                header = False # Allows user to explicitly overwrite without header if needed
            else:
                header = True # Default for 'w' mode is to write header
        
        # Pass the 'index' argument to df.to_csv
        df.to_csv(str(output_filepath), mode=mode, header=header, index=index)
        logger.info(f"DataFrame successfully saved to {output_filepath} (mode='{mode}', header={header}, index={index}).")
    except Exception as e:
        logger.exception(f"Failed to save DataFrame to '{output_filepath}'.")
        raise IOError(f"Error saving DataFrame to '{output_filepath}': {e}") from e


def save_features_to_csv(ImageID: str, features_dict: Dict[str, Any], output_filepath: Path):
    """
    Appends a dictionary of features for a single image to a CSV file,
    ensuring the 'image_id' column is first. This function is designed for
    incremental saving during a batch process.

    If the CSV file does not exist, it will be created with headers.
    If it exists, a new row will be appended without headers.

    Args:
        ImageID (str): 
            The unique ID of the image for which features are being saved.
        features_dict (Dict[str, Any]): 
            A dictionary where keys are feature names and values are their corresponding data.
        output_filepath (Path): 
            The full path, including filename, where the features CSV will be saved.

    Returns:
        None:
            The function does not return any value. It performs a file-saving operation.

    Raises:
        ValueError: If the feature dictionary is empty or contains non-scalar values (e.g., lists).
        TypeError: If `ImageID` is not a string, `features_dict` is not a dictionary,
                   or `output_filepath` is not a `Path` object.
        IOError: If there's an issue writing to the file via `save_dataframe_to_csv`.

    Output:
        - Log: 
            Warning if `features_dict` is empty. Error if saving fails.
        - CSV File:
            A new row appended to the CSV file at `output_filepath`.

    Examples:
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> # Assume logging is set up

        >>> temp_output_file = Path("./temp_features.csv")
        >>> features_1 = {'feature_A': 10.5, 'feature_B': 20.1}
        >>> save_features_to_csv('image_001', features_1, temp_output_file)
 
    Relationships:
        - Dependencies: 
            Relies on `pandas` for DataFrame conversion and `save_dataframe_to_csv` for I/O.
            Uses Python's built-in `logging` module for output.
        - Used by: 
            Primarily used by the feature extraction pipeline (`run_pipeline.py`)
            to save extracted features for individual images incrementally.
    """
    if not isinstance(ImageID, str):
        logger.error(f"Invalid type for ImageID: Expected str, got {type(ImageID)}")
        raise TypeError(f"ImageID must be a string, got {type(ImageID)}")
    if not isinstance(features_dict, dict):
        logger.error(f"Invalid type for features_dict: Expected dict, got {type(features_dict)}")
        raise TypeError(f"features_dict must be a dictionary, got {type(features_dict)}")
    if not isinstance(output_filepath, Path):
        logger.error(f"Invalid type for output_filepath: Expected Path, got {type(output_filepath)}")
        raise TypeError(f"output_filepath must be a pathlib.Path object, got {type(output_filepath)}")

    if not features_dict:
        logger.warning("Feature dictionary is empty. Nothing to save for ImageID: %s", ImageID)
        raise ValueError("Feature dictionary is empty. Cannot save empty features.")

    # Add the ImageID to the features dictionary (create a copy to avoid modifying original dict)
    features_data_for_df = features_dict.copy()
    features_data_for_df['image_id'] = ImageID

    # Create the desired column order with 'image_id' first
    column_order = ['image_id'] + [key for key in features_data_for_df if key != 'image_id']
    
    # Create the DataFrame with the specified column order
    # It's important to wrap features_data_for_df in a list to create a single row DataFrame
    features_df = pd.DataFrame([features_data_for_df], columns=column_order)
    
    # Use the generic save_dataframe_to_csv function
    # It handles directory creation and appends with header only if file doesn't exist.
    save_dataframe_to_csv(features_df, output_filepath, append_mode=True, include_header=None)

def save_cluster_assignments(df: pd.DataFrame, output_filepath: Path):
    """
    Calculates and displays the number of samples in each cluster, then saves
    the image IDs with their cluster assignments to a CSV file.

    This function provides a summary of the clustering results by showing the
    distribution of samples across different identified clusters. It also
    generates a manifest mapping each image to its assigned cluster.

    Args:
        df (pd.DataFrame): 
            DataFrame containing at least 'image_id' and 'cluster_label' columns.
        output_filepath (Path): 
            The full path, including filename, where the cluster assignment CSV will be saved.
            The filename 'image_cluster_map.csv' is typically derived from configuration
            but this function will save to the provided full path.

    Returns:
        Dict[Any, np.intp]: 
            A dictionary mapping each cluster label to its corresponding count of samples.

    Raises:
        ValueError: 
            If the input DataFrame does not contain 'image_id' or 'cluster_label' columns.
        TypeError: 
            If `df` is not a Pandas DataFrame or `output_filepath` is not a `Path` object.
        IOError: 
            If there's an issue writing the CSV file via `save_dataframe_to_csv`.

    Output:
        - Log: 
            Informational messages about cluster sample counts. Errors if saving fails.
        - CSV File: 
            A CSV file created at `output_filepath` containing image IDs and their cluster labels.

    Examples:
        >>> import pandas as pd
        >>> from pathlib import Path

        >>> # Create a dummy DataFrame with cluster assignments
        >>> df_mock = pd.DataFrame({
        ...     'image_id': ['imgA', 'imgB', 'imgC', 'imgD', 'imgE'],
        ...     'cluster_label': [0, 1, 0, 2, 1]
        ... })
        >>> temp_output_file = Path("./temp_image_cluster_map.csv")
        >>>
        >>> counts = save_cluster_assignments(df_mock, temp_output_file)

    Relationships:
        - Dependencies: 
            Relies on `pandas` for DataFrame operations, `numpy` for unique counts,
            and `save_dataframe_to_csv` for I/O. Uses `logging` for output.
        - Used by: 
            The clustering pipeline (`run_pipeline.py`) to record and summarize cluster assignments.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Invalid type for df: Expected pd.DataFrame, got {type(df)}")
        raise TypeError(f"df must be a pandas.DataFrame, got {type(df)}")
    if not isinstance(output_filepath, Path):
        logger.error(f"Invalid type for output_filepath: Expected Path, got {type(output_filepath)}") 
        raise TypeError(f"output_filepath must be a pathlib.Path object, got {type(output_filepath)}") 
    if 'image_id' not in df.columns or 'cluster_label' not in df.columns:
        logger.error("DataFrame must contain 'image_id' and 'cluster_label' columns for saving cluster assignments.")
        raise ValueError("Input DataFrame is missing required columns ('image_id', 'cluster_label').")

    # Count number of samples in each cluster
    cluster_labels_array = df["cluster_label"].values
    unique_labels, counts = np.unique(cluster_labels_array, return_counts=True)
    cluster_counts = {int(label): int(count) for label, count in zip(unique_labels, counts)}
    logger.info("Cluster sample counts: %s", cluster_counts)

    # Extract and save image ID with cluster assignments
    image_clusters_df = df[["image_id", "cluster_label"]]
    
    # Use the generic saving function, overwriting previous results for this file.
    save_dataframe_to_csv(image_clusters_df, output_filepath, append_mode=False)

    return cluster_counts

def generate_cluster_summary(df: pd.DataFrame, output_filepath: Path):
    """
    Generates and saves a summary of numeric statistics (mean, std, count) for each cluster.

    This function calculates descriptive statistics for all numeric features, grouped by
    their assigned cluster label, providing insights into the characteristics of each cluster.
    It's particularly useful for understanding the quantitative differences between discovered
    wound border types.

    Args:
        df (pd.DataFrame): 
            DataFrame that includes numeric feature columns and 'cluster_label' column.
        output_filepath (Path): 
            The full path, including filename, where the summary CSV will be saved.
            The filename 'cluster_summary_stats.csv' is typically derived from configuration
            but this function will save to the provided full path.

    Returns:
        pd.DataFrame: 
            The summary statistics DataFrame (mean, std, count) for each cluster.

    Raises:
        ValueError: If 'cluster_label' column is missing from the DataFrame.
        TypeError: If `df` is not a Pandas DataFrame or `output_filepath` is not a `Path` object.
        IOError: If there's an issue writing the CSV file via `save_dataframe_to_csv`.

    Output:
        - Log:
            Informational messages about summary generation. Errors if saving fails.
        - CSV File: 
            A CSV file created at `output_filepath` containing the aggregated statistics.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pathlib import Path
        >>> df_mock = pd.DataFrame({
        ...     'image_id': ['id1', 'id2', 'id3', 'id4', 'id5'],
        ...     'feature_A': [10, 12, 11, 20, 22],
        ...     'feature_B': [1, 2, 1, 5, 6],
        ...     'cluster_label': [0, 0, 0, 1, 1]
        ... })
        >>> temp_output_file = Path("./temp_cluster_summary_stats.csv")
        >>> summary_df = generate_cluster_summary(df_mock, temp_output_file)

    Relationships:
        - Dependencies: 
            Relies on `pandas` for DataFrame operations, `numpy` for numeric types,
            and `save_dataframe_to_csv` for I/O. Uses `logging` for output.
        - Used by: 
            The clustering pipeline (`run_pipeline.py`)
            to provide a statistical overview of the identified clusters.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Invalid type for df: Expected pd.DataFrame, got {type(df)}")
        raise TypeError(f"df must be a pandas.DataFrame, got {type(df)}")
    if not isinstance(output_filepath, Path):
        logger.error(f"Invalid type for output_filepath: Expected Path, got {type(output_filepath)}") 
        raise TypeError(f"output_filepath must be a pathlib.Path object, got {type(output_filepath)}") 
    if 'cluster_label' not in df.columns:
        logger.error("DataFrame must contain 'cluster_label' column for generating cluster summary.")
        raise ValueError("Input DataFrame is missing required column ('cluster_label').")

    # Select numeric columns for summary (exclude 'image_id' and 'cluster_label' if present)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['image_id', 'cluster_label'], errors='ignore')

    if numeric_cols.empty:
        logger.warning("No numeric feature columns found to generate cluster summary. Returning DataFrame with cluster_label column only.")
        # Only return the DataFrame, do not save a file if there's no actual summary data.
        return pd.DataFrame({'cluster_label': sorted(df["cluster_label"].unique())})

    # Group by cluster label and compute mean, std, count
    cluster_summary = df.groupby("cluster_label")[numeric_cols].agg(['mean', 'std', 'count'])

    # Flatten MultiIndex column names (e.g., 'feature_A', 'mean' -> 'feature_A_mean')
    cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]

    # Reindex to include all clusters (including ones that may have been dropped during aggregation if empty)
    all_clusters = sorted(df["cluster_label"].unique())
    cluster_summary = cluster_summary.reindex(all_clusters).reset_index()

    # Fill missing cells with NaN for clusters with no data after reindexing (e.g., if a cluster has no numeric features)
    cluster_summary.fillna(np.nan, inplace=True)

    # Use the generic saving function, overwriting previous results for this file.
    save_dataframe_to_csv(cluster_summary, output_filepath, append_mode=False)
    
    return cluster_summary

# Generate and save cluster profiles
def generate_cluster_profiles(df: pd.DataFrame, output_filepath: Path):
    """
    Calculates the mean feature values for each cluster and saves the result as a profile table.

    This function provides a "profile" for each identified cluster by computing the average
    value for every feature within that cluster. This helps to quantitatively characterize
    and differentiate distinct wound types discovered during clustering.

    Args:
        df (pd.DataFrame): 
            DataFrame that includes numeric feature columns and 'cluster_label' column.
        output_filepath (Path): 
            The full path, including filename, where the cluster profiles CSV will be saved.
            The filename 'cluster_profiles.csv' is typically derived from configuration
            but this function will save to the provided full path.

    Returns:
        pd.DataFrame: 
            DataFrame containing the mean feature profile of each cluster.

    Raises:
        ValueError: 
            If 'cluster_label' column is missing from the DataFrame.
        TypeError: 
            If `df` is not a Pandas DataFrame or `output_filepath` is not a `Path` object.
        IOError: 
            If there's an issue writing the CSV file via `save_dataframe_to_csv`.

    Output:
        - Log:
            Informational messages about profile generation. Errors if saving fails.
        - CSV File:
            A CSV file created at `output_filepath` containing the mean feature values per cluster.

    Examples:
        >>> import pandas as pd
        >>> from pathlib import Path

        >>> df_mock = pd.DataFrame({
        ...     'image_id': ['id1', 'id2', 'id3', 'id4', 'id5'],
        ...     'feature_X': [10.0, 11.0, 10.5, 20.0, 21.0],
        ...     'feature_Y': [1.0, 1.2, 1.1, 2.0, 2.3],
        ...     'cluster_label': [0, 0, 0, 1, 1]
        ... })
        >>> temp_output_file = Path("./temp_cluster_profiles.csv")
        >>>
        >>> profiles_df = generate_cluster_profiles(df_mock, temp_output_file)
        
    Relationships:
        - Dependencies: 
            Relies on `pandas` for DataFrame operations, `numpy` for numeric types,
            and `save_dataframe_to_csv` for I/O. Uses `logging` for output.
        - Used by: 
            The clustering pipeline (`run_pipeline.py`) to store mean feature values for each cluster.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Invalid type for df: Expected pd.DataFrame, got {type(df)}")
        raise TypeError(f"df must be a pandas.DataFrame, got {type(df)}")
    if not isinstance(output_filepath, Path):
        logger.error(f"Invalid type for output_filepath: Expected Path, got {type(output_filepath)}") 
        raise TypeError(f"output_filepath must be a pathlib.Path object, got {type(output_filepath)}") 
    if 'cluster_label' not in df.columns:
        logger.error("DataFrame must contain 'cluster_label' column for generating cluster profiles.")
        raise ValueError("Input DataFrame is missing required column ('cluster_label').")
    
    # Drop non-feature columns to isolate numerical feature vectors
    # Exclude 'cluster_label' (which is used for grouping) and 'image_id' (which is an identifier).
    feature_columns = df.select_dtypes(include=[np.number]).columns.drop(['cluster_label', 'image_id'], errors='ignore')

    if feature_columns.empty:
        logger.warning("No numeric feature columns found to generate cluster profiles. Returning empty DataFrame.")
        unique_labels = sorted(df["cluster_label"].unique())
        return pd.DataFrame(index=unique_labels, columns=[])

    # Compute mean of each feature per cluster
    cluster_profiles = df.groupby("cluster_label")[feature_columns].mean()

    # Use the generic saving function, overwriting previous results for this file.
    # Set index=True here so the 'cluster_label' index is written as a column
    save_dataframe_to_csv(cluster_profiles, output_filepath, append_mode=False, include_header=True, index=True) 

    return cluster_profiles