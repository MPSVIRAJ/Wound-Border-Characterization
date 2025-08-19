"""
This module provides functions for dimensionality reduction and clustering of image features.

It leverages PaCMAP (Pairwise Controlled Manifold Approximation) for transforming
high-dimensional feature vectors into a low-dimensional space, followed by
HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
to automatically identify clusters and outliers.

Functions:
    - `apply_pacmap`:  Performs dimensionality reduction on a feature set using PaCMAP.
    - `perform_hdbscan_clustering`: Applies the HDBSCAN algorithm to find clusters
        and noise points in a low-dimensional embedding.

Typical use:
    This module is used in the machine learning pipeline after feature extraction
    to prepare the data for visualization and to group similar images based on their
    extracted features. The outputs are then used for subsequent analysis and classification.
"""
import pandas as pd
import pacmap
import hdbscan
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any, Tuple
import logging

# Get a logger instance for this module.
logger = logging.getLogger(__name__)

#-----PaCMAP dimensionality reduction---
def apply_pacmap(df: pd.DataFrame, clustering_params: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Applies PaCMAP for dimensionality reduction on the feature set.

    This function transforms the high-dimensional feature vectors of the dataset into
    a low-dimensional space (typically 2D) for visualization and to improve the
    performance of subsequent clustering algorithms. It first standardizes the features
    using `StandardScaler` to ensure a consistent scale before applying PaCMAP.

    Args:
        df (pd.DataFrame): 
            DataFrame containing the feature vectors, with 'image_id' as the index.
            The DataFrame is expected to have been cleaned of NaNs.
        clustering_params (Dict[str, Any]): 
            Dictionary containing configurable parameters for PaCMAP.
            Expected keys: 'pacmap_n_components', 'pacmap_mn_ratio',
            'pacmap_fp_ratio', and a 'random_state'.

    Returns:
        Optional[np.ndarray]: 
            A NumPy array representing the 2D embedding of the feature data.
            Returns None if an error occurs during the process.

    Raises:
        TypeError: 
            If the input `df` is not a Pandas DataFrame.
        ValueError: 
            If required keys are missing from `clustering_params` or if `df` is empty.
        RuntimeError: 
            If the PaCMAP algorithm fails to run.

    Output:
        - Console/Log:
            Informational messages about the dimensions of the input and output data.
            Error messages for invalid input or algorithm failures.
        - Return Value:
            A NumPy array representing the 2D embedding.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from src.clustering import apply_pacmap
        >>> # Dummy feature data (cleaned, without image_id column)
        >>> dummy_features = pd.DataFrame(np.random.rand(100, 28))
        >>> clustering_params = {'pacmap_n_components': 2, 'pacmap_mn_ratio': 2,
        ...                      'pacmap_fp_ratio': 8, 'random_state': 42}
        >>> embedding = apply_pacmap(dummy_features, clustering_params)

    Relationships:
        - Dependencies:
            - `pandas`: For DataFrame handling.
            - `pacmap`: For the PaCMAP algorithm.
            - `sklearn.preprocessing.StandardScaler`: For feature standardization.
            - `logging`: For outputting messages.
        - Used by:
            The main clustering pipeline in the file 'run_pipeline.py` to prepare data for HDBSCAN.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Input 'df' must be a Pandas DataFrame.")
        raise TypeError("Input 'df' must be a Pandas DataFrame.")
    if df.empty:
        logger.warning("Input DataFrame for PaCMAP is empty. Returning None.")
        return None
    
    # Check for required parameters
    required_params = ['pacmap_n_components', 'pacmap_mn_ratio', 'pacmap_fp_ratio', 'random_state']
    if not all(p in clustering_params for p in required_params):
        logger.error("Missing required parameters for PaCMAP. Check your configuration.")
        raise ValueError("Missing required parameters for PaCMAP.")
    
    logger.info(f"Applying PaCMAP to data of shape {df.shape}.")
    
    try:
        # Standardize the features and apply PaCMAP
        logger.debug("Standardizing features with StandardScaler.")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df.values)
        pmap = pacmap.PaCMAP(
            n_components=clustering_params['pacmap_n_components'],
            n_neighbors=None,
            MN_ratio=clustering_params['pacmap_mn_ratio'],
            FP_ratio=clustering_params['pacmap_fp_ratio'],
            random_state=clustering_params['random_state'],
            distance='euclidean'
        )
        embedding = pmap.fit_transform(features_scaled)
        logger.info(f"PaCMAP dimensionality reduction complete. Output shape: {embedding.shape}.")
        return embedding
    except Exception as e:
        logger.exception("An error occurred during PaCMAP execution.")
        raise RuntimeError(f"PaCMAP execution failed: {e}") from e


#-----HDBSCAN clutering-----
def perform_hdbscan_clustering(embedding: np.ndarray, df: pd.DataFrame, clustering_params: Dict[str, Any]) -> Tuple[pd.DataFrame, int, int, np.ndarray]:
    """
    Performs HDBSCAN clustering on the reduced data embedding.

    This function applies the Hierarchical Density-Based Spatial Clustering of
    Applications with Noise (HDBSCAN) algorithm to identify distinct clusters in the
    input data. It is particularly effective at finding clusters of varying shapes
    and densities and automatically handles outliers (assigning them a label of -1).

    Args:
        embedding (np.ndarray): 
            The 2D embedding of the feature data, typically from PaCMAP.
            Shape: (N, 2).
        df (pd.DataFrame): 
            The original DataFrame used to create the embedding. This is
            used to add the cluster labels back to the original data.
        clustering_params (Dict[str, Any]): 
            Dictionary containing configurable parameters for HDBSCAN.
            Expected keys: 'hdbscan_min_cluster_size',
            'hdbscan_min_samples', 'hdbscan_epsilon'.

    Returns:
        Tuple[pd.DataFrame, int, int, np.ndarray]: A tuple containing:
            - df (pd.DataFrame): The original DataFrame with an added 'cluster_label' column.
            - num_clusters (int): The number of clusters identified.
            - num_noise (int): The number of points classified as noise.
            - cluster_labels (np.ndarray): The raw cluster labels returned by HDBSCAN.

    Raises:
        TypeError: 
            If `embedding` is not a NumPy array or `df` is not a Pandas DataFrame.
        ValueError: 
            If required keys are missing from `clustering_params` or if inputs are empty.
        RuntimeError: 
            If the HDBSCAN algorithm fails to run.

    Output:
        - Console/Log: 
            Informational messages about the number of clusters and noise points.
            Error messages for invalid input or algorithm failures.
        - Return Value: 
            The updated DataFrame and clustering summary.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from src.clustering import perform_hdbscan_clustering
        >>> # Dummy data: A 100x2 embedding and corresponding DataFrame
        >>> dummy_embedding = np.random.rand(100, 2)
        >>> dummy_df = pd.DataFrame(np.random.rand(100, 28))
        >>> clustering_params = {'hdbscan_min_cluster_size': 10, 'hdbscan_min_samples': 5,
        ...                      'hdbscan_epsilon': 0.5}
        >>> df_with_labels, num_clusters, num_noise, labels = perform_hdbscan_clustering(
        ...     dummy_embedding, dummy_df, clustering_params)
 
    Relationships:
        - Dependencies:
            - `pandas`: For DataFrame handling.
            - `hdbscan`: For the HDBSCAN algorithm.
            - `logging`: For outputting messages.
        - Used by: 
            The main clustering pipeline in the 'run_pipeline' file, to assign cluster labels.
    """
    if not isinstance(embedding, np.ndarray) or not isinstance(df, pd.DataFrame):
        logger.error("Inputs must be a NumPy array (embedding) and a Pandas DataFrame (df).")
        raise TypeError("Inputs must be a NumPy array and a Pandas DataFrame.")
    if embedding.size == 0 or df.empty:
        logger.warning("Input data for HDBSCAN is empty. Returning empty data structures.")
        return pd.DataFrame(columns=df.columns), 0, 0, np.array([])
    # Check for required parameters
    required_params = ['hdbscan_min_cluster_size', 'hdbscan_min_samples', 'hdbscan_epsilon']
    if not all(p in clustering_params for p in required_params):
        logger.error("Missing required parameters for HDBSCAN. Check your configuration.")
        raise ValueError("Missing required parameters for HDBSCAN.")
    try:
        # Perform HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size = clustering_params['hdbscan_min_cluster_size'],
            min_samples = clustering_params['hdbscan_min_samples'],
            cluster_selection_epsilon = clustering_params['hdbscan_epsilon'],
            cluster_selection_method='eom',
            core_dist_n_jobs =-1,
            metric = 'euclidean'
        )
        cluster_labels = clusterer.fit_predict(embedding)
        df["cluster_label"] = cluster_labels
    except Exception as e:
        logger.exception("An error occurred during HDBSCAN execution.")
        raise RuntimeError(f"HDBSCAN execution failed: {e}") from e
    
    # Calculate cluster statistics
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise = np.sum(cluster_labels == -1)

    logger.info(f"Clustering complete. Found {num_clusters} clusters and {num_noise} noise points.")
    return df, num_clusters, num_noise, cluster_labels


