"""
Data Visualization Utilities for Wound Border Characterization.

This module contains a collection of functions for visualizing the various stages of the
wound border characterization pipeline. It includes utilities for displaying raw image data,
unrolled depth and RGB strips, dimensionality reduction embeddings, cluster distributions,
and classification performance metrics. All plots are designed to be saved to file,
and can optionally be displayed to the user.

Functions:
    - plot_initial_data : Displays the raw RGB, wound mask, body mask, and depth map.
    - show_unrolled_strip : Visualizes the rectified depth profile and corresponding RGB image.
    - plot_depth_profile : Creates a line plot of the mean depth profile with its standard deviation.
    - plot_profiles_and_fits : Visualizes the mean depth profile, smoothed profile, and curve fits.
    - plot_embedding : Creates a scatter plot of the 2D PaCMAP embedding.
    - plot_clusters : Creates a scatter plot of the HDBSCAN clusters on the embedding.
    - plot_feature_distributions_by_cluster : Generates box plots for key features across clusters.
    - plot_cluster_image_grid : Displays a grid of sample images for each identified cluster.
    - plot_confusion_matrix : Visualizes the confusion matrix for the supervised classifier.
    - plot_feature_importances : Creates a bar chart of feature importance scores.

Typical Use:
    These functions are called by the main application script (`run_pipeline.py`) or utility
    scripts (`ML_Pipeline.py`) to provide visual feedback and to save key results for
    reporting and analysis. Plots are configurable to be saved automatically without
    requiring manual interaction.
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import ConfusionMatrixDisplay
from typing import Dict, Any, List, Optional, Tuple
import logging
import cv2

# Get a logger instance for this module.
logger = logging.getLogger(__name__)


def plot_initial_data(image: np.ndarray, wound_mask: np.ndarray, body_mask: np.ndarray, depth_map: np.ndarray):
    """
    Displays the raw RGB image, wound mask, body mask, and depth map in a grid.

    This is an inspection utility to visually validate that the initial data
    has been loaded correctly.

    Args:
        image (np.ndarray): 
            The loaded RGB image.
        wound_mask (np.ndarray): 
            The loaded wound mask.
        body_mask (np.ndarray): 
            The loaded body mask.
        depth_map (np.ndarray): 
            The loaded depth map.

    Returns:
        None

    Raises:
        TypeError: 
            If any of the inputs are not NumPy arrays.
        ValueError: 
            If the shapes of the inputs are inconsistent.

    Output:
        - Log: 
            A debug message is logged upon successful plotting.
        - Display: 
            A matplotlib window displaying the plots.

    Examples:
        >>> import numpy as np
        >>> from src.plotting import plot_initial_data
        >>> # Create dummy data
        >>> dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> dummy_mask = np.zeros((100, 100), dtype=np.uint8)
        >>> dummy_depth = np.random.rand(100, 100)
        >>> plot_initial_data(dummy_img, dummy_mask, dummy_mask, dummy_depth)
    """

    if not all(isinstance(arr, np.ndarray) for arr in [image, wound_mask, body_mask, depth_map]):
        logger.error("All inputs must be NumPy arrays.")
        raise TypeError("All inputs must be NumPy arrays.")
    if not (image.shape[:2] == wound_mask.shape == body_mask.shape == depth_map.shape):
        logger.error("Input shapes are inconsistent.")
        raise ValueError("Input shapes must be consistent.")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("RGB Image")
    axes[0, 1].imshow(wound_mask, cmap='gray')
    axes[0, 1].set_title("Wound Mask")
    axes[1, 0].imshow(body_mask, cmap='gray')
    axes[1, 0].set_title("Body Mask")
    axes[1, 1].imshow(depth_map, cmap='viridis')
    axes[1, 1].set_title("Depth Map")
    plt.tight_layout()
    plt.show()
    logger.debug("Initial data plot displayed.")


def show_unrolled_strip(rect_depth: np.ndarray, unrolled_image: np.ndarray,
                         d1: int, p1: int, iterations: int):
    """
    Visualizes the rectified depth profile and corresponding RGB image.

    This function displays the unrolled depth strip and the corresponding RGB image
    in a side-by-side view. It highlights the wound border and the width of the
    unrolled strip.

    Args:
        rect_depth (np.ndarray): 
            The rectified depth profile.
        unrolled_image (np.ndarray): 
            The rectified RGB image.
        d1 (int): 
            The position of the wound border in the depth profile.
        p1 (int): 
            The position of the wound border in the RGB image.
        iterations (int): 
            The number of erosion/dilation steps.

    Returns:
        None

    Raises:
        TypeError: 
            If any of the inputs are not NumPy arrays or integers.

    Output:
        - Log:
            A debug message is logged upon successful plotting.
        - Display:
            A matplotlib window displaying the plots.

    Examples:
        >>> import numpy as np
        >>> from src.plotting import show_unrolled_strip
        >>> dummy_depth_strip = np.random.rand(100, 20)
        >>> dummy_rgb_strip = np.random.randint(0, 255, (100, 20, 3), dtype=np.uint8)
        >>> show_unrolled_strip(dummy_depth_strip, dummy_rgb_strip, d1=50, p1=50, iterations=10)
    """
    if not all(isinstance(arr, np.ndarray) for arr in [rect_depth, unrolled_image]):
        logger.error("Inputs must be NumPy arrays.")
        raise TypeError("Inputs must be NumPy arrays.")
    if not all(isinstance(v, int) for v in [d1, p1, iterations]):
        logger.error("Inputs d1, p1, and iterations must be integers.")
        raise TypeError("Inputs d1, p1, and iterations must be integers.")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    ax1.imshow(rect_depth, cmap='viridis')
    ax1.axhline(d1, color='w', linestyle='--')
    ax1.set_title("Rectified Depth Strip")
    ax2.imshow(unrolled_image)
    ax2.axhline(p1, color='w', linestyle='--')
    ax2.set_title("Rectified RGB Strip")
    plt.suptitle(f"Unrolled Strips with {iterations} Iterations")
    plt.tight_layout()
    plt.show()
    logger.debug("Unrolled strips plot displayed.")


# Visualize the mean depth profile with its standard deviation
def plot_depth_profile(mean_profile: np.ndarray, std_profile: np.ndarray, d1: int):
    """
    Visualizes the raw mean depth profile with its standard deviation and the
    estimated wound border position.

    This function creates a line plot of the mean depth profile, and uses a shaded
    area to represent the standard deviation across the strip. It also adds a vertical
    line at the estimated wound border position (d1).

    Args:
        mean_profile (np.ndarray): 
            A 1D array of the mean depth profile.
        std_profile (np.ndarray): 
            A 1D array of the standard deviation profile.
        d1 (int): 
            The index of the wound edge (the baseline contour).

    Returns:
        None

    Raises:
        TypeError: 
            If inputs are not of expected types.
        ValueError: 
            If inputs have inconsistent shapes or are empty.

    Output:
        - Log:
            A debug message is logged upon successful plotting.
        - Display:
            A matplotlib window displaying the plots.

    Examples:
        >>> import numpy as np
        >>> from src.plotting import plot_depth_profile
        >>> dummy_mean = np.linspace(10, 0, 100) + np.random.rand(100)
        >>> dummy_std = np.ones(100) * 0.5
        >>> plot_depth_profile(dummy_mean, dummy_std, d1=50)
    """
    if not all(isinstance(arr, np.ndarray) for arr in [mean_profile, std_profile]):
        logger.error("Inputs mean_profile and std_profile must be NumPy arrays.")
        raise TypeError("Inputs mean_profile and std_profile must be NumPy arrays.")
    if not isinstance(d1, int):
        logger.error("Input d1 must be an integer.")
        raise TypeError("Input d1 must be an integer.")
    if not (mean_profile.shape == std_profile.shape) or mean_profile.size == 0:
        logger.error("Profile inputs must be non-empty and have the same shape.")
        raise ValueError("Profile inputs must be non-empty and have the same shape.")

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(mean_profile, color='blue', linewidth=1, alpha=0.9, label='Mean Profile')
    ax.fill_between(range(len(mean_profile)),
                     mean_profile - std_profile,
                     mean_profile + std_profile,
                     color='lightblue', alpha=0.6, label='Standard Deviation')
    ax.axvline(x=d1, color='red', linestyle='--', linewidth=2, label=f'Wound Border Position (d1={d1})')
    
    ax.set_title('Mean Depth Profile of Periwound Area', fontsize=16)
    ax.set_xlabel('Profile Position (Wound Interior -> Surrounding Skin)', fontsize=12)
    ax.set_ylabel('Mean Corrected Depth', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()
    logger.debug("Depth profile plot displayed.")

    plt.close(fig)


# Visualizing feature fits on the mean depth profile
def plot_profiles_and_fits(mean_profile: np.ndarray, std_profile: np.ndarray,
                           smoothed_profile: np.ndarray, features: Dict[str, Any],
                           d1: int, p1: int, transition_width: int):
    """
    Plots the mean and std deviation profiles, the smoothed profile, and the curve fits.

    This function visualizes the core output of the feature extraction module, showing
    how the linear and sigmoid functions fit the smoothed depth profile across the
    bed, edge, and skin regions.

    Args:
        mean_profile (np.ndarray): 
            The raw mean depth profile.
        std_profile (np.ndarray): 
            The standard deviation profile.
        smoothed_profile (np.ndarray): 
            The smoothed mean depth profile used for fitting.
        features (Dict[str, Any]): 
            A dictionary of extracted features from the profile.
        d1 (int): 
            The index of the wound edge (baseline contour).
        p1 (int): 
            A redundant parameter (likely from original code), but kept for consistency.
        transition_width (int): 
            The width of the edge transition region.

    Returns:
        None

    Raises:
        TypeError: 
            If inputs are not of expected types.
        ValueError: 
            If inputs have inconsistent shapes or are empty.
    Output:
        - Log:
            A debug message is logged upon successful plotting.
        - Display:
            A matplotlib window displaying the plots.

    Examples:
        >>> import numpy as np
        >>> from src.plotting import plot_profiles_and_fits
        >>> dummy_mean = np.linspace(10, 0, 200) + np.random.rand(200)
        >>> dummy_std = np.ones(200) * 0.5
        >>> dummy_smoothed = np.linspace(10, 0, 200)
        >>> dummy_features = {'bed_fit_success': 1, 'bed_slope': -0.1, 'bed_intercept': 10,
        ...                   'edge_fit_success': 1, 'edge_amplitude': 10, 'edge_steepness': 0.5,
        ...                   'edge_midpoint': 100, 'edge_offset': 0,
        ...                   'skin_fit_success': 1, 'skin_slope': 0, 'skin_intercept': 0}
        >>> plot_profiles_and_fits(dummy_mean, dummy_std, dummy_smoothed, dummy_features, d1=100, p1=100, transition_width=50)
    """
    from .feature_extraction import linear_func, sigmoid_func
    
    if not all(isinstance(arr, np.ndarray) for arr in [mean_profile, std_profile, smoothed_profile]):
        logger.error("Inputs must be NumPy arrays.")
        raise TypeError("Inputs must be NumPy arrays.")
    if not all(isinstance(v, (int, float)) for v in [d1, p1, transition_width]):
        logger.error("Inputs d1, p1, and transition_width must be numeric.")
        raise TypeError("Inputs d1, p1, and transition_width must be numeric.")
    if not all(arr.shape == mean_profile.shape for arr in [std_profile, smoothed_profile]):
        logger.error("All profile inputs must have the same shape.")
        raise ValueError("Profile inputs must have the same shape.")

    fig, ax = plt.subplots(figsize=(15, 7))
    
    x = np.arange(len(mean_profile))
    
    # Plot the smoothed profile
    ax.plot(x, smoothed_profile, color='gray', 
             linewidth=2, 
             label='Smoothed Mean Profile')
    
    # Plot the standard deviation as a shaded area around the smoothed profile
    ax.fill_between(x, smoothed_profile - std_profile, 
                     smoothed_profile + std_profile, 
                     color='lightblue', alpha=0.5, 
                     label='Standard Deviation')

    ax.axvline(x=d1, color='red', 
                linestyle='--', linewidth=2, 
                label=f'Wound Edge (baseline contour)')

    # Define regions for plotting fits
    end_bed = d1 - transition_width // 2
    x_bed = np.arange(0, end_bed)
    start_edge = end_bed
    end_edge = min(len(mean_profile), d1 + transition_width // 2)
    x_edge = np.arange(start_edge, end_edge)
    start_skin = end_edge
    x_skin = np.arange(start_skin, len(mean_profile))

    # Plot fits if they were successful (check for success flag and not NaN)
    if features.get('bed_fit_success') == 1 and not pd.isna(features.get('bed_slope')):
        params_bed = [features['bed_slope'], features['bed_intercept']]
        ax.plot(x_bed, linear_func(x_bed, *params_bed), color='blue', 
                 linewidth=3, label="Wound Bed Fit")

    if features.get('edge_fit_success') == 1 and not pd.isna(features.get('edge_amplitude')):
        params_edge = [features['edge_amplitude'], features['edge_steepness'], features['edge_midpoint'], features['edge_offset']]
        ax.plot(x_edge, sigmoid_func(x_edge, *params_edge), 
                 color='green', linewidth=3, 
                 label="Edge Sigmoid Fit")

    if features.get('skin_fit_success') == 1 and not pd.isna(features.get('skin_slope')):
        params_skin = [features['skin_slope'], features['skin_intercept']]
        ax.plot(x_skin, linear_func(x_skin, *params_skin), 
                 color='purple', linewidth=3, 
                 label="Healthy Skin Fit")

    ax.set_title('Mean Depth Profile with Piecewise Curve Fitting',fontsize=16)
    ax.set_xlabel('Width of the rectified strip',fontsize=14)
    ax.set_ylabel('Mean depth',fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(False)
    ax.set_xlim(20, 170)
    
    plt.show()
    logger.debug("Piecewise curve fits plot displayed.")

    plt.close(fig)

        
def plot_embedding(embedding: np.ndarray, save_path: Path):
    """
    Creates and saves a scatter plot of the 2D PaCMAP embedding.

    This function visualizes the feature dataset after dimensionality reduction,
    providing an initial view of the data's inherent structure and potential clusters.

    Args:
        embedding (np.ndarray): 
            The 2D NumPy array of the PaCMAP embedding.
            Expected shape: (N, 2).
        save_path (Path): 
            The full path, including filename, to save the plot.

    Returns:
        None

    Raises:
        TypeError: 
            If `embedding` is not a NumPy array or `save_path` is not a Path object.
        ValueError: 
            If `embedding` does not have 2 dimensions.
        IOError: 
            If there's an issue saving the file.

    Output:
        - Log:
            Informational messages about successful saving. Errors if saving fails.
        - File:
            A PNG file of the plot at `save_path`.

    Examples:
        >>> import numpy as np
        >>> from src.plotting import plot_embedding
        >>> from pathlib import Path
        >>> # Create a dummy embedding and temporary path
        >>> dummy_embedding = np.random.rand(100, 2)
        >>> dummy_path = Path('./dummy_embedding.png')
        >>> plot_embedding(dummy_embedding, dummy_path)
    """
    if not isinstance(embedding, np.ndarray):
        logger.error("Input 'embedding' must be a NumPy array.")
        raise TypeError("Input 'embedding' must be a NumPy array.")
    if embedding.shape[1] != 2:
        logger.error(f"Input 'embedding' must be a 2D array. Shape: {embedding.shape}.")
        raise ValueError("Input 'embedding' must be a 2D array.")
    if not isinstance(save_path, Path):
        logger.error("Input 'save_path' must be a Path object.")
        raise TypeError("Input 'save_path' must be a Path object.")

    fig, ax = plt.subplots()
    ax.scatter(embedding[:, 0], embedding[:, 1], s=1, alpha=0.5)
    ax.set_title("PaCMAP Embedding of Wound Border Features")
    ax.set_xlabel("PaCMAP Dimension 1")
    ax.set_ylabel("PaCMAP Dimension 2")
    
    try:
        fig.savefig(str(save_path), dpi=300)
        logger.info(f"PaCMAP embedding plot saved to {save_path}.")
    except Exception as e:
        logger.exception(f"Failed to save PaCMAP embedding plot to {save_path}.")
        raise IOError(f"Failed to save plot: {e}") from e
    finally:
        plt.show()
        


# Ploting HDBSCAN clusters
def plot_clusters(embedding: np.ndarray, cluster_labels: np.ndarray, save_path: Path):
    """
    Creates and saves a scatter plot of the HDBSCAN clusters on the embedding.

    This function visualizes the output of the HDBSCAN algorithm, with each
    identified cluster represented by a different color. Noise points (labeled -1)
    are shown in a distinct color to provide a clear view of the clustering results.

    Args:
        embedding (np.ndarray): 
            The 2D NumPy array of the PaCMAP embedding.
        cluster_labels (np.ndarray): 
            A 1D NumPy array of cluster labels from HDBSCAN.
        save_path (Path): 
            The full path, including filename, to save the plot.

    Returns:
        None

    Raises:
        TypeError: 
            If inputs are not NumPy arrays or `save_path` is not a Path object.
        ValueError: 
            If input shapes are inconsistent.
        IOError: 
            If there's an issue saving the file.
    Output:
        - Log: 
            Informational messages about successful saving. Errors if saving fails.
        - File:
            A PNG file of the plot at `save_path`.

    Examples:
        >>> import numpy as np
        >>> from src.plotting import plot_clusters
        >>> from pathlib import Path
        >>> dummy_embedding = np.random.rand(100, 2)
        >>> dummy_labels = np.random.randint(-1, 3, 100) # -1, 0, 1, 2
        >>> dummy_path = Path('./dummy_clusters.png')
        >>> plot_clusters(dummy_embedding, dummy_labels, dummy_path)
    """
    if not all(isinstance(arr, np.ndarray) for arr in [embedding, cluster_labels]):
        logger.error("Inputs must be NumPy arrays.")
        raise TypeError("Inputs must be NumPy arrays.")
    if embedding.shape[0] != cluster_labels.shape[0]:
        logger.error("Input shapes are inconsistent.")
        raise ValueError("Input shapes must be consistent.")
    if not isinstance(save_path, Path):
        logger.error("Input 'save_path' must be a Path object.")
        raise TypeError("Input 'save_path' must be a Path object.")

    fig, ax = plt.subplots(figsize=(10, 10))
    unique_labels = np.unique(cluster_labels)
    cmap = plt.get_cmap('viridis', len(unique_labels))

    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1], hue=cluster_labels,
        palette=cmap, s=20, alpha=0.7, ax=ax
    )
    ax.set_title("Wound Border Types Clusters Identified by HDBSCAN + PaCMAP")
    ax.set_xlabel("PaCMAP Dimension 1")
    ax.set_ylabel("PaCMAP Dimension 2")
    ax.legend(
        title='Clusters',
        loc='upper right',
        ncol=2,
        fontsize=8,
        title_fontsize=9,
        labels=[str(label) for label in unique_labels]
    )
    try:
        fig.savefig(str(save_path), dpi=300)
        logger.info(f"HDBSCAN clusters plot saved to {save_path}.")
    except Exception as e:
        logger.exception(f"Failed to save HDBSCAN clusters plot to {save_path}.")
        raise IOError(f"Failed to save plot: {e}") from e
    finally:
        plt.show()

# Plotting feature distribution box plots
def plot_feature_distributions_by_cluster(df: pd.DataFrame, features_to_plot: List[str], save_path: Path):
    """
    Generates and saves a grid of box plots for key features across identified clusters.

    This function provides a visual, statistical overview of the differences between the
    identified wound border types. Each box plot shows the distribution of a feature
    within a cluster, highlighting the unique characteristics of each group.

    Args:
        df (pd.DataFrame): 
            The DataFrame containing feature values and 'cluster_label'.
        features_to_plot (List[str]): 
            A list of feature names to include in the plots.
        save_path (Path): 
            The full path, including filename, to save the plot.

    Returns:
        None

    Raises:
        TypeError: 
            If `df` is not a Pandas DataFrame or `features_to_plot` is not a list.
        ValueError: 
            If required columns are missing from `df`.
        IOError: 
            If there's an issue saving the file.

    Output:
        - Log:
            Informational messages about successful saving. Errors if saving fails.
        - File:
            A PNG file of the plot at `save_path`.

    Examples:
        >>> import pandas as pd
        >>> from src.plotting import plot_feature_distributions_by_cluster
        >>> from pathlib import Path
        >>> dummy_df = pd.DataFrame({
        ...     'feat_A': np.random.rand(100),
        ...     'feat_B': np.random.rand(100),
        ...     'cluster_label': np.random.randint(0, 3, 100)
        ... })
        >>> dummy_path = Path('./dummy_distributions.png')
        >>> plot_feature_distributions_by_cluster(dummy_df, ['feat_A', 'feat_B'], dummy_path)
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Input 'df' must be a Pandas DataFrame.")
        raise TypeError("Input 'df' must be a Pandas DataFrame.")
    if not isinstance(features_to_plot, list):
        logger.error("Input 'features_to_plot' must be a list of strings.")
        raise TypeError("Input 'features_to_plot' must be a list of strings.")
    if 'cluster_label' not in df.columns:
        logger.error("DataFrame must contain 'cluster_label' column.")
        raise ValueError("DataFrame must contain 'cluster_label' column.")

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    for i, feature in enumerate(features_to_plot):
        if feature in df.columns:
            sns.boxplot(x='cluster_label', y=feature, data=df, ax=axes[i])
            axes[i].set_title(f"Distribution of {feature}")
        else:
            logger.warning(f"Feature '{feature}' not found in DataFrame. Skipping plot.")
            axes[i].set_visible(False)
            
    fig.suptitle("Feature Distributions Across Clusters", fontsize=16)
    plt.tight_layout()
    
    try:
        fig.savefig(str(save_path), dpi=300)
        logger.info(f"Feature distribution plots saved to {save_path}.")
    except Exception as e:
        logger.exception(f"Failed to save feature distribution plots to {save_path}.")
        raise IOError(f"Failed to save plot: {e}") from e
    finally:
        plt.close(fig)

# Visualize Sample images from clusters
def plot_cluster_image_grid(cluster_groups: pd.Series, image_dir: Path, save_path: Path, num_samples: int = 3):
    """
    Displays a grid of representative sample images for each identified cluster.

    This function provides a critical qualitative validation step, allowing a user
    to visually inspect whether the computationally identified clusters correspond
    to distinct and meaningful morphological patterns in the wound images.

    Args:
        cluster_groups (pd.Series): 
            A Pandas Series mapping each cluster label to a list of image IDs.
            This is typically the output of `save_cluster_assignments`.
        image_dir (Path): 
            The Path to the directory containing the original RGB image files.
        save_path (Path): 
            The full path, including filename, to save the plot.
        num_samples (int): 
            The number of random images to display for each cluster. Defaults to 3.

    Returns:
        None

    Raises:
        TypeError: 
            If `cluster_groups` is not a Pandas Series or `image_dir` is not a Path object.
        IOError: 
            If there's an issue loading image files or saving the plot.

    Output:
        - Log:
            Informational messages about successful plotting. Warnings for images that fail to load.
          Errors if saving fails.
        - File:
            A PNG file of the plot at `save_path`.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pathlib import Path
        >>> import cv2
        >>> from src.plotting import plot_cluster_image_grid
        >>> # Create a dummy image directory and files
        >>> temp_dir = Path('./temp_images_for_grid'); temp_dir.mkdir(exist_ok=True)
        >>> cv2.imwrite(str(temp_dir / 'imgA.png'), np.zeros((10,10,3), dtype=np.uint8))
        >>> cv2.imwrite(str(temp_dir / 'imgB.png'), np.ones((10,10,3), dtype=np.uint8) * 255)
        >>> # Create a dummy cluster_groups Series
        >>> groups = pd.Series([['imgA'], ['imgB']], index=[0, 1])
        >>> dummy_path = Path('./dummy_image_grid.png')
        >>> plot_cluster_image_grid(groups, temp_dir, dummy_path)
    """
    if not isinstance(cluster_groups, pd.Series):
        logger.error("Input 'cluster_groups' must be a Pandas Series.")
        raise TypeError("Input 'cluster_groups' must be a Pandas Series.")
    if not isinstance(image_dir, Path):
        logger.error("Input 'image_dir' must be a Path object.")
        raise TypeError("Input 'image_dir' must be a Path object.")
    if not isinstance(save_path, Path):
        logger.error("Input 'save_path' must be a Path object.")
        raise TypeError("Input 'save_path' must be a Path object.")

    cluster_labels = cluster_groups.index.tolist()
    num_clusters = len(cluster_labels)
    
    fig, axes = plt.subplots(num_clusters, num_samples, figsize=(5 * num_samples, 5 * num_clusters))
    if num_clusters == 1: # Handle case with only one cluster
        axes = np.array([axes])
        
    for i, label in enumerate(cluster_labels):
        image_ids = cluster_groups[label]
        sample_ids = np.random.choice(image_ids, size=min(num_samples, len(image_ids)), replace=False)
        
        for j, image_id in enumerate(sample_ids):
            image_path = image_dir / f"{image_id}.png"
            if image_path.exists():
                img = cv2.imread(str(image_path))
                if img is not None:
                    axes[i, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    axes[i, j].set_title(f"Cluster {label}")
                    axes[i, j].axis('off')
                else:
                    logger.warning(f"Could not read image file for Cluster {label}: {image_id}.")
            else:
                logger.warning(f"Image file not found for Cluster {label}: {image_id}.")

    plt.tight_layout()
    try:
        fig.savefig(str(save_path), dpi=300)
        logger.info(f"Cluster image grid plot saved to {save_path}.")
    except Exception as e:
        logger.exception(f"Failed to save cluster image grid plot to {save_path}.")
        raise IOError(f"Failed to save plot: {e}") from e
    finally:
        plt.close(fig)

# Showing confusion matrix of test results of random forest classifier
def plot_confusion_matrix(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray, save_path: Path):
    """
    Visualizes and saves the confusion matrix for the supervised classifier.

    This function provides a visual overview of the classifier's performance,
    showing which classes are correctly identified and which are frequently
    misclassified.

    Args:
        model (RandomForestClassifier): 
            The trained classifier model.
        X_test (np.ndarray): 
            Testing features.
        y_test (np.ndarray): 
            Testing target labels.
        save_path (Path): 
            The full path, including filename, to save the plot.

    Returns:
        None

    Raises:
        TypeError: 
            If inputs are not of expected types.
        ValueError: 
            If inputs are empty or have inconsistent shapes.
        IOError: 
            If there's an issue saving the file.

    Output:
        - Console/Log:
            Informational messages about successful saving. Errors if saving fails.
        - File:
            A PNG file of the plot at `save_path`.

    Examples:
        >>> import numpy as np
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from src.classification import plot_confusion_matrix
        >>> from pathlib import Path
        >>> # Dummy data
        >>> model = RandomForestClassifier(random_state=42).fit(np.random.rand(10, 5), np.array([0, 1]*5))
        >>> X_test = np.random.rand(10, 5)
        >>> y_test = np.array([0, 1]*5)
        >>> save_path = Path('./dummy_confusion_matrix.png')
        >>> plot_confusion_matrix(model, X_test, y_test, save_path)

    Relationships:
        - Dependencies:
            Relies on `numpy`, `sklearn.metrics`, and `matplotlib`.
        - Used by:
            The classification pipeline to report model performance.
    """
    if not isinstance(X_test, np.ndarray) or not isinstance(y_test, np.ndarray):
        logger.error("Inputs must be NumPy arrays.")
        raise TypeError("Inputs must be NumPy arrays.")
    if not isinstance(save_path, Path):
        logger.error("Input 'save_path' must be a Path object.")
        raise TypeError("Input 'save_path' must be a Path object.")

    if X_test.size == 0 or X_test.shape[0] != y_test.shape[0]:
        logger.error("Input arrays are empty or have inconsistent shapes.")
        raise ValueError("Input arrays must be non-empty and have consistent shapes.")

    logger.info("Plotting confusion matrix.")
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='viridis', ax=ax, xticks_rotation='vertical')
        ax.set_title("Confusion Matrix for Random Forest Classifier")
        plt.tight_layout()
        
        fig.savefig(str(save_path), dpi=300)
        logger.info(f"Confusion matrix plot saved to {save_path}.")
    except Exception as e:
        logger.exception(f"Failed to save confusion matrix plot to {save_path}.")
        raise IOError(f"Failed to save plot: {e}") from e
    finally:
        plt.close(fig)

def plot_feature_importances(feature_importance_df: pd.DataFrame, save_path: Path):
    """
    Creates and saves a bar chart of feature importance scores from a trained model.

    This function provides a visual ranking of the most influential features for the
    classification task, offering insights into which quantitative metrics are most
    critical for distinguishing between wound border types.

    Args:
        feature_importance_df (pd.DataFrame): 
            A DataFrame with 'feature' and 'importance' columns.
        save_path (Path): 
            The full path, including filename, to save the plot.

    Returns:
        None

    Raises:
        TypeError: 
            If `feature_importance_df` is not a Pandas DataFrame or `save_path` is not a Path object.
        ValueError: 
            If required columns ('feature', 'importance') are missing.
        IOError: 
            If there's an issue saving the file.

    Output:
        - Log:
            Informational messages about successful saving. Errors if saving fails.
        - File:
            A PNG file of the plot at `save_path`.

    Examples:
        >>> import pandas as pd
        >>> from src.plotting import plot_feature_importances
        >>> from pathlib import Path
        >>> dummy_df = pd.DataFrame({
        ...     'feature': ['feat_A', 'feat_B', 'feat_C'],
        ...     'importance': [0.5, 0.3, 0.2]
        ... })
        >>> dummy_path = Path('./dummy_feature_importances.png')
        >>> plot_feature_importances(dummy_df, dummy_path)
    """
    if not isinstance(feature_importance_df, pd.DataFrame):
        logger.error("Input 'feature_importance_df' must be a Pandas DataFrame.")
        raise TypeError("Input 'feature_importance_df' must be a Pandas DataFrame.")
    if not isinstance(save_path, Path):
        logger.error("Input 'save_path' must be a Path object.")
        raise TypeError("Input 'save_path' must be a Path object.")
    if 'feature' not in feature_importance_df.columns or 'importance' not in feature_importance_df.columns:
        logger.error("DataFrame must contain 'feature' and 'importance' columns.")
        raise ValueError("DataFrame must contain 'feature' and 'importance' columns.")

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, ax=ax)
    ax.set_title("Feature Importance Ranking from the Random Forest Model")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    plt.tight_layout()
    
    try:
        fig.savefig(str(save_path), dpi=300)
        logger.info(f"Feature importance plot saved to {save_path}.")
    except Exception as e:
        logger.exception(f"Failed to save feature importance plot to {save_path}.")
        raise IOError(f"Failed to save plot: {e}") from e
    finally:
        plt.close(fig)