"""
This module provides a suite of preprocessing functions essential for wound image analysis.

It encompasses functionalities for filtering wound masks based on quality criteria,
cleaning depth map data using Z-score filtering, correcting depth maps for body
curvature, and unrolling peri-wound regions into standardized strips for feature
extraction.

Functions:
    - `validate_wound_masks`: Filters wound mask images based on area and component count.
    - `zscore_filter`: Applies a Z-score filter to depth maps within body masks to remove outliers.
    - `quad_surface`: A helper function defining the quadratic surface model for curve fitting.
    - `depth_corrction_for_body_curvature`: Corrects depth maps for body's natural curvature using surface fitting.
    - `sample_pixels_from_contour`: Samples image pixels along a mask's longest contour.
    - `unroll_periwound_to_image`: Transforms irregular peri-wound regions into rectangular strips via morphological operations.

Typical use:
    This module is typically used in the early stages of the image processing pipeline,
    after initial data loading, to clean, normalize, and transform raw image and depth
    data into a suitable format for subsequent feature extraction, clustering, and classification.
"""

import cv2
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from scipy.optimize import curve_fit

# Get a logger instance for the module
logger = logging.getLogger(__name__)

def validate_wound_masks(mask_directory: Path, filtering_params: Dict[str, Any]) -> pd.DataFrame:
    """
    Filters wound mask images based on specified quality criteria.

    This function processes image files within a given directory, applying checks for the number of
    connected components and the area of the primary wound region. Only image IDs that satisfy all
    criteria are considered valid and returned as a Pandas DataFrame.
    
    Args:
        mask_directory (Path):
            The path to the folder containing the wound mask images. This should be a resolved Path object.
        filtering_params (Dict[str, Any]): 
            Dictionary containing filtering parameters:
                - 'pixel_area_threshold' (int): Minimum pixel area required for the largest wound component to be considered valid.
                - 'max_wound_components' (int): The expected (and required) number of connected wound components in the mask. Typically 1.

    Returns:
        pd.DataFrame: 
            A Pandas DataFrame with a single column 'image_id' containing
            the IDs of images that meet the filtering criteria. Returns an
            empty DataFrame if no images pass the filter or if the mask
            directory is empty/invalid.

    Raises:
        FileNotFoundError: 
            If the specified `mask_directory` does not exist.
        IOError: 
            If there's an issue reading files from the `mask_directory`.
        TypeError: 
            If input parameters (`mask_directory`, `filtering_params`, `output_filepath`) are not of the expected types.
        ValueError: 
            If `filtering_params` is improperly structured (e.g., missing keys).

    Examples:
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> from src.preprocessing import validate_wound_masks
        >>> from src.config_manager import Config
        >>> Assume config class is defined and has a method to get paths and filtering parameters
        >>> paths = config.get_paths() # load paths from config_manager
        >>> filtering_params = config.get_filtering_params() # load filtering parameters from config_manager
        >>> filter_masks_by_area_and_component_count(paths['wound_masks_dir'],filtering_params)

    Relationships:
        Used by:
            The main application entry point (e.g., `run_pipeline.py`) might call this function
            as part of the initial data preparation phase.
    """
    
    logger.info("Starting mask filtering process.")

    # Extract filtering parameters from the dictionary
    area_threshold = filtering_params['pixel_area_threshold']
    max_components = filtering_params['max_wound_components']

    # Ensure the mask_directory is a Path object and exists
    if not mask_directory.is_dir():
        logger.error(f"Mask directory not found: '{mask_directory}'. Aborting filtering.")
        raise FileNotFoundError(f"Mask directory not found at '{mask_directory}'")

    image_filepaths: List[Path] = []

    try:
        # Iterate over directory contents and filter for valid image files.
        for f in mask_directory.iterdir():
            if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                image_filepaths.append(f)
        logger.info(f"Found {len(image_filepaths)} potential mask files to check.")
    except Exception as e:
        logger.exception(f"Failed to read contents of directory '{mask_directory}'. Aborting filtering.")
        raise IOError(f"Error reading directory '{mask_directory}': {e}") from e
    
    valid_image_ids = []
    
    # Process each mask file
    for filepath_obj in tqdm(image_filepaths):
        try:
            # Read the mask image
            mask = cv2.imread(str(filepath_obj), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Could not read mask file: '{filepath_obj.name}'. Skipping this image.")
                continue
            
            # Find all connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            # Actual wound components are num_labels - 1 (excluding background)
            num_wound_components = num_labels - 1
            
            # Check number of wound components
            if num_wound_components != max_components:
                logger.debug(f"Skipping '{filepath_obj.name}': Found {num_wound_components} components, expected {max_components}.")
                continue
            
            # Ensure at least one foreground component
            if num_labels < 2:
                logger.warning(f"Skipping '{filepath_obj.name}': No foreground components found despite check (num_labels={num_labels}).")
                continue 

            # Get area of the largest component (label 1 is usually the largest non-background)
            component_area = stats[1, cv2.CC_STAT_AREA]

            # Check wound component's area
            if component_area > area_threshold:
                image_id = filepath_obj.stem 
                valid_image_ids.append(image_id)
                logger.debug(f"'{filepath_obj.name}' passed filters. Area: {component_area}, Components: {num_wound_components}.")
            else:
                logger.debug(f"Skipping '{filepath_obj.name}': Area {component_area} is below threshold {area_threshold}.")

        except IndexError:
            logger.exception(f"Failed to extract component area for '{filepath_obj.name}'. Likely unexpected mask structure. Skipping.")
            continue
        except Exception as e:
            logger.exception(f"An unexpected error occurred while processing mask '{filepath_obj.name}'. Skipping this image.")
            continue
    if valid_image_ids:
        filtered_df = pd.DataFrame(valid_image_ids, columns=['image_id'])
        logger.debug(f"Filtering complete. Returning DataFrame with {len(filtered_df)} filtered image IDs.")
        return filtered_df
    else:
        logger.info("No valid image IDs found. Returning an empty DataFrame.")
        return pd.DataFrame(columns=['image_id'])


#--- Z score filter function ---
def zscore_filter(body: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a Z-score filter to the depth map within the body mask to remove outliers.

    This function helps in removing noise (e.g., specular reflections) from the depth data
    by identifying and masking out pixels whose depth values are statistically too far
    from the mean. The body mask is simultaneously updated to reflect the removed outlier regions.

    Args:
        body (np.ndarray): 
            A 2D NumPy array representing the binary body mask (e.g., 0 for background, >0 for body).
            Expected dtype: uint8. Shape: (H, W).
        depth (np.ndarray): 
            A 2D NumPy array representing the depth map values.
            Expected dtype: float32 or uint16 (converted internally). Shape: (H, W).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - body_clensed (np.ndarray): The updated body mask with outlier regions zeroed out.
                                         dtype: uint8. Shape: (H, W).
            - depth_clensed (np.ndarray): The depth map with outlier pixels removed (set to 0).
                                          dtype: Same as input depth. Shape: (H, W).

    Raises:
        TypeError: 
            If input `body` or `depth` are not NumPy arrays.
        ValueError: 
            If input image shapes are inconsistent or empty.
        RuntimeError: 
            If standard deviation is zero causing division by zero in Z-score calculation.

    Output:
        - Console/Log: 
            Debug messages on filter application. Error messages for invalid inputs.
        - Return Value: 
            Two NumPy arrays representing the cleaned body mask and depth map.

    Examples:
        >>> import numpy as np
        >>> from src.preprocessing import zscore_filter
        >>> # Dummy data: A 5x5 depth map with one outlier and a corresponding body mask
        >>> depth_map_in = np.array([[10,10,10,10,10],[10,10,1000,10,10],[10,10,10,10,10],[10,10,10,10,10],[10,10,10,10,10]], dtype=np.float32)
        >>> body_mask_in = np.ones((5,5), dtype=np.uint8) * 255 # All body initially
        >>> cleaned_body, cleaned_depth = zscore_filter(body_mask_in.copy(), depth_map_in.copy())
 
    Relationships:
        - Dependencies: 
            Relies on `numpy` for array operations (`np.ndarray`, `.mean()`, `.std()`)
            and `cv2` for bitwise operations (`cv2.bitwise_and`).
        - Used by: 
            `depth_corrction_for_body_curvature` in this module as a preprocessing step.
    """
    if not isinstance(body, np.ndarray) or not isinstance(depth, np.ndarray):
        logger.error("Input 'body' and 'depth' must be NumPy arrays.")
        raise TypeError("Input 'body' and 'depth' must be NumPy arrays.")
    if body.shape != depth.shape or body.size == 0:
        logger.error(f"Input shapes mismatch or empty: body {body.shape}, depth {depth.shape}.")
        raise ValueError("Input 'body' and 'depth' arrays must have consistent and non-empty shapes.")

    # Convert depth to float for calculations
    depth_float = depth.astype(np.float32) if depth.dtype != np.float32 else depth

    # Calculate Z-score
    depth_mean = depth_float.mean()
    depth_std = depth_float.std()

    # Handle zero standard deviation
    if depth_std == 0:
        logger.warning("Standard deviation of depth is zero in zscore_filter. All values are identical. No Z-score filtering applied.")
        return body.copy(), depth.copy() 

    z_score = (depth_float - depth_mean) / depth_std 
    
    body_clensed = body.copy()
    
    # Filter outliers based on Z-score (outside -2 and 2 std deviations)
    body_clensed[(z_score < -2) | (z_score > 2)] = 0
    
    # Re-mask depth map with updated body mask
    depth_clensed = cv2.bitwise_and(depth, depth, mask=body_clensed)
    
    logger.debug("Z-score filter applied: outliers removed from depth map and body mask.")
    return body_clensed, depth_clensed


#--- Depth correction for body curvature ---
def quad_surface(xy: Tuple[np.ndarray, np.ndarray], a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
    """
    Defines a quadratic surface equation for fitting body curvature.

    This helper function represents the mathematical model $z = ax^2 + by^2 + cxy + dx + ey + f$
    used to approximate the natural curvature of the human body from depth data. It's
    specifically designed to be compatible with `scipy.optimize.curve_fit`.

    Args:
        xy (Tuple[np.ndarray, np.ndarray]): 
            A tuple containing two 2D NumPy arrays:
            (x_coordinates_grid, y_coordinates_grid) for the grid points.
            Note: `curve_fit` expects x and y to be the first two args,
            so (x_coords, y_coords) corresponds to (y_coords_grid, x_coords_grid)
            when called from `depth_corrction_for_body_curvature` due to its setup.
        a, b, c, d, e, f (float): 
            Coefficients of the quadratic surface equation. These are the
            parameters that `curve_fit` will determine.

    Returns:
        np.ndarray: 
            A 2D NumPy array representing the calculated height (z) values for the given coordinates.
            The shape will be that of `x_coordinates_grid` (or `y_coordinates_grid`).
    Output:
        - Return Value: 
            A NumPy array of calculated z-values.

    Examples:
        >>> import numpy as np
        >>> from src.preprocessing import quad_surface
        >>> # Create a simple 2x2 grid for x and y coordinates
        >>> y_coords, x_coords = np.mgrid[0:2, 0:2] # y_coords is rows, x_coords is columns
        >>>
        >>> surface = quad_surface((x_coords, y_coords), a=1, b=2, c=3, d=4, e=5, f=6)

    Relationships:
        - Used by:
             `depth_corrction_for_body_curvature` as the model function for `scipy.optimize.curve_fit`.
    """
    x, y = xy # x and y here are the independent variables for curve_fit
    return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y


def depth_corrction_for_body_curvature(wound: np.ndarray, body: np.ndarray, depth: np.ndarray,
                                       kernel_size: Tuple[int, int] = (20, 20),
                                       dilation_iterations: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Corrects the depth map for the natural curvature of the human body.

    This function isolates the intrinsic topography of the wound by subtracting a
    fitted quadratic surface from the depth map, which represents the body's curvature.
    It applies Z-score filtering first to clean initial depth data outliers.

    Args:
        wound (np.ndarray): 
            A 2D NumPy array representing the binary wound mask (255=wound, 0=background). Shape: (H, W).
        body (np.ndarray): 
            A 2D NumPy array representing the binary body mask (255=body, 0=background). Shape: (H, W).
        depth (np.ndarray): 
            A 2D NumPy array representing the raw depth map. Shape: (H, W).
        kernel_size (Tuple[int, int]): 
            Size of the elliptical kernel for morphological operations. Defaults to (20, 20).
        dilation_iterations (int): 
            Number of iterations for wound mask dilation to define the peri-wound ROI. Defaults to 15.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - body_clensed (np.ndarray): The body mask after Z-score filtering. dtype: uint8.
            - depth_clensed (np.ndarray): The depth map after Z-score filtering (before curvature correction).
            - depth_corrected (np.ndarray): The final depth map corrected for body curvature,
                                            with background and outliers masked out.

    Raises:
        ValueError: 
            If input image shapes are inconsistent or `scipy.optimize.curve_fit` fails to converge or fit.
        TypeError: 
            If input arrays are not NumPy arrays.
        RuntimeError: 
            If image dimensions change unexpectedly during processing.

    Output:
        - Console/Log: 
            Informational messages about filtering and surface fitting steps. Warnings
            if fitting data is insufficient or if `curve_fit` fails. Errors for critical input issues.
        - Return Value: 
            Three NumPy arrays representing the cleaned body, cleaned depth, and curvature-corrected depth.

    Examples:
        >>> import numpy as np
        >>> import cv2
        >>> from src.preprocessing import depth_corrction_for_body_curvature
        >>> # Assume logging is set up

        >>> # Dummy data: A simple wound, body, and depth map (100x100)
        >>> dummy_wound = np.zeros((100,100), dtype=np.uint8); cv2.circle(dummy_wound, (50,50), 10, 255, -1)
        >>> dummy_body = np.ones((100,100), dtype=np.uint8) * 255
        >>> dummy_depth = np.linspace(0, 100, 10000).reshape(100,100).astype(np.float32) # Simulated gradient depth
        >>> dummy_depth[50,50] = 500 # A 'wound' dip to make it interesting

        >>> cleaned_body, cleaned_depth, corrected_depth = depth_corrction_for_body_curvature(
        ...     dummy_wound, dummy_body, dummy_depth)

    Relationships:
        - Dependencies: 
            Calls `zscore_filter()` and `quad_surface()`. Uses `cv2` for morphology
            (`cv2.getStructuringElement`, `cv2.dilate`, `cv2.bitwise_and`),
            `numpy` for array operations, and `scipy.optimize.curve_fit` for surface fitting.
        - Used by: 
            The main application entry point (`run_pipeline.py`)
            during the preprocessing stage, immediately after data loading.
    """
    # Input Validation: Check types and shapes
    if not all(isinstance(arr, np.ndarray) for arr in [wound, body, depth]):
        logger.error("All inputs (wound, body, depth) must be NumPy arrays.")
        raise TypeError("All inputs must be NumPy arrays.")
    if not (wound.shape == body.shape == depth.shape) or wound.size == 0:
        logger.error(f"Input shapes mismatch or empty: wound {wound.shape}, body {body.shape}, depth {depth.shape}.")
        raise ValueError("All input masks and depth map must have the same (H, W) shape and be non-empty.")

    logger.debug("Applying Z-score filter to depth map.")
    # Apply Z-score filter to remove initial noise
    body_clensed, depth_clensed = zscore_filter(body.copy(), depth.copy())

    # Define the kernel for morphological operations
    kernel = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE, ksize = kernel_size)
    
    # Dilate wound mask to create peri-wound ROI
    logger.debug(f"Dilating wound mask {dilation_iterations} times.")
    dilated_wound = cv2.dilate(wound, kernel = kernel, iterations=dilation_iterations)
    
    # Isolate the annulus of peri-wound skin
    dilated_wound = dilated_wound - wound
    
    # Re-mask peri-wound region with Z-score cleaned body mask
    dilated_wound = cv2.bitwise_and(dilated_wound, dilated_wound, mask=body_clensed)   
    
    # Select depth values within peri-wound annulus
    depth_dilated_roi = np.where(dilated_wound == 0, 0, depth_clensed)
    
    # Get coordinates and non-zero depth values from ROI
    y_coords_flat, x_coords_flat = np.nonzero(depth_dilated_roi)
    z_values_flat = depth_dilated_roi[y_coords_flat, x_coords_flat]

    # Check for sufficient data points for fitting
    if len(x_coords_flat) < 6:
        logger.warning("Not enough data points (%d) in peri-wound region for quadratic surface fitting (minimum 6 required). Skipping curvature correction. Returning Z-score cleaned depth.", len(x_coords_flat))
        return body_clensed, depth_clensed, depth_clensed

    logger.debug("Fitting quadratic surface to peri-wound depth data.")
    try:
        # Fit quadratic polynomial surface
        popt, pcov = curve_fit(quad_surface, xdata = (x_coords_flat, y_coords_flat), ydata = z_values_flat)
    except RuntimeError as e:
        logger.warning(f"Quadratic surface fitting failed (RuntimeError: {e}). Skipping curvature correction for this image. Returning Z-score cleaned depth.")
        return body_clensed, depth_clensed, depth_clensed
    except Exception as e: # Catch other unexpected errors during fitting
        logger.exception(f"An unexpected error occurred during quadratic surface fitting. Skipping curvature correction. Returning Z-score cleaned depth.")
        raise IOError(f"Error during quadratic surface fitting: {e}") from e

    # Create new grid matching original depth map dimensions
    h, w = depth.shape
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    
    # Calculate fitted surface over entire grid
    fitted_surface = quad_surface((x_grid, y_grid), *popt)

    # Subtract fitted surface to correct for body curvature
    depth_corrected = depth_clensed - fitted_surface
    
    # Re-mask corrected depth map with cleaned body mask
    depth_corrected = cv2.bitwise_and(depth_corrected, depth_corrected, mask=body_clensed)
    
    logger.debug("Depth map corrected for body curvature.")
    return body_clensed, depth_clensed, depth_corrected
 

# --- Peri-Wound Rectification ---

def sample_pixels_from_contour(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Samples pixels from an image along its longest contour within a given binary mask.

    This utility function extracts pixel values that lie directly on the primary
    contour identified within the mask. It's typically used to get a baseline
    pixel strip along a wound border or other defined boundary for subsequent analysis.

    Args:
        img (np.ndarray): 
            The input image (RGB or grayscale) from which to sample pixels.
            Shape: (H, W) or (H, W, C).
        mask (np.ndarray): 
            A binary mask (uint8, 255=foreground, 0=background) defining the
            region of interest from which contours are extracted. Shape: (H, W).

    Returns:
        np.ndarray: 
            A NumPy array containing the sampled pixels. For color images,
            shape will be (N, 1, C); for grayscale, (N, 1). N is the contour length.

    Raises:
        TypeError: 
            If input `img` or `mask` are not NumPy arrays.
        ValueError: 
            If the mask is empty, has inconsistent shape with `img`, or no
            valid contours are found, or if a contour is degenerate.

    Output:
        - Console/Log: 
            Debug messages on pixel sampling. Error messages for invalid masks/contours.
        - Return Value: 
            A NumPy array of sampled pixel values.

    Example:
        >>> import numpy as np
        >>> import cv2
        >>> from src.preprocessing import sample_pixels_from_contour
        >>> # Dummy image and mask: a white square on a black background
        >>> dummy_img = np.zeros((50,50,3), dtype=np.uint8); dummy_img[10:40, 10:40] = 200 # Grey square
        >>> dummy_mask = np.zeros((50,50), dtype=np.uint8); dummy_mask[10:40, 10:40] = 255 # White square mask
        >>> pixels = sample_pixels_from_contour(dummy_img, dummy_mask)

    Relationships:
        - Dependencies:
            Relies on `cv2` for contour finding (`cv2.findContours`) and `numpy` for array manipulation.
        - Used by: 
            `unroll_periwound_to_image` to get the initial baseline pixel strip and 
            subsequent pixel rings during erosion/dilation.
    """
    if not isinstance(img, np.ndarray) or not isinstance(mask, np.ndarray):
        logger.error("Input 'img' and 'mask' must be NumPy arrays.")
        raise TypeError("Input 'img' and 'mask' must be NumPy arrays.")
    if mask.size == 0 or img.size == 0 or img.shape[:2] != mask.shape:
        logger.error(f"Mask {mask.shape} or image {img.shape} is empty or shapes mismatch.")
        raise ValueError("Input mask and image must be non-empty and have consistent (H, W) shapes.")

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Check if contours are found
    if not contours:
        logger.error("No contours found in the provided mask. Cannot sample pixels.")
        raise ValueError("No contours found in the provided mask.")

    # Select the longest contour
    longest_contour = max(contours, key=len)
    
    # Squeeze to remove single-dimension entries
    contour_coords = np.squeeze(longest_contour)
    
    # Handle degenerate contour
    if contour_coords.ndim == 1:
        logger.warning("Longest contour is 1D (a single point or line segment). Degenerate mask. Cannot sample pixels.")
        raise ValueError("Degenerate contour found (1D). Cannot sample pixels from it.")

    # Sample pixels from image using contour coordinates
    y_indices = contour_coords[:, 1]
    x_indices = contour_coords[:, 0]

    pixs = img[y_indices, x_indices]
    
    # Expand dimensions for strip construction
    pixs = np.expand_dims(pixs, axis=1)
    
    logger.debug(f"Sampled {pixs.shape[0]} pixels from contour.")
    return pixs

def unroll_periwound_to_image(img: np.ndarray, mask: np.ndarray, iterations: int = 100,
                             kernel_size: Tuple[int, int] = (3, 3)) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    "Unrolls" the peri-wound region from an image (RGB or depth map) into a standardized rectangular strip.

    This technique transforms the irregular, ring-like area around a wound into a
    fixed-geometry strip, enabling consistent feature extraction. It achieves this
    by iteratively dilating (outwards from wound) and eroding (inwards from wound)
    the wound mask and sampling pixels at each step. This process helps to flatten
    the wound border profile for quantitative analysis.

    Args:
        img (np.ndarray): 
            The input image (RGB or depth map) to unroll. Shape: (H, W) or (H, W, C).
        mask (np.ndarray): 
            The binary wound mask (uint8, 255=wound, 0=background). Shape: (H, W).
        iterations (int): 
            The maximum number of erosion/dilation steps. This determines
            the potential total width of the unrolled strip. Defaults to 100.
        kernel_size (Tuple[int, int]): 
            Size of the elliptical kernel for morphological operations. Defaults to (3, 3).

    Returns:
        Tuple[np.ndarray, Tuple[int, int]]: A tuple containing:
            - unrolled_strip (np.ndarray): The 2D (for grayscale/depth) or 3D (for RGB) unrolled rectangular strip.
                                           Shape: (Contour_Length, Total_Strip_Width, [Channels]).
            - erosion_dilation_counts (Tuple[int, int]): A tuple (num_eroded_strips, num_dilated_strips)
                                                        representing the effective width of the inner
                                                        (wound bed) and outer (periwound skin) regions.

    Raises:
        TypeError: 
            If input `img` or `mask` are not NumPy arrays.
        ValueError: 
            If input `mask` or `img` is empty or has inconsistent shape, or if `sample_pixels_from_contour` fails internally.
        RuntimeError: 
            If image dimensions change unexpectedly during processing, or if
            unrolled image has an unexpected number of dimensions.

    Output:
        - Console/Log: 
            Informational messages about unrolling progress, and warnings for early stopping
            due to mask issues. Errors for critical input or processing failures.
        - Return Value: 
            The unrolled image strip and a tuple of erosion/dilation counts.

    Examples:
        >>> import numpy as np
        >>> import cv2
        >>> from pathlib import Path
        >>> from src.preprocessing import unroll_periwound_to_image
        >>> # Assume logging is set up
        >>>
        >>> # Dummy data: A simple 100x100 image and wound mask (circle in center)
        >>> dummy_img = np.zeros((100,100,3), dtype=np.uint8); dummy_img[40:60, 40:60] = 200 # Grey square
        >>> dummy_wound_mask = np.zeros((100,100), dtype=np.uint8); cv2.circle(dummy_wound_mask, (50,50), 10, 255, -1)
        >>>
        >>> unrolled_strip, counts = unroll_periwound_to_image(dummy_img, dummy_wound_mask, iterations=10)
  
    Relationships:
        - Dependencies: 
            Calls `sample_pixels_from_contour()`. Uses `cv2` for morphological operations
            (`cv2.getStructuringElement`, `cv2.dilate`, `cv2.erode`, `cv2.resize`) and `numpy` for array manipulation.
        - Used by:
            The main application entry point (`run_pipeline.py`)
            during the preprocessing stage, typically after depth correction.
    """
    if not isinstance(img, np.ndarray) or not isinstance(mask, np.ndarray):
        logger.error("Input 'img' and 'mask' must be NumPy arrays.")
        raise TypeError("Input 'img' and 'mask' must be NumPy arrays.")
    if mask.size == 0 or img.size == 0 or img.shape[:2] != mask.shape:
        logger.error(f"Mask {mask.shape} or image {img.shape} is empty or shapes mismatch.")
        raise ValueError("Input mask and image must be non-empty and have consistent (H, W) shapes.")

    # Define the kernel for morphological operations
    kernel = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE, ksize = kernel_size) 
    
    # Get the base strip of pixels from the initial mask's contour
    logger.debug("Extracting baseline pixels from wound contour.")
    try:
        pixl1 = sample_pixels_from_contour(img, mask)
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to get baseline contour for unrolling: {e}. Cannot unroll image.")
        raise ValueError(f"Baseline contour extraction failed for unrolling: {e}") from e
    
    # Store contour height and initial width (1)
    h, w, *_ = pixl1.shape
    
    # Initialize list of dilated strips; base strip is first
    dilations = [pixl1]
    
    temp_msk = mask.copy()
    num_dilations = 0

    logger.debug(f"Starting iterative dilation (max {iterations} steps).")
    for i in range(iterations):
        prev_temp_msk = temp_msk.copy()
        temp_msk = cv2.dilate(temp_msk, kernel, iterations=1)

        if np.array_equal(prev_temp_msk, temp_msk): # Check if dilation had no effect
            logger.debug(f"Dilation stopped at iteration {i}: Mask no longer expanding.")
            break

        try:
            pixl2 = sample_pixels_from_contour(img, temp_msk)
        except (ValueError, TypeError) as e: # If contour sampling fails
            logger.warning(f"Dilation stopped at iteration {i}: Could not find contour in dilated mask or sampling failed: {e}. Preserving current strip width.")
            break

        # Resize sampled pixels to match original contour height
        pixl2 = cv2.resize(pixl2, (w, h), interpolation=cv2.INTER_CUBIC)
        
        dilations.append(pixl2)
        num_dilations += 1

    logger.debug(f"Completed {num_dilations} dilation steps.")

    # Initialize list of eroded strips
    erosions: List[np.ndarray] = []
    num_erosions = 0
    temp_msk = mask.copy() # Reset mask to original for erosion

    logger.debug(f"Starting iterative erosion (max {iterations} steps).")
    for i in range(iterations):
        prev_temp_msk = temp_msk.copy()
        temp_msk = cv2.erode(temp_msk, kernel, iterations=1)

        # Check if erosion had no effect or mask disappeared/fragmented
        if np.array_equal(prev_temp_msk, temp_msk) or np.sum(temp_msk) == 0:
            logger.debug(f"Erosion stopped at iteration {i}: Mask no longer shrinking or became empty.")
            break
        num_components_after_erode, _ = cv2.connectedComponents(temp_msk)
        if num_components_after_erode > 2: # More than background and one object
            logger.warning(f"Erosion stopped at iteration {i}: Mask fragmented into {num_components_after_erode-1} objects. Preserving integrity.")
            break

        try:
            pixl2 = sample_pixels_from_contour(img, temp_msk)
        except (ValueError, TypeError) as e: # If contour sampling fails
            logger.warning(f"Erosion stopped at iteration {i}: Could not find contour in eroded mask or sampling failed: {e}. Preserving current strip width.")
            break

        # Resize sampled pixels to original contour height
        pixl2 = cv2.resize(pixl2, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Prepend to erosions list for correct stacking order
        erosions.insert(0, pixl2)
        num_erosions += 1
    
    logger.debug(f"Completed {num_erosions} erosion steps.")

    # Concatenate eroded and dilated strips
    if not (erosions + dilations):
        logger.error("No strips generated during unrolling (empty erosions and dilations list). Cannot create unrolled image.")
        raise ValueError("Failed to generate any strips for unrolling.")

    unrolled_image = np.hstack(erosions + dilations)
    
    # Transpose for final unrolled strip shape
    if unrolled_image.ndim == 3: # For a 3D RGB image
        unrolled_strip = unrolled_image.transpose(1, 0, 2)
    elif unrolled_image.ndim == 2: # For a 2D grayscale/depth map
        unrolled_strip = unrolled_image.transpose(1, 0)
    else:
        logger.error(f"Unexpected number of dimensions ({unrolled_image.ndim}) for unrolled image. Expected 2 or 3. Aborting.")
        raise RuntimeError(f"Unrolled image has unexpected dimensions: {unrolled_image.ndim}")
    
    logger.debug(f"Image unrolled into strip of shape {unrolled_strip.shape}. Effective erosion/dilation counts: ({num_erosions}, {num_dilations}).")
    return unrolled_strip, (num_erosions, num_dilations)