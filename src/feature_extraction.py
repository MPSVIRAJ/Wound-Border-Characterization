"""
This module provides a set of functions for extracting quantitative
features from rectified wound images' depth profiles.

It includes utilities for calculating mean and standard deviation profiles,
performing curve fitting with linear and sigmoid models, computing statistical
and spectral features, and applying a Butterworth low-pass filter to smooth data.
The core functionality is encapsulated in a single function that orchestrates
these steps to produce a full feature vector for a given image's profile.

Functions:
- `calculate_depth_profiles`: Calculates mean and standard deviation profiles from a 2D depth strip.
- `r_squared`: Computes the R-squared value for a given curve fit.
- `linear_func`: A helper function defining a linear model for curve fitting.
- `sigmoid_func`: A helper function defining a sigmoid model for curve fitting.
- `get_spectral_features`: Extracts spectral centroid and entropy from a profile segment.
- `get_statistical_features`: Computes statistical moments (mean, std, skew, kurtosis) for a profile segment.
- `butter_lowpass_filter`: Applies a Butterworth low-pass filter to smooth a signal.
- `extract_features_from_profile`: The main function that orchestrates all feature extraction steps.

Typical use:
    This module is a core part of the feature extraction pipeline. It is typically used after
    preprocessing and unrolling a depth map to generate a single, comprehensive feature vector
    for each image, which is then used for downstream clustering and classification.
"""

import numpy as np
import logging
import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt 
from typing import Tuple, Dict, Any, Optional, List

import cv2

logger = logging.getLogger(__name__)

#---Getting the profiles from the rectified depth map---
def calculate_depth_profiles(rect_depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the mean and standard deviation profiles from a rectified depth map.

    This function processes a 2D rectified depth strip by computing the mean and
    standard deviation for each column (the cross-sectional profile). It is designed
    to be a straightforward utility for initial profile generation.

    Args:
        rect_depth (np.ndarray): 
            The 2D NumPy array representing the rectified depth strip.
            This array is expected to have NaNs for masked regions.
            Shape: (Strip_Height, Strip_Width).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - mean_profile (np.ndarray): The 1D mean depth profile.
            - std_profile (np.ndarray): The 1D standard deviation profile.

    Raises:
        TypeError: 
            If `rect_depth` is not a NumPy array.
        ValueError: 
            If `rect_depth` is empty or has an unexpected number of dimensions.

    Output:
        - Console/Log:
            Informational messages about profile dimensions. Errors for invalid inputs.
        - Return Value:
            Two NumPy arrays representing the mean and standard deviation profiles.

    Examples:
        >>> import numpy as np
        >>> from src.feature_extraction import calculate_depth_profiles
        >>> # Assume a 10x100 rectified depth map
        >>> dummy_rect_depth = np.random.rand(10, 100)
        >>> mean_profile, std_profile = calculate_depth_profiles(dummy_rect_depth)

    Relationships:
        - Dependencies:
            - `numpy`: For array operations (`np.ndarray`, `np.nanmean`, `np.nanstd`).
            - `logging`: For outputting messages.
        - Used by: 
            The `run_pipeline.py` script to generate the 1D profile
            from which all features are extracted.
    """
    if not isinstance(rect_depth, np.ndarray):
        logger.error("Input 'rect_depth' must be a NumPy array.")
        raise TypeError("Input 'rect_depth' must be a NumPy array.")
    if rect_depth.ndim != 2 or rect_depth.size == 0:
        logger.error(f"Input 'rect_depth' must be a non-empty 2D array. Shape: {rect_depth.shape}.")
        raise ValueError("Input 'rect_depth' must be a non-empty 2D array.")

    # Calculate mean and standard deviation of each column along axis=1, ignoring NaNs.
    mean_profile = np.nanmean(rect_depth, axis=1)
    std_profile = np.nanstd(rect_depth, axis=1)
    
    logger.debug(f"Calculated depth profiles. Profile shape: {mean_profile.shape}.")
    return mean_profile, std_profile


#--- Fitting and feature extraction from the profiles ---

# -- Helper fuctions
def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the R-squared (coefficient of determination) value for a curve fit.

    The R-squared value measures how well the fitted model explains the variation
    in the actual data. A value closer to 1 indicates a better fit.

    Args:
        y_true (np.ndarray): 
            The actual data values.
        y_pred (np.ndarray): 
            The values predicted by the fitted model.

    Returns:
        float: 
            The R-squared value. Returns 0 if calculation is not possible.

    Raises:
        TypeError: 
            If inputs are not NumPy arrays.
        ValueError: 
            If inputs have different shapes or are empty.

    Output:
        - Console/Log:
            A warning message is logged if the variance is zero.
        - Return Value:
            A floating-point number representing the R-squared value.

    Examples:
        >>> import numpy as np
        >>> from src.feature_extraction import r_squared
        >>> y_actual = np.array([1, 2, 3, 4, 5])
        >>> y_fitted = np.array([1.1, 2.1, 3.2, 4.0, 5.1])
        >>> r2 = r_squared(y_actual, y_fitted)

    Relationships:
        - Dependencies: 
            - `numpy`: For array operations.
            - `logging`: For outputting messages.
        - Used by:
            `extract_features_from_profile` to quantify the goodness of fit
            for the linear and sigmoid curve fits.
    """
    # Check if inputs are NumPy arrays and have the same shape
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        logger.error("Inputs 'y_true' and 'y_pred' must be NumPy arrays.")
        raise TypeError("Inputs must be NumPy arrays.")
    if y_true.shape != y_pred.shape or y_true.size == 0:
        logger.error("Inputs must be non-empty arrays with the same shape.")
        raise ValueError("Inputs must be non-empty arrays with the same shape.")
        
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        logger.warning("Total sum of squares is zero. R-squared is not meaningful. Returning 0.")
        return 0.0
        
    return 1 - (ss_res / ss_tot)


def linear_func(x: np.ndarray, m: float, c: float) -> np.ndarray:
    """
    A helper function that defines a linear model for curve fitting.
    
    This function represents the equation `y = m * x + c` and is specifically designed
    to be compatible with `scipy.optimize.curve_fit`. It takes an array of x-coordinates
    and calculates the corresponding y-coordinates based on the provided slope and intercept.
    
    Args:
        x (np.ndarray): 
            The input independent variable (e.g., array of x-coordinates).
        m (float): 
            The slope of the line.
        c (float): 
            The y-intercept of the line.
    
    Returns:
        np.ndarray: 
            The calculated dependent variable (y) values.
    
    Raises:
        TypeError: 
            If the input `x` is not a NumPy array.
            
    Output:
        - Console/Log:
            Debug messages confirming the calculation.
        - Return Value:
            A NumPy array containing the computed y-values.
    
    Examples:
        >>> import numpy as np
        >>> from src.feature_extraction import linear_func
        >>> x_vals = np.array([0, 1, 2])
        >>> y_vals = linear_func(x_vals, m=2.0, c=1.0)
    
    Relationships:
        - Dependencies:
            Relies on `numpy` for array operations.
        - Used by:
            `extract_features_from_profile` for fitting the wound bed and periwound skin regions.
    """
    if not isinstance(x, np.ndarray):
        logger.error("Input 'x' must be a NumPy array.")
        raise TypeError("Input 'x' must be a NumPy array.")
        
    y_values = m * x + c
    logger.debug(f"Linear function calculated for input array of shape {x.shape}.")
    return y_values


def sigmoid_func(x: np.ndarray, L: float, k: float, x0: float, offset: float) -> np.ndarray:
    """
    A helper function that defines a generalized logistic (sigmoid) model for curve fitting.
    
    This function represents the equation `y = L / (1 + exp(-k * (x - x0))) + offset`.
    It is useful for modeling the S-shaped transition of the wound edge. It's designed
    to be compatible with `scipy.optimize.curve_fit`.
    
    Args:
        x (np.ndarray): 
            The input independent variable (e.g., array of x-coordinates).
        L (float): 
            The curve's maximum value.
        k (float): 
            The steepness or growth rate of the curve.
        x0 (float): 
            The x-value of the curve's midpoint.
        offset (float): 
            The vertical offset of the curve.
    
    Returns:
        np.ndarray: 
            The calculated dependent variable (y) values.
    
    Raises:
        TypeError: 
            If the input `x` is not a NumPy array.
        
    Output:
        - Console/Log:
            Debug messages confirming the calculation.
        - Return Value:
            A NumPy array containing the computed y-values.
    
    Examples:
        >>> import numpy as np
        >>> from src.feature_extraction import sigmoid_func
        >>> x_vals = np.linspace(-10, 10, 100)
        >>> y_vals = sigmoid_func(x_vals, L=10.0, k=0.5, x0=0.0, offset=0.0)

    Relationships:
        - Dependencies:
            - `numpy`: For array operations.
        - Used by:
            `extract_features_from_profile` for fitting the wound edge transition region.
    """
    if not isinstance(x, np.ndarray):
        logger.error("Input 'x' must be a NumPy array.")
        raise TypeError("Input 'x' must be a NumPy array.")
        
    y_values = L / (1 + np.exp(-k * (x - x0))) + offset
    logger.debug(f"Sigmoid function calculated for input array of shape {x.shape}.")
    return y_values


def get_spectral_features(profile_segment: np.ndarray) -> Dict[str, float]:
    """
    Computes spectral features (centroid, entropy) from a profile's power spectrum.

    This function is used to characterize the textural properties of the wound surface.
    Spectral centroid measures the "center of mass" of the spectrum, while spectral
    entropy measures the "peakiness" or randomness of the signal. The function
    is designed to be robust against zero-variance inputs and uses a refined
    method for power spectrum calculation.

    Args:
        profile_segment (np.ndarray): 
            A 1D NumPy array representing the full profile.

    Returns:
        Dict[str, float]: 
            A dictionary containing the spectral centroid and entropy.
            Returns NaN values for both if the input profile is too short or lacks variance.

    Raises:
        TypeError: 
            If `profile_segment` is not a NumPy array.
        ValueError: 
            If `profile_segment` is an empty array.

    Output:
        - Console/Log:
            A warning message is logged if the profile is too short or if the
          power spectrum sum is zero. Debug messages are logged for successful calculation.
        - Return Value:
            A dictionary of spectral features.

    Examples:
        >>> import numpy as np
        >>> from src.feature_extraction import get_spectral_features
        >>> dummy_profile = np.array([1, 2, 3, 4, 5])
        >>> spectral_feats = get_spectral_features(dummy_profile)

    Relationships:
        - Dependencies:
            - `numpy`: For array operations and FFT (`np.fft.fft`, `np.abs`, `np.sum`, etc.).
            - `logging`: For outputting messages.
        - Used by:
            `extract_features_from_profile` for extracting global textural features from the entire profile.
    """
    if not isinstance(profile_segment, np.ndarray):
        logger.error("Input 'profile_segment' must be a NumPy array.")
        raise TypeError("Input 'profile_segment' must be a NumPy array.")
    
    n = len(profile_segment)
    if n < 2: 
        logger.warning("Input profile has fewer than 2 data points. Cannot calculate spectral features.")
        return {'spectral_centroid': np.nan, 'spectral_entropy': np.nan}
    
    fft_vals = np.fft.fft(profile_segment)
    power_spectrum = np.abs(fft_vals[:n // 2]) ** 2
    
    # Calculate frequencies for spectral centroid calculation
    freqs = np.fft.fftfreq(n, d=1)[:n // 2]
    
    # Calculate spectral centroid
    sum_power = np.sum(power_spectrum)
    if sum_power > 0:
        centroid = np.sum(freqs * power_spectrum) / sum_power
        # Calculate spectral entropy
        norm_power = power_spectrum / sum_power
        # Add a small epsilon to avoid log2(0)
        entropy = -np.sum(norm_power * np.log2(norm_power + 1e-9))
        logger.debug("Spectral features calculated successfully.")
    else:
        logger.warning("Power spectrum sum is zero. Spectral centroid and entropy cannot be calculated.")
        centroid = np.nan
        entropy = np.nan
    
    return {'spectral_centroid': centroid, 'spectral_entropy': entropy}


def get_statistical_features(profile: np.ndarray) -> Dict[str, float]:
    """
    Computes statistical features (mean, std, skewness, kurtosis) for a given 1D profile.

    This function quantifies the distribution and shape of the data within a specific
    region of the depth profile. The four moments of the distribution (mean, standard
    deviation, skewness, and kurtosis) are calculated and returned as a dictionary.

    Args:
        profile (np.ndarray): 
            A 1D NumPy array representing a segmented profile region.

    Returns:
        Dict[str, float]: 
            A dictionary containing the calculated statistical features.

    Raises:
        TypeError: 
            If `profile` is not a NumPy array.
        ValueError: 
            If `profile` is an empty array.

    Output:
        - Console/Log:
            A debug message is logged upon successful calculation. A warning is logged
          for an empty input.
        - Return Value:
            A dictionary of statistical features.

    Examples:
        >>> import numpy as np
        >>> from src.feature_extraction import get_statistical_features
        >>> dummy_profile = np.array([1, 2, 3, 4, 5])
        >>> stats = get_statistical_features(dummy_profile)
 
    Relationships:
        - Dependencies:
            - `numpy`: For array operations.
            - `scipy.stats`: For skewness and kurtosis calculations.
            - `logging`: For outputting messages.
        - Used by:
            `extract_features_from_profile` for each of the three profile regions (bed, edge, skin).
    """
    if not isinstance(profile, np.ndarray):
        logger.error("Input 'profile' must be a NumPy array.")
        raise TypeError("Input 'profile' must be a NumPy array.")
    
    if profile.size == 0:
        logger.warning("Input profile is empty. Cannot calculate statistical features.")
        raise ValueError("Input profile array is empty.")
    
    mean_val = np.mean(profile)
    std_val = np.std(profile)
    skew_val = skew(profile)
    kurt_val = kurtosis(profile)

    logger.debug("Statistical features calculated successfully.")
    
    return {
        "mean": mean_val,
        "std": std_val,
        "skew": skew_val,
        "kurtosis": kurt_val
    }

# ---Filtering with a butterworth filter
def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Applies a Butterworth low-pass filter to a 1D signal.

    This function designs and applies a Butterworth digital low-pass filter, which is a signal
    processing technique used to smooth out high-frequency noise from a signal. The `filtfilt`
    function is used to apply the filter, which ensures zero phase shift in the output.

    Args:
        data (np.ndarray): 
            The 1D input signal (e.g., a depth profile).
        cutoff (float): 
            The cutoff frequency of the filter.
        fs (float): 
            The sampling frequency of the signal.
        order (int): 
            The order of the filter. A higher order results in a sharper cutoff. Defaults to 4.

    Returns:
        np.ndarray: 
            The filtered and smoothed 1D signal.

    Raises:
        TypeError: 
            If `data` is not a NumPy array or `cutoff`/`fs` are not numeric types.
        ValueError: 
            If `data` is empty, `cutoff` or `fs` are non-positive, or `order` is non-positive.
        RuntimeError: 
            If the filter design or application fails.

    Output:
        - Console/Log:
            A debug message is logged upon successful filter application. Warnings are
            logged for invalid input values.
        - Return Value:
            A NumPy array containing the filtered signal.

    Examples:
        >>> import numpy as np
        >>> from src.feature_extraction import butter_lowpass_filter
        >>> # Assume a signal with some high-frequency noise
        >>> t = np.linspace(0, 1, 500, endpoint=False)
        >>> sig = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)
        >>> fs = 500
        >>> cutoff_freq = 20
        >>> filtered_sig = butter_lowpass_filter(sig, cutoff=cutoff_freq, fs=fs)

    Relationships:
        - Dependencies:
            - `numpy`: For array operations.
            - `scipy.signal`: For filter design and application (`butter`, `filtfilt`).
            - `logging`: For outputting messages.
        - Used by:
            This filter function can be used to smooth the mean depth profile (e.g., within `calculate_depth_profiles`).
    """
    # Type and value validation
    if not isinstance(data, np.ndarray):
        logger.error("Input 'data' must be a NumPy array.")
        raise TypeError("Input 'data' must be a NumPy array.")
    if data.size == 0:
        logger.error("Input 'data' array is empty.")
        raise ValueError("Input 'data' array is empty.")
    if not all(isinstance(v, (int, float)) for v in [cutoff, fs]):
        logger.error("Cutoff frequency and sampling frequency must be numeric.")
        raise TypeError("Cutoff frequency and sampling frequency must be numeric.")
    if cutoff <= 0 or fs <= 0:
        logger.error("Cutoff frequency and sampling frequency must be positive.")
        raise ValueError("Cutoff frequency and sampling frequency must be positive.")
    if order <= 0:
        logger.warning(f"Filter order is non-positive ({order}). Using default order of 1.")
        order = 1

    try:
        nyq = 0.5 * fs 
        normal_cutoff = cutoff / nyq
        
         # Design the filter
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        # Apply the filter
        y = filtfilt(b, a, data)
        
        logger.debug("Successfully applied Butterworth low-pass filter to the signal.")
        return y
    except Exception as e:
        logger.exception(f"Filter design or application failed for signal of shape {data.shape}.")
        raise RuntimeError(f"Filter design or application failed: {e}") from e


# --- Main Feature Extraction Function ---

def extract_features_from_profile(mean_profile: np.ndarray, d1: int,
                                  feature_params: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Extracts a comprehensive set of quantitative features from a depth profile.

    This is the main function for feature extraction. It segments the profile into
    three regions (wound bed, edge, skin) and applies statistical, spectral, and
    curve-fitting methods to quantify their characteristics. A comprehensive
    feature dictionary is returned.

    Args:
        mean_profile (np.ndarray): 
            The 1D raw mean depth profile.
            Shape: (Profile_Length,).
        d1 (int): 
            The index of the wound edge (the baseline contour).
        feature_params (Dict[str, Any]): 
            A dictionary containing parameters for feature extraction, including 'transition_width'.

    Returns:
        Tuple[Dict[str, Any], bool]: A tuple containing:
            - features (Dict[str, Any]): 
                A dictionary of extracted features. Keys are
                feature names, values are floats or booleans.
            - smoothed_profile (np.ndarray):
                The low-pass filtered version of the mean_profile
            - success (bool): 
                True if feature extraction was successful, False otherwise.

    Raises:
        TypeError: 
            If `mean_profile` is not a NumPy array.
        ValueError: 
            If `mean_profile` is empty or inputs have inconsistent shapes.
        RuntimeError: 
            If profile segmentation fails unexpectedly.

    
    Output:
        - Console/Log:
            Informational messages about each step and warnings for
            unsuccessful curve fitting. Errors for critical input issues.
        - Return Value:
            A dictionary of features and a success flag.

    Examples:
        >>> import numpy as np
        >>> from src.feature_extraction import extract_features_from_profile
        >>> # Assume dummy profiles are generated by calculate_depth_profiles
        >>> dummy_mean = np.linspace(10, 0, 200) # Simple linear transition
        >>> dummy_std = np.ones(200) * 0.5
        >>> # Assume wound edge is at pixel 100
        >>> feature_params = {'transition_width': 50, 'cutoff_freq': 0.1, 'butter_order': 4}
        >>> features_dict, success_flag = extract_features_from_profile(dummy_mean, d1=100, feature_params=feature_params)


    Relationships:
        - Dependencies:
            - `numpy`: For array operations.
            - `scipy.optimize.curve_fit`: For fitting curves.
            - `scipy.stats`: For statistical calculations.
            - `scipy.signal`: For filtering.
            - `logging`: For outputting messages.
            - `r_squared()`, `linear_func()`, `sigmoid_func()`, `get_spectral_features()`,
              `get_statistical_features()`, `butter_lowpass_filter()`: All functions within this module.
        - Used by: 
            The main application entry point (`run_pipeline.py`)
            to generate the final feature vector for a single image.
    """
    if not isinstance(mean_profile, np.ndarray):
        logger.error("Inputs 'mean_profile' must be NumPy arrays.")
        raise TypeError("Inputs must be NumPy arrays.")
    if mean_profile.size == 0:
        logger.error(f"Inputs empty. mean {mean_profile.shape}.")
        raise ValueError("Inputs must be non-empty arrays of the same shape.")

    profile_length = len(mean_profile)
    features: Dict[str, Any] = {}
    success = True

    # --- Smooth mean profile for curve fitting ---
    if profile_length > 10:
        fs = profile_length
        cutoff_freq = feature_params['cutoff_freq']*fs
        butter_order = feature_params['butter_order']
        smoothed_profile = butter_lowpass_filter(mean_profile, cutoff_freq, fs, butter_order)
    else:
        smoothed_profile = mean_profile

    # --- Divide the profile into three regions ---
    # Get parameters from input parameter dictionary
    transition_width = feature_params['transition_width']
    
    # Determine the profile segmentation based on the transition width
    end_bed = max(0, d1 - transition_width // 2)
    start_edge = end_bed
    end_edge = min(profile_length, d1 + transition_width // 2)
    start_skin = end_edge

    logger.debug("Segmenting profile into bed, edge, and skin regions.")
    try:
        profile_bed = smoothed_profile[:end_bed]
        profile_edge = smoothed_profile[start_edge:end_edge]
        profile_skin = smoothed_profile[start_skin:]
    except Exception as e:
        logger.exception("Failed to segment profile. Profile likely malformed.")
        return {}, smoothed_profile, False

    # --- Perform Piecewise Fitting (on Smoothed Profile) ---
    logger.debug("Fitting curves to profile regions.")
    # Fit linear model to bed
    x_bed = np.arange(0, end_bed)
    try:
        if len(x_bed) > 1:
            popt_bed, _ = curve_fit(linear_func, x_bed, profile_bed)
            y_fitted_bed = linear_func(x_bed, *popt_bed)
            features.update({
                "bed_slope": popt_bed[0],
                "bed_intercept": popt_bed[1],
                "bed_r2": r_squared(profile_bed, y_fitted_bed),
                "bed_fit_success": 1
            })
        else:
            logger.warning("Not enough data points in bed region for linear fit. Skipping.")
            features.update({"bed_fit_success": 0})
            success = False
    except (RuntimeError, IndexError):
        logger.warning("Linear curve fitting failed for the wound bed region. Skipping.")
        features.update({"bed_fit_success": 0})
        success = False

    # Fit sigmoid model to edge
    x_edge = np.arange(start_edge, end_edge)
    try:
        if len(x_edge) > 4:
            p0 = [np.max(profile_edge) - np.min(profile_edge), 1.0, np.mean(x_edge), np.min(profile_edge)]
            popt_edge, _ = curve_fit(sigmoid_func, x_edge, profile_edge, p0=p0, maxfev=10000)
            y_fitted_edge = sigmoid_func(x_edge, *popt_edge)
            features.update({
                "edge_amplitude": popt_edge[0],
                "edge_steepness": popt_edge[1],
                "edge_midpoint": popt_edge[2],
                "edge_offset": popt_edge[3],
                "edge_r2": r_squared(profile_edge, y_fitted_edge),
                "edge_fit_success": 1
            })
        else:
            logger.warning("Not enough data points in edge region for sigmoid fit. Skipping.")
            features.update({"edge_fit_success": 0})
            success = False
    except (RuntimeError, IndexError):
        logger.warning("Sigmoid curve fitting failed for the wound edge region. Skipping.")
        features.update({"edge_fit_success": 0})
        success = False

    # Fit linear model to skin
    x_skin = np.arange(start_skin, profile_length)
    try:
        if len(x_skin) > 1:
            popt_skin, _ = curve_fit(linear_func, x_skin, profile_skin)
            y_fitted_skin = linear_func(x_skin, *popt_skin)
            features.update({
                "skin_slope": popt_skin[0],
                "skin_intercept": popt_skin[1],
                "skin_r2": r_squared(profile_skin, y_fitted_skin),
                "skin_fit_success": 1
            })
        else:
            logger.warning("Not enough data points in skin region for linear fit. Skipping.")
            features.update({"skin_fit_success": 0})
            success = False
    except (RuntimeError, IndexError):
        logger.warning("Linear curve fitting failed for the periwound skin region. Skipping.")
        features.update({"skin_fit_success": 0})
        success = False

    # --- Extract Statistical and Spectral Features ---
    # These features will be calculated on the RAW mean profile.
    y_bed_orig = mean_profile[0:end_bed]
    y_edge_orig = mean_profile[start_edge:end_edge]
    y_skin_orig = mean_profile[start_skin:]
    try:
        logger.debug("Extracting statistical features from mean and standard deviation profiles.")
        for name, p in [('bed', y_bed_orig),
                       ('edge', y_edge_orig),
                       ('skin', y_skin_orig)]:
        
            stats_mean = get_statistical_features(p)
            for k, v in stats_mean.items():
                features[f'{name}_{k}'] = v 

    except ValueError as e:
        logger.warning(f"Statistical feature extraction failed on a raw profile region: {e}. Assigning NaNs.")
        for k in ['mean', 'std', 'skew', 'kurtosis']:
            nan_dict = {f'bed_{k}': np.nan, f'edge_{k}': np.nan, f'skin_{k}': np.nan }
            features.update(nan_dict)
        success = False
    
    # --- Extract Spectral Features ---
    try:
        logger.debug("Extracting spectral features from full raw profile.")
        if mean_profile.size > 0:
            spectral_feats = get_spectral_features(mean_profile)
            features.update(spectral_feats)
        else:
            logger.warning("Mean profile is empty, cannot extract spectral features.")
    except ValueError as e:
        logger.warning(f"Spectral feature extraction failed: {e}. Skipping spectral features.")
        success = False

    logger.info(f"Feature extraction complete. Success status: {success}.")
    return features, smoothed_profile, success