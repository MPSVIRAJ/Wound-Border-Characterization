import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis
import cv2

#---Getting the profiles from the rectified depth map---
def calculate_depth_profiles(rect_depth):
    """
    Calculates the mean and standard deviation profiles from a rectified depth map.

    Args:
        rect_depth (np.array): A 2D numpy array of the unrolled depth map,
                               where NaNs may be present.

    Returns:
        tuple: A tuple containing the mean_profile (1D np.array) and
               the std_profile (1D np.array).
    """
    if rect_depth is None or rect_depth.size == 0:
        return np.array([]), np.array([])
        
    mean_profile = np.nanmean(rect_depth, axis=1)
    std_profile = np.nanstd(rect_depth, axis=1)
    
    return mean_profile, std_profile


#--- Fitting and feature extraction from the profiles ---

# -- Helper fuctions
def r_squared(y_true, y_pred):
    """Calculates the R-squared (coefficient of determination) value."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

def linear_func(x, m, c):
    """A simple linear function."""
    return m * x + c

def sigmoid_func(x, L, k, x0, offset):
    """A generalized logistic (sigmoid) function."""
    return L / (1 + np.exp(-k * (x - x0))) + offset

def get_spectral_features(profile_segment):
    """Calculates spectral features from a 1D profile segment."""
    n = len(profile_segment)
    if n < 2: return {'spectral_centroid': np.nan, 'spectral_entropy': np.nan}
    
    fft_vals = np.fft.fft(profile_segment)
    power_spectrum = np.abs(fft_vals[:n // 2]) ** 2
    freqs = np.fft.fftfreq(n, d=1)[:n // 2]
    
    centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else 0.0
    norm_power = power_spectrum / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else np.zeros_like(power_spectrum)
    entropy = -np.sum(norm_power * np.log2(norm_power + 1e-9))
    
    return {'spectral_centroid': centroid, 'spectral_entropy': entropy}

# --- Main Feature Extraction Function ---

def extract_features_from_profile(mean_profile, d1, transition_width):
    """
    Takes a mean depth profile and extracts a comprehensive feature vector.

    Args:
        mean_profile (np.array): The 1D array of the smoothed depth profile.
        d1 (int): The estimated position of the wound border.
        transition_width (int): The width around d1 to define the edge region.

    Returns:
        dict: A dictionary containing all extracted features.
    """
    features = {}

    # --- 1. Divide the profile into three regions ---
    # wound bed
    end_bed = max(0, d1 - transition_width // 2)
    x_bed, y_bed = np.arange(0, end_bed), mean_profile[0:end_bed]
    # edge
    start_edge = end_bed
    end_edge = min(len(mean_profile), d1 + transition_width // 2)
    x_edge, y_edge = np.arange(start_edge, end_edge), mean_profile[start_edge:end_edge]
    # healthy skin
    start_skin = end_edge
    x_skin, y_skin = np.arange(start_skin, len(mean_profile)), mean_profile[start_skin:]

    # --- 2. Perform Piecewise Fitting ---
    try:
        params, _ = curve_fit(linear_func, x_bed, y_bed)
        features.update({'bed_slope': params[0], 
                         'bed_intercept': params[1], 
                         'bed_r2': r_squared(y_bed, linear_func(x_bed, *params)),
                           'bed_fit_success': 1})
    except (RuntimeError, TypeError):
        features.update({'bed_slope': np.nan, 
                         'bed_intercept': np.nan, 
                         'bed_r2': np.nan, 
                         'bed_fit_success': 0})

    try:
        p0 = [np.max(y_edge)-np.min(y_edge), 0.5, np.mean(x_edge), np.min(y_edge)]
        params, _ = curve_fit(sigmoid_func, x_edge, y_edge, p0=p0, maxfev=10000)
        features.update({'edge_amplitude': params[0], 
                         'edge_steepness': params[1], 
                         'edge_midpoint': params[2], 
                         'edge_offset': params[3], 
                         'edge_r2': r_squared(y_edge, sigmoid_func(x_edge, *params)), 
                         'edge_fit_success': 1})
    except (RuntimeError, TypeError):
        features.update({'edge_amplitude': np.nan, 
                         'edge_steepness': np.nan, 
                         'edge_midpoint': np.nan, 
                         'edge_offset': np.nan, 
                         'edge_r2': np.nan, 
                         'edge_fit_success': 0})

    try:
        params, _ = curve_fit(linear_func, x_skin, y_skin)
        features.update({'skin_slope': params[0], 
                         'skin_intercept': params[1], 
                         'skin_r2': r_squared(y_skin, linear_func(x_skin, *params)), 
                         'skin_fit_success': 1})
    except (RuntimeError, TypeError):
        features.update({'skin_slope': np.nan, 
                         'skin_intercept': np.nan, 
                         'skin_r2': np.nan, 
                         'skin_fit_success': 0})

    # --- 3. Extract Statistical and Spectral Features ---
    if len(y_bed) > 0: features.update({'bed_mean': np.mean(y_bed), 
                                        'bed_std': np.std(y_bed), 
                                        'bed_skew': skew(y_bed), 
                                        'bed_kurtosis': kurtosis(y_bed)})
    if len(y_edge) > 0: features.update({'edge_mean': np.mean(y_edge), 
                                         'edge_std': np.std(y_edge), 
                                         'edge_skew': skew(y_edge), 
                                         'edge_kurtosis': kurtosis(y_edge)})
    if len(y_skin) > 0: features.update({'skin_mean': np.mean(y_skin), 
                                         'skin_std': np.std(y_skin), 
                                         'skin_skew': skew(y_skin), 
                                         'skin_kurtosis': kurtosis(y_skin)})
    if len(y_edge) > 0: features.update(get_spectral_features(y_edge))
    
    return features