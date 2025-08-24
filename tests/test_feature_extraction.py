"""
This module contains unit tests for the feature extraction functions defined in `src.feature_extraction`.

It uses pytest to verify the correctness, robustness, and proper error/warning handling
of the functions that generate statistical and spectral features from unrolled depth profiles.
The tests cover various scenarios, including successful feature extraction, handling of
invalid inputs, edge cases with short or zero-variance profiles, and simulated failures
during curve fitting.

Functions:
    - `setup_test_logging`: A pytest fixture to configure logging capture.
    - All `test_*` functions: Unit tests for each public and private function within the
        `feature_extraction` module, covering functionality like `calculate_depth_profiles`,
        `r_squared`, `get_spectral_features`, and the main `extract_features_from_profile` pipeline.

Typical use:
    This module is designed to be executed by `pytest` as part of the project's
    automated test suite. It ensures the reliability and accuracy of the feature
    extraction components, which are crucial for the subsequent clustering and
    classification tasks.
"""

import pytest
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt
import logging
import sys
from unittest.mock import patch, Mock

# Import functions from src/feature_extraction.py
from src.feature_extraction import (
    calculate_depth_profiles,
    r_squared,
    linear_func,
    sigmoid_func,
    get_spectral_features,
    get_statistical_features,
    butter_lowpass_filter,
    extract_features_from_profile
)

# --- Fixtures ---

@pytest.fixture(autouse=True)
def setup_test_logging(caplog):
    """
    Sets up basic logging for tests to capture messages.
    'autouse=True' means this fixture runs automatically for all tests.
    """
    logging.getLogger().setLevel(logging.DEBUG)
    if logging.getLogger().hasHandlers():
        logging.getLogger().handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    
    caplog.set_level(logging.DEBUG, logger="src.feature_extraction")

# --- Tests for calculate_depth_profiles ---

def test_calculate_depth_profiles_success():
    """
    GIVEN: a 2D numpy array with valid data.
    WHEN: calculate_depth_profiles is called.
    THEN: It should return two 1D arrays for the mean and standard deviation.
    """
    rect_depth = np.array([
        [10, 20, 30, 40, 50],
        [11, 21, 31, 41, 51],
        [12, 22, 32, 42, 52]
    ])
    expected_mean = np.array([30, 31, 32])
    expected_std = np.array([14.14213562, 14.14213562, 14.14213562])

    mean_profile, std_profile = calculate_depth_profiles(rect_depth)

    assert mean_profile.shape == (3,)
    assert std_profile.shape == (3,)
    np.testing.assert_allclose(mean_profile, expected_mean)
    np.testing.assert_allclose(std_profile, expected_std)


def test_calculate_depth_profiles_with_nan():
    """
    GIVEN: a 2D numpy array with NaN values.
    WHEN: calculate_depth_profiles is called.
    THEN: It should correctly calculate mean and std, ignoring NaNs.
    """
    rect_depth = np.array([
        [10, 20, np.nan, 40, 50],
        [11, 21, 31, 41, np.nan],
        [np.nan, 22, 32, 42, 52]
    ])
    
    # Calculate expected values dynamically to avoid floating-point discrepancies
    expected_mean = np.nanmean(rect_depth, axis=1)
    expected_std = np.nanstd(rect_depth, axis=1)

    mean_profile, std_profile = calculate_depth_profiles(rect_depth)

    np.testing.assert_allclose(mean_profile, expected_mean, rtol=1e-5)
    np.testing.assert_allclose(std_profile, expected_std, rtol=1e-5)


def test_calculate_depth_profiles_single_row():
    """
    GIVEN: a 2D numpy array with a single row.
    WHEN: calculate_depth_profiles is called.
    THEN: It should return two 1D arrays of shape (1,).
    """
    rect_depth = np.array([[10, 20, 30, 40, 50]])
    expected_mean = np.array([30])
    expected_std = np.array([14.14213562])

    mean_profile, std_profile = calculate_depth_profiles(rect_depth)
    
    assert mean_profile.shape == (1,)
    assert std_profile.shape == (1,)
    np.testing.assert_allclose(mean_profile, expected_mean)
    np.testing.assert_allclose(std_profile, expected_std)


def test_calculate_depth_profiles_type_error():
    """
    GIVEN: A non-numpy array input.
    WHEN: calculate_depth_profiles is called.
    THEN: It should raise a TypeError and log an error.
    """
    with pytest.raises(TypeError) as excinfo:
        calculate_depth_profiles([1, 2, 3])

    assert "Input 'rect_depth' must be a NumPy array." in str(excinfo.value)


def test_calculate_depth_profiles_value_error():
    """
    GIVEN: An empty or non-2D numpy array input.
    WHEN: calculate_depth_profiles is called.
    THEN: It should raise a ValueError and log an error.
    """
    with pytest.raises(ValueError) as excinfo:
        calculate_depth_profiles(np.array([]))

    assert "Input 'rect_depth' must be a non-empty 2D array." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        calculate_depth_profiles(np.array([1, 2, 3]))

    assert "Input 'rect_depth' must be a non-empty 2D array." in str(excinfo.value)


# --- Tests for r_squared ---

def test_r_squared_perfect_fit():
    """
    GIVEN: Perfect match between true and predicted values.
    WHEN: r_squared is called.
    THEN: It should return 1.0.
    """
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    assert r_squared(y_true, y_pred) == 1.0


def test_r_squared_imperfect_fit():
    """
    GIVEN: A non-perfect fit.
    WHEN: r_squared is called.
    THEN: It should return a value less than 1.0.
    """
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 2.1, 3.1])
    # Expected value calculated manually
    expected_r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2))
    assert r_squared(y_true, y_pred) == pytest.approx(expected_r2)


def test_r_squared_zero_variance():
    """
    GIVEN: Inputs with zero variance.
    WHEN: r_squared is called.
    THEN: It should return 0.0 and log a warning.
    """
    y_true = np.array([5, 5, 5])
    y_pred = np.array([5, 5, 5])
    
    with patch('src.feature_extraction.logger.warning') as mock_warning:
        result = r_squared(y_true, y_pred)
    
    assert result == 0.0
    mock_warning.assert_called_with("Total sum of squares is zero. R-squared is not meaningful. Returning 0.")


def test_r_squared_type_error():
    """
    GIVEN: non-numpy array inputs.
    WHEN: r_squared is called.
    THEN: It should raise a TypeError.
    """
    with pytest.raises(TypeError):
        r_squared([1, 2], np.array([1, 2]))


def test_r_squared_value_error():
    """
    GIVEN: arrays with different shapes.
    WHEN: r_squared is called.
    THEN: It should raise a ValueError.
    """
    with pytest.raises(ValueError):
        r_squared(np.array([1, 2]), np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        r_squared(np.array([]), np.array([]))

# --- Tests for linear_func ---

def test_linear_func_success():
    """
    GIVEN: valid x, m, and c values.
    WHEN: linear_func is called.
    THEN: It should return the correct y values.
    """
    x = np.array([0, 1, 2, 3])
    m, c = 2.0, 1.0
    expected_y = np.array([1.0, 3.0, 5.0, 7.0])
    y = linear_func(x, m, c)
    np.testing.assert_allclose(y, expected_y)

# --- Tests for sigmoid_func ---

def test_sigmoid_func_success():
    """
    GIVEN: valid x, L, k, x0, and offset values.
    WHEN: sigmoid_func is called.
    THEN: It should return the correct y values.
    """
    x = np.array([-5, 0, 5])
    L, k, x0, offset = 10.0, 1.0, 0.0, 0.0
    
    # Calculate expected values dynamically to avoid floating-point discrepancies
    expected_y = L / (1 + np.exp(-k * (x - x0))) + offset
    
    y = sigmoid_func(x, L, k, x0, offset)
    np.testing.assert_allclose(y, expected_y, rtol=1e-5)

# --- Tests for get_spectral_features ---

def test_get_spectral_features_success():
    """
    GIVEN: A valid 1D profile.
    WHEN: get_spectral_features is called.
    THEN: It should return a dictionary with spectral centroid and entropy.
    """
    fs = 100
    t = np.arange(0, 100) / fs
    profile = np.sin(2 * np.pi * 5 * t) + np.random.randn(100) * 0.1 
    
    features = get_spectral_features(profile)
    
    assert 'spectral_centroid' in features
    assert 'spectral_entropy' in features
    assert not np.isnan(features['spectral_centroid'])
    assert not np.isnan(features['spectral_entropy'])
    
    # Check if centroid is somewhat close to the expected frequency (5 Hz)
    assert features['spectral_centroid'] > 0 and features['spectral_centroid'] < 0.5 * fs


def test_get_spectral_features_short_profile():
    """
    GIVEN: A profile with fewer than 2 data points.
    WHEN: get_spectral_features is called.
    THEN: It should return NaN values and log a warning.
    """
    profile = np.array([1])
    
    with patch('src.feature_extraction.logger.warning') as mock_warning:
        features = get_spectral_features(profile)
    
    assert np.isnan(features['spectral_centroid'])
    assert np.isnan(features['spectral_entropy'])
    mock_warning.assert_called_with("Input profile has fewer than 2 data points. Cannot calculate spectral features.")


def test_get_spectral_features_zero_power_spectrum():
    """
    GIVEN: A zero-power profile (e.g., all zeros or constant values).
    WHEN: get_spectral_features is called.
    THEN: It should return NaN values and log a warning.
    """
    profile = np.zeros(10)
    
    with patch('src.feature_extraction.logger.warning') as mock_warning:
        features = get_spectral_features(profile)
    
    assert np.isnan(features['spectral_centroid'])
    assert np.isnan(features['spectral_entropy'])
    mock_warning.assert_called_with("Power spectrum sum is zero. Spectral centroid and entropy cannot be calculated.")

# --- Tests for get_statistical_features ---

def test_get_statistical_features_success():
    """
    GIVEN: A 1D numpy array.
    WHEN: get_statistical_features is called.
    THEN: It should return a dictionary with mean, std, skew, and kurtosis.
    """
    profile = np.array([1, 2, 3, 4, 5])
    expected_mean = 3.0
    expected_std = np.std(profile)
    expected_skew = skew(profile)
    expected_kurtosis = kurtosis(profile)

    stats = get_statistical_features(profile)
    
    assert stats['mean'] == expected_mean
    assert stats['std'] == pytest.approx(expected_std)
    assert stats['skew'] == pytest.approx(expected_skew)
    assert stats['kurtosis'] == pytest.approx(expected_kurtosis)


def test_get_statistical_features_single_element_array():
    """
    GIVEN: A 1D numpy array with a single element.
    WHEN: get_statistical_features is called.
    THEN: It should return predictable values (e.g., std=0, skew/kurtosis=0).
    """
    profile = np.array([10.0])
    
    stats = get_statistical_features(profile)
    
    assert stats['mean'] == 10.0
    assert stats['std'] == 0.0
    assert np.isnan(stats['skew'])
    assert np.isnan(stats['kurtosis'])


def test_get_statistical_features_value_error():
    """
    GIVEN: An empty array.
    WHEN: get_statistical_features is called.
    THEN: It should raise a ValueError.
    """
    with pytest.raises(ValueError) as excinfo:
        get_statistical_features(np.array([]))
    assert "Input profile array is empty." in str(excinfo.value)

# --- Tests for butter_lowpass_filter ---

def test_butter_lowpass_filter_success():
    """
    GIVEN: A noisy signal and valid filter parameters.
    WHEN: butter_lowpass_filter is called.
    THEN: It should return a smoothed signal.
    """
    fs = 100.0  
    cutoff = 20.0  
    t = np.arange(100) / fs
    data = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(100)
    
    filtered_data = butter_lowpass_filter(data, cutoff, fs)
    
    assert np.var(filtered_data) < np.var(data)
    assert np.allclose(data[0], filtered_data[0], atol=0.1) 


def test_butter_lowpass_filter_value_error_empty_data():
    """
    GIVEN: An empty data array.
    WHEN: butter_lowpass_filter is called.
    THEN: It should raise a ValueError.
    """
    with pytest.raises(ValueError) as excinfo:
        butter_lowpass_filter(np.array([]), 10, 100)
    assert "Input 'data' array is empty." in str(excinfo.value)


def test_butter_lowpass_filter_value_error_non_positive_params():
    """
    GIVEN: Non-positive cutoff or fs.
    WHEN: butter_lowpass_filter is called.
    THEN: It should raise a ValueError.
    """
    with pytest.raises(ValueError) as excinfo:
        butter_lowpass_filter(np.array([1, 2]), 0, 100)
    assert "Cutoff frequency and sampling frequency must be positive." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        butter_lowpass_filter(np.array([1, 2]), 10, -1)
    assert "Cutoff frequency and sampling frequency must be positive." in str(excinfo.value)


def test_butter_lowpass_filter_type_error():
    """
    GIVEN: Invalid input types.
    WHEN: butter_lowpass_filter is called.
    THEN: It should raise a TypeError.
    """
    with pytest.raises(TypeError) as excinfo:
        butter_lowpass_filter([1, 2], 10, 100)
    assert "Input 'data' must be a NumPy array." in str(excinfo.value)


def test_butter_lowpass_filter_warning_non_positive_order():
    """
    GIVEN: A non-positive filter order.
    WHEN: butter_lowpass_filter is called.
    THEN: It should log a warning and use the default order.
    """
    data = np.random.randn(100)
    with patch('src.feature_extraction.logger.warning') as mock_warning:
        filtered_data = butter_lowpass_filter(data, 10, 100, order=0)
    
    assert filtered_data.shape == data.shape
    mock_warning.assert_called_with("Filter order is non-positive (0). Using default order of 1.")

# --- Tests for extract_features_from_profile ---

def test_extract_features_from_profile_success():
    """
    GIVEN: A valid mean profile and d1 index.
    WHEN: extract_features_from_profile is called.
    THEN: It should return a dictionary of features and a success flag of True.
    """
    profile_length = 200
    d1 = 100 
    feature_params = {
        'transition_width': 50, 
        'cutoff_freq': 0.1,
        'butter_order': 4
    }

    end_bed = d1 - feature_params['transition_width'] // 2 
    start_edge = end_bed 
    end_edge = d1 + feature_params['transition_width'] // 2 
    start_skin = end_edge 

    bed_segment = np.linspace(10.0, 8.0, end_bed)
    x_edge_segment = np.linspace(-5.0, 5.0, end_edge - start_edge)
    edge_segment = sigmoid_func(x_edge_segment, L=5.0, k=1.0, x0=0.0, offset=3.0) 

    skin_segment = np.linspace(3.0, 1.0, profile_length - start_skin)

    mean_profile = np.concatenate([
        bed_segment,
        edge_segment,
        skin_segment
    ])
    mean_profile += np.random.randn(profile_length) * 0.05
    assert len(mean_profile) == profile_length

    features, smoothed_profile, success = extract_features_from_profile(mean_profile, d1, feature_params)

    assert success is True 
    assert isinstance(features, dict)
    assert "bed_slope" in features
    assert "edge_steepness" in features
    assert "skin_mean" in features
    assert "spectral_centroid" in features
    assert features.get("bed_fit_success") == 1
    assert features.get("edge_fit_success") == 1
    assert features.get("skin_fit_success") == 1
    assert smoothed_profile.shape == mean_profile.shape


def test_extract_features_from_profile_empty_profile():
    """
    GIVEN: An empty mean profile.
    WHEN: extract_features_from_profile is called.
    THEN: It should raise a ValueError and log an error.
    """
    mean_profile = np.array([])
    d1 = 0
    feature_params = {}
    
    with pytest.raises(ValueError) as excinfo:
        extract_features_from_profile(mean_profile, d1, feature_params)
    assert "Inputs must be non-empty arrays of the same shape." in str(excinfo.value)


def test_extract_features_from_profile_fitting_failure_bed(caplog, monkeypatch):
    """
    GIVEN: A scenario where the linear fit for the bed region fails.
    WHEN: extract_features_from_profile is called.
    THEN: It should log a warning and return success=False.
    """
    profile_length = 200
    mean_profile = np.linspace(10, 0, profile_length) + np.random.randn(profile_length) * 0.1
    d1 = 100
    feature_params = {
        'transition_width': 50,
        'cutoff_freq': 0.1,
        'butter_order': 4
    }

    def mock_curve_fit(func, x, y, *args, **kwargs):
        if func == linear_func and np.all(x < d1 - feature_params['transition_width']//2 +1):
            raise RuntimeError("Bed fit failed: Mock error")
        return curve_fit(func, x, y, *args, **kwargs)

    monkeypatch.setattr('src.feature_extraction.curve_fit', mock_curve_fit)

    features, _, success = extract_features_from_profile(mean_profile, d1, feature_params)
    
    assert success is False
    assert features.get('bed_fit_success') == 0
    assert "Linear curve fitting failed for the wound bed region. Skipping." in caplog.text


def test_extract_features_from_profile_stats_failure(caplog, monkeypatch):
    """
    GIVEN: A scenario where get_statistical_features fails.
    WHEN: extract_features_from_profile is called.
    THEN: It should log a warning, return success=False, and fill stats with NaNs.
    """
    profile_length = 200
    mean_profile = np.linspace(10, 0, profile_length) + np.random.randn(profile_length) * 0.1
    d1 = 100
    feature_params = {
        'transition_width': 50,
        'cutoff_freq': 0.1,
        'butter_order': 4
    }

    def mock_get_stats(profile):
        raise ValueError("Mock stats failure")

    monkeypatch.setattr('src.feature_extraction.get_statistical_features', mock_get_stats)

    features, _, success = extract_features_from_profile(mean_profile, d1, feature_params)

    assert success is False
    assert "Statistical feature extraction failed on a raw profile region: Mock stats failure. Assigning NaNs." in caplog.text
    assert np.isnan(features.get('bed_mean'))
    assert np.isnan(features.get('edge_mean'))
    assert np.isnan(features.get('skin_mean'))


def test_extract_features_from_profile_spectral_failure(caplog, monkeypatch):
    """
    GIVEN: A scenario where get_spectral_features fails.
    WHEN: extract_features_from_profile is called.
    THEN: It should log a warning, return success=False, and skip spectral features.
    """
    profile_length = 200
    mean_profile = np.linspace(10, 0, profile_length) + np.random.randn(profile_length) * 0.1
    d1 = 100
    feature_params = {
        'transition_width': 50,
        'cutoff_freq': 0.1,
        'butter_order': 4
    }
    def mock_get_spectral(*args, **kwargs):
        raise ValueError("Mock spectral failure")
    
    monkeypatch.setattr('src.feature_extraction.get_spectral_features', mock_get_spectral)

    features, _, success = extract_features_from_profile(mean_profile, d1, feature_params)

    assert success is False
    assert "Spectral feature extraction failed: Mock spectral failure. Skipping spectral features." in caplog.text
    assert 'spectral_centroid' not in features 


def test_extract_features_from_profile_profile_too_short(caplog):
    """
    GIVEN: A profile that is too short for a successful fit.
    WHEN: extract_features_from_profile is called.
    THEN: It should return a success=False and log a warning for the failed fit.
    """
    profile_length = 5
    mean_profile = np.linspace(10, 0, profile_length)
    d1 = 2
    feature_params = {
        'transition_width': 2,
        'cutoff_freq': 0.1,
        'butter_order': 4
    }
 
    with patch('src.feature_extraction.butter_lowpass_filter') as mock_filter:
        features, smoothed_profile, success = extract_features_from_profile(mean_profile, d1, feature_params)
        
        mock_filter.assert_not_called()
    
    assert success is False
    assert features.get('bed_fit_success') == 0
    assert "Not enough data points in bed region for linear fit. Skipping." in caplog.text


def test_extract_features_from_profile_empty_bed_region(caplog):
    """
    GIVEN: A profile and d1 such that the bed region is empty or too short.
    WHEN: extract_features_from_profile is called.
    THEN: The bed fit should be skipped, and a warning should be logged.
    """
    profile_length = 100
    mean_profile = np.linspace(10, 0, profile_length)
    d1 = 1 
    feature_params = {
        'transition_width': 10,
        'cutoff_freq': 0.1,
        'butter_order': 4
    }

    features, _, success = extract_features_from_profile(mean_profile, d1, feature_params)
    
    assert features.get('bed_fit_success') == 0
    assert "Not enough data points in bed region for linear fit. Skipping." in caplog.text


def test_extract_features_from_profile_empty_edge_region(caplog):
    """
    GIVEN: A profile and d1 such that the edge region is too short.
    WHEN: extract_features_from_profile is called.
    THEN: The sigmoid fit should be skipped, and a warning should be logged.
    """
    profile_length = 100
    mean_profile = np.linspace(10, 0, profile_length)
    d1 = 50 
    feature_params = {
        'transition_width': 2, 
        'cutoff_freq': 0.1,
        'butter_order': 4
    }

    features, _, success = extract_features_from_profile(mean_profile, d1, feature_params)
    
    assert features.get('edge_fit_success') == 0
    assert "Not enough data points in edge region for sigmoid fit. Skipping." in caplog.text


def test_extract_features_from_profile_empty_skin_region(caplog):
    """
    GIVEN: A profile and d1 such that the skin region is empty or too short.
    WHEN: extract_features_from_profile is called.
    THEN: The skin fit should be skipped, and a warning should be logged.
    """
    profile_length = 100
    mean_profile = np.linspace(10, 0, profile_length)
    d1 = 99 
    feature_params = {
        'transition_width': 10,
        'cutoff_freq': 0.1,
        'butter_order': 4
    }

    features, _, success = extract_features_from_profile(mean_profile, d1, feature_params)
    
    assert features.get('skin_fit_success') == 0
    assert "Not enough data points in skin region for linear fit. Skipping." in caplog.text


def test_extract_features_from_profile_non_numpy_input():
    """
    GIVEN: A non-numpy array input for mean_profile.
    WHEN: extract_features_from_profile is called.
    THEN: It should raise a TypeError.
    """
    d1 = 50
    feature_params = {
        'transition_width': 10,
        'cutoff_freq': 0.1,
        'butter_order': 4
    }

    with pytest.raises(TypeError) as excinfo:
        extract_features_from_profile([1, 2, 3], d1, feature_params)
    assert "Inputs must be NumPy arrays." in str(excinfo.value)


def test_extract_features_from_profile_fitting_failure_sigmoid(caplog, monkeypatch):
    """
    GIVEN: A scenario where the sigmoid fit for the edge region fails.
    WHEN: extract_features_from_profile is called.
    THEN: It should log a warning and return success=False.
    """
    profile_length = 200
    mean_profile = np.linspace(10, 0, profile_length) + np.random.randn(profile_length) * 0.1
    d1 = 100
    feature_params = {
        'transition_width': 50,
        'cutoff_freq': 0.1,
        'butter_order': 4
    }

    def mock_curve_fit(func, x, y, *args, **kwargs):
        if func == sigmoid_func:
            raise RuntimeError("Sigmoid fit failed: Mock error")
        return curve_fit(func, x, y, *args, **kwargs)

    monkeypatch.setattr('src.feature_extraction.curve_fit', mock_curve_fit)

    features, _, success = extract_features_from_profile(mean_profile, d1, feature_params)
    
    assert success is False
    assert features.get('edge_fit_success') == 0
    assert "Sigmoid curve fitting failed for the wound edge region. Skipping." in caplog.text


def test_extract_features_from_profile_fitting_failure_skin(caplog, monkeypatch):
    """
    GIVEN: A scenario where the linear fit for the skin region fails.
    WHEN: extract_features_from_profile is called.
    THEN: It should log a warning and return success=False.
    """
    profile_length = 200
    mean_profile = np.linspace(10, 0, profile_length) + np.random.randn(profile_length) * 0.1
    d1 = 100
    feature_params = {
        'transition_width': 50,
        'cutoff_freq': 0.1,
        'butter_order': 4
    }

    call_count = 0
    def mock_curve_fit(func, x, y, *args, **kwargs):
        nonlocal call_count
        if func == linear_func:
            call_count += 1
            if call_count == 2: 
                raise RuntimeError("Skin fit failed: Mock error")
        return curve_fit(func, x, y, *args, **kwargs)

    monkeypatch.setattr('src.feature_extraction.curve_fit', mock_curve_fit)

    features, _, success = extract_features_from_profile(mean_profile, d1, feature_params)
    
    assert success is False
    assert features.get('skin_fit_success') == 0
    assert "Linear curve fitting failed for the periwound skin region. Skipping." in caplog.text

# --- Additional Tests for linear_func ---

def test_linear_func_type_error():
    """
    GIVEN: A non-numpy array input for x.
    WHEN: linear_func is called.
    THEN: It should raise a TypeError and log an error.
    """
    with pytest.raises(TypeError) as excinfo:
        linear_func([0, 1, 2], 2.0, 1.0)
    assert "Input 'x' must be a NumPy array." in str(excinfo.value)


def test_linear_func_zero_values():
    """
    GIVEN: Zero slope and intercept values.
    WHEN: linear_func is called.
    THEN: It should return an array of zeros.
    """
    x = np.array([1, 2, 3, 4])
    result = linear_func(x, m=0.0, c=0.0)
    expected = np.array([0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result, expected)


def test_linear_func_negative_slope():
    """
    GIVEN: A negative slope value.
    WHEN: linear_func is called.
    THEN: It should return correct decreasing y values.
    """
    x = np.array([0, 1, 2])
    result = linear_func(x, m=-2.0, c=5.0)
    expected = np.array([5.0, 3.0, 1.0])
    np.testing.assert_allclose(result, expected)


# --- Additional Tests for sigmoid_func ---

def test_sigmoid_func_type_error():
    """
    GIVEN: A non-numpy array input for x.
    WHEN: sigmoid_func is called.
    THEN: It should raise a TypeError and log an error.
    """
    with pytest.raises(TypeError) as excinfo:
        sigmoid_func([0, 1, 2], 10.0, 1.0, 0.0, 0.0)
    assert "Input 'x' must be a NumPy array." in str(excinfo.value)


def test_sigmoid_func_extreme_values():
    """
    GIVEN: Extreme parameter values for the sigmoid function.
    WHEN: sigmoid_func is called.
    THEN: It should handle the computation without errors.
    """
    x = np.array([-100, 0, 100])
    # Test with extreme steepness
    result = sigmoid_func(x, L=1.0, k=100.0, x0=0.0, offset=0.0)
    assert len(result) == 3
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_sigmoid_func_zero_steepness():
    """
    GIVEN: Zero steepness parameter.
    WHEN: sigmoid_func is called.
    THEN: It should return constant values equal to L/2 + offset.
    """
    x = np.array([-5, 0, 5])
    L, k, x0, offset = 10.0, 0.0, 0.0, 2.0
    result = sigmoid_func(x, L, k, x0, offset)
    expected = np.full(3, L/2 + offset)  # Should be 7.0 for all values
    np.testing.assert_allclose(result, expected)


# --- Additional Tests for butter_lowpass_filter ---

def test_butter_lowpass_filter_type_error_numeric_params():
    """
    GIVEN: Non-numeric cutoff or fs parameters.
    WHEN: butter_lowpass_filter is called.
    THEN: It should raise a TypeError.
    """
    data = np.array([1, 2, 3, 4, 5])
    
    with pytest.raises(TypeError) as excinfo:
        butter_lowpass_filter(data, cutoff="10", fs=100)
    assert "Cutoff frequency and sampling frequency must be numeric." in str(excinfo.value)
    
    with pytest.raises(TypeError) as excinfo:
        butter_lowpass_filter(data, cutoff=10, fs="100")
    assert "Cutoff frequency and sampling frequency must be numeric." in str(excinfo.value)


def test_butter_lowpass_filter_runtime_error():
    """
    GIVEN: Parameters that cause filter design to fail.
    WHEN: butter_lowpass_filter is called.
    THEN: It should raise a RuntimeError and log the exception.
    """
    data = np.array([1, 2, 3])
    
    # Mock butter to raise an exception
    with patch('src.feature_extraction.butter') as mock_butter:
        mock_butter.side_effect = ValueError("Mock filter design error")
        
        with pytest.raises(RuntimeError) as excinfo:
            butter_lowpass_filter(data, cutoff=10, fs=100)
        
        assert "Filter design or application failed" in str(excinfo.value)


def test_butter_lowpass_filter_filtfilt_error():
    """
    GIVEN: Parameters that cause filtfilt application to fail.
    WHEN: butter_lowpass_filter is called.
    THEN: It should raise a RuntimeError and log the exception.
    """
    data = np.array([1, 2, 3])
    
    # Mock filtfilt to raise an exception
    with patch('src.feature_extraction.filtfilt') as mock_filtfilt:
        mock_filtfilt.side_effect = ValueError("Mock filter application error")
        
        with pytest.raises(RuntimeError) as excinfo:
            butter_lowpass_filter(data, cutoff=10, fs=100)
        
        assert "Filter design or application failed" in str(excinfo.value)


def test_butter_lowpass_filter_high_cutoff_frequency():
    """
    GIVEN: A cutoff frequency higher than the Nyquist frequency.
    WHEN: butter_lowpass_filter is called.
    THEN: It should raise a RuntimeError due to invalid normalized cutoff.
    """
    fs = 100.0
    cutoff = 60.0  # Higher than Nyquist (50 Hz)
    data = np.random.randn(100)
    
    # Should raise an error because normalized cutoff > 1
    with pytest.raises(RuntimeError) as excinfo:
        butter_lowpass_filter(data, cutoff, fs)
    assert "Filter design or application failed" in str(excinfo.value)


# --- Additional Tests for extract_features_from_profile ---

def test_extract_features_from_profile_no_filtering_short_profile():
    """
    GIVEN: A very short profile that skips butterworth filtering.
    WHEN: extract_features_from_profile is called.
    THEN: It should use the original profile without filtering.
    """
    profile_length = 8  # Less than 10
    mean_profile = np.linspace(10, 0, profile_length)
    d1 = 4
    feature_params = {
        'transition_width': 2,
        'cutoff_freq': 0.1,
        'butter_order': 4
    }
    
    with patch('src.feature_extraction.butter_lowpass_filter') as mock_filter:
        features, smoothed_profile, success = extract_features_from_profile(mean_profile, d1, feature_params)
        
        # Filter should not be called for short profiles
        mock_filter.assert_not_called()
        np.testing.assert_array_equal(smoothed_profile, mean_profile)


def test_extract_features_from_profile_edge_region_boundary_conditions():
    """
    GIVEN: A profile where the edge region boundaries are at the extremes.
    WHEN: extract_features_from_profile is called.
    THEN: It should handle the boundary conditions correctly.
    """
    profile_length = 100
    mean_profile = np.linspace(10, 0, profile_length)
    d1 = 0  # Edge at the very beginning
    feature_params = {
        'transition_width': 10,
        'cutoff_freq': 0.1,
        'butter_order': 4
    }
    
    features, _, success = extract_features_from_profile(mean_profile, d1, feature_params)
    
    # Should handle this gracefully
    assert isinstance(features, dict)
    # Bed region should be empty, so bed_fit_success should be 0
    assert features.get('bed_fit_success') == 0


def test_extract_features_from_profile_d1_at_end():
    """
    GIVEN: A profile where d1 is at the very end.
    WHEN: extract_features_from_profile is called.
    THEN: It should handle this boundary condition correctly.
    """
    profile_length = 100
    mean_profile = np.linspace(10, 0, profile_length)
    d1 = profile_length - 1  # Edge at the very end
    feature_params = {
        'transition_width': 10,
        'cutoff_freq': 0.1,
        'butter_order': 4
    }
    
    features, _, success = extract_features_from_profile(mean_profile, d1, feature_params)
    
    # Should handle this gracefully
    assert isinstance(features, dict)
    # Skin region should be very small or empty
    assert features.get('skin_fit_success') == 0


def test_extract_features_from_profile_missing_feature_params():
    """
    GIVEN: A feature_params dictionary missing required keys.
    WHEN: extract_features_from_profile is called.
    THEN: It should raise a KeyError.
    """
    mean_profile = np.array([1, 2, 3, 4, 5])
    d1 = 2
    feature_params = {}  # Missing required keys
    
    with pytest.raises(KeyError):
        extract_features_from_profile(mean_profile, d1, feature_params)


def test_extract_features_from_profile_negative_d1():
    """
    GIVEN: A negative d1 value.
    WHEN: extract_features_from_profile is called.
    THEN: It should handle this edge case without crashing.
    """
    profile_length = 100
    mean_profile = np.linspace(10, 0, profile_length)
    d1 = -10  # Negative d1
    feature_params = {
        'transition_width': 20,
        'cutoff_freq': 0.1,
        'butter_order': 4
    }
    
    features, _, success = extract_features_from_profile(mean_profile, d1, feature_params)
    
    # Should handle this without crashing
    assert isinstance(features, dict)
    # Most fits should fail due to invalid segmentation
    assert not success


def test_extract_features_from_profile_all_regions_too_small(caplog):
    """
    GIVEN: A profile and parameters where all three regions are too small for fitting.
    WHEN: extract_features_from_profile is called.
    THEN: All fit_success flags should be 0 and success should be False.
    """
    profile_length = 6  # Very small profile
    mean_profile = np.linspace(10, 0, profile_length)
    d1 = 3
    feature_params = {
        'transition_width': 4,  # Transition width larger than useful regions
        'cutoff_freq': 0.1,
        'butter_order': 4
    }
    
    features, _, success = extract_features_from_profile(mean_profile, d1, feature_params)
    
    assert success is False
    # Check that at least most regions fail to fit
    failed_fits = sum([
        features.get('bed_fit_success', 0) == 0,
        features.get('edge_fit_success', 0) == 0,
        features.get('skin_fit_success', 0) == 0
    ])
    assert failed_fits >= 2  # At least 2 out of 3 regions should fail


def test_extract_features_from_profile_curve_fit_index_error(caplog, monkeypatch):
    """
    GIVEN: A scenario where curve_fit raises an IndexError.
    WHEN: extract_features_from_profile is called.
    THEN: It should handle the IndexError gracefully and log a warning.
    """
    profile_length = 200
    mean_profile = np.linspace(10, 0, profile_length) + np.random.randn(profile_length) * 0.1
    d1 = 100
    feature_params = {
        'transition_width': 50,
        'cutoff_freq': 0.1,
        'butter_order': 4
    }

    # Import curve_fit to use in the mock
    from scipy.optimize import curve_fit
    
    original_curve_fit = curve_fit
    call_count = 0
    
    def mock_curve_fit(func, x, y, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Target the first call to linear_func (bed region)
        if func.__name__ == 'linear_func' and call_count == 1:
            raise IndexError("Mock index error")
        return original_curve_fit(func, x, y, *args, **kwargs)

    monkeypatch.setattr('src.feature_extraction.curve_fit', mock_curve_fit)

    features, _, success = extract_features_from_profile(mean_profile, d1, feature_params)
    
    assert success is False
    assert features.get('bed_fit_success') == 0
    assert "Linear curve fitting failed for the wound bed region. Skipping." in caplog.text


def test_extract_features_from_profile_runtime_exception_handling():
    """
    GIVEN: A profile that causes a general runtime exception during segmentation.
    WHEN: extract_features_from_profile is called with invalid parameters.
    THEN: It should catch the exception and return appropriate failure response.
    """
    mean_profile = np.array([])  # Empty profile should be caught by earlier validation
    d1 = 50
    feature_params = {
        'transition_width': 10,
        'cutoff_freq': 0.1,
        'butter_order': 4
    }
    
    with pytest.raises(ValueError) as excinfo:
        extract_features_from_profile(mean_profile, d1, feature_params)
    assert "Inputs must be non-empty arrays of the same shape." in str(excinfo.value)


def test_extract_features_from_profile_empty_mean_profile_stats():
    """
    GIVEN: A profile where one of the regions ends up empty for stats calculation.
    WHEN: extract_features_from_profile is called.
    THEN: It should handle the empty region gracefully.
    """
    profile_length = 20
    mean_profile = np.linspace(10, 0, profile_length)
    d1 = 19  # Very close to the end
    feature_params = {
        'transition_width': 2,
        'cutoff_freq': 0.1,
        'butter_order': 4
    }
    
    features, _, success = extract_features_from_profile(mean_profile, d1, feature_params)
    
    # Should handle gracefully even if some regions are empty
    assert isinstance(features, dict)


# --- Test for get_spectral_features edge cases ---

def test_get_spectral_features_type_error():
    """
    GIVEN: A non-numpy array input.
    WHEN: get_spectral_features is called.
    THEN: It should raise a TypeError.
    """
    with pytest.raises(TypeError) as excinfo:
        get_spectral_features([1, 2, 3, 4, 5])
    assert "Input 'profile_segment' must be a NumPy array." in str(excinfo.value)


# --- Test for get_statistical_features edge cases ---

def test_get_statistical_features_type_error():
    """
    GIVEN: A non-numpy array input.
    WHEN: get_statistical_features is called.
    THEN: It should raise a TypeError.
    """
    with pytest.raises(TypeError) as excinfo:
        get_statistical_features([1, 2, 3, 4, 5])
    assert "Input 'profile' must be a NumPy array." in str(excinfo.value)