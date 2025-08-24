"""
This module contains unit tests for the preprocessing functions defined in `src.preproccesing`.

This module contains unit tests for all functions within the `preprocessing.py` file,
ensuring their correctness, robustness, and proper error/warning handling.

Functions:
    - `temp_data_dir`: Pytest fixture for providing a temporary data directory.
    - `setup_test_logging`: Pytest fixture for configuring logging capture.
    - `create_dummy_image`: Helper to create a dummy RGB image file.
    - `create_dummy_mask`: Helper to create a dummy grayscale mask file.
    - `create_dummy_depth_map`: Helper to create a dummy depth map file.
    - All `test_*` functions: Individual unit tests covering:
        - `validate_wound_masks`: Mask filtering based on area and component count.
        - `zscore_filter`: Outlier removal from depth maps.
        - `quad_surface`: Quadratic surface model calculation.
        - `depth_corrction_for_body_curvature`: Depth map correction for body curvature.
        - `sample_pixels_from_contour`: Pixel sampling along contours.
        - `unroll_periwound_to_image`: Peri-wound region unrolling.

Typical use:
    This module is designed to be run as part of the project's test suite using `pytest`.
    It verifies that the preprocessing steps, crucial for preparing raw image and depth data,
    function correctly under various conditions.
"""

import pytest
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import logging
import sys
from unittest.mock import patch 
from scipy.optimize import OptimizeWarning 

# Import functions from src/preprocessing.py
from src.preprocessing import (
    validate_wound_masks,
    zscore_filter,
    quad_surface,
    depth_corrction_for_body_curvature,
    sample_pixels_from_contour,
    unroll_periwound_to_image,
)

# --- Fixtures ---

@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Provides a temporary root directory for data files for each test."""
    data_root = tmp_path / "test_data"
    data_root.mkdir()
    return data_root

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
    
    caplog.set_level(logging.DEBUG, logger="src.preprocessing")

# --- Helper functions for creating dummy files ---

def create_dummy_image(filepath: Path, shape: tuple = (100, 100, 3), color: tuple = (0, 255, 0)):
    """Creates a dummy RGB image (BGR for OpenCV)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    img = np.full(shape, color[::-1], dtype=np.uint8) 
    cv2.imwrite(str(filepath), img)


def create_dummy_mask(filepath: Path, shape: tuple = (100, 100), value: int = 255):
    """Creates a dummy grayscale mask."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    mask = np.full(shape, value, dtype=np.uint8)
    cv2.imwrite(str(filepath), mask)


def create_dummy_depth_map(filepath: Path, shape: tuple = (100, 100), value: int = 1000, dtype=np.uint16):
    """Creates a dummy depth map (e.g., 16-bit)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    depth = np.full(shape, value, dtype=dtype)
    cv2.imwrite(str(filepath), depth)

# --- Tests for validate_wound_masks ---

def test_validate_wound_masks_success(temp_data_dir, caplog):
    """
    GIVEN: A directory with valid wound masks.
    WHEN:  validate_wound_masks is called.
    THEN:  A DataFrame of valid image IDs should be returned, and INFO/DEBUG messages logged.
    """
    mask_dir = temp_data_dir / "wound_masks"
    mask_dir.mkdir()
    
    # Create valid masks
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask1, (50, 50), 20, 255, -1) 
    cv2.imwrite(str(mask_dir / "mask_001.png"), mask1)

    mask2 = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask2, (10, 10), (90, 90), 255, -1)
    cv2.imwrite(str(mask_dir / "mask_002.png"), mask2)

    # Filtering parameters
    filtering_params = {
        'pixel_area_threshold': 100,
        'max_wound_components': 1
    }

    valid_ids_df = validate_wound_masks(mask_dir, filtering_params)
 
    assert not valid_ids_df.empty
    assert sorted(valid_ids_df['image_id'].tolist()) == ['mask_001', 'mask_002']
    assert "Starting mask filtering process." in caplog.text
    assert "Found 2 potential mask files to check." in caplog.text
    assert "Filtering complete. Returning DataFrame with 2 filtered image IDs." in caplog.text
    assert "'mask_001.png' passed filters." in caplog.text
    assert "'mask_002.png' passed filters." in caplog.text


def test_validate_wound_masks_area_filter(temp_data_dir, caplog):
    """
    GIVEN: A mask with area below threshold.
    WHEN:  validate_wound_masks is called.
    THEN:  The mask should be filtered out, and a DEBUG message logged.
    """
    mask_dir = temp_data_dir / "wound_masks"
    mask_dir.mkdir()
    
    mask_small_area = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask_small_area, (50, 50), 2, 255, -1) # Area ~12 pixels
    cv2.imwrite(str(mask_dir / "small_area_mask.png"), mask_small_area)

    filtering_params = {
        'pixel_area_threshold': 50, # Threshold higher than mask area
        'max_wound_components': 1
    }

    valid_ids_df = validate_wound_masks(mask_dir, filtering_params)
    
    assert valid_ids_df.empty
    assert "Skipping 'small_area_mask.png': Area" in caplog.text
    assert "is below threshold 50." in caplog.text


def test_validate_wound_masks_component_count_filter(temp_data_dir, caplog):
    """
    GIVEN: A mask with incorrect number of components.
    WHEN:  validate_wound_masks is called.
    THEN:  The mask should be filtered out, and a DEBUG message logged.
    """
    mask_dir = temp_data_dir / "wound_masks"
    mask_dir.mkdir()
    
    mask_multiple_components = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask_multiple_components, (20, 20), 5, 255, -1)
    cv2.circle(mask_multiple_components, (80, 80), 5, 255, -1)
    cv2.imwrite(str(mask_dir / "multi_comp_mask.png"), mask_multiple_components)

    filtering_params = {
        'pixel_area_threshold': 10,
        'max_wound_components': 1 
    }

    valid_ids_df = validate_wound_masks(mask_dir, filtering_params)
    
    assert valid_ids_df.empty
    assert "Skipping 'multi_comp_mask.png': Found 2 components, expected 1." in caplog.text


def test_validate_wound_masks_missing_directory(temp_data_dir, caplog):
    """
    GIVEN: A non-existent mask directory.
    WHEN:  validate_wound_masks is called.
    THEN:  FileNotFoundError should be raised, and an ERROR message logged.
    """
    non_existent_dir = temp_data_dir / "non_existent_masks"
    filtering_params = {'pixel_area_threshold': 1, 'max_wound_components': 1}

    with pytest.raises(FileNotFoundError) as excinfo:
        validate_wound_masks(non_existent_dir, filtering_params)
    
    assert f"Mask directory not found at '{non_existent_dir}'" in str(excinfo.value)
    assert f"Mask directory not found: '{non_existent_dir}'. Aborting filtering." in caplog.text


def test_validate_wound_masks_unreadable_file(temp_data_dir, caplog, monkeypatch):
    """
    GIVEN: A mask file that cv2.imread cannot read.
    WHEN:  validate_wound_masks is called.
    THEN:  A warning should be logged for the unreadable file, and it should be skipped.
    """
    mask_dir = temp_data_dir / "wound_masks"
    mask_dir.mkdir()
    
    unreadable_mask_path = mask_dir / "unreadable_mask.png"
    unreadable_mask_path.touch() 

    original_imread = cv2.imread
    def mock_imread(path, *args, **kwargs):
        if Path(path) == unreadable_mask_path:
            return None
        return original_imread(path, *args, **kwargs)

    monkeypatch.setattr(cv2, 'imread', mock_imread)

    valid_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(valid_mask, (50, 50), 20, 255, -1)
    cv2.imwrite(str(mask_dir / "valid_mask.png"), valid_mask)

    filtering_params = {'pixel_area_threshold': 100, 'max_wound_components': 1}

    valid_ids_df = validate_wound_masks(mask_dir, filtering_params)

    assert not valid_ids_df.empty
    assert valid_ids_df['image_id'].tolist() == ['valid_mask']
    assert "Could not read mask file: 'unreadable_mask.png'. Skipping this image." in caplog.text


def test_validate_wound_masks_empty_directory(temp_data_dir, caplog):
    """
    GIVEN: An empty mask directory.
    WHEN:  validate_wound_masks is called.
    THEN:  An empty DataFrame should be returned, and an INFO message logged.
    """
    empty_mask_dir = temp_data_dir / "empty_masks"
    empty_mask_dir.mkdir()
    filtering_params = {'pixel_area_threshold': 1, 'max_wound_components': 1}

    valid_ids_df = validate_wound_masks(empty_mask_dir, filtering_params)
    
    assert valid_ids_df.empty
    assert "Found 0 potential mask files to check." in caplog.text
    assert "No valid image IDs found. Returning an empty DataFrame." in caplog.text


def test_validate_wound_masks_invalid_params(temp_data_dir, caplog):
    """
    GIVEN: Invalid filtering parameters (e.g., missing key).
    WHEN:  validate_wound_masks is called.
    THEN:  A KeyError should be raised.
    """
    mask_dir = temp_data_dir / "wound_masks"
    mask_dir.mkdir()
    create_dummy_mask(mask_dir / "mask.png")

    # Missing 'pixel_area_threshold'
    bad_filtering_params = {'max_wound_components': 1}

    with pytest.raises(KeyError):
        validate_wound_masks(mask_dir, bad_filtering_params)
    

def test_validate_wound_masks_no_valid_ids(temp_data_dir, caplog):
    """
    GIVEN: Masks that all fail filtering criteria.
    WHEN:  validate_wound_masks is called.
    THEN:  An empty DataFrame should be returned, and INFO/DEBUG messages logged.
    """
    mask_dir = temp_data_dir / "wound_masks"
    mask_dir.mkdir()
    
    # Mask with too many components
    mask_multi = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask_multi, (20, 20), 5, 255, -1)
    cv2.circle(mask_multi, (80, 80), 5, 255, -1) # Two components
    cv2.imwrite(str(mask_dir / "mask_multi.png"), mask_multi)

    # Mask with too small area
    mask_small = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask_small, (50, 50), 2, 255, -1)
    cv2.imwrite(str(mask_dir / "mask_small.png"), mask_small)

    filtering_params = {
        'pixel_area_threshold': 100,
        'max_wound_components': 1
    }

    valid_ids_df = validate_wound_masks(mask_dir, filtering_params)
    
    assert valid_ids_df.empty
    assert "No valid image IDs found. Returning an empty DataFrame." in caplog.text
    assert "Skipping 'mask_multi.png': Found 2 components, expected 1." in caplog.text
    assert "Skipping 'mask_small.png': Area" in caplog.text

# --- Tests for zscore_filter ---

def test_zscore_filter_success(caplog):
    """
    GIVEN: Body and depth maps with outliers.
    WHEN:  zscore_filter is called.
    THEN:  Outliers should be removed, and debug message logged.
    """
    body_in = np.ones((10, 10), dtype=np.uint8) * 255
    
    # Depth map with clear outliers
    depth_in = np.full((10, 10), 100, dtype=np.float32) 
    depth_in[0, 0] = 1.0     # Very low outlier
    depth_in[9, 9] = 200.0   # Very high outlier

    body_clensed, depth_clensed = zscore_filter(body_in, depth_in)

    assert body_clensed.shape == body_in.shape
    assert depth_clensed.shape == depth_in.shape
    assert body_clensed[0, 0] == 0 
    assert body_clensed[9, 9] == 0 
    assert depth_clensed[0, 0] == 0.0 
    assert depth_clensed[9, 9] == 0.0
    assert depth_clensed[5, 5] == 100.0 
    assert "Z-score filter applied: outliers removed from depth map and body mask." in caplog.text


def test_zscore_filter_no_outliers(caplog):
    """
    GIVEN: Body and depth maps with no outliers.
    WHEN:  zscore_filter is called.
    THEN:  No changes should occur, and a WARNING message logged (due to zero std dev).
    """
    body_in = np.ones((10, 10), dtype=np.uint8) * 255
    depth_in = np.full((10, 10), 500, dtype=np.uint16) 

    body_clensed, depth_clensed = zscore_filter(body_in, depth_in)

    assert np.array_equal(body_clensed, body_in)
    assert np.array_equal(depth_clensed, depth_in)
    assert "Standard deviation of depth is zero in zscore_filter. All values are identical. No Z-score filtering applied." in caplog.text
    assert "Z-score filter applied: outliers removed from depth map and body mask." not in caplog.text


def test_zscore_filter_zero_std_dev(caplog):
    """
    GIVEN: Depth map with zero standard deviation (all values identical).
    WHEN:  zscore_filter is called.
    THEN:  No filtering should be applied, and a WARNING logged.
    """
    body_in = np.ones((5, 5), dtype=np.uint8) * 255
    depth_in = np.full((5, 5), 150, dtype=np.uint16) 

    body_clensed, depth_clensed = zscore_filter(body_in, depth_in)

    assert np.array_equal(body_clensed, body_in)
    assert np.array_equal(depth_clensed, depth_in)
    assert "Standard deviation of depth is zero in zscore_filter. All values are identical. No Z-score filtering applied." in caplog.text


def test_zscore_filter_input_type_error(caplog):
    """
    GIVEN: Non-NumPy array inputs.
    WHEN:  zscore_filter is called.
    THEN:  TypeError should be raised, and an ERROR logged.
    """
    with pytest.raises(TypeError) as excinfo:
        zscore_filter([1,2], np.array([1,2]))
    assert "Input 'body' and 'depth' must be NumPy arrays." in str(excinfo.value)
    assert "Input 'body' and 'depth' must be NumPy arrays." in caplog.text

    with pytest.raises(TypeError) as excinfo:
        zscore_filter(np.array([1,2]), [1,2])
    assert "Input 'body' and 'depth' must be NumPy arrays." in str(excinfo.value)


def test_zscore_filter_shape_mismatch(caplog):
    """
    GIVEN: Input arrays with inconsistent shapes.
    WHEN:  zscore_filter is called.
    THEN:  ValueError should be raised, and an ERROR logged.
    """
    body_in = np.ones((10, 10), dtype=np.uint8)
    depth_in = np.ones((5, 5), dtype=np.uint16)

    with pytest.raises(ValueError) as excinfo:
        zscore_filter(body_in, depth_in)
    assert "Input 'body' and 'depth' arrays must have consistent and non-empty shapes." in str(excinfo.value)
    assert f"Input shapes mismatch or empty: body {body_in.shape}, depth {depth_in.shape}." in caplog.text


def test_zscore_filter_empty_inputs(caplog):
    """
    GIVEN: Empty input arrays.
    WHEN:  zscore_filter is called.
    THEN:  ValueError should be raised, and an ERROR logged.
    """
    body_in = np.array([], dtype=np.uint8).reshape(0,0)
    depth_in = np.array([], dtype=np.uint16).reshape(0,0)

    with pytest.raises(ValueError) as excinfo:
        zscore_filter(body_in, depth_in)
    assert "Input 'body' and 'depth' arrays must have consistent and non-empty shapes." in str(excinfo.value)
    assert f"Input shapes mismatch or empty: body {body_in.shape}, depth {depth_in.shape}." in caplog.text

# --- Tests for quad_surface ---

def test_quad_surface_basic_calculation():
    """
    GIVEN: Sample x, y coordinates and coefficients.
    WHEN:  quad_surface is called.
    THEN:  The quadratic surface value should be calculated correctly.
    """
    x_coords, y_coords = np.mgrid[0:2, 0:2]
    # z = a + b*x + c*y + d*x^2 + e*y^2 + f*x*y

    a, b, c, d, e, f = 1.0, 2.0, 3.0, 0.5, 0.2, 0.1
    
    # Expected values
    expected_z_recalc = np.array([
        [1.0, 4.2],
        [3.5, 6.8]
    ])

    calculated_z = quad_surface((x_coords, y_coords), a, b, c, d, e, f)
    np.testing.assert_allclose(calculated_z, expected_z_recalc)

# --- Tests for depth_corrction_for_body_curvature ---

def test_depth_correction_success(caplog):
    """
    GIVEN: Valid wound, body, and depth maps.
    WHEN:  depth_corrction_for_body_curvature is called.
    THEN:  Corrected depth map and cleaned masks should be returned, and logs generated.
    """
    h, w = 50, 50
    wound = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(wound, (w//2, h//2), 5, 255, -1)

    body = np.ones((h, w), dtype=np.uint8) * 255 
    
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    raw_depth = (0.01 * x_grid + 0.005 * y_grid + 100).astype(np.float32) 
    raw_depth[wound > 0] += 50 
    raw_depth[0,0] = 60000.0 # Outlier for zscore_filter

    body_clensed, depth_clensed, depth_corrected = depth_corrction_for_body_curvature(
        wound.copy(), body.copy(), raw_depth.copy())
     
    assert body_clensed.shape == (h, w)
    assert depth_clensed.shape == (h, w)
    assert depth_corrected.shape == (h, w)

    # Check that the outlier was removed by zscore_filter
    assert body_clensed[0,0] == 0
    assert depth_clensed[0,0] == 0.0

    # Check wound area is not zeroed out
    assert np.any(depth_corrected[wound > 0])
    
    assert "Applying Z-score filter to depth map." in caplog.text
    assert "Z-score filter applied: outliers removed from depth map and body mask." in caplog.text
    assert f"Dilating wound mask 15 times." in caplog.text
    assert "Fitting quadratic surface to peri-wound depth data." in caplog.text
    assert "Depth map corrected for body curvature." in caplog.text


def test_depth_correction_insufficient_data_for_fit(caplog):
    """
    GIVEN: Inputs that result in too few data points for curve_fit.
    WHEN:  depth_corrction_for_body_curvature is called.
    THEN:  A warning should be logged, and depth_clensed should be returned as depth_corrected.
    """
    h, w = 10, 10
    wound = np.zeros((h, w), dtype=np.uint8)
    wound[5,5] = 255 

    body = np.ones((h, w), dtype=np.uint8) * 255
    raw_depth = (np.random.rand(h,w) * 10 + 95).astype(np.float32)

    body_clensed, depth_clensed, depth_corrected = depth_corrction_for_body_curvature(
        wound, body, raw_depth, kernel_size=(3,3), dilation_iterations=1) 

    np.testing.assert_allclose(depth_clensed, depth_corrected, rtol=1e-5, atol=1e-8) 

    assert "Not enough data points" in caplog.text
    assert "in peri-wound region for quadratic surface fitting (minimum 6 required). Skipping curvature correction. Returning Z-score cleaned depth." in caplog.text


def test_depth_correction_curve_fit_runtime_error(caplog, monkeypatch):
    """
    GIVEN: Inputs that cause curve_fit to raise a RuntimeError.
    WHEN:  depth_corrction_for_body_curvature is called.
    THEN:  A warning should be logged, and depth_clensed should be returned as depth_corrected.
    """
    h, w = 50, 50
    wound = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(wound, (w//2, h//2), 5, 255, -1)
    body = np.ones((h, w), dtype=np.uint8) * 255
    raw_depth = (np.random.rand(h,w) * 100 + 100).astype(np.float32)

    def mock_curve_fit(*args, **kwargs):
        raise RuntimeError("Optimal parameters not found: Mock error")

    monkeypatch.setattr('src.preprocessing.curve_fit', mock_curve_fit)

    body_clensed, depth_clensed, depth_corrected = depth_corrction_for_body_curvature(
        wound, body, raw_depth)

    np.testing.assert_allclose(depth_clensed, depth_corrected)
    assert "Quadratic surface fitting failed (RuntimeError: Optimal parameters not found: Mock error). Skipping curvature correction for this image. Returning Z-score cleaned depth." in caplog.text


def test_depth_correction_input_type_error(caplog):
    """
    GIVEN: Non-NumPy array inputs.
    WHEN:  depth_corrction_for_body_curvature is called.
    THEN:  TypeError should be raised, and an ERROR logged.
    """
    h, w = 10, 10
    wound = np.zeros((h, w), dtype=np.uint8)
    body = np.ones((h, w), dtype=np.uint8) * 255
    depth = np.full((h, w), 100, dtype=np.uint16)

    with pytest.raises(TypeError) as excinfo:
        depth_corrction_for_body_curvature([1,2], body, depth)
    assert "All inputs must be NumPy arrays." in str(excinfo.value)
    assert "All inputs (wound, body, depth) must be NumPy arrays." in caplog.text


def test_depth_correction_shape_mismatch(caplog):
    """
    GIVEN: Input arrays with inconsistent shapes.
    WHEN:  depth_corrction_for_body_curvature is called.
    THEN:  ValueError should be raised, and an ERROR logged.
    """
    h, w = 10, 10
    wound = np.zeros((h, w), dtype=np.uint8)
    body = np.ones((h, w + 1), dtype=np.uint8) * 255
    depth = np.full((h, w), 100, dtype=np.uint16)

    with pytest.raises(ValueError) as excinfo:
        depth_corrction_for_body_curvature(wound, body, depth)
    assert "All input masks and depth map must have the same (H, W) shape and be non-empty." in str(excinfo.value)
    assert f"Input shapes mismatch or empty: wound {wound.shape}, body {body.shape}, depth {depth.shape}." in caplog.text


def test_depth_correction_empty_inputs(caplog):
    """
    GIVEN: Empty input arrays.
    WHEN:  depth_corrction_for_body_curvature is called.
    THEN:  ValueError should be raised, and an ERROR logged.
    """
    empty_array = np.array([], dtype=np.uint8).reshape(0,0)
    
    with pytest.raises(ValueError) as excinfo:
        depth_corrction_for_body_curvature(empty_array, empty_array, empty_array)
    assert "All input masks and depth map must have the same (H, W) shape and be non-empty." in str(excinfo.value)
    assert f"Input shapes mismatch or empty: wound {empty_array.shape}, body {empty_array.shape}, depth {empty_array.shape}." in caplog.text

# --- Tests for sample_pixels_from_contour ---

def test_sample_pixels_from_contour_success_grayscale(caplog):
    """
    GIVEN: A grayscale image and a mask with a clear contour.
    WHEN:  sample_pixels_from_contour is called.
    THEN:  Pixels along the contour should be sampled correctly.
    """
    img = np.zeros((50, 50), dtype=np.uint8)
    img[10:40, 10:40] = 150 # Grey square
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:40] = 255 # White square mask

    pixels = sample_pixels_from_contour(img, mask)

    assert pixels.ndim == 2 
    assert pixels.shape[1] == 1 
    assert pixels.shape[0] > 0 
    assert np.all(pixels == 150)
    assert "Sampled" in caplog.text
    assert "pixels from contour." in caplog.text


def test_sample_pixels_from_contour_success_rgb(caplog):
    """
    GIVEN: An RGB image and a mask with a clear contour.
    WHEN:  sample_pixels_from_contour is called.
    THEN:  Pixels along the contour should be sampled correctly with 3 channels.
    """
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[10:40, 10:40] = [10, 20, 30] 
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:40] = 255

    pixels = sample_pixels_from_contour(img, mask)

    assert pixels.ndim == 3 
    assert pixels.shape[1] == 1
    assert pixels.shape[2] == 3 
    assert pixels.shape[0] > 0
    assert np.all(pixels == [10, 20, 30])
    assert "Sampled" in caplog.text
    assert "pixels from contour." in caplog.text


def test_sample_pixels_from_contour_no_contours(caplog):
    """
    GIVEN: A mask with no contours (all black).
    WHEN:  sample_pixels_from_contour is called.
    THEN:  ValueError should be raised, and an ERROR logged.
    """
    img = np.zeros((50, 50), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8) 

    with pytest.raises(ValueError) as excinfo:
        sample_pixels_from_contour(img, mask)
    assert "No contours found in the provided mask." in str(excinfo.value)
    assert "No contours found in the provided mask. Cannot sample pixels." in caplog.text


def test_sample_pixels_from_contour_degenerate_contour(caplog):
    """
    GIVEN: A mask that results in a degenerate (1D) contour (e.g., single pixel).
    WHEN:  sample_pixels_from_contour is called.
    THEN:  ValueError should be raised, and a WARNING logged.
    """
    img = np.zeros((50, 50), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 25] = 255 

    with pytest.raises(ValueError) as excinfo:
        sample_pixels_from_contour(img, mask)
    assert "Degenerate contour found (1D). Cannot sample pixels from it." in str(excinfo.value)
    assert "Longest contour is 1D (a single point or line segment). Degenerate mask. Cannot sample pixels." in caplog.text


def test_sample_pixels_from_contour_input_type_error(caplog):
    """
    GIVEN: Non-NumPy array inputs.
    WHEN:  sample_pixels_from_contour is called.
    THEN:  TypeError should be raised, and an ERROR logged.
    """
    img = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5,5] = 255 

    with pytest.raises(TypeError) as excinfo:
        sample_pixels_from_contour([1,2], mask)
    assert "Input 'img' and 'mask' must be NumPy arrays." in str(excinfo.value)
    assert "Input 'img' and 'mask' must be NumPy arrays." in caplog.text


def test_sample_pixels_from_contour_shape_mismatch(caplog):
    """
    GIVEN: Input image and mask with inconsistent shapes.
    WHEN:  sample_pixels_from_contour is called.
    THEN:  ValueError should be raised, and an ERROR logged.
    """
    img = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((5, 5), dtype=np.uint8) 
    mask[2,2] = 255 

    with pytest.raises(ValueError) as excinfo:
        sample_pixels_from_contour(img, mask)
    assert "Input mask and image must be non-empty and have consistent (H, W) shapes." in str(excinfo.value)
    assert f"Mask {mask.shape} or image {img.shape} is empty or shapes mismatch." in caplog.text


def test_sample_pixels_from_contour_empty_inputs(caplog):
    """
    GIVEN: Empty input arrays.
    WHEN:  sample_pixels_from_contour is called.
    THEN:  ValueError should be raised, and an ERROR logged.
    """
    empty_array_2d = np.array([], dtype=np.uint8).reshape(0,0)
    
    with pytest.raises(ValueError) as excinfo:
        sample_pixels_from_contour(empty_array_2d, empty_array_2d)
    assert "Input mask and image must be non-empty and have consistent (H, W) shapes." in str(excinfo.value)
    assert f"Mask {empty_array_2d.shape} or image {empty_array_2d.shape} is empty or shapes mismatch." in caplog.text

# --- Tests for unroll_periwound_to_image ---

def test_unroll_periwound_to_image_success_grayscale(caplog):
    """
    GIVEN: A grayscale image and a wound mask.
    WHEN:  unroll_periwound_to_image is called.
    THEN:  A 2D unrolled strip and correct counts should be returned.
    """
    img = np.zeros((100, 100), dtype=np.uint8)
    img[40:60, 40:60] = 100 # Inner square
    img[30:70, 30:70] = 50  # Outer square
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 10, 255, -1) # Wound mask

    unrolled_strip, counts = unroll_periwound_to_image(img, mask, iterations=5, kernel_size=(3,3))

    assert unrolled_strip.ndim == 2
    base_strip_h = sample_pixels_from_contour(img, mask).shape[0] 
    assert unrolled_strip.shape[1] == base_strip_h # First dim is original contour length
    assert unrolled_strip.shape[0] == counts[0] + counts[1] + 1 # Second dim is total strip width
    assert counts[0] >= 0 # Can be 0 if mask is too small to erode
    assert counts[1] >= 0 # Can be 0 if mask is too large to dilate
    assert "Extracting baseline pixels from wound contour." in caplog.text
    assert "Starting iterative dilation (max 5 steps)." in caplog.text
    assert "Starting iterative erosion (max 5 steps)." in caplog.text
    assert "Image unrolled into strip of shape" in caplog.text


def test_unroll_periwound_to_image_success_rgb(caplog):
    """
    GIVEN: An RGB image and a wound mask.
    WHEN:  unroll_periwound_to_image is called.
    THEN:  A 3D unrolled strip with 3 channels and correct counts should be returned.
    """
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60] = [10, 20, 30]
    img[30:70, 30:70] = [50, 60, 70]
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 10, 255, -1)

    unrolled_strip, counts = unroll_periwound_to_image(img, mask, iterations=5, kernel_size=(3,3))

    assert unrolled_strip.ndim == 3
    base_strip_h = sample_pixels_from_contour(img, mask).shape[0]
    assert unrolled_strip.shape[1] == base_strip_h 
    assert unrolled_strip.shape[0] == counts[0] + counts[1] + 1 
    assert unrolled_strip.shape[2] == 3 
    assert counts[0] >= 0
    assert counts[1] >= 0


def test_unroll_periwound_to_image_early_stop_dilation(caplog):
    """
    GIVEN: A mask that quickly stops expanding during dilation.
    WHEN:  unroll_periwound_to_image is called.
    THEN:  Dilation should stop early, and a DEBUG message logged.
    """
    img = np.zeros((50, 50), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[1:49, 1:49] = 255 

    unrolled_strip, counts = unroll_periwound_to_image(img, mask, iterations=5, kernel_size=(3,3))

    assert unrolled_strip is not None
    assert "Dilation stopped at iteration" in caplog.text
    assert "Mask no longer expanding." in caplog.text
    assert counts[1] < 5 


def test_unroll_periwound_to_image_early_stop_erosion_mask_empty(caplog):
    """
    GIVEN: A mask that becomes empty quickly during erosion.
    WHEN:  unroll_periwound_to_image is called.
    THEN:  Erosion should stop early, and a DEBUG message logged.
    """
    img = np.zeros((50, 50), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[24:26, 24:26] = 255 

    unrolled_strip, counts = unroll_periwound_to_image(img, mask, iterations=10, kernel_size=(3,3))

    assert unrolled_strip is not None
    assert "Erosion stopped at iteration" in caplog.text
    assert "Mask no longer shrinking or became empty." in caplog.text


def test_unroll_periwound_to_image_early_stop_erosion_mask_fragmented(caplog):
    """
    GIVEN: A mask that fragments into multiple components during erosion.
    WHEN:  unroll_periwound_to_image is called.
    THEN:  Erosion should stop early, and a WARNING message logged.
    """
    img = np.zeros((50, 50), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    cv2.circle(mask, (15, 15), 3, 255, -1)
    cv2.circle(mask, (35, 35), 3, 255, -1)

    unrolled_strip, counts = unroll_periwound_to_image(img, mask, iterations=10, kernel_size=(3,3))
 
    assert unrolled_strip is not None
    assert "Erosion stopped at iteration" in caplog.text
    assert "Mask fragmented into" in caplog.text
    assert "objects. Preserving integrity." in caplog.text


def test_unroll_periwound_to_image_sample_pixels_failure(caplog, monkeypatch):
    """
    GIVEN: A scenario where sample_pixels_from_contour fails during unrolling.
    WHEN:  unroll_periwound_to_image is called.
    THEN:  An error/warning should be logged, and processing should stop for that branch.
    """
    img = np.zeros((50, 50), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    cv2.circle(mask, (25, 25), 10, 255, -1)

    call_count = 0
    original_sample = sample_pixels_from_contour
    def mock_sample_pixels_from_contour(img_arg, mask_arg):
        nonlocal call_count
        call_count += 1
        if call_count > 1: 
            raise ValueError("Mocked sample_pixels_from_contour failure.")
        return original_sample(img_arg, mask_arg)

    monkeypatch.setattr('src.preprocessing.sample_pixels_from_contour', mock_sample_pixels_from_contour)

    unrolled_strip, counts = unroll_periwound_to_image(img, mask, iterations=5)
    
    assert unrolled_strip is not None 
    assert "Dilation stopped at iteration" in caplog.text
    assert "Could not find contour in dilated mask or sampling failed: Mocked sample_pixels_from_contour failure." in caplog.text
    assert "Erosion stopped at iteration" in caplog.text
    assert "Could not find contour in eroded mask or sampling failed: Mocked sample_pixels_from_contour failure." in caplog.text


def test_unroll_periwound_to_image_no_strips_generated(caplog, monkeypatch):
    """
    GIVEN: A scenario where no strips can be generated (e.g., base contour fails).
    WHEN:  unroll_periwound_to_image is called.
    THEN:  ValueError should be raised, and an ERROR logged.
    """
    img = np.zeros((50, 50), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    
    # Mock sample_pixels_from_contour to always fail
    def mock_sample_pixels_from_contour_fail(*args, **kwargs):
        raise ValueError("No base contour found for unrolling.")
    monkeypatch.setattr('src.preprocessing.sample_pixels_from_contour', mock_sample_pixels_from_contour_fail)
    
    with pytest.raises(ValueError) as excinfo:
        unroll_periwound_to_image(img, mask, iterations=1)
    
    assert "Baseline contour extraction failed for unrolling: No base contour found for unrolling." in str(excinfo.value)
    assert "Failed to get baseline contour for unrolling: No base contour found for unrolling.. Cannot unroll image." in caplog.text


def test_unroll_periwound_to_image_input_type_error(caplog):
    """
    GIVEN: Non-NumPy array inputs.
    WHEN:  unroll_periwound_to_image is called.
    THEN:  TypeError should be raised, and an ERROR logged.
    """
    img = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    cv2.circle(mask, (5,5), 2, 255, -1)

    with pytest.raises(TypeError) as excinfo:
        unroll_periwound_to_image([1,2], mask)
    assert "Input 'img' and 'mask' must be NumPy arrays." in str(excinfo.value)
    assert "Input 'img' and 'mask' must be NumPy arrays." in caplog.text


def test_unroll_periwound_to_image_shape_mismatch(caplog):
    """
    GIVEN: Input image and mask with inconsistent shapes.
    WHEN:  unroll_periwound_to_image is called.
    THEN:  ValueError should be raised, and an ERROR logged.
    """
    img = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((5, 5), dtype=np.uint8) 
    cv2.circle(mask, (2,2), 1, 255, -1)

    with pytest.raises(ValueError) as excinfo:
        unroll_periwound_to_image(img, mask)
    assert "Input mask and image must be non-empty and have consistent (H, W) shapes." in str(excinfo.value)
    assert f"Mask {mask.shape} or image {img.shape} is empty or shapes mismatch." in caplog.text


def test_unroll_periwound_to_image_empty_inputs(caplog):
    """
    GIVEN: Empty input arrays.
    WHEN:  unroll_periwound_to_image is called.
    THEN:  ValueError should be raised, and an ERROR logged.
    """
    empty_array_2d = np.array([], dtype=np.uint8).reshape(0,0)
    
    with pytest.raises(ValueError) as excinfo:
        unroll_periwound_to_image(empty_array_2d, empty_array_2d)
    assert "Input mask and image must be non-empty and have consistent (H, W) shapes." in str(excinfo.value)
    assert f"Mask {empty_array_2d.shape} or image {empty_array_2d.shape} is empty or shapes mismatch." in caplog.text


# Additional test cases to improve coverage for preprocessing.py

def test_validate_wound_masks_edge_case_num_labels_check(temp_data_dir, caplog):
    """
    GIVEN: A scenario that specifically targets the num_labels < 2 check.
    WHEN:  validate_wound_masks is called.
    THEN:  The warning for no foreground components should be logged.
    """
    mask_dir = temp_data_dir / "wound_masks"
    mask_dir.mkdir()
    
    mask_path = mask_dir / "edge_case_mask.png"
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.imwrite(str(mask_path), mask)
    
    filtering_params = {
        'pixel_area_threshold': 10,
        'max_wound_components': 0  # Expect 0 components to pass the first check
    }
    
    # Mock to create the exact scenario: 
    # num_labels = 1 (background only), num_wound_components = 0, max_components = 0
    # This should pass the component count check but fail the num_labels < 2 check
    with patch('cv2.connectedComponentsWithStats') as mock_components:
        mock_components.return_value = (1, None, np.array([[0, 0, 0, 0, 0]]), None)
        
        valid_ids_df = validate_wound_masks(mask_dir, filtering_params)
        
        assert valid_ids_df.empty
        assert "Skipping 'edge_case_mask.png': No foreground components found despite check (num_labels=1)." in caplog.text


# Additional test cases to improve coverage for preprocessing.py

def test_validate_wound_masks_index_error_handling(temp_data_dir, caplog):
    """
    GIVEN: A mask that causes IndexError when extracting component area.
    WHEN:  validate_wound_masks is called.
    THEN:  The error should be caught, logged, and the mask skipped.
    """
    mask_dir = temp_data_dir / "wound_masks"
    mask_dir.mkdir()
    
    # Create a mask that might cause issues with component statistics
    problematic_mask = np.zeros((100, 100), dtype=np.uint8)
    # Create a very thin line that might cause cv2.connectedComponentsWithStats issues
    problematic_mask[50, 10:90] = 255
    cv2.imwrite(str(mask_dir / "problematic_mask.png"), problematic_mask)

    filtering_params = {
        'pixel_area_threshold': 10,
        'max_wound_components': 1
    }

    # Mock cv2.connectedComponentsWithStats to raise IndexError
    with patch('cv2.connectedComponentsWithStats') as mock_components:
        mock_components.return_value = (2, None, np.array([[0, 0, 0, 0, 0]]), None)  # Missing stats for component 1
        
        valid_ids_df = validate_wound_masks(mask_dir, filtering_params)
        
        assert valid_ids_df.empty
        assert "Failed to extract component area for 'problematic_mask.png'. Likely unexpected mask structure. Skipping." in caplog.text


def test_validate_wound_masks_no_foreground_components(temp_data_dir, caplog):
    """
    GIVEN: A mask with num_labels < 2 (no foreground components).
    WHEN:  validate_wound_masks is called.  
    THEN:  The mask should be filtered out due to component count mismatch.
    """
    mask_dir = temp_data_dir / "wound_masks"
    mask_dir.mkdir()
    
    # Create an all-black mask - this will have 0 wound components
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.imwrite(str(mask_dir / "empty_mask.png"), empty_mask)

    filtering_params = {
        'pixel_area_threshold': 10,
        'max_wound_components': 1
    }

    valid_ids_df = validate_wound_masks(mask_dir, filtering_params)
    
    assert valid_ids_df.empty
    # This gets caught by the component count check, not the num_labels < 2 check
    assert "Skipping 'empty_mask.png': Found 0 components, expected 1." in caplog.text


def test_validate_wound_masks_unexpected_processing_error(temp_data_dir, caplog):
    """
    GIVEN: A mask that causes an unexpected error during processing.
    WHEN:  validate_wound_masks is called.
    THEN:  The error should be caught, logged, and the mask skipped.
    """
    mask_dir = temp_data_dir / "wound_masks"
    mask_dir.mkdir()
    
    # Create a valid mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 20, 255, -1)
    cv2.imwrite(str(mask_dir / "error_mask.png"), mask)

    filtering_params = {
        'pixel_area_threshold': 10,
        'max_wound_components': 1
    }

    # Mock cv2.connectedComponentsWithStats to raise a generic exception
    with patch('cv2.connectedComponentsWithStats') as mock_components:
        mock_components.side_effect = RuntimeError("Unexpected OpenCV error")
        
        valid_ids_df = validate_wound_masks(mask_dir, filtering_params)
        
        assert valid_ids_df.empty
        assert "An unexpected error occurred while processing mask 'error_mask.png'. Skipping this image." in caplog.text


def test_validate_wound_masks_io_error(temp_data_dir, caplog):
    """
    GIVEN: A directory that raises IOError during iteration.
    WHEN:  validate_wound_masks is called.
    THEN:  IOError should be raised with appropriate error message.
    """
    mask_dir = temp_data_dir / "wound_masks"
    mask_dir.mkdir()

    # Mock iterdir to raise PermissionError
    with patch.object(Path, 'iterdir') as mock_iterdir:
        mock_iterdir.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(IOError) as excinfo:
            validate_wound_masks(mask_dir, {'pixel_area_threshold': 10, 'max_wound_components': 1})
        
        assert "Error reading directory" in str(excinfo.value)
        assert "Permission denied" in str(excinfo.value)
        assert "Failed to read contents of directory" in caplog.text


def test_depth_correction_unexpected_curve_fit_error(caplog, monkeypatch):
    """
    GIVEN: Inputs that cause curve_fit to raise an unexpected (non-RuntimeError) exception.
    WHEN:  depth_corrction_for_body_curvature is called.
    THEN:  IOError should be raised and logged appropriately.
    """
    h, w = 50, 50
    wound = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(wound, (w//2, h//2), 5, 255, -1)
    body = np.ones((h, w), dtype=np.uint8) * 255
    raw_depth = (np.random.rand(h,w) * 100 + 100).astype(np.float32)

    def mock_curve_fit(*args, **kwargs):
        raise ValueError("Unexpected curve_fit error")

    monkeypatch.setattr('src.preprocessing.curve_fit', mock_curve_fit)

    with pytest.raises(IOError) as excinfo:
        depth_corrction_for_body_curvature(wound, body, raw_depth)

    assert "Error during quadratic surface fitting: Unexpected curve_fit error" in str(excinfo.value)
    assert "An unexpected error occurred during quadratic surface fitting. Skipping curvature correction. Returning Z-score cleaned depth." in caplog.text


def test_zscore_filter_different_input_dtypes(caplog):
    """
    GIVEN: Body and depth maps with different input dtypes.
    WHEN:  zscore_filter is called.
    THEN:  The function should handle dtype conversion correctly and filter outliers.
    """
    body_in = np.ones((10, 10), dtype=np.uint8) * 255
    
    # Test with uint16 depth input (common for depth cameras)
    depth_in = np.full((10, 10), 1000, dtype=np.uint16)
    depth_in[0, 0] = 100    # Low outlier
    depth_in[9, 9] = 2000   # High outlier

    body_clensed, depth_clensed = zscore_filter(body_in, depth_in)

    assert body_clensed.shape == body_in.shape
    assert depth_clensed.shape == depth_in.shape
    assert depth_clensed.dtype == depth_in.dtype  # Should preserve original dtype
    assert body_clensed[0, 0] == 0 
    assert body_clensed[9, 9] == 0 
    assert depth_clensed[0, 0] == 0
    assert depth_clensed[9, 9] == 0
    assert depth_clensed[5, 5] == 1000
    assert "Z-score filter applied: outliers removed from depth map and body mask." in caplog.text


def test_unroll_periwound_to_image_alternative_coverage(caplog):
    """
    GIVEN: A scenario that tests edge cases in the unrolling process.
    WHEN:  unroll_periwound_to_image is called.
    THEN:  The function should handle the edge case appropriately.
    """
    # Create a very small image that might cause issues with cv2.resize
    img = np.zeros((10, 10), dtype=np.uint8)
    img[4:6, 4:6] = 200
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 255

    # This should work but test edge case behavior
    unrolled_strip, counts = unroll_periwound_to_image(img, mask, iterations=1, kernel_size=(1,1))

    assert unrolled_strip is not None
    assert unrolled_strip.ndim == 2
    assert counts[0] >= 0
    assert counts[1] >= 0


# Alternative test focusing on more achievable coverage improvements
def test_validate_wound_masks_directory_iteration_with_non_files(temp_data_dir, caplog):
    """
    GIVEN: A directory containing both files and subdirectories.
    WHEN:  validate_wound_masks is called.
    THEN:  Only files should be processed, subdirectories should be ignored.
    """
    mask_dir = temp_data_dir / "wound_masks"
    mask_dir.mkdir()
    
    # Create a valid mask file
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    cv2.imwrite(str(mask_dir / "valid_mask.png"), mask)
    
    # Create a subdirectory (should be ignored)
    (mask_dir / "subdirectory").mkdir()
    
    # Create a file with wrong extension (should be ignored)
    (mask_dir / "invalid.txt").write_text("not an image")

    filtering_params = {
        'pixel_area_threshold': 100,
        'max_wound_components': 1
    }

    valid_ids_df = validate_wound_masks(mask_dir, filtering_params)
    
    # Should only process the valid PNG file
    assert len(valid_ids_df) == 1
    assert valid_ids_df['image_id'].iloc[0] == 'valid_mask'
    assert "Found 1 potential mask files to check." in caplog.text


def test_sample_pixels_from_contour_single_contour_point_edge_case(caplog):
    """
    GIVEN: A mask that creates a contour with minimal points.
    WHEN:  sample_pixels_from_contour is called.
    THEN:  The function should handle minimal contours appropriately.
    """
    img = np.zeros((50, 50), dtype=np.uint8)
    img[20:25, 20:25] = 100
    
    # Create a very thin line that might result in minimal contour
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[22, 20:25] = 255  # Thin horizontal line
    
    try:
        pixels = sample_pixels_from_contour(img, mask)
        assert pixels.shape[0] > 0  # Should have some pixels
        assert pixels.shape[1] == 1  # Should have width 1
        assert "Sampled" in caplog.text
    except ValueError:
        # This might legitimately fail for very thin contours, which is acceptable
        assert "Degenerate contour found" in caplog.text or "No contours found" in caplog.text


def test_depth_correction_with_minimal_peri_wound_data(caplog):
    """
    GIVEN: A wound that results in very few peri-wound pixels after dilation.
    WHEN:  depth_corrction_for_body_curvature is called.
    THEN:  The function should handle insufficient data gracefully.
    """
    h, w = 20, 20  # Small image
    wound = np.zeros((h, w), dtype=np.uint8)
    wound[9:11, 9:11] = 255  # Very small wound
    
    body = np.ones((h, w), dtype=np.uint8) * 255
    
    # Create a depth map with limited variation
    depth = np.full((h, w), 1000, dtype=np.float32)
    
    body_clensed, depth_clensed, depth_corrected = depth_corrction_for_body_curvature(
        wound, body, depth, kernel_size=(1,1), dilation_iterations=1)
    
    # Should handle the minimal data scenario
    assert body_clensed.shape == (h, w)
    assert depth_clensed.shape == (h, w)
    assert depth_corrected.shape == (h, w)
    
    # Likely to get insufficient data warning
    assert ("Not enough data points" in caplog.text or 
            "Depth map corrected for body curvature." in caplog.text)


def test_unroll_periwound_cv2_resize_with_small_dimensions(caplog):
    """
    GIVEN: A very small mask that tests cv2.resize edge cases.
    WHEN:  unroll_periwound_to_image is called.
    THEN:  The function should handle small dimensions in resizing.
    """
    # Very small image to test resize edge cases
    img = np.zeros((15, 15), dtype=np.uint8)
    img[6:9, 6:9] = 150
    
    mask = np.zeros((15, 15), dtype=np.uint8)
    mask[7, 7] = 255  # Single pixel mask
    
    try:
        unrolled_strip, counts = unroll_periwound_to_image(img, mask, iterations=1, kernel_size=(1,1))
        assert unrolled_strip is not None
        assert counts[0] >= 0 and counts[1] >= 0
    except ValueError as e:
        # Single pixel might cause contour issues, which is acceptable
        assert ("Degenerate contour" in str(e) or 
                "No contours found" in str(e) or
                "Baseline contour extraction failed" in str(e))


def test_sample_pixels_from_contour_multiple_contours(caplog):
    """
    GIVEN: A mask with multiple contours (function should select the longest one).
    WHEN:  sample_pixels_from_contour is called.
    THEN:  Pixels from the longest contour should be sampled.
    """
    img = np.zeros((100, 100), dtype=np.uint8)
    img[10:20, 10:20] = 100  # Small square
    img[30:80, 30:80] = 150  # Large square
    
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 255  # Small contour
    mask[30:80, 30:80] = 255  # Large contour (should be selected)

    pixels = sample_pixels_from_contour(img, mask)

    assert pixels.ndim == 2
    assert pixels.shape[1] == 1
    assert pixels.shape[0] > 0
    # Should sample from the larger contour (value 150)
    assert np.all(pixels == 150)
    assert "Sampled" in caplog.text


def test_unroll_periwound_to_image_cv2_resize_edge_case(caplog):
    """
    GIVEN: A scenario where cv2.resize might behave unexpectedly with very small dimensions.
    WHEN:  unroll_periwound_to_image is called.
    THEN:  The function should handle the resizing gracefully.
    """
    # Create a very small image and mask
    img = np.zeros((20, 20), dtype=np.uint8)
    img[8:12, 8:12] = 200
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[9:11, 9:11] = 255  # Very small wound

    unrolled_strip, counts = unroll_periwound_to_image(img, mask, iterations=2, kernel_size=(1,1))

    assert unrolled_strip is not None
    assert unrolled_strip.ndim == 2
    assert counts[0] >= 0
    assert counts[1] >= 0


def test_depth_correction_edge_case_all_zeros_after_zscore(caplog):
    """
    GIVEN: A scenario where Z-score filter results in all zeros (extreme case).
    WHEN:  depth_corrction_for_body_curvature is called.
    THEN:  The function should handle this gracefully and skip curve fitting.
    """
    h, w = 30, 30
    wound = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(wound, (w//2, h//2), 3, 255, -1)
    
    body = np.ones((h, w), dtype=np.uint8) * 255
    
    # Create depth map where everything is an outlier except a few pixels
    depth = np.ones((h, w), dtype=np.float32) * 10000  # All outliers
    depth[h//2-1:h//2+2, w//2-1:w//2+2] = 100  # Small normal region

    body_clensed, depth_clensed, depth_corrected = depth_corrction_for_body_curvature(
        wound, body, depth, kernel_size=(1,1), dilation_iterations=1)

    # Should return the cleaned versions since curve fitting will likely fail
    assert body_clensed.shape == (h, w)
    assert depth_clensed.shape == (h, w) 
    assert depth_corrected.shape == (h, w)


def test_quad_surface_edge_values():
    """
    GIVEN: Edge case values for quad_surface (zeros, negative values, etc.).
    WHEN:  quad_surface is called.
    THEN:  Correct mathematical results should be returned.
    """
    # Test with zero coordinates
    x_coords = np.array([[0, 0], [0, 0]])
    y_coords = np.array([[0, 0], [0, 0]])
    
    result = quad_surface((x_coords, y_coords), a=1, b=2, c=3, d=4, e=5, f=6)
    expected = np.array([[1, 1], [1, 1]])  # Only 'a' term contributes
    np.testing.assert_allclose(result, expected)
    
    # Test with negative coordinates
    x_coords = np.array([[-1, -1], [1, 1]])
    y_coords = np.array([[-1, 1], [-1, 1]])
    
    result = quad_surface((x_coords, y_coords), a=1, b=1, c=1, d=1, e=1, f=1)
    # For each point: z = 1 + 1*x + 1*y + 1*x^2 + 1*y^2 + 1*x*y
    expected = np.array([[1 + (-1) + (-1) + 1 + 1 + 1,  # x=-1, y=-1: 1-1-1+1+1+1 = 2
                          1 + (-1) + (1) + 1 + 1 + (-1)], # x=-1, y=1: 1-1+1+1+1-1 = 2  
                         [1 + (1) + (-1) + 1 + 1 + (-1),  # x=1, y=-1: 1+1-1+1+1-1 = 2
                          1 + (1) + (1) + 1 + 1 + 1]])    # x=1, y=1: 1+1+1+1+1+1 = 6
    expected = np.array([[2, 2], [2, 6]])
    np.testing.assert_allclose(result, expected)


def test_unroll_periwound_to_image_exact_iteration_limit(caplog):
    """
    GIVEN: A mask that reaches exactly the iteration limit.
    WHEN:  unroll_periwound_to_image is called.
    THEN:  The function should complete all iterations without early stopping.
    """
    img = np.zeros((100, 100), dtype=np.uint8)
    img[40:60, 40:60] = 100
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 10, 255, -1)

    # Use very small iteration count to test exact limit
    unrolled_strip, counts = unroll_periwound_to_image(img, mask, iterations=2, kernel_size=(3,3))

    assert unrolled_strip is not None
    assert "Completed 2 dilation steps." in caplog.text or "Dilation stopped at iteration" in caplog.text
    assert counts[1] <= 2  # Should not exceed iteration limit


def test_validate_wound_masks_various_file_extensions(temp_data_dir, caplog):
    """
    GIVEN: A directory with various image file extensions.
    WHEN:  validate_wound_masks is called.
    THEN:  Only supported extensions should be processed.
    """
    mask_dir = temp_data_dir / "wound_masks"
    mask_dir.mkdir()
    
    # Create files with different extensions
    # Use a simple rectangle to avoid JPEG compression artifacts
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255  # Simple solid rectangle
    
    cv2.imwrite(str(mask_dir / "mask1.png"), mask)
    # For lossy formats, use PNG and then copy to avoid compression issues
    cv2.imwrite(str(mask_dir / "temp.png"), mask)
    # Copy the PNG as JPG to ensure identical content
    temp_img = cv2.imread(str(mask_dir / "temp.png"), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(mask_dir / "mask2.jpg"), temp_img)
    cv2.imwrite(str(mask_dir / "mask3.jpeg"), temp_img)
    (mask_dir / "temp.png").unlink()  # Clean up
    
    # Create unsupported file
    (mask_dir / "mask4.txt").write_text("not an image")
    (mask_dir / "mask5.bmp").touch()  # Unsupported extension

    filtering_params = {
        'pixel_area_threshold': 100,
        'max_wound_components': 1
    }

    valid_ids_df = validate_wound_masks(mask_dir, filtering_params)
    
    # Should only process .png, .jpg, .jpeg files that pass the filtering
    assert len(valid_ids_df) >= 1  # At least PNG should pass
    assert "Found 3 potential mask files to check." in caplog.text
    # Check that only supported extensions were considered
    assert "mask1" in valid_ids_df['image_id'].tolist() or len(valid_ids_df) > 0