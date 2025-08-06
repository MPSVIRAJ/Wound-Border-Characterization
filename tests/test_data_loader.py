"""
This module contains unit tests for the data loading and preparation functions
defined in `src.data_loader`.

It employs pytest to verify the correctness, robustness, and error handling of
functions that load images, masks, depth maps, process feature CSVs, and manage
cluster assignments. Tests cover various scenarios including successful loading,
missing critical files, unreadable files, shape mismatches, missing configuration
keys, and handling of empty or malformed CSV data.

Functions:
    - `temp_data_root`: Pytest fixture providing a temporary base directory for data.
    - `subdirs_config`: Pytest fixture providing a standard subdirectory configuration.
    - `setup_test_logging`: Pytest fixture for configuring logging capture during tests.
    - `create_dummy_image`: Helper to create a dummy RGB image file.
    - `create_dummy_mask`: Helper to create a dummy grayscale mask file.
    - `create_dummy_depth_map`: Helper to create a dummy depth map file.
    - `create_dummy_csv`: Helper to create a dummy CSV file from a DataFrame.
    - All `test_*` functions: Individual unit tests for `data_loader`, `load_and_clean_features`,
  and `load_cluster_groups`, following the GIVEN/WHEN/THEN format.

Typical use:
    This module is designed to be executed by `pytest` as part of the project's
    automated testing suite. It ensures the reliability and integrity of the data
    loading components, which are crucial for the downstream processing and analysis pipelines.
"""

import pytest
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import logging
import sys

# Import functions from src/data_loader.py

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import (
    data_loader,
    load_and_clean_features,
    load_cluster_groups,
)

# --- Fixtures ---

@pytest.fixture
def temp_data_root(tmp_path: Path) -> Path:
    """Provides a temporary root directory for data files for each test."""
    data_root = tmp_path / "test_data"
    data_root.mkdir()
    return data_root


@pytest.fixture
def subdirs_config() -> dict:
    """Provides a standard subdirectory configuration."""
    return {
        "images_subdir": "images",
        "wound_masks_subdir": "wound_masks",
        "body_mask_subdir": "body_mask",
        "depth_maps_subdir": "depth_maps",
        "marker_mask_subdir": "marker_mask",
    }


@pytest.fixture(autouse=True)
def setup_test_logging(caplog):
    """
    Sets up basic logging for tests to capture messages.
    'autouse=True' means this fixture runs automatically for all tests.
    """
    # Configure root logger to capture all messages
    logging.getLogger().setLevel(logging.DEBUG)

    if logging.getLogger().hasHandlers():
        logging.getLogger().handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    
    # Capture logs from the specific logger used in src.data_loader
    caplog.set_level(logging.DEBUG, logger="src.data_loader")

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


def create_dummy_csv(filepath: Path, df: pd.DataFrame, index: bool = False):
    """Creates a dummy CSV file from a DataFrame."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(filepath), index=index)

# --- Tests for data_loader ---

def test_data_loader_success(temp_data_root, subdirs_config, caplog):
    """
    GIVEN: A valid ImageID and all critical and optional files exist.
    WHEN:  data_loader is called.
    THEN:  All data should be loaded correctly, masks applied, and INFO messages logged.
    """

    image_id = "test_image"
    
    (temp_data_root / subdirs_config['images_subdir']).mkdir()
    (temp_data_root / subdirs_config['wound_masks_subdir']).mkdir()
    (temp_data_root / subdirs_config['body_mask_subdir']).mkdir()
    (temp_data_root / subdirs_config['depth_maps_subdir']).mkdir()
    (temp_data_root / subdirs_config['marker_mask_subdir']).mkdir()

    create_dummy_image(temp_data_root / subdirs_config['images_subdir'] / f"{image_id}.png", color=(255, 0, 0)) 
    create_dummy_mask(temp_data_root / subdirs_config['wound_masks_subdir'] / f"{image_id}.png")
    create_dummy_mask(temp_data_root / subdirs_config['body_mask_subdir'] / f"{image_id}.png", value=255) 
    create_dummy_depth_map(temp_data_root / subdirs_config['depth_maps_subdir'] / f"{image_id}.png", value=5000)
    
    marker_mask_path = temp_data_root / subdirs_config['marker_mask_subdir'] / f"{image_id}.png"
    marker_mask = np.zeros((100, 100), dtype=np.uint8)
    marker_mask[10:20, 10:20] = 255 
    cv2.imwrite(str(marker_mask_path), marker_mask)

    loaded_data = data_loader(image_id, temp_data_root, subdirs_config)

    assert loaded_data is not None
    assert isinstance(loaded_data['image'], np.ndarray)
    assert loaded_data['image'].shape == (100, 100, 3)
    assert np.all(loaded_data['image'][0,0] == [255, 0, 0]) 

    assert isinstance(loaded_data['wound'], np.ndarray)
    assert loaded_data['wound'].shape == (100, 100)
    assert np.all(loaded_data['wound'] == 255)

    assert isinstance(loaded_data['body'], np.ndarray)
    assert loaded_data['body'].shape == (100, 100)
    assert loaded_data['body'][0,0] == 255
    assert loaded_data['body'][15,15] == 0 

    assert isinstance(loaded_data['depth'], np.ndarray)
    assert loaded_data['depth'].shape == (100, 100)
    assert loaded_data['depth'][0,0] == 5000 
    assert loaded_data['depth'][15,15] == 0 

    assert f"Successfully loaded all critical files for {image_id}." in caplog.text
    assert f"Body mask applied to depth map for {image_id}." in caplog.text
    assert f"Marker mask applied to depth map and body mask for {image_id}." in caplog.text


def test_data_loader_missing_critical_image(temp_data_root, subdirs_config, caplog):
    """
    GIVEN: A missing main image file.
    WHEN:  data_loader is called.
    THEN:  FileNotFoundError should be raised, and an ERROR message logged.
    """

    image_id = "missing_image"
    # Create other necessary dirs
    (temp_data_root / subdirs_config['wound_masks_subdir']).mkdir(parents=True, exist_ok=True)
    (temp_data_root / subdirs_config['body_mask_subdir']).mkdir(parents=True, exist_ok=True)
    (temp_data_root / subdirs_config['depth_maps_subdir']).mkdir(parents=True, exist_ok=True)
    (temp_data_root / subdirs_config['marker_mask_subdir']).mkdir(parents=True, exist_ok=True) 

    with pytest.raises(FileNotFoundError) as excinfo:
        data_loader(image_id, temp_data_root, subdirs_config)
    
    expected_path = temp_data_root / subdirs_config['images_subdir'] / f"{image_id}.png"
    assert f"Image file not found: {expected_path}" in str(excinfo.value)
    assert f"Image file not found at {expected_path}. Terminating processing for this ID." in caplog.text


def test_data_loader_unreadable_image(temp_data_root, subdirs_config, caplog, monkeypatch):
    """
    GIVEN: An image file that cv2.imread cannot read.
    WHEN:  data_loader is called.
    THEN:  IOError should be raised, and an ERROR message logged.
    """

    image_id = "unreadable_image"
    image_path = temp_data_root / subdirs_config['images_subdir'] / f"{image_id}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.touch() 

    # Mock cv2.imread to return None
    original_imread = cv2.imread
    def mock_imread(path, *args, **kwargs):
        if Path(path) == image_path:
            return None
        return original_imread(path, *args, **kwargs)
    monkeypatch.setattr(cv2, 'imread', mock_imread)

    # Create other necessary files
    create_dummy_mask(temp_data_root / subdirs_config['wound_masks_subdir'] / f"{image_id}.png")
    create_dummy_mask(temp_data_root / subdirs_config['body_mask_subdir'] / f"{image_id}.png")
    create_dummy_depth_map(temp_data_root / subdirs_config['depth_maps_subdir'] / f"{image_id}.png")
    (temp_data_root / subdirs_config['marker_mask_subdir']).mkdir()

    with pytest.raises(IOError) as excinfo:
        data_loader(image_id, temp_data_root, subdirs_config)
    
    assert f"Failed to read image file: {image_path}" in str(excinfo.value)
    assert f"Image file at {image_path} could not be read. Terminating processing for this ID." in caplog.text


def test_data_loader_missing_optional_marker_mask(temp_data_root, subdirs_config, caplog):
    """
    GIVEN: A missing optional marker mask file.
    WHEN:  data_loader is called.
    THEN:  All critical data should still load, a WARNING logged, and marker mask defaults to zeros.
    """

    image_id = "no_marker"

    create_dummy_image(temp_data_root / subdirs_config['images_subdir'] / f"{image_id}.png")
    create_dummy_mask(temp_data_root / subdirs_config['wound_masks_subdir'] / f"{image_id}.png")
    create_dummy_mask(temp_data_root / subdirs_config['body_mask_subdir'] / f"{image_id}.png", value=255)
    create_dummy_depth_map(temp_data_root / subdirs_config['depth_maps_subdir'] / f"{image_id}.png", value=100)
    
    # Intentionally don't create marker mask file; ensure parent dir exists
    (temp_data_root / subdirs_config['marker_mask_subdir']).mkdir(parents=True, exist_ok=True)

    loaded_data = data_loader(image_id, temp_data_root, subdirs_config)

    assert loaded_data is not None
    assert isinstance(loaded_data['depth'], np.ndarray)
    assert np.all(loaded_data['depth'] == 100) 
    assert f"Marker mask file not found at {temp_data_root / subdirs_config['marker_mask_subdir'] / f'{image_id}.png'}. Defaulting marker mask to all black (no masking effect)." in caplog.text
    assert f"Marker mask for {image_id} is all zeros/empty. No marker masking applied." in caplog.text


def test_data_loader_marker_mask_shape_mismatch(temp_data_root, subdirs_config, caplog):
    """
    GIVEN: A marker mask with a shape that doesn't match the depth map.
    WHEN:  data_loader is called.
    THEN:  All critical data should load, a WARNING logged, and marker masking skipped.
    """
    image_id = "mismatch_marker"
    
    create_dummy_image(temp_data_root / subdirs_config['images_subdir'] / f"{image_id}.png")
    create_dummy_mask(temp_data_root / subdirs_config['wound_masks_subdir'] / f"{image_id}.png")
    create_dummy_mask(temp_data_root / subdirs_config['body_mask_subdir'] / f"{image_id}.png", value=255) 
    create_dummy_depth_map(temp_data_root / subdirs_config['depth_maps_subdir'] / f"{image_id}.png", value=100)

    marker_mask_path = temp_data_root / subdirs_config['marker_mask_subdir'] / f"{image_id}.png"
    create_dummy_mask(marker_mask_path, shape=(50, 50), value=255) 

    loaded_data = data_loader(image_id, temp_data_root, subdirs_config)

    assert loaded_data is not None
    assert isinstance(loaded_data['depth'], np.ndarray)
    assert np.all(loaded_data['depth'] == 100) 
    assert f"Marker mask shape (50, 50) does not match depth map shape (100, 100). Skipping marker masking for {image_id}." in caplog.text
    assert f"Marker mask for {image_id} is all zeros/empty. No marker masking applied." not in caplog.text 


def test_data_loader_body_mask_shape_mismatch(temp_data_root, subdirs_config, caplog):
    """
    GIVEN: A body mask with a shape that doesn't match the depth map.
    WHEN:  data_loader is called.
    THEN:  All critical data should load, a WARNING logged, and body masking skipped for depth.
    """
 
    image_id = "mismatch_body"
    
    create_dummy_image(temp_data_root / subdirs_config['images_subdir'] / f"{image_id}.png")
    create_dummy_mask(temp_data_root / subdirs_config['wound_masks_subdir'] / f"{image_id}.png")
    create_dummy_mask(temp_data_root / subdirs_config['body_mask_subdir'] / f"{image_id}.png", shape=(50, 50), value=255) # Mismatched shape
    create_dummy_depth_map(temp_data_root / subdirs_config['depth_maps_subdir'] / f"{image_id}.png", value=100)

    (temp_data_root / subdirs_config['marker_mask_subdir']).mkdir()

    loaded_data = data_loader(image_id, temp_data_root, subdirs_config)

    assert loaded_data is not None
    assert isinstance(loaded_data['depth'], np.ndarray)
    assert np.all(loaded_data['depth'] == 100)
    assert f"Body mask shape (50, 50) does not match depth map shape (100, 100). Skipping body masking for {image_id}." in caplog.text
    assert f"Marker mask file not found at {temp_data_root / subdirs_config['marker_mask_subdir'] / f'{image_id}.png'}. Defaulting marker mask to all black (no masking effect)." in caplog.text
    assert f"Marker mask for {image_id} is all zeros/empty. No marker masking applied." in caplog.text


def test_data_loader_missing_subdir_config_key(temp_data_root, caplog):
    """
    GIVEN: A subdirs_config missing a critical key.
    WHEN:  data_loader is called.
    THEN:  A ValueError should be raised.
    """

    image_id = "test_id"
    # Missing 'images_subdir'
    bad_subdirs_config = {
        "wound_masks_subdir": "wound_masks",
        "body_mask_subdir": "body_mask",
        "depth_maps_subdir": "depth_maps",
        "marker_mask_subdir": "marker_mask",
    }

    with pytest.raises(ValueError) as excinfo:
        data_loader(image_id, temp_data_root, bad_subdirs_config)
    assert "Missing subdirectory configuration key in subdirs_config: 'images_subdir'" in str(excinfo.value)
    assert "Missing subdirectory configuration key: 'images_subdir'. Please check your subdirs_config." in caplog.text


def test_data_loader_bgr_to_rgb_conversion(temp_data_root, subdirs_config):
    """
    GIVEN: An image file.
    WHEN:  data_loader loads it.
    THEN:  The loaded image should be in RGB format (not BGR).
    """

    image_id = "rgb_test"
    test_color = (255, 0, 0) 
    create_dummy_image(temp_data_root / subdirs_config['images_subdir'] / f"{image_id}.png", color=test_color)
    create_dummy_mask(temp_data_root / subdirs_config['wound_masks_subdir'] / f"{image_id}.png")
    create_dummy_mask(temp_data_root / subdirs_config['body_mask_subdir'] / f"{image_id}.png")
    create_dummy_depth_map(temp_data_root / subdirs_config['depth_maps_subdir'] / f"{image_id}.png")
    (temp_data_root / subdirs_config['marker_mask_subdir']).mkdir() 

    loaded_data = data_loader(image_id, temp_data_root, subdirs_config)

    assert loaded_data is not None
    assert np.all(loaded_data['image'][0, 0] == test_color)


# --- Tests for load_and_clean_features ---

def test_load_and_clean_features_success(temp_data_root, caplog):
    """
    GIVEN: A CSV file with feature data, including some NaNs.
    WHEN:  load_and_clean_features is called.
    THEN:  DataFrame should be loaded and cleaned, image IDs and features extracted correctly.
           INFO messages should be logged.
    """

    features_df_data = {
        'image_id': ['img1', 'img2', 'img3', 'img4'],
        'feat_A': [1.1, 2.2, np.nan, 4.4],
        'feat_B': [5.5, 6.6, 7.7, 8.8],
    }
    features_csv_path = temp_data_root / "comprehensive_features.csv"
    create_dummy_csv(features_csv_path, pd.DataFrame(features_df_data))

    df_clean, image_ids, features = load_and_clean_features(features_csv_path)

    assert isinstance(df_clean, pd.DataFrame)
    assert len(df_clean) == 3 
    assert df_clean['image_id'].tolist() == ['img1', 'img2', 'img4']
    assert np.array_equal(image_ids, np.array(['img1', 'img2', 'img4']))
    assert features.shape == (3, 2)
    assert np.array_equal(features, np.array([[1.1, 5.5], [2.2, 6.6], [4.4, 8.8]]))
    assert f"Loaded {len(features_df_data['image_id'])} feature rows from '{features_csv_path}'." in caplog.text
    assert f"{len(df_clean)} rows remaining after dropping NaNs." in caplog.text
    assert "Features loaded and cleaned successfully." in caplog.text


def test_load_and_clean_features_empty_csv(temp_data_root, caplog):
    """
    GIVEN: An empty CSV file.
    WHEN:  load_and_clean_features is called.
    THEN:  Empty DataFrame and arrays should be returned, and a WARNING logged.
    """

    empty_csv_path = temp_data_root / "empty_features.csv"
    empty_csv_path.parent.mkdir(parents=True, exist_ok=True)
    empty_csv_path.touch() 

    df_clean, image_ids, features = load_and_clean_features(empty_csv_path)

    assert isinstance(df_clean, pd.DataFrame)
    assert df_clean.empty
    assert 'image_id' in df_clean.columns 
    assert np.array_equal(image_ids, np.array([]))
    assert features.shape == (0,0)
    assert f"CSV file at '{empty_csv_path}' is empty. Returning empty data structures." in caplog.text


def test_load_and_clean_features_file_not_found(temp_data_root, caplog):
    """
    GIVEN: A non-existent CSV file.
    WHEN:  load_and_clean_features is called.
    THEN:  FileNotFoundError should be raised, and an ERROR logged.
    """
 
    non_existent_path = temp_data_root / "non_existent.csv"

    with pytest.raises(FileNotFoundError) as excinfo:
        load_and_clean_features(non_existent_path)
    assert f"Feature CSV file not found: {non_existent_path}" in str(excinfo.value)
    assert f"CSV file not found at '{non_existent_path}'. Please ensure the path is correct or generate the file first." in caplog.text


def test_load_and_clean_features_missing_image_id_column(temp_data_root, caplog):
    """
    GIVEN: A CSV file missing the 'image_id' column.
    WHEN:  load_and_clean_features is called.
    THEN:  IOError (as it's re-raised) should be raised, and an ERROR logged.
    """

    df_missing_id = pd.DataFrame({'feature_A': [1.1, 2.2], 'feature_B': [5.5, 6.6]})
    csv_path = temp_data_root / "missing_id.csv"
    create_dummy_csv(csv_path, df_missing_id)

    with pytest.raises(IOError) as excinfo:
        load_and_clean_features(csv_path)
    assert f"Required 'image_id' column not found in '{csv_path}'." in str(excinfo.value)
    assert f"Required 'image_id' column not found in loaded CSV: '{csv_path}'." in caplog.text

# --- Tests for load_cluster_groups ---

def test_load_cluster_groups_success(temp_data_root, caplog):
    """
    GIVEN: A CSV file with image_id and cluster_label.
    WHEN:  load_cluster_groups is called.
    THEN:  Cluster groups and DataFrame should be loaded correctly, and INFO logged.
    """

    cluster_map_data = {
        'image_id': ['imgA', 'imgB', 'imgC', 'imgD', 'imgE'],
        'cluster_label': [0, 1, 0, 2, 1]
    }
    cluster_map_path = temp_data_root / "image_cluster_map.csv"
    create_dummy_csv(cluster_map_path, pd.DataFrame(cluster_map_data))

    cluster_groups, cluster_map_df = load_cluster_groups(cluster_map_path)

    assert isinstance(cluster_groups, pd.Series)
    assert sorted(cluster_groups.index.tolist()) == [0, 1, 2]
    assert cluster_groups[0] == ['imgA', 'imgC']
    assert cluster_groups[1] == ['imgB', 'imgE']
    assert cluster_groups[2] == ['imgD']

    assert isinstance(cluster_map_df, pd.DataFrame)
    assert cluster_map_df.equals(pd.DataFrame(cluster_map_data))
    assert f"Loading cluster map from '{cluster_map_path}'." in caplog.text
    assert f"Successfully loaded cluster groups for 3 clusters." in caplog.text


def test_load_cluster_groups_empty_csv(temp_data_root, caplog):
    """
    GIVEN: An empty CSV file for cluster groups.
    WHEN:  load_cluster_groups is called.
    THEN:  Empty Series and DataFrame should be returned, and a WARNING logged.
    """

    empty_csv_path = temp_data_root / "empty_cluster_map.csv"
    empty_csv_path.parent.mkdir(parents=True, exist_ok=True)
    empty_csv_path.touch()

    cluster_groups, cluster_map_df = load_cluster_groups(empty_csv_path)

    assert isinstance(cluster_groups, pd.Series)
    assert cluster_groups.empty
    assert isinstance(cluster_map_df, pd.DataFrame)
    assert cluster_map_df.empty
    assert cluster_map_df.columns.tolist() == ['image_id', 'cluster_label']
    assert f"CSV file at '{empty_csv_path}' is empty. Returning empty data structures for cluster groups." in caplog.text


def test_load_cluster_groups_file_not_found(temp_data_root, caplog):
    """
    GIVEN: A non-existent CSV file for cluster groups.
    WHEN:  load_cluster_groups is called.
    THEN:  FileNotFoundError should be raised, and an ERROR logged.
    """

    non_existent_path = temp_data_root / "non_existent_cluster_map.csv"

    with pytest.raises(FileNotFoundError) as excinfo:
        load_cluster_groups(non_existent_path)
    assert f"Cluster map CSV file not found: {non_existent_path}" in str(excinfo.value)
    assert f"Cluster map CSV file not found at '{non_existent_path}'. Cannot load cluster groups." in caplog.text


def test_load_cluster_groups_missing_columns(temp_data_root, caplog):
    """
    GIVEN: A CSV file for cluster groups missing required columns.
    WHEN:  load_cluster_groups is called.
    THEN:  IOError (as it's re-raised) should be raised, and an ERROR logged.
    """

    df_bad_cols = pd.DataFrame({'ImageID': ['id1', 'id2'], 'Label': [0, 1]})
    csv_path = temp_data_root / "bad_cluster_map.csv"
    create_dummy_csv(csv_path, df_bad_cols)

    with pytest.raises(IOError) as excinfo:
        load_cluster_groups(csv_path)
    assert f"Required 'image_id' or 'cluster_label' column not found in '{csv_path}'." in str(excinfo.value)
    assert f"Required 'image_id' or 'cluster_label' column not found in loaded CSV: '{csv_path}'." in caplog.text