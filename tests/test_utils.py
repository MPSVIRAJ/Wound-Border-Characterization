"""
This module contains unit tests for the utility functions defined in `src.utils`.

It utilizes pytest to verify the correctness and robustness of functions related
to DataFrame saving, feature management, and cluster analysis. Tests cover
successful operations, various edge cases, and error handling for invalid inputs
or unexpected file system issues.

Functions:
    - `temp_output_dir`: A pytest fixture providing a temporary directory for test output files.
    - `setup_test_logging`: A pytest fixture to configure logging for capturing messages during tests.
    - `create_dummy_csv`: A helper function to create simple CSV files for testing purposes.
    - Test functions (e.g., `test_save_dataframe_to_csv_overwrite`, `test_save_features_to_csv_success`,
    `test_save_cluster_assignments_success`, `test_generate_cluster_summary_success`,
    `test_generate_cluster_profiles_success`): Comprehensive tests for each utility function.

Typical use:
    This module is designed to be executed by `pytest` as part of the project's
    continuous integration and development workflow. It ensures the reliability
    of core utility functions crucial for data processing and pipeline execution.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import logging # For caplog
import sys

# Import functions from src/utils.py
from src.utils import (
    save_dataframe_to_csv,
    save_features_to_csv,
    save_cluster_assignments,
    generate_cluster_summary,
    generate_cluster_profiles,
)

# --- Fixture for temporary output directories ---
@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Provides a temporary directory for output files for each test."""
    output_dir = tmp_path / "output_files"
    output_dir.mkdir()
    return output_dir


# --- Fixture for setting up basic logging (needed for tests using caplog) ---
@pytest.fixture(autouse=True)
def setup_test_logging(caplog):
    """
    Sets up basic logging for tests to capture messages.
    'autouse=True' means this fixture runs automatically for all tests.
    """
    # Configure root logger to capture all messages
    logging.getLogger().setLevel(logging.DEBUG)
    # Ensure handlers are cleared to prevent duplicate messages across test runs
    if logging.getLogger().hasHandlers():
        logging.getLogger().handlers.clear()
    handler = logging.StreamHandler(sys.stdout) 
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    
    caplog.set_level(logging.DEBUG, logger="src.utils") 

# --- Helper for creating dummy CSVs ---
def create_dummy_csv(file_path: Path, df: pd.DataFrame):
    """Helper to write a DataFrame to a CSV file."""
    df.to_csv(str(file_path), index=False)

# --- Tests for save_dataframe_to_csv ---

def test_save_dataframe_to_csv_overwrite(temp_output_dir: Path, caplog):
    """
    GIVEN: A DataFrame and a non-existent output path for overwrite mode.
    WHEN:  save_dataframe_to_csv is called with append_mode=False.
    THEN:  A new CSV file should be created with the DataFrame content and header.
           An INFO message should be logged.
    """

    df_data = pd.DataFrame({'col1': [1, 2], 'col2': ['A', 'B']})
    output_file = temp_output_dir / "overwrite_test.csv"

    save_dataframe_to_csv(df_data, output_file, append_mode=False)

    assert output_file.exists()
    assert output_file.read_text() == "col1,col2\n1,A\n2,B\n"
    assert "INFO" in caplog.text
    assert f"DataFrame successfully saved to {output_file} (mode='w', header=True, index=False)." in caplog.text


def test_save_dataframe_to_csv_append_existing_no_header(temp_output_dir: Path, caplog):
    """
    GIVEN: An existing CSV file, a new DataFrame, and append_mode=True with include_header=False.
    WHEN:  save_dataframe_to_csv is called.
    THEN:  The new DataFrame content should be appended without a new header.
           An INFO message should be logged.
    """

    initial_df = pd.DataFrame({'col1': [1, 2], 'col2': ['A', 'B']})
    output_file = temp_output_dir / "append_test.csv"
    create_dummy_csv(output_file, initial_df)
    
    df_to_append = pd.DataFrame({'col1': [3], 'col2': ['C']})

    save_dataframe_to_csv(df_to_append, output_file, append_mode=True, include_header=False)

    expected_content = "col1,col2\n1,A\n2,B\n3,C\n"
    assert output_file.read_text() == expected_content
    assert "INFO" in caplog.text
    assert f"DataFrame successfully saved to {output_file} (mode='a', header=False, index=False)." in caplog.text


def test_save_dataframe_to_csv_append_non_existing_auto_header(temp_output_dir: Path, caplog):
    """
    GIVEN: A non-existent output path and append_mode=True with include_header=None (auto).
    WHEN:  save_dataframe_to_csv is called.
    THEN:  A new CSV file should be created with the DataFrame content and header.
           An INFO message should be logged.
    """

    df_data = pd.DataFrame({'col1': [10, 20]})
    output_file = temp_output_dir / "auto_header_test.csv"

    save_dataframe_to_csv(df_data, output_file, append_mode=True, include_header=None)

    assert output_file.exists()
    assert output_file.read_text() == "col1\n10\n20\n"
    assert "INFO" in caplog.text
    assert f"DataFrame successfully saved to {output_file} (mode='a', header=True, index=False)." in caplog.text


def test_save_dataframe_to_csv_type_error_df(temp_output_dir: Path, caplog):
    """
    GIVEN: A non-DataFrame object as input.
    WHEN:  save_dataframe_to_csv is called.
    THEN:  A TypeError should be raised, and an ERROR message should be logged.
    """
    not_a_df = [1, 2, 3]
    output_file = temp_output_dir / "type_error.csv"

    with pytest.raises(TypeError) as excinfo:
        save_dataframe_to_csv(not_a_df, output_file)
    assert "df must be a pandas.DataFrame" in str(excinfo.value)
    assert "ERROR" in caplog.text
    assert f"Invalid type for df: Expected pd.DataFrame, got {type(not_a_df)}" in caplog.text


def test_save_dataframe_to_csv_type_error_path(caplog): 
    """
    GIVEN: A non-Path object as output_filepath.
    WHEN:  save_dataframe_to_csv is called.
    THEN:  A TypeError should be raised, and an ERROR message should be logged.
    """

    df_data = pd.DataFrame({'col1': [1]})
    not_a_path = "/invalid/path"

    with pytest.raises(TypeError) as excinfo:
        save_dataframe_to_csv(df_data, not_a_path)
    assert "output_filepath must be a pathlib.Path object" in str(excinfo.value)
    assert "ERROR" in caplog.text
    assert f"Invalid type for output_filepath: Expected Path, got {type(not_a_path)}" in caplog.text


def test_save_dataframe_to_csv_io_error(temp_output_dir: Path, caplog, monkeypatch):
    """
    GIVEN: A DataFrame and an output path that causes an IOError (e.g., permission denied).
    WHEN:  save_dataframe_to_csv is called.
    THEN:  An IOError should be raised, and an EXCEPTION should be logged.
    """

    df_data = pd.DataFrame({'col1': [1]})
    output_file = temp_output_dir / "no_permission.csv"

    def mock_to_csv(*args, **kwargs):
        raise OSError("Permission denied: Mock error for testing IOError")
    
    monkeypatch.setattr(pd.DataFrame, 'to_csv', mock_to_csv)

    with pytest.raises(IOError) as excinfo:
        save_dataframe_to_csv(df_data, output_file)
    
    assert "Error saving DataFrame to" in str(excinfo.value)
    assert "Permission denied" in str(excinfo.value)
    assert "ERROR" in caplog.text
    assert "Failed to save DataFrame" in caplog.text
    assert "Permission denied: Mock error for testing IOError" in caplog.text 

# --- Tests for save_features_to_csv ---

def test_save_features_to_csv_success(temp_output_dir: Path, caplog):
    """
    GIVEN: A valid image ID, features dictionary, and output path.
    WHEN:  save_features_to_csv is called for the first time.
    THEN:  A CSV file should be created with 'image_id' as the first column,
           and an INFO message should be logged indicating header=True.
    """

    image_id = "test_img_001"
    features_dict = {'feat_A': 0.5, 'feat_B': 1.2}
    output_file = temp_output_dir / "features.csv"

    save_features_to_csv(image_id, features_dict, output_file)

    assert output_file.exists()
    df_saved = pd.read_csv(output_file)
    assert df_saved.columns.tolist() == ['image_id', 'feat_A', 'feat_B']
    assert df_saved['image_id'].iloc[0] == image_id
    assert df_saved['feat_A'].iloc[0] == features_dict['feat_A']
    assert "INFO" in caplog.text
    assert f"DataFrame successfully saved to {output_file} (mode='a', header=True, index=False)." in caplog.text


def test_save_features_to_csv_append(temp_output_dir: Path, caplog):
    """
    GIVEN: An existing features CSV file and new features.
    WHEN:  save_features_to_csv is called subsequently.
    THEN:  New features should be appended without a new header,
           and an INFO message should be logged indicating header=False.
    """

    image_id_1 = "test_img_001"
    features_dict_1 = {'feat_A': 0.5, 'feat_B': 1.2}
    output_file = temp_output_dir / "features_append.csv" 
    save_features_to_csv(image_id_1, features_dict_1, output_file) 

    image_id_2 = "test_img_002"
    features_dict_2 = {'feat_A': 0.6, 'feat_B': 1.3}

    caplog.clear()

    save_features_to_csv(image_id_2, features_dict_2, output_file)

    df_saved = pd.read_csv(output_file)
    assert len(df_saved) == 2
    assert df_saved['image_id'].tolist() == [image_id_1, image_id_2]
    assert "INFO" in caplog.text
    assert f"DataFrame successfully saved to {output_file} (mode='a', header=False, index=False)." in caplog.text


def test_save_features_to_csv_empty_dict(temp_output_dir: Path, caplog):
    """
    GIVEN: An empty features dictionary.
    WHEN:  save_features_to_csv is called.
    THEN:  A ValueError should be raised, and a WARNING message should be logged.
    """

    image_id = "empty_features_id"
    features_dict = {}
    output_file = temp_output_dir / "empty_features.csv"

    with pytest.raises(ValueError) as excinfo:
        save_features_to_csv(image_id, features_dict, output_file)
    assert "Feature dictionary is empty." in str(excinfo.value)
    assert "WARNING" in caplog.text
    assert "Feature dictionary is empty. Nothing to save for ImageID: empty_features_id" in caplog.text
    assert not output_file.exists() 


def test_save_features_to_csv_type_error_image_id(temp_output_dir: Path, caplog):
    """
    GIVEN: Non-string ImageID.
    WHEN:  save_features_to_csv is called.
    THEN:  TypeError should be raised.
    """
    dummy_output_file = temp_output_dir / "dummy.csv"

    with pytest.raises(TypeError) as excinfo:
        save_features_to_csv(123, {'a':1}, dummy_output_file)
    assert "ImageID must be a string" in str(excinfo.value)

# --- Tests for save_cluster_assignments ---

def test_save_cluster_assignments_success(temp_output_dir: Path, caplog):
    """
    GIVEN: A DataFrame with image_id and cluster_label.
    WHEN:  save_cluster_assignments is called.
    THEN:  A CSV file should be created with image IDs and cluster labels,
           and cluster counts should be returned.
           INFO messages should be logged.
    """

    df_mock = pd.DataFrame({
        'image_id': ['imgA', 'imgB', 'imgC', 'imgD', 'imgE'],
        'cluster_label': [0, 1, 0, 2, 1],
        'other_col': [1,2,3,4,5] 
    })
    output_file = temp_output_dir / "image_cluster_map.csv"

    cluster_counts = save_cluster_assignments(df_mock, output_file) 

    assert output_file.exists()
    df_saved = pd.read_csv(output_file)
    assert df_saved.columns.tolist() == ['image_id', 'cluster_label']
    assert df_saved['image_id'].tolist() == ['imgA', 'imgB', 'imgC', 'imgD', 'imgE']
    assert df_saved['cluster_label'].tolist() == [0, 1, 0, 2, 1]
    
    assert cluster_counts == {0: 2, 1: 2, 2: 1}
    assert "INFO" in caplog.text
    assert "Cluster sample counts: {0: 2, 1: 2, 2: 1}" in caplog.text
    assert f"DataFrame successfully saved to {output_file} (mode='w', header=True, index=False)." in caplog.text


def test_save_cluster_assignments_missing_columns(temp_output_dir: Path, caplog):
    """
    GIVEN: A DataFrame missing 'image_id' or 'cluster_label'.
    WHEN:  save_cluster_assignments is called.
    THEN:  A ValueError should be raised, and an ERROR message logged.
    """

    df_mock = pd.DataFrame({'id': [1], 'label': [0]})
    output_file = temp_output_dir / "image_cluster_map.csv" 

    with pytest.raises(ValueError) as excinfo:
        save_cluster_assignments(df_mock, output_file) 
    assert "Input DataFrame is missing required columns" in str(excinfo.value)
    assert "ERROR" in caplog.text
    assert "DataFrame must contain 'image_id' and 'cluster_label' columns" in caplog.text
    assert not output_file.exists()


# --- Tests for generate_cluster_summary ---

def test_generate_cluster_summary_success(temp_output_dir: Path, caplog):
    """
    GIVEN: A DataFrame with numeric features and cluster_label.
    WHEN:  generate_cluster_summary is called.
    THEN:  A CSV file should be created with summary statistics per cluster,
           and a DataFrame should be returned.
           INFO messages should be logged.
    """
    df_mock = pd.DataFrame({
        'image_id': ['id1', 'id2', 'id3', 'id4', 'id5'],
        'feature_A': [10, 12, 11, 20, 22],
        'feature_B': [1, 2, 1, 5, 6],
        'cluster_label': [0, 0, 0, 1, 1]
    })
    output_file = temp_output_dir / "cluster_summary_stats.csv"

    summary_df = generate_cluster_summary(df_mock, output_file)

    assert output_file.exists()
    df_saved = pd.read_csv(output_file)
    
    assert not summary_df.empty
    assert summary_df.columns.tolist() == ['cluster_label', 'feature_A_mean', 'feature_A_std', 'feature_A_count', 'feature_B_mean', 'feature_B_std', 'feature_B_count']
    assert summary_df.iloc[0]['cluster_label'] == 0
    assert summary_df.iloc[0]['feature_A_mean'] == pytest.approx(11.0)
    assert summary_df.iloc[1]['cluster_label'] == 1
    assert summary_df.iloc[1]['feature_A_mean'] == pytest.approx(21.0)
    
    assert "INFO" in caplog.text
    assert f"DataFrame successfully saved to {output_file} (mode='w', header=True, index=False)." in caplog.text


def test_generate_cluster_summary_missing_cluster_label(temp_output_dir: Path, caplog):
    """
    GIVEN: A DataFrame missing 'cluster_label' column.
    WHEN:  generate_cluster_summary is called.
    THEN:  A ValueError should be raised, and an ERROR message logged.
    """

    df_mock = pd.DataFrame({'feature_A': [10, 12]})
    output_file = temp_output_dir / "cluster_summary_stats.csv" 

    with pytest.raises(ValueError) as excinfo:
        generate_cluster_summary(df_mock, output_file) 
    assert "Input DataFrame is missing required column" in str(excinfo.value)
    assert "ERROR" in caplog.text
    assert "DataFrame must contain 'cluster_label' column" in caplog.text
    assert not output_file.exists()


def test_generate_cluster_summary_no_numeric_features(temp_output_dir: Path, caplog):
    """
    GIVEN: A DataFrame with only non-numeric columns besides cluster_label.
    WHEN:  generate_cluster_summary is called.
    THEN:  An empty DataFrame with 'cluster_label' should be returned.
           A WARNING message should be logged if no numeric features are found.
           No CSV file should be created.
    """
    df_mock = pd.DataFrame({
        'image_id': ['id1', 'id2'],
        'category': ['catA', 'catB'],
        'cluster_label': [0, 1]
    })
    output_file = temp_output_dir / "cluster_summary_stats.csv" 

    summary_df = generate_cluster_summary(df_mock, output_file)
    
    assert not summary_df.empty 
    assert summary_df.columns.tolist() == ['cluster_label'] 
    assert summary_df['cluster_label'].tolist() == [0, 1] 
    assert not output_file.exists() 
    assert "WARNING" in caplog.text
    assert "No numeric feature columns found to generate cluster summary. Returning DataFrame with cluster_label column only." in caplog.text 
    caplog.clear() 
    assert f"DataFrame successfully saved to {output_file}" not in caplog.text

# --- Tests for generate_cluster_profiles ---

def test_generate_cluster_profiles_success(temp_output_dir: Path, caplog):
    """
    GIVEN: A DataFrame with numeric features and cluster_label.
    WHEN:  generate_cluster_profiles is called.
    THEN:  A CSV file should be created with mean feature profiles per cluster,
           and a DataFrame should be returned.
           INFO messages should be logged.
    """

    df_mock = pd.DataFrame({
        'image_id': ['id1', 'id2', 'id3', 'id4', 'id5'],
        'feature_X': [10.0, 11.0, 10.5, 20.0, 21.0],
        'feature_Y': [1.0, 1.2, 1.1, 2.0, 2.3],
        'cluster_label': [0, 0, 0, 1, 1]
    })
    output_file = temp_output_dir / "cluster_profiles.csv" 

    profiles_df = generate_cluster_profiles(df_mock, output_file) 
   
    assert output_file.exists()
    df_saved = pd.read_csv(output_file, index_col='cluster_label')

    assert not profiles_df.empty
    assert profiles_df.index.tolist() == [0, 1]
    assert profiles_df.columns.tolist() == ['feature_X', 'feature_Y']

    np.testing.assert_allclose(profiles_df['feature_X'].values, df_saved['feature_X'].values)
    np.testing.assert_allclose(profiles_df['feature_Y'].values, df_saved['feature_Y'].values)
    
    assert "INFO" in caplog.text
    assert f"DataFrame successfully saved to {output_file} (mode='w', header=True, index=True)." in caplog.text


def test_generate_cluster_profiles_missing_cluster_label(temp_output_dir: Path, caplog):
    """
    GIVEN: A DataFrame missing 'cluster_label' column.
    WHEN:  generate_cluster_profiles is called.
    THEN:  A ValueError should be raised, and an ERROR message logged.
    """

    df_mock = pd.DataFrame({'feature_X': [1.0, 2.0]})
    output_file = temp_output_dir / "cluster_profiles.csv" 

    with pytest.raises(ValueError) as excinfo:
        generate_cluster_profiles(df_mock, output_file)
    assert "Input DataFrame is missing required column" in str(excinfo.value)
    assert "ERROR" in caplog.text
    assert "DataFrame must contain 'cluster_label' column" in caplog.text
    assert not output_file.exists()


def test_generate_cluster_profiles_no_numeric_features(temp_output_dir: Path, caplog):
    """
    GIVEN: A DataFrame with no numeric feature columns (only image_id and cluster_label).
    WHEN:  generate_cluster_profiles is called.
    THEN:  An empty DataFrame with appropriate columns should be returned.
           A WARNING message should be logged.
           No CSV file should be created.
    """

    df_mock = pd.DataFrame({
        'image_id': ['id1', 'id2'],
        'cluster_label': [0, 1]
    })
    output_file = temp_output_dir / "cluster_profiles.csv" 

    profiles_df = generate_cluster_profiles(df_mock, output_file)
 
    assert profiles_df.empty
    assert not output_file.exists()
    assert profiles_df.columns.tolist() == []
    assert profiles_df.index.tolist() == [0, 1]
    assert "WARNING" in caplog.text
    assert "No numeric feature columns found to generate cluster profiles. Returning empty DataFrame." in caplog.text
    caplog.clear() 
    assert f"DataFrame successfully saved to {output_file}" not in caplog.text


def test_save_dataframe_to_csv_io_error(temp_output_dir, caplog, monkeypatch):
    """
    GIVEN: A scenario where writing to a file fails.
    WHEN:  save_dataframe_to_csv is called.
    THEN:  It should log an error and not crash.
    """
    df_data = pd.DataFrame({'col1': [1]})
    output_file = temp_output_dir / "no_permission.csv"

    def mock_to_csv(*args, **kwargs):
        raise OSError("Permission denied: Mock error for testing IOError")

    monkeypatch.setattr(pd.DataFrame, 'to_csv', mock_to_csv)

    with pytest.raises(IOError) as excinfo:
        save_dataframe_to_csv(df_data, output_file)
    
    assert "Error saving DataFrame to" in str(excinfo.value)
    assert "Permission denied" in str(excinfo.value)
    assert "ERROR" in caplog.text
    assert "Failed to save DataFrame" in caplog.text


def test_save_cluster_assignments_key_error(temp_output_dir, caplog):
    """
    GIVEN: A DataFrame missing the 'cluster_label' column.
    WHEN:  save_cluster_assignments is called.
    THEN:  It should raise a ValueError and log an error.
    """

    df_missing_col = pd.DataFrame({'image_id': ['img1', 'img2']})
    filepath = temp_output_dir / "assignments.csv"
    with pytest.raises(ValueError):
        save_cluster_assignments(df_missing_col, filepath)
    assert "DataFrame must contain 'image_id' and 'cluster_label' columns" in caplog.text


def test_save_cluster_assignments_empty_df(temp_output_dir):
    """
    GIVEN: An empty DataFrame.
    WHEN:  save_cluster_assignments is called.
    THEN:  It should return an empty dictionary and save a file with only headers.
    """
    df_empty = pd.DataFrame({'image_id': [], 'cluster_label': []})
    filepath = temp_output_dir / "empty_assignments.csv"

    counts = save_cluster_assignments(df_empty, filepath)
    # Assert that the returned counts dictionary is empty
    assert counts == {}
    # Verify that the file was created correctly
    assert filepath.exists()
    saved_content = pd.read_csv(filepath)
    assert saved_content.empty
    assert list(saved_content.columns) == ['image_id', 'cluster_label']


def test_generate_cluster_summary_empty_df(temp_output_dir, caplog):
    """
    GIVEN: An empty DataFrame.
    WHEN:  generate_cluster_summary is called.
    THEN:  It should return an empty DataFrame and log the correct warning.
    """
    df_empty = pd.DataFrame(columns=['cluster_label', 'feat1'])
    filepath = temp_output_dir / "summary.csv"
    summary_df = generate_cluster_summary(df_empty, filepath)
    assert summary_df.empty
    assert "Input DataFrame is empty. Cannot generate cluster summary." in caplog.text

def test_save_dataframe_to_csv_type_error(temp_output_dir):
    """
    GIVEN: A list instead of a DataFrame.
    WHEN:  save_dataframe_to_csv is called.
    THEN:  It should raise a TypeError.
    """
    # Pass a list instead of a pandas DataFrame
    not_a_df = [1, 2, 3]
    filepath = temp_output_dir / "test.csv"
    
    with pytest.raises(TypeError):
        save_dataframe_to_csv(not_a_df, filepath)