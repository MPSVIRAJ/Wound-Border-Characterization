"""
This module contains unit tests for the clustering module (`src.clustering.py`).

It employs pytest to verify the correctness, robustness, and proper error/warning handling
of the functions that perform dimensionality reduction and clustering. Tests cover a wide
range of scenarios, including successful execution, handling of empty or invalid inputs,
missing configuration parameters, and simulating failures from external libraries.

Functions:
    - `setup_test_logging`: A pytest fixture for configuring logging capture.
    - `TestClustering` (class): A test class that groups related tests for `apply_pacmap`
  and `perform_hdbscan_clustering`, utilizing mocking for robust and isolated testing.

Typical use:
    This module is designed to be executed as part of the project's automated test suite
    using `pytest`. It ensures the reliability and integrity of the clustering components,
    which are a critical part of the machine learning pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import logging
import sys
from unittest.mock import patch, Mock

# Import functions from src/clustering.py
from src.clustering import (
    apply_pacmap,
    perform_hdbscan_clustering,
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
    
    caplog.set_level(logging.DEBUG, logger="src.clustering")


# --- Mocks for external libraries (PaCMAP and HDBSCAN) to simulate failures ---

# Patching these at the module level for consistency
@patch('src.clustering.pacmap')
@patch('src.clustering.hdbscan')
class TestClustering:

    # --- Tests for apply_pacmap ---
    
    def test_apply_pacmap_success(self, mock_hdbscan, mock_pacmap):
        """
        GIVEN: A valid DataFrame and valid clustering parameters.
        WHEN: apply_pacmap is called.
        THEN: It should return a 2D NumPy array embedding.
        """

        df = pd.DataFrame(np.random.rand(100, 50))
        clustering_params = {'pacmap_n_components': 2, 'pacmap_mn_ratio': 0.5,
                             'pacmap_fp_ratio': 2.0, 'random_state': 42}
        
        mock_pacmap.PaCMAP.return_value.fit_transform.return_value = np.random.rand(100, 2)
        embedding = apply_pacmap(df, clustering_params)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (100, 2)
        mock_pacmap.PaCMAP.assert_called_once()
        
        
    def test_apply_pacmap_empty_dataframe_input(self, mock_hdbscan, mock_pacmap, caplog):
        """
        GIVEN: An empty DataFrame.
        WHEN: apply_pacmap is called.
        THEN: It should return None and log a warning.
        """

        df = pd.DataFrame()
        clustering_params = {'pacmap_n_components': 2, 'pacmap_mn_ratio': 0.5,
                             'pacmap_fp_ratio': 2.0, 'random_state': 42}
        
        embedding = apply_pacmap(df, clustering_params)
        
        assert embedding is None
        assert "Input DataFrame for PaCMAP is empty. Returning None." in caplog.text
        mock_pacmap.PaCMAP.assert_not_called()
        
        
    def test_apply_pacmap_non_dataframe_input(self, mock_hdbscan, mock_pacmap):
        """
        GIVEN: A non-DataFrame input.
        WHEN: apply_pacmap is called.
        THEN: It should raise a TypeError.
        """

        non_df = np.random.rand(10, 10)
        clustering_params = {'pacmap_n_components': 2, 'pacmap_mn_ratio': 0.5,
                             'pacmap_fp_ratio': 2.0, 'random_state': 42}
        
        with pytest.raises(TypeError) as excinfo:
            apply_pacmap(non_df, clustering_params)

        assert "Input 'df' must be a Pandas DataFrame." in str(excinfo.value)
        
        
    def test_apply_pacmap_missing_params(self, mock_hdbscan, mock_pacmap):
        """
        GIVEN: A clustering_params dict with a missing key.
        WHEN: apply_pacmap is called.
        THEN: It should raise a ValueError.
        """

        df = pd.DataFrame(np.random.rand(100, 50))
        clustering_params = {'pacmap_n_components': 2, 'pacmap_mn_ratio': 0.5,
                             'pacmap_fp_ratio': 2.0} 

        with pytest.raises(ValueError) as excinfo:
            apply_pacmap(df, clustering_params)

        assert "Missing required parameters for PaCMAP." in str(excinfo.value)
        
        
    def test_apply_pacmap_runtime_error(self, mock_hdbscan, mock_pacmap, caplog):
        """
        GIVEN: A scenario where the PaCMAP algorithm raises an internal error.
        WHEN: apply_pacmap is called.
        THEN: It should raise a RuntimeError and log the exception.
        """

        df = pd.DataFrame(np.random.rand(100, 50))
        clustering_params = {'pacmap_n_components': 2, 'pacmap_mn_ratio': 0.5,
                             'pacmap_fp_ratio': 2.0, 'random_state': 42}
        
        # Mock PaCMAP's fit_transform to raise an error
        mock_pacmap.PaCMAP.return_value.fit_transform.side_effect = RuntimeError("PaCMAP failed")
        
        with pytest.raises(RuntimeError) as excinfo:
            apply_pacmap(df, clustering_params)

        assert "PaCMAP execution failed: PaCMAP failed" in str(excinfo.value)
        assert "An error occurred during PaCMAP execution." in caplog.text


    def test_apply_pacmap_nan_input(self, mock_hdbscan, mock_pacmap, caplog, monkeypatch):
        """
        GIVEN: A DataFrame with NaN values.
        WHEN: apply_pacmap is called.
        THEN: The StandardScaler should fail, and the function should raise a RuntimeError
              and log the appropriate error message.
        """
        df = pd.DataFrame(np.random.rand(10, 5))
        df.iloc[0, 0] = np.nan
        clustering_params = {'pacmap_n_components': 2, 'pacmap_mn_ratio': 0.5,
                             'pacmap_fp_ratio': 2.0, 'random_state': 42}
        
        def mock_fit_transform_fail(*args, **kwargs):
            raise ValueError("Input contains NaN, infinity or a value too large for dtype('float64')")

        monkeypatch.setattr('src.clustering.StandardScaler.fit_transform', mock_fit_transform_fail)

        with pytest.raises(RuntimeError) as excinfo:
            apply_pacmap(df, clustering_params)

        assert "Input contains NaN, infinity or a value too large for dtype('float64')" in str(excinfo.value)
        assert "An error occurred during PaCMAP execution." in caplog.text


    def test_apply_pacmap_small_number_of_features(self, mock_hdbscan, mock_pacmap):
        """
        GIVEN: A DataFrame with a small number of features.
        WHEN: apply_pacmap is called.
        THEN: It should return a valid embedding without issues.
        """
        df = pd.DataFrame(np.random.rand(10, 3)) # 3 features
        clustering_params = {'pacmap_n_components': 2, 'pacmap_mn_ratio': 0.5,
                             'pacmap_fp_ratio': 2.0, 'random_state': 42}
        
        mock_pacmap.PaCMAP.return_value.fit_transform.return_value = np.random.rand(10, 2)
        
        embedding = apply_pacmap(df, clustering_params)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (10, 2)


    # --- Tests for perform_hdbscan_clustering ---
    
    def test_perform_hdbscan_clustering_success(self, mock_hdbscan, mock_pacmap, caplog):
        """
        GIVEN: A valid embedding and DataFrame, with valid parameters.
        WHEN: perform_hdbscan_clustering is called.
        THEN: It should return the labeled DataFrame, cluster counts, and labels.
        """

        embedding = np.random.rand(100, 2)
        df = pd.DataFrame(np.random.rand(100, 50))
        clustering_params = {'hdbscan_min_cluster_size': 10, 'hdbscan_min_samples': 5, 'hdbscan_epsilon': 0.5}
        
        mock_hdbscan.HDBSCAN.return_value.fit_predict.return_value = np.array([0, 0, 1, 1, -1] * 20)
        
        df_with_labels, num_clusters, num_noise, cluster_labels = perform_hdbscan_clustering(
            embedding, df, clustering_params
        )
        
        assert isinstance(df_with_labels, pd.DataFrame)
        assert 'cluster_label' in df_with_labels.columns
        assert num_clusters == 2
        assert num_noise == 20
        assert cluster_labels.shape == (100,)
        mock_hdbscan.HDBSCAN.assert_called_once()
        assert "Clustering complete. Found 2 clusters and 20 noise points." in caplog.text
        
        
    def test_perform_hdbscan_clustering_empty_embedding_input(self, mock_hdbscan, mock_pacmap, caplog):
        """
        GIVEN: An empty embedding array.
        WHEN: perform_hdbscan_clustering is called.
        THEN: It should return empty data structures and log a warning.
        """

        embedding = np.array([])
        df = pd.DataFrame(np.random.rand(100, 50))
        clustering_params = {'hdbscan_min_cluster_size': 10, 'hdbscan_min_samples': 5, 'hdbscan_epsilon': 0.5}
        
        df_with_labels, num_clusters, num_noise, cluster_labels = perform_hdbscan_clustering(
            embedding, df, clustering_params
        )
        
        assert df_with_labels.empty
        assert num_clusters == 0
        assert num_noise == 0
        assert cluster_labels.size == 0
        assert "Input data for HDBSCAN is empty. Returning empty data structures." in caplog.text
        mock_hdbscan.HDBSCAN.assert_not_called()
        

    def test_perform_hdbscan_clustering_empty_dataframe_input(self, mock_hdbscan, mock_pacmap, caplog):
        """
        GIVEN: An empty DataFrame input.
        WHEN: perform_hdbscan_clustering is called.
        THEN: It should return empty data structures and log a warning.
        """
  
        embedding = np.random.rand(100, 2)
        df = pd.DataFrame()
        clustering_params = {'hdbscan_min_cluster_size': 10, 'hdbscan_min_samples': 5, 'hdbscan_epsilon': 0.5}
        
        df_with_labels, num_clusters, num_noise, cluster_labels = perform_hdbscan_clustering(
            embedding, df, clustering_params
        )

        assert df_with_labels.empty
        assert num_clusters == 0
        assert num_noise == 0
        assert cluster_labels.size == 0
        assert "Input data for HDBSCAN is empty. Returning empty data structures." in caplog.text
        mock_hdbscan.HDBSCAN.assert_not_called()
        
        
    def test_perform_hdbscan_clustering_non_ndarray_embedding(self, mock_hdbscan, mock_pacmap):
        """
        GIVEN: A non-NumPy array embedding input.
        WHEN: perform_hdbscan_clustering is called.
        THEN: It should raise a TypeError.
        """

        non_ndarray_embedding = [1, 2, 3]
        df = pd.DataFrame(np.random.rand(100, 50))
        clustering_params = {'hdbscan_min_cluster_size': 10, 'hdbscan_min_samples': 5, 'hdbscan_epsilon': 0.5}
        
        with pytest.raises(TypeError) as excinfo:
            perform_hdbscan_clustering(non_ndarray_embedding, df, clustering_params)

        assert "Inputs must be a NumPy array and a Pandas DataFrame." in str(excinfo.value)

        

    def test_perform_hdbscan_clustering_missing_params(self, mock_hdbscan, mock_pacmap):
        """
        GIVEN: A clustering_params dict with a missing key.
        WHEN: perform_hdbscan_clustering is called.
        THEN: It should raise a ValueError.
        """

        embedding = np.random.rand(100, 2)
        df = pd.DataFrame(np.random.rand(100, 50))
        clustering_params = {'hdbscan_min_cluster_size': 10, 'hdbscan_min_samples': 5} # Missing hdbscan_epsilon

        with pytest.raises(ValueError) as excinfo:
            perform_hdbscan_clustering(embedding, df, clustering_params)
        assert "Missing required parameters for HDBSCAN." in str(excinfo.value)
        
        
    def test_perform_hdbscan_clustering_runtime_error(self, mock_hdbscan, mock_pacmap, caplog):
        """
        GIVEN: A scenario where the HDBSCAN algorithm raises an internal error.
        WHEN: perform_hdbscan_clustering is called.
        THEN: It should raise a RuntimeError and log the exception.
        """

        embedding = np.random.rand(100, 2)
        df = pd.DataFrame(np.random.rand(100, 50))
        clustering_params = {'hdbscan_min_cluster_size': 10, 'hdbscan_min_samples': 5, 'hdbscan_epsilon': 0.5}
        
        mock_hdbscan.HDBSCAN.return_value.fit_predict.side_effect = RuntimeError("HDBSCAN failed")

        with pytest.raises(RuntimeError) as excinfo:
            perform_hdbscan_clustering(embedding, df, clustering_params)
        assert "HDBSCAN execution failed: HDBSCAN failed" in str(excinfo.value)
        assert "An error occurred during HDBSCAN execution." in caplog.text
        

    def test_perform_hdbscan_clustering_no_clusters_found(self, mock_hdbscan, mock_pacmap, caplog):
        """
        GIVEN: An embedding that results in no clusters (all noise).
        WHEN: perform_hdbscan_clustering is called.
        THEN: It should report 0 clusters and a log message.
        """

        embedding = np.random.rand(10, 2)
        df = pd.DataFrame(np.random.rand(10, 2))
        clustering_params = {'hdbscan_min_cluster_size': 10, 'hdbscan_min_samples': 5, 'hdbscan_epsilon': 0.5}
        
        mock_hdbscan.HDBSCAN.return_value.fit_predict.return_value = np.full(10, -1)
        
        df_with_labels, num_clusters, num_noise, cluster_labels = perform_hdbscan_clustering(
            embedding, df, clustering_params
        )

        assert num_clusters == 0
        assert num_noise == 10
        assert "Clustering complete. Found 0 clusters and 10 noise points." in caplog.text


    def test_perform_hdbscan_clustering_single_cluster(self, mock_hdbscan, mock_pacmap, caplog):
        """
        GIVEN: An embedding that results in a single cluster.
        WHEN: perform_hdbscan_clustering is called.
        THEN: It should report 1 cluster.
        """

        embedding = np.random.rand(100, 2)
        df = pd.DataFrame(np.random.rand(100, 50))
        clustering_params = {'hdbscan_min_cluster_size': 10, 'hdbscan_min_samples': 5, 'hdbscan_epsilon': 0.5}
        
        mock_hdbscan.HDBSCAN.return_value.fit_predict.return_value = np.full(100, 0)
        
        df_with_labels, num_clusters, num_noise, cluster_labels = perform_hdbscan_clustering(
            embedding, df, clustering_params
        )

        assert num_clusters == 1
        assert num_noise == 0
        assert "Clustering complete. Found 1 clusters and 0 noise points." in caplog.text


    def test_perform_hdbscan_clustering_mismatched_shapes(self, mock_hdbscan, mock_pacmap):
        """
        GIVEN: An embedding and DataFrame with a mismatched number of rows.
        WHEN: perform_hdbscan_clustering is called.
        THEN: It should raise a RuntimeError.
        """
        embedding = np.random.rand(100, 2) 
        df = pd.DataFrame(np.random.rand(99, 50)) 
        clustering_params = {'hdbscan_min_cluster_size': 10, 'hdbscan_min_samples': 5, 'hdbscan_epsilon': 0.5}
        
        mock_hdbscan.HDBSCAN.return_value.fit_predict.return_value = np.array([0] * 100) 
        
        with pytest.raises(RuntimeError) as excinfo:
            perform_hdbscan_clustering(embedding, df, clustering_params)
        
        assert "HDBSCAN execution failed" in str(excinfo.value)
        assert "Length of values (100) does not match length of index (99)" in str(excinfo.value)
    
    
    def test_perform_hdbscan_clustering_large_min_cluster_size(self, mock_hdbscan, mock_pacmap, caplog):
        """
        GIVEN: HDBSCAN parameters that are too large for the dataset.
        WHEN: perform_hdbscan_clustering is called.
        THEN: It should result in all points being classified as noise (-1).
        """
        embedding = np.random.rand(10, 2)
        df = pd.DataFrame(np.random.rand(10, 50))
        clustering_params = {'hdbscan_min_cluster_size': 11, 'hdbscan_min_samples': 5, 'hdbscan_epsilon': 0.5} 

        mock_hdbscan.HDBSCAN.return_value.fit_predict.return_value = np.full(10, -1)
        
        df_with_labels, num_clusters, num_noise, cluster_labels = perform_hdbscan_clustering(
            embedding, df, clustering_params
        )

        assert num_clusters == 0
        assert num_noise == 10
        assert "Clustering complete. Found 0 clusters and 10 noise points." in caplog.text
        

    def test_perform_hdbscan_clustering_df_already_has_label_column(self, mock_hdbscan, mock_pacmap):
        """
        GIVEN: A DataFrame that already has a 'cluster_label' column.
        WHEN: perform_hdbscan_clustering is called.
        THEN: The old column should be overwritten with new labels.
        """
        embedding = np.random.rand(10, 2)
        df = pd.DataFrame(np.random.rand(10, 50))
        df['cluster_label'] = np.full(10, 99) 
        clustering_params = {'hdbscan_min_cluster_size': 2, 'hdbscan_min_samples': 2, 'hdbscan_epsilon': 0.5}
        
        new_labels = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, -1])
        mock_hdbscan.HDBSCAN.return_value.fit_predict.return_value = new_labels
        
        df_with_labels, _, _, _ = perform_hdbscan_clustering(embedding, df, clustering_params)

        assert 'cluster_label' in df_with_labels.columns
        assert np.array_equal(df_with_labels['cluster_label'].values, new_labels)
        assert not np.any(df_with_labels['cluster_label'].values == 99)