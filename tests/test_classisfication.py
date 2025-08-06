"""
Test suite for the classification module (src/classification.py).

This module contains unit tests for all functions within the `classification.py` file,
ensuring their correctness, robustness, and proper error/warning handling.

Functions:
    - `setup_test_logging`: A pytest fixture for configuring logging capture.
    - `dummy_model_path`: A pytest fixture for creating a dummy machine learning model file.
    - `TestClassification` (class): A test class that groups related tests for all
  functions in the `classification` module, utilizing mocking for robust and isolated testing.

Typical use:
    This module is designed to be executed as part of the project's automated test suite
    using `pytest`. It ensures the reliability and integrity of the classification components,
    which are crucial for the final wound border type prediction.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report
from typing import Optional, Dict, Any, Tuple, List
import logging
import sys
from unittest.mock import patch, Mock
import joblib 
from pathlib import Path

# Import functions from src/classification.py
from src.classification import (
    label_clustered_data,
    prepare_training_data_splits,
    train_classifier,
    evaluate_classifier,
    get_feature_importances,
    predict_wound_border_type 
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
    
    caplog.set_level(logging.DEBUG, logger="src.classification")

@pytest.fixture
def dummy_model_path(tmp_path) -> Path:
    """Creates a dummy joblib model file for testing."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "dummy_model.joblib"
    
    # Create a simple dummy model and save it
    dummy_model = RandomForestClassifier(n_estimators=1, random_state=42)
    X = np.array([[1,2],[3,4],[5,6],[7,8]])
    y = np.array([0,1,0,1])
    dummy_model.fit(X,y)
    joblib.dump(dummy_model, model_path)
    return model_path

# --- Test Class for Classification Functions ---
class TestClassification:

    # --- Tests for label_clustered_data ---

    def test_label_clustered_data_success(self):
        """
        GIVEN: Valid features_df, cluster_map_df, and label_map.
        WHEN: label_clustered_data is called.
        THEN: It should merge data and apply descriptive labels correctly.
        """
        features_df = pd.DataFrame({
            'image_id': ['A', 'B', 'C', 'D'],
            'feat1': [10, 20, 30, 40],
            'feat2': [1, 2, 3, 4]
        })
        cluster_map_df = pd.DataFrame({
            'image_id': ['A', 'B', 'C', 'D'],
            'cluster_label': [0, 1, 0, -1]
        })
        label_map = {0: 'Healthy', 1: 'Inflamed', -1: 'Outlier'}

        expected_df = pd.DataFrame({
            'image_id': ['A', 'B', 'C', 'D'],
            'feat1': [10, 20, 30, 40],
            'feat2': [1, 2, 3, 4],
            'cluster_label': [0, 1, 0, -1],
            'wound_type': ['Healthy', 'Inflamed', 'Healthy', 'Outlier']
        })

        result_df = label_clustered_data(features_df, cluster_map_df, label_map)
        pd.testing.assert_frame_equal(result_df, expected_df)
        assert 'wound_type' in result_df.columns

    def test_label_clustered_data_empty_features_df(self, caplog):
        """
        GIVEN: An empty features_df.
        WHEN: label_clustered_data is called.
        THEN: It should return an empty DataFrame and log a warning.
        """
        features_df = pd.DataFrame(columns=['image_id', 'feat1'])
        cluster_map_df = pd.DataFrame({'image_id': ['A'], 'cluster_label': [0]})
        label_map = {0: 'Healthy'}

        result_df = label_clustered_data(features_df, cluster_map_df, label_map)
        assert result_df.empty
        assert "Input DataFrame is empty. Returning empty DataFrame." in caplog.text

    def test_label_clustered_data_empty_cluster_map_df(self, caplog):
        """
        GIVEN: An empty cluster_map_df.
        WHEN: label_clustered_data is called.
        THEN: It should return an empty DataFrame and log a warning.
        """
        features_df = pd.DataFrame({'image_id': ['A'], 'feat1': [1]})
        cluster_map_df = pd.DataFrame(columns=['image_id', 'cluster_label'])
        label_map = {0: 'Healthy'}

        result_df = label_clustered_data(features_df, cluster_map_df, label_map)
        assert result_df.empty
        assert "Input DataFrame is empty. Returning empty DataFrame." in caplog.text

    def test_label_clustered_data_missing_image_id_features_df(self, caplog):
        """
        GIVEN: features_df missing 'image_id' column.
        WHEN: label_clustered_data is called.
        THEN: It should raise a KeyError.
        """
        features_df = pd.DataFrame({'feat1': [10]})
        cluster_map_df = pd.DataFrame({'image_id': ['A'], 'cluster_label': [0]})
        label_map = {0: 'Healthy'}

        with pytest.raises(KeyError) as excinfo:
            label_clustered_data(features_df, cluster_map_df, label_map)
        assert "Missing column for merge: 'image_id'" in str(excinfo.value)
        assert "Missing column 'image_id' for merge operation." in caplog.text

    def test_label_clustered_data_missing_cluster_label_cluster_map_df(self, caplog):
        """
        GIVEN: cluster_map_df missing 'cluster_label' column.
        WHEN: label_clustered_data is called.
        THEN: It should raise a KeyError (from .map operation).
        """
        features_df = pd.DataFrame({'image_id': ['A'], 'feat1': [10]})
        cluster_map_df = pd.DataFrame({'image_id': ['A']}) 
        label_map = {0: 'Healthy'}

        with pytest.raises(KeyError) as excinfo:
            label_clustered_data(features_df, cluster_map_df, label_map)
        assert "'cluster_label'" in str(excinfo.value) 

    def test_label_clustered_data_empty_label_map(self, caplog):
        """
        GIVEN: An empty label_map.
        WHEN: label_clustered_data is called.
        THEN: It should raise a ValueError.
        """
        features_df = pd.DataFrame({'image_id': ['A'], 'feat1': [10]})
        cluster_map_df = pd.DataFrame({'image_id': ['A'], 'cluster_label': [0]})
        label_map = {}

        with pytest.raises(ValueError) as excinfo:
            label_clustered_data(features_df, cluster_map_df, label_map)
        assert "A non-empty label map dictionary is required." in str(excinfo.value)

    def test_label_clustered_data_non_dataframe_input(self):
        """
        GIVEN: Non-DataFrame inputs.
        WHEN: label_clustered_data is called.
        THEN: It should raise a TypeError.
        """
        with pytest.raises(TypeError) as excinfo:
            label_clustered_data([1,2], pd.DataFrame(), {})
        assert "Inputs must be Pandas DataFrames." in str(excinfo.value)

    # --- Tests for prepare_training_data_splits ---

    @patch('src.classification.train_test_split')
    def test_prepare_training_data_splits_success(self, mock_train_test_split):
        """
        GIVEN: A valid DataFrame with 'wound_type' and classification parameters.
        WHEN: prepare_training_data_splits is called.
        THEN: It should split data correctly and return arrays and df.
        """
        df_mock = pd.DataFrame({
            'image_id': ['A', 'B', 'C', 'D', 'E', 'F'],
            'feat1': [1, 2, 3, 4, 5, 6],
            'feat2': [10, 20, 30, 40, 50, 60],
            'cluster_label': [0, 1, 0, 1, 0, -1],
            'wound_type': ['TypeA', 'TypeB', 'TypeA', 'TypeB', 'TypeA', 'Outlier']
        })
        classification_params = {'test_size': 0.5, 'random_state': 42}

        X_train_df_mock_return = pd.DataFrame({'feat1': [1,3,5], 'feat2': [10,30,50]})
        X_test_df_mock_return = pd.DataFrame({'feat1': [2,4], 'feat2': [20,40]})
        y_train_df_mock_return = pd.Series(['TypeA', 'TypeA', 'TypeA'])
        y_test_df_mock_return = pd.Series(['TypeB', 'TypeB'])

        mock_train_test_split.return_value = (
            X_train_df_mock_return,
            X_test_df_mock_return,
            y_train_df_mock_return,
            y_test_df_mock_return,
        )
        X_train, X_test, y_train, y_test, X_train_df = prepare_training_data_splits(df_mock, classification_params)

        assert X_train.shape == (3, 2)
        assert X_test.shape == (2, 2)
        assert y_train.shape == (3,)
        assert y_test.shape == (2,)
        assert isinstance(X_train_df, pd.DataFrame)
        assert 'image_id' not in X_train_df.columns
        assert 'cluster_label' not in X_train_df.columns
        assert 'wound_type' not in X_train_df.columns
        
        assert 'Outlier' not in y_train and 'Outlier' not in y_test
        mock_train_test_split.assert_called_once()
        assert mock_train_test_split.call_args[1]['stratify'] is not None


    def test_prepare_training_data_splits_empty_df_input(self, caplog):
        """
        GIVEN: An empty DataFrame input.
        WHEN: prepare_training_data_splits is called.
        THEN: It should raise a ValueError and log a warning.
        """
        df_empty = pd.DataFrame(columns=['image_id', 'feat1', 'wound_type', 'cluster_label']) 
        classification_params = {'test_size': 0.5, 'random_state': 42}

        with pytest.raises(ValueError) as excinfo:
            prepare_training_data_splits(df_empty, classification_params)
        assert "Input DataFrame is empty." in str(excinfo.value)
        assert "Input DataFrame is empty. Cannot split data." in caplog.text

    def test_prepare_training_data_splits_missing_wound_type_column(self, caplog):
        """
        GIVEN: DataFrame missing 'wound_type' column.
        WHEN: prepare_training_data_splits is called.
        THEN: It should raise a ValueError.
        """
        df_no_wound_type = pd.DataFrame({'image_id': ['A'], 'feat1': [1], 'cluster_label': [0]}) 
        classification_params = {'test_size': 0.5, 'random_state': 42}

        with pytest.raises(ValueError) as excinfo:
            prepare_training_data_splits(df_no_wound_type, classification_params)
        assert "DataFrame is missing the 'wound_type' column." in str(excinfo.value)
        assert "DataFrame must contain 'wound_type' column for splitting." in caplog.text

    def test_prepare_training_data_splits_no_valid_rows_after_filtering(self, caplog):
        """
        GIVEN: DataFrame where all rows are 'Outlier' or become empty after dropna.
        WHEN: prepare_training_data_splits is called.
        THEN: It should return empty arrays and log a warning.
        """
        df_all_outlier = pd.DataFrame({
            'image_id': ['A', 'B'],
            'feat1': [1, 2],
            'cluster_label': [-1, -1], 
            'wound_type': ['Outlier', 'Outlier']
        })
        classification_params = {'test_size': 0.5, 'random_state': 42}

        X_train, X_test, y_train, y_test, X_train_df = prepare_training_data_splits(df_all_outlier, classification_params)
        
        assert X_train.size == 0
        assert X_test.size == 0
        assert y_train.size == 0
        assert y_test.size == 0
        assert X_train_df.empty
        assert "No valid rows remaining after removing NaNs for training." in caplog.text

    def test_prepare_training_data_splits_non_dataframe_input(self):
        """
        GIVEN: Non-DataFrame input.
        WHEN: prepare_training_data_splits is called.
        THEN: It should raise a TypeError.
        """
        with pytest.raises(TypeError) as excinfo:
            prepare_training_data_splits([1,2], {'test_size': 0.5, 'random_state': 42})
        assert "Input 'df' must be a Pandas DataFrame." in str(excinfo.value)

    def test_prepare_training_data_splits_single_row(self, caplog):
        """
        GIVEN: A DataFrame with a single valid row.
        WHEN: prepare_training_data_splits is called.
        THEN: It should handle the split gracefully (though split might not be possible).
        """
        df_single_row = pd.DataFrame({
            'image_id': ['A'],
            'feat1': [10],
            'cluster_label': [0],
            'wound_type': ['TypeA']
        })
        classification_params = {'test_size': 0.5, 'random_state': 42}

        with pytest.raises(ValueError) as excinfo:
            X_train, X_test, y_train, y_test, X_train_df = prepare_training_data_splits(df_single_row, classification_params)
        
        assert "resulting train set will be empty" in str(excinfo.value)


    # --- Tests for train_classifier ---

    def test_train_classifier_success(self, tmp_path): 
        """
        GIVEN: Valid training data and classification parameters.
        WHEN: train_classifier is called.
        THEN: It should return a trained RandomForestClassifier model.
        """
        X_train = np.random.rand(10, 5)
        y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        classification_params = {'random_state': 42, 'random_forest_n_estimators': 10}
        
        dummy_save_path = tmp_path / "dummy_model.joblib"

        model = train_classifier(X_train, y_train, classification_params, dummy_save_path) 
        
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 10
        assert model.random_state == 42
        assert hasattr(model, 'estimators_') 

    def test_train_classifier_empty_input(self, caplog, tmp_path): 
        """
        GIVEN: Empty training data.
        WHEN: train_classifier is called.
        THEN: It should raise a ValueError.
        """
        X_train = np.array([])
        y_train = np.array([])
        classification_params = {'random_state': 42, 'random_forest_n_estimators': 10}
        dummy_save_path = tmp_path / "dummy_model.joblib" 

        with pytest.raises(ValueError) as excinfo:

            train_classifier(X_train, y_train, classification_params, dummy_save_path) 
        
        assert "Input arrays are empty or have inconsistent shapes." in str(excinfo.value)
        assert "Input arrays are empty or have inconsistent shapes." in caplog.text

    def test_train_classifier_inconsistent_shapes(self, caplog, tmp_path): 
        """
        GIVEN: Training data with inconsistent shapes.
        WHEN: train_classifier is called.
        THEN: It should raise a ValueError.
        """
        X_train = np.random.rand(10, 5)
        y_train = np.array([0, 1])
        classification_params = {'random_state': 42, 'random_forest_n_estimators': 10}
        dummy_save_path = tmp_path / "dummy_model.joblib" 

        with pytest.raises(ValueError) as excinfo:
            train_classifier(X_train, y_train, classification_params, dummy_save_path)
        
        assert "Input arrays are empty or have inconsistent shapes." in str(excinfo.value)
        assert "Input arrays are empty or have inconsistent shapes." in caplog.text

    def test_train_classifier_non_numpy_input(self, tmp_path):
        """
        GIVEN: Non-NumPy array inputs.
        WHEN: train_classifier is called.
        THEN: It should raise a TypeError.
        """
        classification_params = {'random_state': 42, 'random_forest_n_estimators': 10}
        dummy_save_path = tmp_path / "dummy_model.joblib"

        with pytest.raises(TypeError) as excinfo:
            train_classifier([1,2], np.array([0]), classification_params, dummy_save_path) 
        assert "Inputs must be NumPy arrays." in str(excinfo.value)

    @patch('src.classification.RandomForestClassifier')
    def test_train_classifier_runtime_error(self, MockRFClassifier, caplog, tmp_path): 
        """
        GIVEN: RandomForestClassifier.fit raises an error during training.
        WHEN: train_classifier is called.
        THEN: It should raise a RuntimeError and log the exception.
        """
        X_train = np.random.rand(10, 5)
        y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        classification_params = {'random_state': 42, 'random_forest_n_estimators': 10}
        
        dummy_save_path = tmp_path / "dummy_model.joblib"
        MockRFClassifier.return_value.fit.side_effect = RuntimeError("Mock training failure")
        
        with pytest.raises(RuntimeError) as excinfo:
            train_classifier(X_train, y_train, classification_params, dummy_save_path) 
        
        assert "Model training failed: Mock training failure" in str(excinfo.value)
        assert "An error occurred during model training." in caplog.text

    # --- Tests for evaluate_classifier ---

    def test_evaluate_classifier_success(self):
        """
        GIVEN: A trained model and valid test data.
        WHEN: evaluate_classifier is called.
        THEN: It should return a dictionary with accuracy and classification report.
        """
        X_test = np.array([[1,2], [3,4], [5,6], [7,8]])
        y_test = np.array([0, 1, 0, 1])
        
        mock_model = Mock(spec=RandomForestClassifier)
        mock_model.predict.return_value = np.array([0, 1, 0, 1])

        results = evaluate_classifier(mock_model, X_test, y_test)
        
        assert isinstance(results, dict)
        assert results['accuracy'] == 1.0
        assert 'classification_report' in results
        assert results['classification_report']['accuracy'] == 1.0
        assert results['model'] == mock_model 


    def test_evaluate_classifier_empty_input(self, caplog):
        """
        GIVEN: Empty test data.
        WHEN: evaluate_classifier is called.
        THEN: It should raise a ValueError.
        """
        mock_model = Mock(spec=RandomForestClassifier)
        X_test = np.array([])
        y_test = np.array([])

        with pytest.raises(ValueError) as excinfo:
            evaluate_classifier(mock_model, X_test, y_test)
        
        assert "Input arrays must be non-empty and have consistent shapes." in str(excinfo.value)
        assert "Input arrays must be non-empty and have consistent shapes." in caplog.text


    def test_evaluate_classifier_inconsistent_shapes(self, caplog):
        """
        GIVEN: Test data with inconsistent shapes.
        WHEN: evaluate_classifier is called.
        THEN: It should raise a ValueError.
        """
        mock_model = Mock(spec=RandomForestClassifier)
        X_test = np.random.rand(10, 5)
        y_test = np.array([0, 1]) 
        
        with pytest.raises(ValueError) as excinfo:
            evaluate_classifier(mock_model, X_test, y_test)
        
        assert "Input arrays must be non-empty and have consistent shapes." in str(excinfo.value)
        assert "Input arrays must be non-empty and have consistent shapes." in caplog.text


    def test_evaluate_classifier_non_numpy_input(self):
        """
        GIVEN: Non-NumPy array inputs for X_test or y_test.
        WHEN: evaluate_classifier is called.
        THEN: It should raise a TypeError.
        """
        mock_model = Mock(spec=RandomForestClassifier)
        with pytest.raises(TypeError) as excinfo:
            evaluate_classifier(mock_model, [1,2], np.array([0]))
        assert "Inputs must be NumPy arrays." in str(excinfo.value)


    def test_evaluate_classifier_non_rf_model_input(self):
        """
        GIVEN: A non-RandomForestClassifier model.
        WHEN: evaluate_classifier is called.
        THEN: It should raise a TypeError.
        """
        mock_model = Mock() 
        X_test = np.random.rand(5,2)
        y_test = np.array([0,1,0,1,0])
        with pytest.raises(TypeError) as excinfo:
            evaluate_classifier(mock_model, X_test, y_test)
        assert "Input 'model' must be a trained RandomForestClassifier." in str(excinfo.value)


    @patch('src.classification.classification_report')
    def test_evaluate_classifier_runtime_error(self, mock_class_report, caplog):
        """
        GIVEN: An error occurs during prediction or report generation.
        WHEN: evaluate_classifier is called.
        THEN: It should raise a RuntimeError and log the exception.
        """
        mock_model = Mock(spec=RandomForestClassifier)
        mock_model.predict.side_effect = RuntimeError("Mock prediction failure")
        X_test = np.random.rand(10, 5)
        y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with pytest.raises(RuntimeError) as excinfo:
            evaluate_classifier(mock_model, X_test, y_test)
        assert "Model prediction or evaluation failed: Mock prediction failure" in str(excinfo.value)
        assert "An error occurred during model prediction or evaluation." in caplog.text

    # --- Tests for get_feature_importances ---

    def test_get_feature_importances_success(self):
        """
        GIVEN: A trained model with feature importances and a valid X_train_df.
        WHEN: get_feature_importances is called.
        THEN: It should return a DataFrame of feature importances.
        """
        X_train_df = pd.DataFrame(np.random.rand(10, 3), columns=['featA', 'featB', 'featC'])
        mock_model = Mock(spec=RandomForestClassifier)
        mock_model.feature_importances_ = np.array([0.3, 0.6, 0.1]) 
        
        result_df = get_feature_importances(mock_model, X_train_df)
        
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.columns.tolist() == ['feature', 'importance']
        assert result_df['feature'].tolist() == ['featB', 'featA', 'featC'] 
        assert result_df['importance'].tolist() == [0.6, 0.3, 0.1]


    def test_get_feature_importances_empty_df(self, caplog):
        """
        GIVEN: An empty X_train_df.
        WHEN: get_feature_importances is called.
        THEN: It should return None and log a warning.
        """
        mock_model = Mock(spec=RandomForestClassifier)
        mock_model.feature_importances_ = np.array([0.1, 0.2]) 
        X_train_df = pd.DataFrame()

        result = get_feature_importances(mock_model, X_train_df)
        assert result is None
        assert "Input DataFrame is empty. Skipping feature importance calculation." in caplog.text


    def test_get_feature_importances_model_no_attribute(self, caplog):
        """
        GIVEN: A model without 'feature_importances_' attribute.
        WHEN: get_feature_importances is called.
        THEN: It should return None and log a warning.
        """
        mock_model = Mock(spec=RandomForestClassifier) 
        del mock_model.feature_importances_ 
        X_train_df = pd.DataFrame(np.random.rand(10, 2))

        result = get_feature_importances(mock_model, X_train_df)
        assert result is None
        assert "Model does not have 'feature_importances_' attribute. Skipping feature importance calculation." in caplog.text


    def test_get_feature_importances_non_rf_model_input(self):
        """
        GIVEN: A non-RandomForestClassifier model.
        WHEN: get_feature_importances is called.
        THEN: It should raise a TypeError.
        """
        mock_model = Mock() 
        X_train_df = pd.DataFrame(np.random.rand(10, 2))

        with pytest.raises(TypeError) as excinfo:
            get_feature_importances(mock_model, X_train_df)

        assert "Input 'model' must be a trained RandomForestClassifier." in str(excinfo.value)


    def test_get_feature_importances_non_dataframe_input(self):
        """
        GIVEN: Non-DataFrame input for X_train_df.
        WHEN: get_feature_importances is called.
        THEN: It should raise a TypeError.
        """
        mock_model = Mock(spec=RandomForestClassifier)
        mock_model.feature_importances_ = np.array([0.1, 0.2])

        with pytest.raises(TypeError) as excinfo:
            get_feature_importances(mock_model, [1,2,3])

        assert "Input 'X_train_df' must be a Pandas DataFrame." in str(excinfo.value)


    def test_get_feature_importances_runtime_error(self, caplog, monkeypatch):
        """
        GIVEN: An error occurs during DataFrame creation or sorting within get_feature_importances.
        WHEN: get_feature_importances is called.
        THEN: It should raise a RuntimeError and log the exception.
        """
        mock_model = Mock(spec=RandomForestClassifier)
        mock_model.feature_importances_ = np.array([0.3, 0.6, 0.1])
        X_train_df = pd.DataFrame(np.random.rand(10, 3), columns=['featA', 'featB', 'featC'])
        
        monkeypatch.setattr(pd.DataFrame, 'sort_values', Mock(side_effect=RuntimeError("Mock sorting failure")))

        with pytest.raises(RuntimeError) as excinfo:
            get_feature_importances(mock_model, X_train_df)
        
        assert "Feature importance calculation failed: Mock sorting failure" in str(excinfo.value)
        assert "An error occurred during feature importance calculation." in caplog.text


    # --- Tests for predict_wound_border_type ---

    @patch('src.classification.joblib')
    def test_predict_wound_border_type_success(self, mock_joblib, dummy_model_path):
        """
        GIVEN: A valid model path and new features DataFrame.
        WHEN: predict_wound_border_type is called.
        THEN: It should return a DataFrame with predictions and probabilities.
        """

        mock_model = Mock(spec=RandomForestClassifier)
        mock_model.predict.return_value = np.array([0, 1])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])
        mock_joblib.load.return_value = mock_model

        new_features_df = pd.DataFrame({
            'image_id': ['img_001', 'img_002'],
            'feat1': [1.0, 2.0],
            'feat2': [3.0, 4.0]
        })

        expected_df = pd.DataFrame({
            'image_id': ['img_001', 'img_002'],
            'predicted_label': [0, 1],
            'probability': [0.9, 0.8]
        })

        result_df = predict_wound_border_type(dummy_model_path, new_features_df)
        
        pd.testing.assert_frame_equal(result_df, expected_df)
        mock_joblib.load.assert_called_once_with(str(dummy_model_path))
        mock_model.predict.assert_called_once()
        mock_model.predict_proba.assert_called_once()


    def test_predict_wound_border_type_empty_features_df(self, dummy_model_path, caplog):
        """
        GIVEN: An empty new_features_df.
        WHEN: predict_wound_border_type is called.
        THEN: It should return an empty DataFrame and log a warning.
        """
        new_features_df = pd.DataFrame(columns=['image_id', 'feat1', 'feat2'])
        result_df = predict_wound_border_type(dummy_model_path, new_features_df)
        
        assert result_df.empty
        assert result_df.columns.tolist() == ['image_id', 'predicted_label', 'probability']
        
        assert "Input features DataFrame is empty. Returning empty prediction DataFrame." in caplog.text
        assert "Prediction complete." not in caplog.text


    def test_predict_wound_border_type_model_file_not_found(self, tmp_path, caplog):
        """
        GIVEN: A non-existent model_path.
        WHEN: predict_wound_border_type is called.
        THEN: It should raise a FileNotFoundError.
        """
        non_existent_model_path = tmp_path / "non_existent_model.joblib"
        new_features_df = pd.DataFrame({'image_id': ['img1'], 'feat1': [1.0]})

        with pytest.raises(FileNotFoundError) as excinfo:
            predict_wound_border_type(non_existent_model_path, new_features_df)
        assert f"Model file not found at: {non_existent_model_path}" in str(excinfo.value)
        assert f"Model file not found at: {non_existent_model_path}" in caplog.text


    def test_predict_wound_border_type_non_path_model_path(self):
        """
        GIVEN: A non-Path object for model_path.
        WHEN: predict_wound_border_type is called.
        THEN: It should raise a TypeError.
        """
        new_features_df = pd.DataFrame({'image_id': ['img1'], 'feat1': [1.0]})
        with pytest.raises(TypeError) as excinfo:
            predict_wound_border_type("invalid/path.joblib", new_features_df)

        assert "Input 'model_path' must be a Path object." in str(excinfo.value)


    def test_predict_wound_border_type_non_dataframe_features_df(self, dummy_model_path):
        """
        GIVEN: A non-DataFrame object for new_features_df.
        WHEN: predict_wound_border_type is called.
        THEN: It should raise a TypeError.
        """
        with pytest.raises(TypeError) as excinfo:
            predict_wound_border_type(dummy_model_path, [1,2,3])

        assert "Input 'new_features_df' must be a Pandas DataFrame." in str(excinfo.value)


    def test_predict_wound_border_type_missing_image_id_column(self, dummy_model_path, caplog):
        """
        GIVEN: new_features_df missing 'image_id' column.
        WHEN: predict_wound_border_type is called.
        THEN: It should raise a RuntimeError (wrapping the ValueError) and log the error.
        """
        new_features_df = pd.DataFrame({'feat1': [1.0, 2.0]}) 
        
        with pytest.raises(RuntimeError) as excinfo: 
            predict_wound_border_type(dummy_model_path, new_features_df)
        
        assert "Prediction failed: Features DataFrame is missing the 'image_id' column." in str(excinfo.value)
        assert "Features DataFrame is missing the 'image_id' column." in caplog.text


    @patch('src.classification.joblib')
    def test_predict_wound_border_type_joblib_load_failure(self, mock_joblib, dummy_model_path, caplog):
        """
        GIVEN: joblib.load raises an error.
        WHEN: predict_wound_border_type is called.
        THEN: It should raise a RuntimeError.
        """
        mock_joblib.load.side_effect = Exception("Mock joblib load error")
        new_features_df = pd.DataFrame({'image_id': ['img1'], 'feat1': [1.0]})

        with pytest.raises(RuntimeError) as excinfo:
            predict_wound_border_type(dummy_model_path, new_features_df)

        assert "Prediction failed: Mock joblib load error" in str(excinfo.value)
        assert "An error occurred during prediction." in caplog.text


    @patch('src.classification.joblib')
    def test_predict_wound_border_type_model_predict_failure(self, mock_joblib, dummy_model_path, caplog):
        """
        GIVEN: The loaded model's predict method fails.
        WHEN: predict_wound_border_type is called.
        THEN: It should raise a RuntimeError.
        """
        mock_model = Mock(spec=RandomForestClassifier)
        mock_model.predict.side_effect = Exception("Mock predict failure")
        mock_joblib.load.return_value = mock_model

        new_features_df = pd.DataFrame({'image_id': ['img1'], 'feat1': [1.0]})

        with pytest.raises(RuntimeError) as excinfo:
            predict_wound_border_type(dummy_model_path, new_features_df)

        assert "Prediction failed: Mock predict failure" in str(excinfo.value)
        assert "An error occurred during prediction." in caplog.text


    @patch('src.classification.joblib')
    def test_predict_wound_border_type_model_predict_proba_failure(self, mock_joblib, dummy_model_path, caplog):
        """
        GIVEN: The loaded model's predict_proba method fails.
        WHEN: predict_wound_border_type is called.
        THEN: It should raise a RuntimeError.
        """
        mock_model = Mock(spec=RandomForestClassifier)
        mock_model.predict.return_value = np.array([0]) 
        mock_model.predict_proba.side_effect = Exception("Mock predict_proba failure")
        mock_joblib.load.return_value = mock_model

        new_features_df = pd.DataFrame({'image_id': ['img1'], 'feat1': [1.0]})

        with pytest.raises(RuntimeError) as excinfo:
            predict_wound_border_type(dummy_model_path, new_features_df)

        assert "Prediction failed: Mock predict_proba failure" in str(excinfo.value)
        assert "An error occurred during prediction." in caplog.text


    def test_predict_wound_border_type_single_feature_column(self, dummy_model_path):
        """
        GIVEN: new_features_df with only one feature column (besides image_id).
        WHEN: predict_wound_border_type is called.
        THEN: It should process correctly.
        """
        mock_model = Mock(spec=RandomForestClassifier)
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
        with patch('src.classification.joblib') as mock_joblib:
            mock_joblib.load.return_value = mock_model

            new_features_df = pd.DataFrame({
                'image_id': ['img_001'],
                'feat1': [1.0]
            })

            expected_df = pd.DataFrame({
                'image_id': ['img_001'],
                'predicted_label': [0],
                'probability': [0.9]
            })

            result_df = predict_wound_border_type(dummy_model_path, new_features_df)
            pd.testing.assert_frame_equal(result_df, expected_df)
            mock_model.predict.assert_called_once_with(np.array([[1.0]])) 