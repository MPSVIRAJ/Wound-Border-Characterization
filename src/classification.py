"""
Supervised Classification and Performance Evaluation.

This module provides a robust pipeline for supervised machine learning, using the
unsupervised clusters identified in the previous stage as ground truth labels.
It includes functions for preparing the data, training a Random Forest classifier,
evaluating its performance, and analyzing the importance of the extracted features.
The module is designed to be fully configurable, with all algorithm parameters
specified in the application's configuration.

Functions:
    - label_clustered_data : Merges features with cluster assignments and assigns descriptive labels.
    - prepare_training_data_splits : Splits the labeled dataset into training and testing sets.
    - train_classifier : Trains and saves a Random Forest classifier model.
    - evaluate_classifier : Evaluates the performance of the trained model.
    - get_feature_importances : Calculates and returns the feature importance scores from the model.
    - predict_wound_border_type : Classifies new wound images using a pre-trained model.

Typical Use:
    This module is typically used in the main application script (`run_pipeline.py`)
    after the clustering stage. The main script orchestrates a sequence of calls to
    these functions to train a classifier on the clustered data and report its
    performance.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from typing import Optional, Dict, Any, Tuple, List
import logging
from pathlib import Path

# Get a logger instance for this module.
logger = logging.getLogger(__name__)

def label_clustered_data(features_df: pd.DataFrame, cluster_map_df: pd.DataFrame, label_map: Dict[int, str]) -> pd.DataFrame:
    """
    Merges feature data with cluster assignments and assigns descriptive wound type labels.

    This function combines the extracted features with the cluster labels identified by HDBSCAN.
    It then uses a provided `label_map` to replace the numerical cluster labels with
    more descriptive strings, preparing the final dataset for supervised classification.

    Args:
        features_df (pd.DataFrame): 
            DataFrame containing extracted features.
            Must include an 'image_id' column.
        cluster_map_df (pd.DataFrame): 
            DataFrame containing image-cluster assignments.
            Must include 'image_id' and 'cluster_label' columns.
        label_map (Dict[int, str]): 
            A dictionary that maps numerical cluster labels to
            descriptive string labels.

    Returns:
        pd.DataFrame: 
            Merged and labeled DataFrame (df_merged).

    Raises:
        TypeError: 
            If inputs are not of the expected DataFrame types.
        ValueError: 
            If required columns ('image_id', 'cluster_label') are missing or if
            the label map is empty.
        KeyError: 
            If a required column is missing from one of the DataFrames.

    Output:
        - Console/Log:
            Informational messages about the merging process and outlier removal.
        - Return Value:
            A Pandas DataFrame ready for supervised learning.

    Examples:
        >>> import pandas as pd
        >>> from src.classification import label_clustered_data
        >>> # Dummy data
        >>> df_features = pd.DataFrame({'image_id': ['A', 'B'], 'feat1': [1, 2]})
        >>> df_clusters = pd.DataFrame({'image_id': ['A', 'B'], 'cluster_label': [0, -1]})
        >>> label_map = {0: 'Type A', -1: 'Outlier'}
        >>> labeled_data = label_clustered_data(df_features, df_clusters, label_map)

    Relationships:
        - Dependencies:
            Relies on `pandas` for DataFrame manipulation.
        - Used by:
            The classification pipeline to prepare the input for model training.
    """
    if not isinstance(features_df, pd.DataFrame) or not isinstance(cluster_map_df, pd.DataFrame):
        logger.error("Inputs must be Pandas DataFrames.")
        raise TypeError("Inputs must be Pandas DataFrames.")
    if features_df.empty or cluster_map_df.empty:
        logger.warning("Input DataFrame is empty. Returning empty DataFrame.")
        return pd.DataFrame()
    if not isinstance(label_map, dict) or not label_map:
        logger.error("A non-empty label map dictionary is required.")
        raise ValueError("A non-empty label map dictionary is required.")

    logger.info("Merging feature data with cluster assignments and applying descriptive labels.")
    
    # Merge on image_id
    try:
        df_merged = pd.merge(features_df, cluster_map_df, on='image_id')
        logger.debug(f"Merged DataFrame shape: {df_merged.shape}.")
    except KeyError as e:
        logger.error(f"Missing column 'image_id' for merge operation. Error: {e}")
        raise KeyError(f"Missing column for merge: {e}") from e
    
    # Apply the descriptive labels using a map
    df_merged['wound_type'] = df_merged['cluster_label'].map(label_map)
    logger.debug("Descriptive labels applied to clusters.")

    return df_merged


def prepare_training_data_splits(
        df: pd.DataFrame, 
        classification_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:

    """
    Prepares and splits the labeled dataset into training and testing sets.

    This function takes the final labeled DataFrame, separates the feature columns (X)
    from the target variable (y), and then uses a stratified split to ensure that the
    distribution of wound types is preserved in both the training and test sets.

    Args:
        df (pd.DataFrame): 
            The labeled DataFrame to be split.
        classification_params (Dict[str, Any]): 
            A dictionary containing parameters for
            the classification pipeline, including
            'test_size' and 'random_state'.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]: A tuple containing:
            - X_train (np.ndarray): Training features.
            - X_test (np.ndarray): Testing features.
            - y_train (np.ndarray): Training target labels.
            - y_test (np.ndarray): Testing target labels.
            - X_train_df (pd.DataFrame): Training features as a DataFrame for feature importance.

    Raises:
        TypeError: 
            If `df` is not a Pandas DataFrame.
        ValueError: 
            If required columns are missing or if `df` is empty.

    Output:
        - Console/Log:
            Informational messages about the split and the resulting shapes of the datasets.
        - Return Value:
            Four NumPy arrays representing the split data.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from src.classification import prepare_training_data_splits
        >>> # Dummy data
        >>> df_mock = pd.DataFrame({
        ...     'image_id': ['A', 'B', 'C', 'D'], 'feat1': [1, 2, 3, 4],
        ...     'wound_type': ['Type A', 'Type A', 'Type B', 'Type B']
        ... })
        >>> classification_params = {'test_size': 0.5, 'random_state': 42}
        >>> X_train, X_test, y_train, y_test = prepare_training_data_splits(df_mock, classification_params=classification_params)
        >>> print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    Relationships:
        - Dependencies:
            Relies on `pandas` and `sklearn.model_selection.train_test_split`.
        - Used by:
            The classification pipeline to prepare data for `train_classifier`.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Input 'df' must be a Pandas DataFrame.")
        raise TypeError("Input 'df' must be a Pandas DataFrame.")
    if df.empty:
        logger.warning("Input DataFrame is empty. Cannot split data.")
        raise ValueError("Input DataFrame is empty.")
    if 'wound_type' not in df.columns:
        logger.error("DataFrame must contain 'wound_type' column for splitting.")
        raise ValueError("DataFrame is missing the 'wound_type' column.")
    
    test_size = classification_params['test_size']
    random_state = classification_params['random_state']

    # Filter out the 'Outlier / Anomaly' wound type
    df_final = df[df['wound_type'] != 'Outlier'].copy()
    logger.info(f"Original number of rows after filtering '{'Outlier / Anomaly'}': {len(df_final)}")

    # remove row with NaN value.
    df_final_cleaned = df_final.dropna()
    logger.info(f"Number of rows after removing NaNs: {len(df_final_cleaned)}")

    feature_columns = df.drop(columns=['image_id', 'cluster_label', 'wound_type'], errors='ignore').columns
    if df_final_cleaned.empty:
        logger.warning("No valid rows remaining after removing NaNs for training.")
        return np.array([]), np.array([]), np.array([]), np.array([]), pd.DataFrame(columns=feature_columns)

    y = df_final_cleaned['wound_type']
    
    X = df_final_cleaned.drop(columns=['image_id', 'cluster_label', 'wound_type']) 

    logger.info("Splitting dataset into training and testing sets.")
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    X_train, X_test, y_train, y_test = X_train_df.values, X_test_df.values, y_train_df.values, y_test_df.values

    logger.info(f"Data split complete. Training set shape: {X_train.shape}, Test set shape: {X_test.shape}.")
    
    return X_train, X_test, y_train, y_test, X_train_df


def train_classifier(X_train: np.ndarray, y_train: np.ndarray, classification_params: Dict[str, Any], save_path: Path) -> RandomForestClassifier:
    """
    Trains a Random Forest classifier model.

    This function initializes a `RandomForestClassifier` with a fixed set of hyperparameters
    and trains it on the provided training data.

    Args:
        X_train (np.ndarray): 
            Training features.
        y_train (np.ndarray): 
            Training target labels.
        classification_params (Dict[str, Any]): 
            A dictionary containing parameters for
            the classification pipeline, including
            'random_state' and 'random_forest_n_estimators'.
        save_path (Path): 
            The full path, including filename, to save the trained model.

    Returns:
        RandomForestClassifier: 
            The trained classifier model.

    Raises:
        TypeError: 
            If inputs are not NumPy arrays.
        ValueError: 
            If input arrays are empty or have inconsistent shapes.
        RuntimeError: 
            If model training fails.

    Output:
        - Console/Log:
            Informational messages about model training and performance.
        - Return Value:
            The trained classifier model.

    Examples:
        >>> import numpy as np
        >>> from src.classification import train_classifier
        >>> # Dummy data
        >>> X_train = np.random.rand(10, 5)
        >>> y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        >>> classification_params = {'random_state': 42, 'random_forest_n_estimators': 100}
        >>> model = train_classifier(X_train, y_train, classification_params)

    Relationships:
        - Dependencies:
            Relies on `numpy` and `sklearn.ensemble.RandomForestClassifier` and 'joblib.
        - Used by:
            The classification pipeline to train the model.
    """
    if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
        logger.error("Inputs must be NumPy arrays.")
        raise TypeError("Inputs must be NumPy arrays.")
    if X_train.size == 0 or X_train.shape[0] != y_train.shape[0]:
        error_message = "Input arrays are empty or have inconsistent shapes."
        logger.error(error_message)
        raise ValueError(error_message) 

    n_estimators = classification_params['random_forest_n_estimators']
    random_state = classification_params['random_state']

    logger.info("Training Random Forest classifier model.")

    try:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        
        # Save the model to a file
        joblib.dump(model, str(save_path))
        
        logger.info(f"Model training complete and saved to {save_path}.")
    except Exception as e:
        logger.exception("An error occurred during model training.")
        raise RuntimeError(f"Model training failed: {e}") from e

    return model


def evaluate_classifier(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates the performance of the trained classifier model on a test set.

    This function calculates several standard classification metrics, including
    overall accuracy and a detailed classification report, providing a comprehensive
    view of the model's performance.

    Args:
        model (RandomForestClassifier): 
            The trained classifier model.
        X_test (np.ndarray): 
            Testing features.
        y_test (np.ndarray): 
            Testing target labels.

    Returns:
        Dict[str, Any]: 
            A dictionary containing the accuracy score, the full classification
            report, and the model itself.

    Raises:
        TypeError: 
            If inputs are not of expected types.
        ValueError: 
            If input arrays are empty or have inconsistent shapes.
        RuntimeError: 
            If model prediction or evaluation fails.

    Output:
        - Console/Log:
            Informational messages about model performance.
        - Return Value:
            A dictionary of evaluation metrics.

    Examples:
        >>> import numpy as np
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from src.classification import evaluate_classifier
        >>> # Dummy data
        >>> model = RandomForestClassifier(random_state=42).fit(np.random.rand(10, 5), np.array([0, 1]*5))
        >>> X_test = np.random.rand(10, 5)
        >>> y_test = np.array([0, 1]*5)
        >>> results = evaluate_classifier(model, X_test, y_test)

    Relationships:
        - Dependencies:
            Relies on `numpy` and `sklearn.metrics` for evaluation.
        - Used by:
            The classification pipeline to report model performance.
    """
    if not isinstance(model, RandomForestClassifier):
        logger.error("Input 'model' must be a trained RandomForestClassifier.")
        raise TypeError("Input 'model' must be a trained RandomForestClassifier.")
    if not isinstance(X_test, np.ndarray) or not isinstance(y_test, np.ndarray):
        logger.error("Inputs must be NumPy arrays.")
        raise TypeError("Inputs must be NumPy arrays.")
    if X_test.size == 0 or X_test.shape[0] != y_test.shape[0]:
        error_message = "Input arrays must be non-empty and have consistent shapes." 
        logger.error(error_message)
        raise ValueError(error_message)

    logger.info("Evaluating model performance on test set.")
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        logger.info(f"Overall Model Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))

        results = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'model': model
        }
        return results
    except Exception as e:
        logger.exception("An error occurred during model prediction or evaluation.")
        raise RuntimeError(f"Model prediction or evaluation failed: {e}") from e


def get_feature_importances(model: RandomForestClassifier, X_train_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Calculates and returns feature importances from a trained model.

    This function extracts the feature importance scores from a trained `RandomForestClassifier`
    model, and returns them as a DataFrame. This is useful for understanding which features
    are most influential for the model's predictions.

    Args:
        model (RandomForestClassifier): 
            The trained classifier model.
        X_train_df (pd.DataFrame): 
            Training features.

    Returns:
        Optional[pd.DataFrame]: 
            A DataFrame with 'feature' and 'importance' columns,
            sorted by importance in descending order, or None if the model
            does not support feature importances.

    Raises:
        TypeError: 
            If inputs are not of expected types.
        ValueError: 
            If input DataFrame is empty or missing required columns.

    Output:
        - Console/Log: 
            Informational messages about the top features and any warnings if the
            model does not support feature importances.
        - Return Value:
            A DataFrame of feature importances.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from src.classification import get_feature_importances
        >>> # Dummy data
        >>> X_train = pd.DataFrame(np.random.rand(10, 5), columns=[f'feat{i}' for i in range(5)])
        >>> y_train = np.array([0, 1]*5)
        >>> model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
        >>> importances_df = get_feature_importances(model, X_train)

    Relationships:
    - Dependencies:
        - `pandas`: For DataFrame manipulation.
        - `numpy`: For array operations.
        - `sklearn.ensemble.RandomForestClassifier`: The type of model expected.
        - `logging`: For outputting messages.
    - Used by:
        - The classification pipeline, to report feature importance.
    """
    if not isinstance(model, RandomForestClassifier):
        logger.error("Input 'model' must be a trained RandomForestClassifier.")
        raise TypeError("Input 'model' must be a trained RandomForestClassifier.")
    if not isinstance(X_train_df, pd.DataFrame):
        logger.error("Input 'X_train_df' must be a Pandas DataFrame.")
        raise TypeError("Input 'X_train_df' must be a Pandas DataFrame.")
    if X_train_df.empty:
        logger.warning("Input DataFrame is empty. Skipping feature importance calculation.")
        return None
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have 'feature_importances_' attribute. Skipping feature importance calculation.")
        return None

    try:
        feature_names = X_train_df.columns
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        logger.info("Top 10 Most Important Features:\n%s", feature_importance_df.head(10))
        
        return feature_importance_df
    except Exception as e:
        logger.exception("An error occurred during feature importance calculation.")
        raise RuntimeError(f"Feature importance calculation failed: {e}") from e
    

def predict_wound_border_type(model_path: Path, new_features_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Classifies new wound images using a pre-trained model and returns the prediction.

    This function loads a pre-trained `RandomForestClassifier` model from a file,
    loads and standardizes the new features from a CSV file, and then uses the model
    to predict the wound border type. The result is returned as a DataFrame containing
    the original image IDs and the predicted labels.

    Args:
        model_path (Path): 
            The full path to the saved model file (e.g., '.joblib').
        new_features_path (Path): 
            The full path to the CSV file containing the new feature vectors.
            The CSV file must have an 'image_id' column.

    Returns:
        Optional[pd.DataFrame]: 
            A DataFrame with 'image_id', 'predicted_label', and 'probability'.
            Returns None if the prediction fails.

    Raises:
        FileNotFoundError: 
            If the model or feature file does not exist.
        IOError: 
            If there is an issue loading the model or feature file.
        TypeError: 
            If inputs are not of expected types.

    Output:
        - Console/Log:
            Informational messages about the prediction and the result.
        - Return Value:
            A DataFrame of prediction results.

    Examples:
        >>> import pandas as pd
        >>> from pathlib import Path
        >>> from src.classification import predict_wound_border_type
        >>> # Dummy data setup
        >>> dummy_model_path = Path('./dummy_model.joblib')
        >>> dummy_features_path = Path('./dummy_features.csv')
        >>> label_map = {0: 'Type A', 1: 'Type B'}
        >>> prediction_df = predict_wound_border_type(dummy_model_path, dummy_features_path, label_map)
    
    Relationships:
        - Dependencies:
            - `joblib`: For loading the pre-trained model.
            - `pandas`: For DataFrame manipulation.
            - `numpy`: For array operations.
            - `sklearn.ensemble.RandomForestClassifier`: The type of model expected.
            - `logging`: For outputting messages.
        - Used by:
            - The classification pipeline, to classify new, unseen images.
    """
    if not isinstance(model_path, Path):
        logger.error("Input 'model_path' must be a Path object.")
        raise TypeError("Input 'model_path' must be a Path object.")
    if not isinstance(new_features_df, pd.DataFrame):
        logger.error("Input 'new_features_df' must be a Pandas DataFrame.")
        raise TypeError("Input 'new_features_df' must be a Pandas DataFrame.")
    if new_features_df.empty:
        logger.warning("Input features DataFrame is empty. Returning empty prediction DataFrame.")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=['image_id', 'predicted_label', 'probability'])
    if not model_path.exists():
        logger.error(f"Model file not found at: {model_path}")
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        model = joblib.load(str(model_path))
        
        if 'image_id' not in new_features_df.columns:
            logger.error("Features DataFrame is missing the 'image_id' column.")
            raise ValueError("Features DataFrame is missing the 'image_id' column.")

        # Drop non-feature columns
        features_to_predict = new_features_df.drop(columns=['image_id']).values
        image_ids = new_features_df['image_id']
        
        scaled_features = features_to_predict # Placeholder for a complete pipeline

        predictions = model.predict(scaled_features)
        probabilities = np.max(model.predict_proba(scaled_features), axis=1)
       
        result_df = pd.DataFrame({
            'image_id': image_ids,
            'predicted_label': predictions,
            'probability': probabilities
        })
        logger.info("Prediction complete.")
        return result_df
    except Exception as e:
        logger.exception("An error occurred during prediction.")
        raise RuntimeError(f"Prediction failed: {e}") from e

