"""
This module contains unit tests for the `Config` class available in `src.config_manager`.

It uses pytest to rigorously verify the functionality of the configuration management
system, including file loading, data retrieval, path resolution, and comprehensive
error handling scenarios.

Functions:
    - `create_dummy_config_file`: A helper to generate temporary JSON configuration files for tests.
    - `temp_config_path`: A pytest fixture providing a temporary file path for config tests.
    - `test_config_load_success`: Verifies successful configuration loading.
    - `test_config_file_not_found`: Tests error handling for missing config files.
    - `test_config_invalid_json`: Checks error handling for malformed JSON config files.
    - `test_config_get_existing_option`: Confirms correct retrieval of existing config options.
    - `test_config_get_missing_option_with_fallback`: Validates fallback behavior for missing options.
    - `test_config_get_missing_option_no_fallback`: Tests error raising for missing options without fallback.
    - `test_config_get_paths_accuracy`: Ensures accurate resolution of all defined paths.
    - `test_config_get_section_params`: Verifies retrieval of entire configuration sections.
    - `test_get_descriptive_labels_success_basic`: Tests successful retrieval and conversion of basic descriptive labels.
    - `test_get_descriptive_labels_empty_section`: Verifies handling of empty or missing 'DescriptiveLabels' section.
    - `test_get_descriptive_labels_non_integer_convertible_keys`: Tests error handling for non-integer keys in labels.
    - `test_get_descriptive_labels_mixed_value_types`: Ensures correct handling of various value types in labels.

Typical use:
    This module is designed to be run as part of the project's test suite using `pytest`.
    It ensures the `Config` class behaves as expected under various conditions,
    contributing to the overall stability and reliability of the application's configuration handling.
"""

import pytest
import json
from pathlib import Path
import os
import sys
import logging

# Add project root to sys.path for module discovery
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root)) 

from src.config_manager import Config # Import the Config class

# --- Helper function to create a dummy config.json ---
def create_dummy_config_file(file_path: Path, content: dict):
    """Writes a dictionary as JSON to a file."""
    with open(file_path, 'w') as f:
        json.dump(content, f, indent=2)

# --- Pytest Fixture for a temporary config file ---
@pytest.fixture
def temp_config_path(tmp_path: Path) -> Path:
    """
    Provides a temporary path for a config.json file.
    `tmp_path` is a built-in pytest fixture for temporary directories.
    """
    return tmp_path / "config.json"

# --- Test Cases for Config Class ---
class TestConfig: 

    @pytest.fixture(autouse=True)
    def setup_test_logging(self, caplog): 
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
        
        caplog.set_level(logging.DEBUG, logger="src.config_manager")


    def test_config_load_success(self, temp_config_path: Path, caplog):
        """
        GIVEN: A valid config.json file exists.
        WHEN:  Config class is instantiated with its path.
        THEN:  The config should load successfully and its data should match the file content.
            An INFO message should be logged.
        """
        config_content = {
            "DataPaths": {"data_path": "./data"},
            "Filtering": {"threshold": 100}
        }
        create_dummy_config_file(temp_config_path, config_content)

        with caplog.at_level(logging.INFO): 
            config = Config(str(temp_config_path))

        assert config.data == config_content
        assert "Configuration loaded successfully" in caplog.text
        assert "INFO" in caplog.text


    def test_config_file_not_found(self, tmp_path: Path, caplog):
        """
        GIVEN: A non-existent config file.
        WHEN:  Config class is instantiated.
        THEN:  A FileNotFoundError should be raised, and a CRITICAL message should be logged.
        """
        non_existent_path = tmp_path / "non_existent_config.json"

        with pytest.raises(FileNotFoundError) as excinfo:
            with caplog.at_level(logging.ERROR): 
                Config(str(non_existent_path))
        
        assert f"Configuration file not found at: {non_existent_path}" in str(excinfo.value)
        assert "CRITICAL" in caplog.text 
        assert f"Configuration file not found at: '{non_existent_path}'." in caplog.text 


    def test_config_invalid_json(self, temp_config_path: Path, caplog): 
        """
        GIVEN: A file exists but contains invalid JSON content.
        WHEN:  Config class is instantiated.
        THEN:  A json.JSONDecodeError should be raised, and a CRITICAL message should be logged.
        """
        temp_config_path.write_text("{'DataPaths': {'data_path': './data'") 
        
        with pytest.raises(json.JSONDecodeError):
            with caplog.at_level(logging.ERROR): 
                Config(str(temp_config_path))
        
        assert "CRITICAL" in caplog.text
        assert "Could not decode JSON from" in caplog.text


    def test_config_get_existing_option(self, temp_config_path: Path):
        """
        GIVEN: A config loaded with a known section and option.
        WHEN:  The get method is called for that option.
        THEN:  The correct value should be returned.
        """

        config_content = {"Section": {"Option": "value"}}
        create_dummy_config_file(temp_config_path, config_content)
        config = Config(str(temp_config_path))

        value = config.get("Section", "Option")

        assert value == "value"


    def test_config_get_missing_option_with_fallback(self, temp_config_path: Path, caplog): 
        """
        GIVEN: A config loaded without a specific option.
        WHEN:  The get method is called for that option with a fallback.
        THEN:  The fallback value should be returned, and a WARNING should be logged.
        """
        config_content = {"Section": {"ExistingOption": "existing_value"}}
        create_dummy_config_file(temp_config_path, config_content)

        config = Config(str(temp_config_path)) 

        with caplog.at_level(logging.WARNING): 
            value = config.get("Section", "MissingOption", fallback="default_value")

        assert value == "default_value"
        assert "WARNING" in caplog.text
        assert "Option 'MissingOption' not found in section 'Section'. Using fallback value: default_value." in caplog.text


    def test_config_get_missing_option_no_fallback(self, temp_config_path: Path, caplog): 
        """
        GIVEN: A config loaded without a specific option and no fallback is provided.
        WHEN:  The get method is called for that option.
        THEN:  A ValueError should be raised, and an ERROR should be logged.
        """
        config_content = {"Section": {"ExistingOption": "existing_value"}}
        create_dummy_config_file(temp_config_path, config_content)
        config = Config(str(temp_config_path)) 

        with pytest.raises(ValueError) as excinfo:
            with caplog.at_level(logging.ERROR): 
                config.get("Section", "MissingOption")
        
        assert "Missing required config option: [Section]MissingOption" in str(excinfo.value)
        assert "'MissingOption'" in str(excinfo.value) 
        assert "ERROR" in caplog.text
        assert "Missing required configuration option: [Section]MissingOption. No fallback provided." in caplog.text


    def test_config_get_paths_accuracy(self, tmp_path: Path):
        """
        GIVEN: A config.json with standard data paths and subdirs, and a temporary project structure.
        WHEN:  The get_paths method is called.
        THEN:  All returned Path objects should be correct, resolved, and absolute paths
               matching the expected temporary structure.
        """
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "images").mkdir()
        (tmp_path / "data" / "wound_masks").mkdir()
        (tmp_path / "data" / "body_masks").mkdir()
        (tmp_path / "data" / "depth_maps").mkdir()
        (tmp_path / "data" / "marker_masks").mkdir()
        (tmp_path / "metadata").mkdir()
        (tmp_path / "outputs").mkdir()
        
        config_content = {
            "DataPaths": {
                "data_path": "./data",
                "metadata_path": "./metadata",
                "output_path": "./outputs",
                "image_manifest_filtered_filename": "image_index_filtered.csv",
                "comprehensive_features_filename": "1_comprehensive_features.csv",      
                "image_cluster_map_filename": "2_image_cluster_map.csv",      
                "cluster_summary_filename": "3_cluster_summary.csv",
                "cluster_profile_filename": "4_cluster_profiles.csv",
                "features_with_labels_filename": "5_features_with_labels.csv",
                "pacmap_graph_filename": "6_PACMAP_graph.png",
                "hdbscan_graph_filename": "7_HDBSCAN_cluster_graph.png",
                "claster_sample_filename": "8_samples_for_clusters.png",
                "feature_distribution_plot_filename": "9_feature_distribution_boxplot.png",
                "confution_matrix_plot_filename": "10_confusion_matrix.png",
                "feature_importance_filename": "11_feature_importance.png",
                "random_forest_model_filename": "random_forest_model.joblib"
            },
            "subdirs": {
                "wound_masks_subdir": "wound_masks",
                "images_subdir": "images",
                "body_mask_subdir": "body_masks",
                "depth_maps_subdir": "depth_maps",
                "marker_mask_subdir": "marker_masks"
            },
            "Filtering": {}, "FeatureExtraction": {}, "Clustering": {}, "Classification": {} 
        }
        config_file = tmp_path / "config.json"
        create_dummy_config_file(config_file, config_content)
        
        # Change current working directory to tmp_path so relative paths resolve correctly
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            config = Config(str(config_file.name)) 
        
            paths = config.get_paths()
            
            assert paths['base_data_dir'] == (tmp_path / "data").resolve()
            assert paths['base_metadata_dir'] == (tmp_path / "metadata").resolve()
            assert paths['base_output_dir'] == (tmp_path / "outputs").resolve()

            assert paths['images_dir'] == (tmp_path / "data" / "images").resolve()
            assert paths['wound_masks_dir'] == (tmp_path / "data" / "wound_masks").resolve()
            assert paths['body_mask_dir'] == (tmp_path / "data" / "body_masks").resolve()
            assert paths['depth_maps_dir'] == (tmp_path / "data" / "depth_maps").resolve()
            assert paths['marker_mask_dir'] == (tmp_path / "data" / "marker_masks").resolve()

            assert paths['filtered_manifest_path'] == (tmp_path / "metadata" / "image_index_filtered.csv").resolve()
            assert paths['comprehensive_features_csv'] == (tmp_path / "outputs" / "1_comprehensive_features.csv").resolve()
            assert paths['image_cluster_map_csv'] == (tmp_path / "outputs" / "2_image_cluster_map.csv").resolve()
            assert paths['cluster_summary_csv'] == (tmp_path / "outputs" / "3_cluster_summary.csv").resolve()
            assert paths['cluster_profiles_csv'] == (tmp_path / "outputs" / "4_cluster_profiles.csv").resolve()
            assert paths['features_with_labels_csv'] == (tmp_path / "outputs" / "5_features_with_labels.csv").resolve()
            assert paths['pacmap_graph'] == (tmp_path / "outputs" / "6_PACMAP_graph.png").resolve()
            assert paths['hdbscan_graph'] == (tmp_path / "outputs" / "7_HDBSCAN_cluster_graph.png").resolve()
            assert paths['samples_for_clusters'] == (tmp_path / "outputs" / "8_samples_for_clusters.png").resolve()
            assert paths['feature_distribution_graph'] == (tmp_path / "outputs" / "9_feature_distribution_boxplot.png").resolve()
            assert paths['confusion_matrix_plot'] == (tmp_path / "outputs" / "10_confusion_matrix.png").resolve()
            assert paths['feature_importance_graph'] == (tmp_path / "outputs" / "11_feature_importance.png").resolve()

        finally:
            # Restore the original working directory
            os.chdir(old_cwd)


    def test_config_get_section_params(self, temp_config_path: Path):
        """
        GIVEN: A config loaded with various sections.
        WHEN:  Convenience methods like get_filtering_params are called.
        THEN:  The correct dictionary for that section should be returned.
        """

        config_content = {
            "Filtering": {"threshold": 100, "components": 1},
            "FeatureExtraction": {"iterations": 50},
            "Clustering": {"param_c": 1}, 
            "Classification": {"param_class": "rf"}, 
            "subdirs": {"images_subdir": "img"}, 
            "DataPaths": {} 
        }
        create_dummy_config_file(temp_config_path, config_content)
        config = Config(str(temp_config_path))

        filtering_params = config.get_filtering_params()
        feature_params = config.get_feature_extraction_params()
        clustering_params = config.get_clustering_params()
        classification_params = config.get_classification_params()
        subdirs_params = config.get_subdirs_params()

        assert filtering_params == {"threshold": 100, "components": 1}
        assert feature_params == {"iterations": 50}
        assert clustering_params == {"param_c": 1}
        assert classification_params == {"param_class": "rf"}
        assert subdirs_params == {"images_subdir": "img"}


    def test_get_descriptive_labels_success_basic(self):
        """
        GIVEN: A Config instance with a 'DescriptiveLabels' section containing string keys that are integers.
        WHEN: get_descriptive_labels is called.
        THEN: It should return a dictionary with integer keys and correct values.
        """
        config = Config()
        config.data = {
            'DescriptiveLabels': {
                '0': 'Cluster A',
                '1': 'Cluster B',
                '10': 'Noise'
            }
        }
        
        labels = config.get_descriptive_labels()
        
        assert isinstance(labels, dict)
        assert labels == {0: 'Cluster A', 1: 'Cluster B', 10: 'Noise'}
        assert all(isinstance(k, int) for k in labels.keys())


    def test_get_descriptive_labels_empty_section(self):
        """
        GIVEN: A Config instance where 'DescriptiveLabels' section is empty or missing.
        WHEN: get_descriptive_labels is called.
        THEN: It should return an empty dictionary.
        """
        config = Config()
        config.data = {'DescriptiveLabels': {}}
        labels = config.get_descriptive_labels()
        assert labels == {}

        config.data = {} 
        labels = config.get_descriptive_labels()
        assert labels == {}


    def test_get_descriptive_labels_non_integer_convertible_keys(self):
        """
        GIVEN: A Config instance with 'DescriptiveLabels' containing keys that cannot be converted to integers.
        WHEN: get_descriptive_labels is called.
        THEN: It should raise a ValueError (from int() conversion).
        """
        config = Config()
        config.data = {
            'DescriptiveLabels': {
                '0': 'Cluster A',
                'invalid_key': 'Some label',
                '1': 'Cluster B'
            }
        }
        
        with pytest.raises(ValueError) as excinfo:
            config.get_descriptive_labels()
        
        assert "invalid literal for int()" in str(excinfo.value)


    def test_get_descriptive_labels_mixed_value_types(self):
        """
        GIVEN: A Config instance with 'DescriptiveLabels' containing mixed value types.
        WHEN: get_descriptive_labels is called.
        THEN: It should correctly convert keys to integers and preserve value types.
        """
        config = Config()
        config.data = {
            'DescriptiveLabels': {
                '0': 'String Label',
                '1': 100,
                '2': True,
                '3': [1, 2, 3]
            }
        }
        
        labels = config.get_descriptive_labels()
        
        assert labels == {0: 'String Label', 1: 100, 2: True, 3: [1, 2, 3]}
        assert all(isinstance(k, int) for k in labels.keys())
        assert isinstance(labels[0], str)
        assert isinstance(labels[1], int)
        assert isinstance(labels[2], bool)
        assert isinstance(labels[3], list)
