"""
This module provides a robust configuration management system for the project.

It defines the `Config` class, which handles loading, accessing, and organizing
application settings from a JSON configuration file. This centralizes all
configuration parameters, including file paths, filtering thresholds, and model
parameters, making the application more flexible and easier to manage.

Functions: 
    - `Config`: The main class providing methods to load, access, and manage application configurations.

Typical use:
    This module is typically used at the application's startup to load all necessary
    configuration parameters and file paths, providing a single source of truth for
    settings throughout the pipeline.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Get a logger instance for this module
logger = logging.getLogger(__name__)

class Config:
    """
    Manages configuration settings loaded from a JSON file.

    This class provides a structured way to load, access, and manage application
    configuration parameters from a JSON file. It centralizes configuration logic,
    making it easy to retrieve parameters by section and option, and to resolve
    file paths.

    Args:
        config_filepath (str): 
            The relative or absolute path to the JSON configuration file.
            Defaults to 'config.json' in the current working directory.

    Attributes:
        config_filepath (Path): 
            The resolved Path object pointing to the configuration file.
        data (Dict[str, Any]): 
            A dictionary holding the parsed content of the JSON file.

    Raises:
        FileNotFoundError: 
            If the specified configuration file does not exist.
        json.JSONDecodeError: 
            If the configuration file is found but is not valid JSON.
        ValueError: 
            If a required configuration option is missing when accessed without a fallback.

    Methods:
        - `_load_config()`: Internal method to read and parse the JSON file.
        - `get(section, option, fallback)`: Retrieves a specific configuration value.
        - `get_paths()`: Constructs and returns resolved `pathlib.Path` objects for all
          file and directory paths defined in the configuration.
        - `get_filtering_params()`, `get_feature_extraction_params()`, etc.:
          Convenience methods to return specific sections of the configuration as dictionaries.

    Output:
        - Log: 
            Informational messages about successful configuration loading, warnings
            if optional configuration options are missing (and a fallback is used),
            and errors for critical issues (e.g., file not found, invalid format).

    Example:
        # Example : Basic configuration loading and access
        >>> # Assuming a 'config.json' exists in the root with:
        >>> # {"DataPaths": {"data_path": "./data"}, "Filtering": {"threshold": 100}}
        >>> from pathlib import Path
        >>> from src.config_manager import Config
        >>> # Assume logging is set up: from src.logging_setup import setup_logging; setup_logging(log_level="INFO")

        >>> config = Config("config.json")
        >>> data_path = config.get("DataPaths", "data_path")
        >>> threshold = config.get("Filtering", "threshold")
        >>> print(f"Loaded data path: {data_path}, threshold: {threshold}")
        Expected output:
        Loaded data path: ./data, threshold: 100

    Relationships:
        - Dependencies: 
            Relies on Python's built-in `json` module for parsing,
            `pathlib.Path` for path management, and `logging` for output.
        - Used by: 
            The main application entry point (`run_pipeline.py`)
            to load and manage all application parameters and file paths.
    """
    def __init__(self, config_filepath: str = 'config.json'):
        """
        Initializes the Config manager by loading settings from a JSON file.

        Args:
            config_filepath (str): 
                The path to the JSON configuration file. Defaults to 'config.json'.

        Raises:
            FileNotFoundError: 
                If the specified `config_filepath` does not exist.
            json.JSONDecodeError: 
                If the configuration file is not valid JSON.
            IOError: 
                If there's an issue reading the file.
        """
        self.config_filepath: Path = Path(config_filepath)
        self.data: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """
        Loads the configuration from the JSON file.

        Methods:
            - Reads the JSON file specified by `self.config_filepath`.
            - Parses the content into `self.data` dictionary.

        Raises:
            FileNotFoundError: 
                If the configuration file does not exist.
            json.JSONDecodeError: 
                If the file content is not valid JSON.
            IOError: 
                If an unexpected error occurs during file reading.
        """
        if not self.config_filepath.exists():
            logger.critical(f"Configuration file not found at: '{self.config_filepath}'.")
            raise FileNotFoundError(f"Configuration file not found at: {self.config_filepath}")
        try:
            with open(self.config_filepath, 'r') as f:
                self.data = json.load(f)
            logger.info(f"Configuration loaded successfully from {self.config_filepath}.")
        except json.JSONDecodeError as e:
            logger.critical(f"Error: Could not decode JSON from '{self.config_filepath}'. Please check file format. {e}")
            raise json.JSONDecodeError(f"Invalid JSON format in '{self.config_filepath}': {e.msg}", e.doc, e.pos) from e
        except Exception as e:
            logger.critical(f"An unexpected error occurred while loading config from '{self.config_filepath}': {e}")
            raise IOError(f"Error reading configuration file '{self.config_filepath}': {e}") from e

    def get(self, section: str, option: str, fallback: Any = None) -> Any:
        """
        Retrieves a configuration value from a specific section and option.

        Args:
            section (str): The name of the section (e.g., 'DataPaths', 'Filtering').
            option (str): The name of the option within the section (e.g., 'data_path', 'threshold').
            fallback (Any, optional): A default value to return if the option is not found.
                                      If None and the option is not found, a ValueError is raised.

        Returns:
            Any: The value of the specified configuration option.

        Raises:
            ValueError: If the option is not found and no `fallback` value is provided.

        Methods:
            - Attempts to access the value using dictionary lookup.
            - If `KeyError` occurs and `fallback` is provided, returns `fallback`.
            - If `KeyError` occurs and no `fallback` is provided, raises `ValueError`.

        Output:
            - Log: A warning message is logged if an option is not found but a fallback is used.
        """
        try:
            return self.data[section][option]
        except KeyError as e:
            if fallback is not None:
                logger.warning(f"Option '{option}' not found in section '{section}'. Using fallback value: {fallback}.")
                return fallback
            else:
                logger.error(f"Missing required configuration option: [{section}]{option}. No fallback provided.")
                raise ValueError(f"Missing required config option: [{section}]{option}: {e}") from e

    def get_paths(self) -> Dict[str, Path]:
        """
        Constructs and returns a dictionary of all resolved Path objects needed by the pipeline.
        This centralizes all path creation logic and ensures OS-independent path handling.

        Returns:
            Dict[str, Path]: 
                A dictionary where keys are descriptive path names
                (e.g., 'base_data_dir', 'filtered_manifest_path') and
                values are resolved `pathlib.Path` objects.

        Methods:
            - Retrieves base directory paths (e.g., `data_path`, `metadata_path`) and
              subdirectory names from the configuration.
            - Uses `pathlib.Path` and its `/` operator to construct full, absolute paths
              for all relevant files and directories. `resolve()` is used to get absolute paths.

        Examples:
            >>> from src.config_manager import Config
            >>> from pathlib import Path
            >>> # Assume config.json in project root with "DataPaths":{"data_path": "./data"}
            >>> config = Config("config.json")
            >>> paths = config.get_paths()
            >>> print(paths['base_data_dir'])
            Expected output:
            /absolute/path/to/your/project/data

        Relationships:
            - Used by: 
                The main application entry point (`run_pipeline.py`)
                to retrieve all necessary file system references.
        """
        data_paths_config = self.data['DataPaths']
        subdirs_config = self.data['subdirs']

        paths = {}

        # Base directories
        paths['base_data_dir'] = Path(data_paths_config['data_path']).resolve()
        paths['base_metadata_dir'] = Path(data_paths_config['metadata_path']).resolve()
        paths['base_output_dir'] = Path(data_paths_config['output_path']).resolve()

        # Full paths for main files (joining base dir with filename)
        paths['filtered_manifest_path'] = paths['base_metadata_dir'] / data_paths_config['image_manifest_filtered_filename']
        paths['comprehensive_features_csv'] = paths['base_output_dir'] / data_paths_config['comprehensive_features_filename']
        paths['image_cluster_map_csv'] = paths['base_output_dir'] / data_paths_config['image_cluster_map_filename']
        paths['cluster_summary_csv'] = paths['base_output_dir'] / data_paths_config['cluster_summary_filename']
        paths['cluster_profiles_csv'] = paths['base_output_dir'] / data_paths_config['cluster_profile_filename']
        paths['features_with_labels_csv'] = paths['base_output_dir'] / data_paths_config['features_with_labels_filename']
        
        # Full paths for figurs 
        paths['pacmap_graph'] = paths['base_output_dir'] / data_paths_config['pacmap_graph_filename']
        paths['hdbscan_graph'] = paths['base_output_dir'] / data_paths_config['hdbscan_graph_filename']
        paths['samples_for_clusters'] = paths['base_output_dir'] / data_paths_config['claster_sample_filename']
        paths['feature_distribution_graph'] = paths['base_output_dir'] / data_paths_config['feature_distribution_plot_filename']
        paths['confusion_matrix_plot'] = paths['base_output_dir'] / data_paths_config['confution_matrix_plot_filename']
        paths['feature_importance_graph'] = paths['base_output_dir'] / data_paths_config['feature_importance_filename']
        
        # Path for the Trained Model
        paths['random_forest_model_path'] = paths['base_output_dir'] / data_paths_config['random_forest_model_filename']
        
        # Full paths for subdirectories (joined with base_data_dir)
        paths['wound_masks_dir'] = paths['base_data_dir'] / subdirs_config['wound_masks_subdir']
        paths['images_dir'] = paths['base_data_dir'] / subdirs_config['images_subdir']
        paths['body_mask_dir'] = paths['base_data_dir'] / subdirs_config['body_mask_subdir']
        paths['depth_maps_dir'] = paths['base_data_dir'] / subdirs_config['depth_maps_subdir']
        paths['marker_mask_dir'] = paths['base_data_dir'] / subdirs_config['marker_mask_subdir']

        return paths

    def get_filtering_params(self) -> Dict[str, Any]:
        """
        Returns the filtering parameters section from the configuration.

        Returns:
            Dict[str, Any]: 
                A dictionary containing parameters related to image filtering (e.g., area thresholds).

        Methods:
            - Directly retrieves the 'Filtering' section from the loaded configuration data.

        Examples:
            >>> from src.config_manager import Config
            >>> # Assume config.json has {"Filtering": {"threshold": 100}}
            >>> config = Config("config.json")
            >>> params = config.get_filtering_params()
            >>> print(params)
            Expected output:
            {'threshold': 100}

        Relationships:
            - Used by: 
                The data filtering stage (e.g., `filter_masks_by_area_and_component_count`)
                to apply specific quality criteria.
        """
        return self.data['Filtering']

    def get_feature_extraction_params(self) -> Dict[str, Any]:
        """
        Returns the feature extraction parameters section from the configuration.

        Returns:
            Dict[str, Any]: A dictionary containing parameters for feature extraction (e.g., unroll iterations).

        Methods:
            - Directly retrieves the 'FeatureExtraction' section.
        """
        return self.data['FeatureExtraction']

    def get_clustering_params(self) -> Dict[str, Any]:
        """
        Returns the clustering parameters section from the configuration.

        Returns:
            Dict[str, Any]: 
                A dictionary containing parameters for clustering algorithms (e.g., HDBSCAN parameters).

        Methods:
            - Directly retrieves the 'Clustering' section.
        """
        return self.data['Clustering']

    def get_classification_params(self) -> Dict[str, Any]:
        """
        Returns the classification parameters section from the configuration.

        Returns:
            Dict[str, Any]: 
                A dictionary containing parameters for classification models (e.g., Random Forest settings).

        Methods:
            - Directly retrieves the 'Classification' section.
        """
        return self.data['Classification']

    def get_subdirs_params(self) -> Dict[str, str]:
        """
        Returns the subdirectory names and mappings from the configuration.

        Returns:
            Dict[str, str]: 
                A dictionary containing mappings for subdirectory names.

        Methods:
            - Directly retrieves the 'subdirs' section.
        """
        return self.data['subdirs']
    
    def get_descriptive_labels(self) -> Dict[int, str]:
        """
        Returns the descriptive labels section from the configuration.
        """
        # Note: JSON keys are strings, so we need to convert them to integers.
        label_map_str = self.data.get('DescriptiveLabels', {})
        label_map_int = {int(k): v for k, v in label_map_str.items()}
        return label_map_int