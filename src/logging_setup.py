"""
This module provides a centralized and configurable logging setup for the project.

It defines the `setup_logging` function, which allows for easy configuration of
console and file-based logging, ensuring consistent log formatting and levels
across the application.

Functions:
    - `setup_logging`: Configures the global root logger for console and optional file output.

Typical use:
    This module is intended to be called once at the application's startup to
    establish the primary logging configuration for the entire project.

"""
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(log_level: str = "INFO", log_file_path: Optional[Path] = None):
    """
    Sets up the global logging configuration for the project.

    This function configures the root logger to output messages to both the console
    and, optionally, to a specified file. It ensures that logs include timestamps,
    log levels, and the origin module name, facilitating debugging and monitoring.
    It also clears any existing handlers to prevent duplicate log messages if called
    multiple times.

       Args:
        log_level (str, optional):
            The minimum level of messages to log for the root logger
            (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
            Messages below this level will be ignored. Case-insensitive.
            Defaults to 'INFO'.
        log_file_path (Optional[Path], optional):
            If a `Path` object is provided, logs will be written to this file,
            appending to it if it exists. The necessary parent directories will
            be created if they don't exist. If `None` (default), logs will only
            go to the console.

    Returns:
        None:
            The function does not return any value; it configures the global
            logging system as a side effect.

    Raises:
        ValueError:
            If `log_level` is not a valid logging level string. 
            # Note: Specific IOError on file handling is caught internally and logged, but not re-raised.

    Examples:
        >>> import logging
        >>> from pathlib import Path
        >>>
        >>> # Example 1: Set up console logging only at INFO level
        >>> setup_logging(log_level="INFO")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("This is an info message to console.")
        >>> logger.debug("This debug message will not appear (level is INFO).")
        >>>
        >>> # Example 2: Set up logging to a file and console at DEBUG level
        >>> log_file = Path("/tmp/my_application.log")
        >>> setup_logging(log_level="DEBUG", log_file_path=log_file)
        >>> logger_file = logging.getLogger(__name__)
        >>> logger_file.debug("This debug message will go to console and file.")

    Relationships:
        Used by:
            Expected to be called once by the main entry point of the application
            at startup (e.g., `run_pipeline.py`).
        Affects:
            All `logging.getLogger(__name__)` instances throughout the application
            will inherit and adhere to this configuration.
    
    Notes:
        - This function modifies the global root logger.
        - Existing log handlers are cleared upon each call to prevent duplicate
          log output if `setup_logging` is invoked multiple times during application
          runtime.
        - If file logging setup fails (e.g., due to permission errors), an error
          is logged, and the function falls back to console-only logging.
    """
    # Create a logger instance (the root logger in this case)
    logger = logging.getLogger()
    logger.setLevel(log_level.upper())

    # Clear any existing handlers to prevent duplicate messages if setup_logging is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define a formatter for log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler: Sends log messages to standard output (console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level.upper())
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler: Sends log messages to a specified file
    if log_file_path:
        try:
            log_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            file_handler = logging.FileHandler(log_file_path, mode='a') # 'a' for append mode
            file_handler.setLevel(log_level.upper())
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Fallback to console if file logging setup fails
            logger.error(f"Failed to set up file logging to {log_file_path}: {e}")
            logger.warning("Proceeding with console-only logging.")
    # Inform about successful setup (this will be the first message output by the logger)
    logger.info(f"Logging configured with level: {log_level.upper()}" +
                (f" and file output to: {log_file_path}" if log_file_path else ""))
