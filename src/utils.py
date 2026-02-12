"""
Utility Functions Module

Provides essential utilities:
- File I/O (CSV, Excel, JSON, Parquet)
- Logging configuration
- Seed management for reproducibility
- Data sanitization and cleaning
- Path management
"""

import os
import sys
import json
import yaml
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from logging.handlers import RotatingFileHandler

# File I/O handlers
class FileHandler:
    """
    Handles file input/output operations
    
    Supports: CSV, Excel, JSON, Parquet, Pickle
    """
    
    @staticmethod
    def read_file(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Read file based on extension
        
        Args:
            filepath: Path to file
            **kwargs: Additional arguments for pandas readers
            
        Returns:
            DataFrame
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        extension = filepath.suffix.lower()
        
        try:
            if extension == '.csv':
                return pd.read_csv(filepath, **kwargs)
            elif extension in ['.xlsx', '.xls']:
                return pd.read_excel(filepath, **kwargs)
            elif extension == '.json':
                return pd.read_json(filepath, **kwargs)
            elif extension == '.parquet':
                return pd.read_parquet(filepath, **kwargs)
            elif extension == '.pkl':
                return pd.read_pickle(filepath, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
        
        except Exception as e:
            logging.error(f"Failed to read {filepath}: {e}")
            raise
    
    @staticmethod
    def write_file(
        data: pd.DataFrame,
        filepath: Union[str, Path],
        **kwargs
    ):
        """
        Write file based on extension
        
        Args:
            data: DataFrame to write
            filepath: Output path
            **kwargs: Additional arguments for pandas writers
        """
        filepath = Path(filepath)
        
        # Create directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        extension = filepath.suffix.lower()
        
        try:
            if extension == '.csv':
                data.to_csv(filepath, index=False, **kwargs)
            elif extension in ['.xlsx', '.xls']:
                data.to_excel(filepath, index=False, **kwargs)
            elif extension == '.json':
                data.to_json(filepath, **kwargs)
            elif extension == '.parquet':
                data.to_parquet(filepath, **kwargs)
            elif extension == '.pkl':
                data.to_pickle(filepath, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
            
            logging.info(f"File written successfully: {filepath}")
        
        except Exception as e:
            logging.error(f"Failed to write {filepath}: {e}")
            raise
    
    @staticmethod
    def read_config(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Read configuration file (YAML or JSON)
        
        Args:
            filepath: Path to config file
            
        Returns:
            Configuration dictionary
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        extension = filepath.suffix.lower()
        
        try:
            with open(filepath, 'r') as f:
                if extension in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                elif extension == '.json':
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {extension}")
        
        except Exception as e:
            logging.error(f"Failed to read config {filepath}: {e}")
            raise
    
    @staticmethod
    def write_config(
        config: Dict[str, Any],
        filepath: Union[str, Path]
    ):
        """
        Write configuration file (YAML or JSON)
        
        Args:
            config: Configuration dictionary
            filepath: Output path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        extension = filepath.suffix.lower()
        
        try:
            with open(filepath, 'w') as f:
                if extension in ['.yaml', '.yml']:
                    yaml.dump(config, f, default_flow_style=False)
                elif extension == '.json':
                    json.dump(config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config format: {extension}")
            
            logging.info(f"Config written successfully: {filepath}")
        
        except Exception as e:
            logging.error(f"Failed to write config {filepath}: {e}")
            raise
    
    @staticmethod
    def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Get file information
        
        Args:
            filepath: Path to file
            
        Returns:
            Dictionary with file info
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            return {'exists': False}
        
        stat = filepath.stat()
        
        return {
            'exists': True,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'extension': filepath.suffix,
            'name': filepath.name,
            'parent': str(filepath.parent)
        }


class LoggerConfig:
    """
    Logging configuration manager
    
    Sets up consistent logging across the application
    """
    
    @staticmethod
    def setup_logger(
        name: str = "synthetic_data",
        level: int = logging.INFO,
        log_file: Optional[Union[str, Path]] = None,
        log_to_console: bool = True,
        log_format: Optional[str] = None
    ) -> logging.Logger:
        """
        Setup and configure logger
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Optional file path for file logging
            log_to_console: Whether to log to console
            log_format: Custom log format
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Default format
        if log_format is None:
            log_format = (
                '%(asctime)s - %(name)s - %(levelname)s - '
                '%(filename)s:%(lineno)d - %(message)s'
            )
        
        formatter = logging.Formatter(log_format)
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def get_logger(name: str = "synthetic_data") -> logging.Logger:
        """
        Get existing logger or create default one
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        logger = logging.getLogger(name)
        
        if not logger.handlers:
            # Setup default logger
            LoggerConfig.setup_logger(name)
        
        return logger


class SeedManager:
    """
    Manages random seeds for reproducibility
    
    Ensures consistent random number generation across numpy, pandas, and Python
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize seed manager
        
        Args:
            seed: Random seed (None for random)
        """
        self.seed = seed
        self._original_seed = seed
    
    def set_seed(self, seed: Optional[int] = None):
        """
        Set random seed globally
        
        Args:
            seed: Random seed (uses stored seed if None)
        """
        if seed is None:
            seed = self.seed
        
        if seed is not None:
            # Python random
            import random
            random.seed(seed)
            
            # Numpy
            np.random.seed(seed)
            
            # Pandas (uses numpy internally, but set explicitly)
            # Note: pandas doesn't have direct seed, uses numpy
            
            logging.info(f"Random seed set to: {seed}")
        else:
            logging.info("No seed set - using random initialization")
    
    def get_seed(self) -> Optional[int]:
        """Get current seed"""
        return self.seed
    
    def reset_seed(self):
        """Reset to original seed"""
        self.set_seed(self._original_seed)
    
    def generate_seed(self) -> int:
        """Generate a new random seed"""
        import random
        new_seed = random.randint(0, 2**32 - 1)
        logging.info(f"Generated new seed: {new_seed}")
        return new_seed


class DataSanitizer:
    """
    Sanitizes and cleans data
    
    Handles missing values, outliers, invalid data, etc.
    """
    
    @staticmethod
    def remove_pii(
        data: pd.DataFrame,
        pii_columns: List[str],
        replacement: str = "[REDACTED]"
    ) -> pd.DataFrame:
        """
        Remove PII from data
        
        Args:
            data: DataFrame
            pii_columns: Columns containing PII
            replacement: Replacement value
            
        Returns:
            Sanitized DataFrame
        """
        sanitized = data.copy()
        
        for col in pii_columns:
            if col in sanitized.columns:
                sanitized[col] = replacement
        
        logging.info(f"Removed PII from {len(pii_columns)} columns")
        return sanitized
    
    @staticmethod
    def handle_missing_values(
        data: pd.DataFrame,
        strategy: str = "drop",
        fill_value: Any = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values
        
        Args:
            data: DataFrame
            strategy: 'drop', 'fill', 'forward', 'backward'
            fill_value: Value for fill strategy
            columns: Specific columns (None for all)
            
        Returns:
            Cleaned DataFrame
        """
        cleaned = data.copy()
        
        if columns is None:
            columns = cleaned.columns.tolist()
        
        if strategy == 'drop':
            cleaned = cleaned.dropna(subset=columns)
        elif strategy == 'fill':
            cleaned[columns] = cleaned[columns].fillna(fill_value)
        elif strategy == 'forward':
            cleaned[columns] = cleaned[columns].fillna(method='ffill')
        elif strategy == 'backward':
            cleaned[columns] = cleaned[columns].fillna(method='bfill')
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logging.info(f"Handled missing values using strategy: {strategy}")
        return cleaned
    
    @staticmethod
    def remove_outliers(
        data: pd.DataFrame,
        columns: List[str],
        method: str = "iqr",
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers from numeric columns
        
        Args:
            data: DataFrame
            columns: Columns to check
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame without outliers
        """
        cleaned = data.copy()
        
        for col in columns:
            if col not in cleaned.columns:
                continue
            
            if method == "iqr":
                Q1 = cleaned[col].quantile(0.25)
                Q3 = cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (cleaned[col] >= lower_bound) & (cleaned[col] <= upper_bound)
                cleaned = cleaned[mask]
            
            elif method == "zscore":
                from scipy import stats
                z_scores = np.abs(stats.zscore(cleaned[col].dropna()))
                mask = z_scores < threshold
                cleaned = cleaned[mask]
        
        removed = len(data) - len(cleaned)
        logging.info(f"Removed {removed} outliers using {method} method")
        
        return cleaned
    
    @staticmethod
    def normalize_text(
        data: pd.Series,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_extra_spaces: bool = True
    ) -> pd.Series:
        """
        Normalize text data
        
        Args:
            data: Text series
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            remove_extra_spaces: Remove extra whitespace
            
        Returns:
            Normalized series
        """
        normalized = data.copy()
        
        if lowercase:
            normalized = normalized.str.lower()
        
        if remove_punctuation:
            import string
            normalized = normalized.str.replace(f'[{string.punctuation}]', '', regex=True)
        
        if remove_extra_spaces:
            normalized = normalized.str.replace(r'\s+', ' ', regex=True).str.strip()
        
        return normalized
    
    @staticmethod
    def validate_data_types(
        data: pd.DataFrame,
        schema: Dict[str, str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate data types match schema
        
        Args:
            data: DataFrame to validate
            schema: Expected schema {column: type}
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for col, expected_type in schema.items():
            if col not in data.columns:
                errors.append(f"Missing column: {col}")
                continue
            
            actual_type = str(data[col].dtype)
            
            # Simple type matching
            type_matches = {
                'int': ['int64', 'int32', 'int16', 'int8'],
                'float': ['float64', 'float32', 'float16'],
                'string': ['object', 'string'],
                'datetime': ['datetime64'],
                'bool': ['bool']
            }
            
            if expected_type in type_matches:
                if actual_type not in type_matches[expected_type]:
                    errors.append(
                        f"Column {col}: expected {expected_type}, got {actual_type}"
                    )
        
        is_valid = len(errors) == 0
        return is_valid, errors


class PathManager:
    """
    Manages paths and directories
    
    Provides utilities for path operations
    """
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_project_root() -> Path:
        """Get project root directory"""
        # Assume this file is in src/
        return Path(__file__).parent.parent
    
    @staticmethod
    def get_data_dir() -> Path:
        """Get data directory"""
        return PathManager.get_project_root() / "data"
    
    @staticmethod
    def get_output_dir() -> Path:
        """Get output directory"""
        output_dir = PathManager.get_project_root() / "outputs"
        PathManager.ensure_dir(output_dir)
        return output_dir
    
    @staticmethod
    def get_temp_dir() -> Path:
        """Get temporary directory"""
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / "synthetic_data"
        PathManager.ensure_dir(temp_dir)
        return temp_dir
    
    @staticmethod
    def clean_filename(filename: str) -> str:
        """
        Clean filename to be filesystem-safe
        
        Args:
            filename: Original filename
            
        Returns:
            Cleaned filename
        """
        import re
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove leading/trailing spaces and dots
        filename = filename.strip('. ')
        return filename
    
    @staticmethod
    def get_unique_filename(
        directory: Union[str, Path],
        base_name: str,
        extension: str
    ) -> Path:
        """
        Get unique filename by adding counter if needed
        
        Args:
            directory: Directory path
            base_name: Base filename
            extension: File extension
            
        Returns:
            Unique filepath
        """
        directory = Path(directory)
        extension = extension if extension.startswith('.') else f'.{extension}'
        
        filepath = directory / f"{base_name}{extension}"
        
        if not filepath.exists():
            return filepath
        
        # Add counter
        counter = 1
        while True:
            filepath = directory / f"{base_name}_{counter}{extension}"
            if not filepath.exists():
                return filepath
            counter += 1


# Convenience functions
def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
):
    """
    Quick logging setup
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    LoggerConfig.setup_logger(level=level, log_file=log_file)


def set_global_seed(seed: int):
    """
    Quick global seed setup
    
    Args:
        seed: Random seed
    """
    manager = SeedManager(seed)
    manager.set_seed()


def read_data(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Quick data reading
    
    Args:
        filepath: File path
        **kwargs: Additional arguments
        
    Returns:
        DataFrame
    """
    return FileHandler.read_file(filepath, **kwargs)


def write_data(data: pd.DataFrame, filepath: Union[str, Path], **kwargs):
    """
    Quick data writing
    
    Args:
        data: DataFrame
        filepath: Output path
        **kwargs: Additional arguments
    """
    FileHandler.write_file(data, filepath, **kwargs)


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Testing utilities...")
    
    # Test seed management
    seed_manager = SeedManager(42)
    seed_manager.set_seed()
    
    # Generate some random data
    data = pd.DataFrame({
        'a': np.random.randn(100),
        'b': np.random.randn(100),
        'c': ['test'] * 100
    })
    
    logger.info(f"Generated data shape: {data.shape}")
    
    # Test file I/O
    output_path = PathManager.get_temp_dir() / "test_data.csv"
    write_data(data, output_path)
    logger.info(f"Written to: {output_path}")
    
    # Read back
    data_read = read_data(output_path)
    logger.info(f"Read data shape: {data_read.shape}")
    
    # Get file info
    info = FileHandler.get_file_info(output_path)
    logger.info(f"File info: {info}")
    
    # Test sanitizer
    sanitizer = DataSanitizer()
    
    # Add some missing values
    data_with_missing = data.copy()
    data_with_missing.loc[0:10, 'a'] = np.nan
    
    cleaned = sanitizer.handle_missing_values(data_with_missing, strategy='drop')
    logger.info(f"Cleaned data shape: {cleaned.shape}")
    
    # Test path management
    logger.info(f"Project root: {PathManager.get_project_root()}")
    logger.info(f"Data dir: {PathManager.get_data_dir()}")
    logger.info(f"Output dir: {PathManager.get_output_dir()}")
    
    logger.info("All tests passed!")
