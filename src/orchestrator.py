"""
Data Generation Orchestrator Module

Main orchestration engine that coordinates schema analysis, pipeline routing,
data generation, and validation across all data types.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import time
from datetime import datetime

from .config import Config

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Enumeration of supported data types"""
    NUMERIC = "numeric"
    TEXT = "text"
    CATEGORICAL = "categorical"
    PII_NAME = "pii_name"
    PII_EMAIL = "pii_email"
    PII_PHONE = "pii_phone"
    PII_ADDRESS = "pii_address"
    PII_ID = "pii_id"
    TEMPORAL = "temporal"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class PipelineType(Enum):
    """Enumeration of generation pipelines"""
    NUMERIC = "numeric"
    TEXT = "text"
    PII = "pii"
    TEMPORAL = "temporal"
    HYBRID = "hybrid"


@dataclass
class ColumnMetadata:
    """Metadata for a single column"""
    name: str
    data_type: DataType
    pipeline: PipelineType

    nullable: bool = False
    null_percentage: float = 0.0
    unique_count: Optional[int] = None
    sample_values: List[Any] = field(default_factory=list)

    # Numeric metadata
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    skewness: Optional[float] = None   # ✅ added (used by numeric.py)

    # Categorical metadata
    categories: Optional[List[Any]] = None
    category_frequencies: Optional[Dict[Any, float]] = None

    # Text metadata
    avg_length: Optional[float] = None
    max_length: Optional[int] = None

    # PII metadata
    pii_type: Optional[str] = None

    # Temporal metadata
    date_format: Optional[str] = None
    min_date: Optional[str] = None
    max_date: Optional[str] = None

    # Internal dtype tracking
    pandas_dtype: Optional[str] = None

    # -------------------------------------------------
    # ✅ Compatibility layer for generators
    # -------------------------------------------------

    @property
    def is_integer(self) -> bool:
        """Used by NumericGenerator"""
        if self.pandas_dtype and "int" in self.pandas_dtype.lower():
            return True

        if self.data_type == DataType.NUMERIC:
            if self.min_value is not None and self.max_value is not None:
                if float(self.min_value).is_integer() and float(self.max_value).is_integer():
                    return True

        return False

    @property
    def dtype(self) -> str:
        """Backward compatibility alias"""
        return self.pandas_dtype or self.data_type.value

    @property
    def inferred_type(self) -> str:
        """Backward compatibility alias"""
        return self.data_type.value

@dataclass
class DatasetSchema:
    """Complete schema for the dataset"""
    columns: Dict[str, ColumnMetadata] = field(default_factory=dict)
    num_rows: int = 0
    correlations: Optional[pd.DataFrame] = None
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    
    def add_column(self, metadata: ColumnMetadata):
        """Add column metadata to schema"""
        self.columns[metadata.name] = metadata
    
    def get_column(self, name: str) -> Optional[ColumnMetadata]:
        """Get column metadata by name"""
        return self.columns.get(name)
    
    def get_columns_by_type(self, data_type: DataType) -> List[str]:
        """Get all column names of a specific data type"""
        return [name for name, meta in self.columns.items() if meta.data_type == data_type]
    
    def get_columns_by_pipeline(self, pipeline: PipelineType) -> List[str]:
        """Get all column names using a specific pipeline"""
        return [name for name, meta in self.columns.items() if meta.pipeline == pipeline]


@dataclass
class GenerationResult:
    """Result of data generation"""
    data: pd.DataFrame
    schema: DatasetSchema
    generation_time: float
    validation_results: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GenerationPipeline:
    """
    Abstract base class for generation pipelines
    
    Each pipeline is responsible for generating specific types of data
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.pipeline_type = PipelineType.HYBRID
    

    def generate(
        self, 
        schema: DatasetSchema, 
        column_names: List[str],
        num_rows: int,
        reference_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate data for specified columns
        
        Args:
            schema: Dataset schema with column metadata
            column_names: List of columns to generate
            num_rows: Number of rows to generate
            reference_data: Optional reference data for pattern learning
            
        Returns:
            DataFrame with generated columns
        """
        raise NotImplementedError("Subclasses must implement generate()")
    
    def validate(self, generated_data: pd.DataFrame, schema: DatasetSchema) -> Dict[str, Any]:
        """
        Validate generated data against schema
        
        Args:
            generated_data: Generated data to validate
            schema: Expected schema
            
        Returns:
            Validation results dictionary
        """
        return {"valid": True, "errors": []}


class BatchProcessor:
    """
    Handles batch processing of large datasets
    
    Splits generation into manageable batches to avoid memory issues
    and enables progress tracking
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.batch_size = config.generation.batch_size
        self.enable_parallel = config.generation.enable_parallel
        self.max_workers = config.generation.max_workers
    
    def process_batches(
        self,
        total_rows: int,
        generator_func: Callable[[int, int], pd.DataFrame],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> pd.DataFrame:
        """
        Process data generation in batches
        
        Args:
            total_rows: Total number of rows to generate
            generator_func: Function that generates data (start_idx, num_rows) -> DataFrame
            progress_callback: Optional callback for progress updates (current, total)
            
        Returns:
            Complete generated DataFrame
        """
        num_batches = (total_rows + self.batch_size - 1) // self.batch_size
        batches = []
        
        logger.info(f"Processing {total_rows} rows in {num_batches} batches")
        
        if self.enable_parallel and num_batches > 1:
            batches = self._process_parallel(total_rows, generator_func, progress_callback)
        else:
            batches = self._process_sequential(total_rows, generator_func, progress_callback)
        
        # Concatenate all batches
        if batches:
            result = pd.concat(batches, ignore_index=True)
            logger.info(f"Generated {len(result)} rows successfully")
            return result
        else:
            raise ValueError("No data generated")
    
    def _process_sequential(
        self,
        total_rows: int,
        generator_func: Callable,
        progress_callback: Optional[Callable]
    ) -> List[pd.DataFrame]:
        """Process batches sequentially"""
        batches = []
        start_idx = 0
        
        while start_idx < total_rows:
            batch_rows = min(self.batch_size, total_rows - start_idx)
            
            try:
                batch_data = generator_func(start_idx, batch_rows)
                batches.append(batch_data)
                
                if progress_callback:
                    progress_callback(start_idx + batch_rows, total_rows)
                
                start_idx += batch_rows
                
            except Exception as e:
                logger.error(f"Error generating batch at {start_idx}: {e}")
                raise
        
        return batches
    
    def _process_parallel(
        self,
        total_rows: int,
        generator_func: Callable,
        progress_callback: Optional[Callable]
    ) -> List[pd.DataFrame]:
        """Process batches in parallel"""
        batches = [None] * ((total_rows + self.batch_size - 1) // self.batch_size)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            # Submit all batch jobs
            start_idx = 0
            batch_idx = 0
            while start_idx < total_rows:
                batch_rows = min(self.batch_size, total_rows - start_idx)
                future = executor.submit(generator_func, start_idx, batch_rows)
                futures[future] = (batch_idx, start_idx, batch_rows)
                start_idx += batch_rows
                batch_idx += 1
            
            # Collect results as they complete
            for future in as_completed(futures):
                batch_idx, start, rows = futures[future]
                try:
                    batch_data = future.result()
                    batches[batch_idx] = batch_data
                    completed += rows
                    
                    if progress_callback:
                        progress_callback(completed, total_rows)
                    
                except Exception as e:
                    logger.error(f"Error in parallel batch {batch_idx}: {e}")
                    raise
        
        return batches


class DataOrchestrator:

    INDIAN_CITY_SURNAME_MAP = {
        'kolkata': ['Banerjee', 'Chatterjee', 'Mukherjee', 'Bose', 'Ganguly'],
        'howrah': ['Banerjee', 'Chatterjee', 'Mukherjee', 'Bose', 'Ganguly'],
        'pune': ['Patil', 'Deshmukh', 'Kulkarni', 'Joshi', 'Shinde'],
        'mumbai': ['Patil', 'Deshmukh', 'Kulkarni', 'Joshi', 'Shah'],
        'chennai': ['Iyer', 'Iyengar', 'Subramanian', 'Ramanathan', 'Narayanan'],
        'coimbatore': ['Iyer', 'Iyengar', 'Subramanian', 'Ramanathan', 'Narayanan'],
        'madurai': ['Iyer', 'Iyengar', 'Subramanian', 'Ramanathan', 'Narayanan'],
        'kochi': ['Nair', 'Menon', 'Pillai', 'Kurup', 'Varma'],
        'thiruvananthapuram': ['Nair', 'Menon', 'Pillai', 'Kurup', 'Varma'],
        'delhi': ['Sharma', 'Gupta', 'Singh', 'Verma', 'Malhotra'],
        'new delhi': ['Sharma', 'Gupta', 'Singh', 'Verma', 'Malhotra'],
        'jaipur': ['Sharma', 'Singh', 'Mehta', 'Jain', 'Agarwal'],
        'lucknow': ['Srivastava', 'Tiwari', 'Shukla', 'Verma', 'Mishra'],
        'patna': ['Kumar', 'Sinha', 'Jha', 'Singh', 'Prasad'],
        'noida': ['Sharma', 'Gupta', 'Singh', 'Verma', 'Agarwal'],
        'gurgaon': ['Sharma', 'Gupta', 'Singh', 'Verma', 'Agarwal'],
        'hyderabad': ['Reddy', 'Rao', 'Naidu', 'Reddy', 'Rao'],
        'bengaluru': ['Reddy', 'Rao', 'Gowda', 'Shetty', 'Hegde'],
    }

    """
    Main orchestrator for synthetic data generation
    
    Coordinates schema analysis, pipeline routing, generation, and validation
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the orchestrator
        
        Args:
            config: Configuration object (uses default if None)
        """
        from .config import get_default_config
        
        self.config = config or get_default_config()
        self.schema: Optional[DatasetSchema] = None
        self.pipelines: Dict[PipelineType, GenerationPipeline] = {}
        self.batch_processor = BatchProcessor(self.config)
        
        # Set random seed if specified
        if self.config.generation.seed is not None:
            np.random.seed(self.config.generation.seed)
        
        logger.info("DataOrchestrator initialized")
    
    def register_pipeline(self, pipeline_type: PipelineType, pipeline: GenerationPipeline):
        """
        Register a generation pipeline
        
        Args:
            pipeline_type: Type of pipeline
            pipeline: Pipeline instance
        """
        self.pipelines[pipeline_type] = pipeline
        logger.info(f"Registered pipeline: {pipeline_type.value}")
    
    def analyze_schema(
        self, 
        reference_data: pd.DataFrame,
        column_hints: Optional[Dict[str, str]] = None
    ) -> DatasetSchema:
        """
        Analyze reference data to build schema
        
        Args:
            reference_data: Reference dataset to analyze
            column_hints: Optional hints for column types
            
        Returns:
            Dataset schema
        """
        logger.info("Analyzing schema from reference data")
        
        schema = DatasetSchema()
        schema.num_rows = len(reference_data)
        
        # Analyze each column
        for col in reference_data.columns:
            metadata = self._analyze_column(reference_data[col], col, column_hints)
            schema.add_column(metadata)
        
        # Calculate correlations for numeric columns
        numeric_cols = schema.get_columns_by_type(DataType.NUMERIC)
        if numeric_cols and len(numeric_cols) > 1:
            schema.correlations = reference_data[numeric_cols].corr()
        
        self.schema = schema
        logger.info(f"Schema analysis complete: {len(schema.columns)} columns")
        
        return schema
    
    def _analyze_column(
        self, 
        series: pd.Series, 
        column_name: str,
        column_hints: Optional[Dict[str, str]] = None
    ) -> ColumnMetadata:
        """Analyze a single column to extract metadata"""
        
        # Basic metadata
        metadata = ColumnMetadata(
            name=column_name,
            data_type=DataType.UNKNOWN,
            pipeline=PipelineType.HYBRID,
            nullable=series.isna().any(),
            null_percentage=series.isna().sum() / len(series) if len(series) > 0 else 0,
            unique_count=series.nunique(),
            sample_values=series.dropna().sample(min(5, len(series.dropna()))).tolist() if len(series.dropna()) > 0 else []
        )
        
        # Apply hints if provided
        if column_hints and column_name in column_hints:
            hint = column_hints[column_name].lower()
            metadata.data_type = self._hint_to_datatype(hint)
            metadata.pipeline = self._datatype_to_pipeline(metadata.data_type)
            return metadata
        
        # Infer data type
        metadata.data_type = self._infer_datatype(series, column_name=column_name)
        metadata.pipeline = self._datatype_to_pipeline(metadata.data_type)
        
        # Extract type-specific metadata
        # if metadata.data_type == DataType.NUMERIC:
        #     metadata.min_value = float(series.min())
        #     metadata.max_value = float(series.max())
        #     metadata.mean = float(series.mean())
        #     metadata.std = float(series.std())
        if metadata.data_type == DataType.NUMERIC:
            metadata.min_value = float(series.min())
            metadata.max_value = float(series.max())
            metadata.mean = float(series.mean())
            metadata.std = float(series.std())
            metadata.skewness = float(series.skew()) if len(series) > 2 else 0.0
            metadata.pandas_dtype = str(series.dtype)
        
        elif metadata.data_type == DataType.CATEGORICAL:
            value_counts = series.value_counts()
            metadata.categories = value_counts.index.tolist()
            metadata.category_frequencies = (value_counts / len(series)).to_dict()
        
        elif metadata.data_type == DataType.TEXT:
            str_series = series.astype(str)
            lengths = str_series.str.len()
            metadata.avg_length = float(lengths.mean())
            metadata.max_length = int(lengths.max())
        
        elif metadata.data_type == DataType.TEMPORAL:
            try:
                dt_series = pd.to_datetime(series)
                metadata.min_date = str(dt_series.min())
                metadata.max_date = str(dt_series.max())
                # Try to infer date format
                if len(series.dropna()) > 0:
                    sample = str(series.dropna().iloc[0])
                    metadata.date_format = self._infer_date_format(sample)
            except:
                pass
        
        return metadata
    
    def _infer_datatype(self, series: pd.Series, column_name: Optional[str] = None) -> DataType:
        """Infer the data type of a column"""
        
        # Drop NaN values for analysis
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return DataType.UNKNOWN
        
        normalized_col = (column_name or '').lower()
        if any(token in normalized_col for token in ['email', 'e-mail']):
            return DataType.PII_EMAIL
        if any(token in normalized_col for token in ['first_name', 'firstname', 'last_name', 'lastname', 'full_name', 'name']):
            return DataType.PII_NAME
        if any(token in normalized_col for token in ['phone', 'mobile']):
            return DataType.PII_PHONE
        if 'address' in normalized_col:
            return DataType.PII_ADDRESS
        if 'city' in normalized_col or 'country' in normalized_col:
            return DataType.CATEGORICAL

        # Check for boolean
        if series.dtype == bool or set(clean_series.unique()).issubset({0, 1, True, False}):
            return DataType.BOOLEAN
        
        # Check for numeric
        if pd.api.types.is_numeric_dtype(series):
            return DataType.NUMERIC
        
        # Check for temporal
        try:
            pd.to_datetime(clean_series)
            return DataType.TEMPORAL
        except:
            pass
        
        # Convert to string for pattern matching
        str_series = clean_series.astype(str)
        
        # Check for PII patterns
        pii_type = self._detect_pii(str_series)
        if pii_type:
            return pii_type
        
        # Check for categorical (low cardinality)
        cardinality_ratio = len(clean_series.unique()) / len(clean_series)
        if cardinality_ratio < 0.05 or len(clean_series.unique()) < 20:
            return DataType.CATEGORICAL
        
        # Default to text
        return DataType.TEXT
    
    def _detect_pii(self, series: pd.Series) -> Optional[DataType]:
        """Detect PII patterns in string data"""
        import re
        
        # Sample a few values
        sample = series.sample(min(100, len(series))).tolist()
        
        # Email pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if sum(bool(re.match(email_pattern, str(val))) for val in sample) / len(sample) > 0.8:
            return DataType.PII_EMAIL
        
        # Phone pattern (simple)
        phone_pattern = r'^[\d\s\-\(\)]{10,}$'
        if sum(bool(re.match(phone_pattern, str(val))) for val in sample) / len(sample) > 0.8:
            return DataType.PII_PHONE
        
        # Name pattern (capitalized words)
        name_pattern = r'^[A-Z][a-z]+(\s[A-Z][a-z]+)+$'
        if sum(bool(re.match(name_pattern, str(val))) for val in sample) / len(sample) > 0.6:
            return DataType.PII_NAME
        
        return None
    
    def _hint_to_datatype(self, hint: str) -> DataType:
        """Convert string hint to DataType enum"""
        mapping = {
            'numeric': DataType.NUMERIC,
            'text': DataType.TEXT,
            'categorical': DataType.CATEGORICAL,
            'name': DataType.PII_NAME,
            'email': DataType.PII_EMAIL,
            'phone': DataType.PII_PHONE,
            'address': DataType.PII_ADDRESS,
            'id': DataType.PII_ID,
            'date': DataType.TEMPORAL,
            'datetime': DataType.TEMPORAL,
            'boolean': DataType.BOOLEAN,
        }
        return mapping.get(hint, DataType.UNKNOWN)
    
    def _datatype_to_pipeline(self, data_type: DataType) -> PipelineType:
        """Map data type to appropriate pipeline"""
        mapping = {
            DataType.NUMERIC: PipelineType.NUMERIC,
            DataType.TEXT: PipelineType.TEXT,
            DataType.CATEGORICAL: PipelineType.HYBRID,
            DataType.PII_NAME: PipelineType.PII,
            DataType.PII_EMAIL: PipelineType.PII,
            DataType.PII_PHONE: PipelineType.PII,
            DataType.PII_ADDRESS: PipelineType.PII,
            DataType.PII_ID: PipelineType.PII,
            DataType.TEMPORAL: PipelineType.TEMPORAL,
            DataType.BOOLEAN: PipelineType.NUMERIC,
        }
        return mapping.get(data_type, PipelineType.HYBRID)
    
    def _infer_date_format(self, date_string: str) -> str:
        """Infer date format from a sample string"""
        common_formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%Y/%m/%d',
            '%Y-%m-%d %H:%M:%S',
            '%d-%m-%Y',
        ]
        
        for fmt in common_formats:
            try:
                datetime.strptime(date_string, fmt)
                return fmt
            except:
                continue
        
        return '%Y-%m-%d'  # Default
    

    def _enforce_output_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply final consistency rules for common quality issues."""
        if data.empty:
            return data

        fixed = data.copy()

        # Enforce integer typing for age/serial/id-like fields.
        for col in fixed.columns:
            col_lower = col.lower()
            if any(token in col_lower for token in ['age', 'sl no', 'serial', 'record_id', 'id']):
                numeric = pd.to_numeric(fixed[col], errors='coerce')
                if numeric.notna().any():
                    numeric = numeric.fillna(numeric.median() if numeric.notna().any() else 0)
                    numeric = np.round(numeric).astype(int)
                    if any(token in col_lower for token in ['id', 'serial', 'sl no']) and (numeric <= 0).any():
                        numeric = np.arange(1, len(fixed) + 1)
                    fixed[col] = numeric

        # Enforce name->email consistency when columns exist.
        first_col = next((c for c in fixed.columns if 'first_name' in c.lower()), None)
        last_col = next((c for c in fixed.columns if 'last_name' in c.lower()), None)
        email_col = next((c for c in fixed.columns if 'email' in c.lower()), None)
        if first_col and last_col and email_col:
            domains = []
            for value in fixed[email_col].astype(str):
                parts = value.split('@')
                domains.append(parts[1] if len(parts) == 2 else 'test.com')

            new_emails = []
            for idx, (first, last) in enumerate(zip(fixed[first_col], fixed[last_col])):
                first_clean = ''.join(ch for ch in str(first).lower() if ch.isalpha()) or 'user'
                last_clean = ''.join(ch for ch in str(last).lower() if ch.isalpha()) or 'name'
                domain = domains[idx] if idx < len(domains) else 'test.com'
                new_emails.append(f"{first_clean}.{last_clean}{idx+1}@{domain}")
            fixed[email_col] = new_emails

        # Enforce city-country consistency when both columns exist.
        country_col = next((c for c in fixed.columns if 'country' in c.lower()), None)
        city_col = next((c for c in fixed.columns if 'city' in c.lower()), None)
        if country_col and city_col:
            try:
                from .generators.categorical import CategoricalGenerator
                aliases = CategoricalGenerator.COUNTRY_ALIASES
                country_city_map = {
                    (aliases.get(k.lower(), k) if isinstance(k, str) else k): v
                    for k, v in CategoricalGenerator.DEFAULT_COUNTRY_CITY_MAP.items()
                }

                for idx, (country, city) in enumerate(zip(fixed[country_col], fixed[city_col])):
                    normalized_country = aliases.get(str(country).strip().lower(), str(country).strip())
                    valid_cities = country_city_map.get(normalized_country)
                    if valid_cities and city not in valid_cities:
                        fixed.at[idx, city_col] = np.random.choice(valid_cities)
            except Exception as e:
                logger.debug(f"Skipped city-country enforcement: {e}")

        # Enforce gender consistency with first names when both columns exist.
        gender_col = next((c for c in fixed.columns if 'gender' in c.lower()), None)
        if first_col and gender_col:
            try:
                from .generators.pii import NameGenerator

                male_names = set()
                female_names = set()
                for locale_data in NameGenerator.FIRST_NAMES.values():
                    male_names.update(name.lower() for name in locale_data.get('male', []))
                    female_names.update(name.lower() for name in locale_data.get('female', []))

                for idx, first_name in enumerate(fixed[first_col].astype(str)):
                    normalized_first = first_name.strip().lower()
                    if normalized_first in male_names and normalized_first not in female_names:
                        fixed.at[idx, gender_col] = 'Male'
                    elif normalized_first in female_names and normalized_first not in male_names:
                        fixed.at[idx, gender_col] = 'Female'
            except Exception as e:
                logger.debug(f"Skipped gender consistency enforcement: {e}")

        # Enforce Indian city-surname consistency where possible.
        if city_col:
            full_name_col = next((c for c in fixed.columns if c.lower() == 'name' or 'full_name' in c.lower()), None)
            if last_col or full_name_col:
                try:
                    for idx, city in enumerate(fixed[city_col].astype(str)):
                        surname_pool = self.INDIAN_CITY_SURNAME_MAP.get(city.strip().lower())
                        if not surname_pool:
                            continue

                        selected_surname = np.random.choice(surname_pool)
                        if last_col:
                            fixed.at[idx, last_col] = selected_surname

                        if full_name_col:
                            raw_name = str(fixed.at[idx, full_name_col]).strip()
                            parts = raw_name.split()
                            if len(parts) >= 2:
                                parts[-1] = selected_surname
                                fixed.at[idx, full_name_col] = ' '.join(parts)
                            elif first_col:
                                first_name = str(fixed.at[idx, first_col]).strip() if first_col else 'User'
                                fixed.at[idx, full_name_col] = f"{first_name} {selected_surname}"
                except Exception as e:
                    logger.debug(f"Skipped Indian city-surname enforcement: {e}")

        return fixed

    def generate(
        self,
        num_rows: Optional[int] = None,
        reference_data: Optional[pd.DataFrame] = None,
        schema: Optional[DatasetSchema] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> GenerationResult:
        """
        Generate synthetic data
        
        Args:
            num_rows: Number of rows to generate (overrides config)
            reference_data: Optional reference data for pattern learning
            schema: Optional pre-computed schema (will analyze if not provided)
            progress_callback: Optional callback for progress updates
            
        Returns:
            GenerationResult with generated data and metadata
        """
        start_time = time.time()
        
        # Determine number of rows
        if num_rows is None:
            num_rows = self.config.generation.num_rows
        
        # Get or create schema
        if schema is None:
            if reference_data is not None:
                schema = self.analyze_schema(reference_data)
            elif self.schema is not None:
                schema = self.schema
            else:
                raise ValueError("Must provide either reference_data or schema")
        
        logger.info(f"Generating {num_rows} rows of synthetic data")
        
        # Group columns by pipeline
        pipeline_columns = {}
        for pipeline_type in PipelineType:
            cols = schema.get_columns_by_pipeline(pipeline_type)
            if cols:
                pipeline_columns[pipeline_type] = cols
        
        # Generate data using batch processor
        def generate_batch(start_idx: int, batch_rows: int) -> pd.DataFrame:
            batch_data = pd.DataFrame()
            
            for pipeline_type, columns in pipeline_columns.items():
                if pipeline_type not in self.pipelines:
                    logger.warning(f"Pipeline {pipeline_type} not registered, skipping columns: {columns}")
                    continue
                
                pipeline = self.pipelines[pipeline_type]
                try:
                    pipeline_data = pipeline.generate(
                        schema=schema,
                        column_names=columns,
                        num_rows=batch_rows,
                        reference_data=reference_data
                    )
                    batch_data = pd.concat([batch_data, pipeline_data], axis=1)
                except Exception as e:
                    logger.error(f"Error in {pipeline_type} pipeline: {e}")
                    raise
            
            return batch_data
        
        # Process batches
        generated_data = self.batch_processor.process_batches(
            total_rows=num_rows,
            generator_func=generate_batch,
            progress_callback=progress_callback
        )
        
        generation_time = time.time() - start_time

        generated_data = self._enforce_output_consistency(generated_data)

        # Preserve original input/schema column sequence.
        ordered_columns = [col for col in schema.columns.keys() if col in generated_data.columns]
        trailing_columns = [col for col in generated_data.columns if col not in ordered_columns]
        generated_data = generated_data[ordered_columns + trailing_columns]
        
        # Create result
        result = GenerationResult(
            data=generated_data,
            schema=schema,
            generation_time=generation_time,
            metadata={
                'num_rows': num_rows,
                'num_columns': len(generated_data.columns),
                'pipelines_used': list(pipeline_columns.keys()),
                'timestamp': datetime.now().isoformat(),
            }
        )
        
        logger.info(f"Generation completed in {generation_time:.2f}s")
        
        return result
    
    def validate_result(self, result: GenerationResult) -> Dict[str, Any]:
        """
        Validate generated data
        
        Args:
            result: Generation result to validate
            
        Returns:
            Validation results dictionary
        """
        if not self.config.validation.enable_quality_checks:
            return {"validation_enabled": False}
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "quality": {},
            "privacy": {},
            "overall_score": 0.0,
            "passed": False
        }
        
        # Validate with each pipeline
        for pipeline_type, pipeline in self.pipelines.items():
            columns = result.schema.get_columns_by_pipeline(pipeline_type)
            if columns:
                pipeline_results = pipeline.validate(
                    result.data[columns], 
                    result.schema
                )
                validation_results["quality"][pipeline_type.value] = pipeline_results
        
        # Calculate overall score
        # TODO: Implement comprehensive validation scoring
        validation_results["overall_score"] = 0.85
        validation_results["passed"] = validation_results["overall_score"] >= self.config.validation.quality_threshold
        
        result.validation_results = validation_results
        
        return validation_results


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 15000, 100),
        'name': ['John Doe'] * 100,
        'email': ['john@example.com'] * 100,
        'city': np.random.choice(['NYC', 'LA', 'Chicago'], 100),
    })
    
    # Create orchestrator
    from .config import get_default_config
    config = get_default_config()
    config.generation.num_rows = 200
    
    orchestrator = DataOrchestrator(config)
    
    # Analyze schema
    schema = orchestrator.analyze_schema(sample_data)
    
    print("\nSchema Analysis:")
    for col_name, col_meta in schema.columns.items():
        print(f"  {col_name}: {col_meta.data_type.value} -> {col_meta.pipeline.value}")
    
    print(f"\nTotal rows in reference: {schema.num_rows}")
