"""
Configuration Management Module

Handles loading, validation, and merging of configuration files
with support for presets and user-defined overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for data generation parameters"""
    num_rows: int = 1000
    seed: Optional[int] = None
    batch_size: int = 100
    enable_parallel: bool = True
    max_workers: int = 4
    

@dataclass
class NumericConfig:
    """Configuration for numeric data generation"""
    method: str = "distribution"  # distribution, correlation, sdv
    preserve_correlations: bool = True
    distribution_fitting: str = "auto"  # auto, normal, uniform, exponential
    outlier_handling: str = "preserve"  # preserve, remove, clip
    decimal_places: Optional[int] = None
    range_enforcement: bool = True


@dataclass
class TextConfig:
    """Configuration for text data generation"""
    llm_provider: str = "anthropic"  # anthropic, openai, huggingface
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 500
    temperature: float = 0.7
    use_templates: bool = False
    preserve_length: bool = True
    length_variance: float = 0.2  # ±20% of original length


@dataclass
class PIIConfig:
    """Configuration for PII generation"""
    locale: str = "en_US"
    gender_distribution: Dict[str, float] = field(default_factory=lambda: {"male": 0.5, "female": 0.5})
    anonymization_level: str = "full"  # full, partial, none
    consistency_check: bool = True
    realistic_patterns: bool = True


@dataclass
class TemporalConfig:
    """Configuration for temporal data generation"""
    preserve_patterns: bool = True  # weekends, seasonality
    business_days_only: bool = False
    timezone: str = "UTC"
    date_format: Optional[str] = None


@dataclass
class ValidationConfig:
    """Configuration for validation and quality checks"""
    enable_quality_checks: bool = True
    enable_privacy_checks: bool = True
    quality_threshold: float = 0.8
    privacy_threshold: float = 0.7
    k_anonymity: int = 5
    fail_on_validation: bool = False
    

@dataclass
class KnowledgeConfig:
    """Configuration for domain knowledge integration"""
    enable_rag: bool = False
    knowledge_base_path: Optional[str] = None
    domain: Optional[str] = None  # fashion, finance, healthcare, retail
    enforce_constraints: bool = True


@dataclass
class Config:
    """Main configuration class combining all sub-configurations"""
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    numeric: NumericConfig = field(default_factory=NumericConfig)
    text: TextConfig = field(default_factory=TextConfig)
    pii: PIIConfig = field(default_factory=PIIConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    
    # Custom column-level overrides
    column_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def merge(self, other: 'Config') -> 'Config':
        """Merge another configuration into this one (other takes precedence)"""
        merged = deepcopy(self)
        
        # Merge each sub-config
        for key in ['generation', 'numeric', 'text', 'pii', 'temporal', 'validation', 'knowledge']:
            if hasattr(other, key):
                other_config = getattr(other, key)
                merged_config = getattr(merged, key)
                
                # Update non-None values
                for field_name, field_value in asdict(other_config).items():
                    if field_value is not None:
                        setattr(merged_config, field_name, field_value)
        
        # Merge column configs
        merged.column_configs.update(other.column_configs)
        
        return merged


class ConfigLoader:
    """Loads and manages configuration from various sources"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the configuration loader
        
        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            # Default to data/presets directory
            self.config_dir = Path(__file__).parent.parent / "data" / "presets"
        else:
            self.config_dir = Path(config_dir)
        
        self.presets = self._load_presets()
    
    def _load_presets(self) -> Dict[str, Config]:
        """Load all available preset configurations"""
        presets = {}
        
        if not self.config_dir.exists():
            logger.warning(f"Config directory not found: {self.config_dir}")
            return presets
        
        for preset_file in self.config_dir.glob("*.yaml"):
            preset_name = preset_file.stem
            try:
                presets[preset_name] = self.load_from_file(preset_file)
                logger.info(f"Loaded preset: {preset_name}")
            except Exception as e:
                logger.error(f"Failed to load preset {preset_name}: {e}")
        
        return presets
    
    def load_from_file(self, filepath: Union[str, Path]) -> Config:
        """
        Load configuration from a YAML file
        
        Args:
            filepath: Path to the YAML configuration file
            
        Returns:
            Config object
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return self._dict_to_config(config_dict)
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> Config:
        """
        Load configuration from a dictionary
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config object
        """
        return self._dict_to_config(config_dict)
    
    def load_preset(self, preset_name: str) -> Config:
        """
        Load a preset configuration by name
        
        Args:
            preset_name: Name of the preset (e.g., 'default', 'analytics')
            
        Returns:
            Config object
        """
        if preset_name not in self.presets:
            available = ", ".join(self.presets.keys())
            raise ValueError(f"Preset '{preset_name}' not found. Available: {available}")
        
        return deepcopy(self.presets[preset_name])
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> Config:
        """Convert dictionary to Config object"""
        config = Config()
        
        # Map dictionary keys to config objects
        config_mapping = {
            'generation': GenerationConfig,
            'numeric': NumericConfig,
            'text': TextConfig,
            'pii': PIIConfig,
            'temporal': TemporalConfig,
            'validation': ValidationConfig,
            'knowledge': KnowledgeConfig,
        }
        
        for key, config_class in config_mapping.items():
            if key in config_dict:
                setattr(config, key, config_class(**config_dict[key]))
        
        # Handle column-level configs
        if 'column_configs' in config_dict:
            config.column_configs = config_dict['column_configs']
        
        return config
    
    def merge_configs(self, base: Config, override: Union[Config, Dict[str, Any], str]) -> Config:
        """
        Merge configurations with override taking precedence
        
        Args:
            base: Base configuration
            override: Override configuration (Config object, dict, or preset name)
            
        Returns:
            Merged Config object
        """
        if isinstance(override, str):
            # Load preset
            override = self.load_preset(override)
        elif isinstance(override, dict):
            # Convert dict to Config
            override = self.load_from_dict(override)
        
        return base.merge(override)
    
    def save_config(self, config: Config, filepath: Union[str, Path]):
        """
        Save configuration to a YAML file
        
        Args:
            config: Configuration to save
            filepath: Path to save the file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.to_dict()
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to: {filepath}")
    
    def list_presets(self) -> List[str]:
        """Get list of available preset names"""
        return list(self.presets.keys())


class ConfigValidator:
    """Validates configuration parameters"""
    
    @staticmethod
    def validate(config: Config) -> tuple[bool, List[str]]:
        """
        Validate configuration parameters
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate generation config
        if config.generation.num_rows <= 0:
            errors.append("num_rows must be positive")
        
        if config.generation.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if config.generation.max_workers <= 0:
            errors.append("max_workers must be positive")
        
        # Validate numeric config
        valid_methods = ["distribution", "correlation", "sdv"]
        if config.numeric.method not in valid_methods:
            errors.append(f"numeric.method must be one of {valid_methods}")
        
        valid_distributions = ["auto", "normal", "uniform", "exponential", "lognormal", "gamma"]
        if config.numeric.distribution_fitting not in valid_distributions:
            errors.append(f"numeric.distribution_fitting must be one of {valid_distributions}")
        
        # Validate text config
        valid_providers = ["anthropic", "openai", "huggingface", "local"]
        if config.text.llm_provider not in valid_providers:
            errors.append(f"text.llm_provider must be one of {valid_providers}")
        
        if config.text.max_tokens <= 0:
            errors.append("text.max_tokens must be positive")
        
        if not 0 <= config.text.temperature <= 2:
            errors.append("text.temperature must be between 0 and 2")
        
        if not 0 <= config.text.length_variance <= 1:
            errors.append("text.length_variance must be between 0 and 1")
        
        # Validate PII config
        valid_locales = ["en_US", "en_GB", "fr_FR", "de_DE", "es_ES", "ja_JP", "zh_CN"]
        if config.pii.locale not in valid_locales:
            errors.append(f"pii.locale must be one of {valid_locales}")
        
        # Check gender distribution sums to 1
        gender_sum = sum(config.pii.gender_distribution.values())
        if not (0.99 <= gender_sum <= 1.01):  # Allow small floating point errors
            errors.append("pii.gender_distribution must sum to 1.0")
        
        # Validate validation config
        if not 0 <= config.validation.quality_threshold <= 1:
            errors.append("validation.quality_threshold must be between 0 and 1")
        
        if not 0 <= config.validation.privacy_threshold <= 1:
            errors.append("validation.privacy_threshold must be between 0 and 1")
        
        if config.validation.k_anonymity < 2:
            errors.append("validation.k_anonymity must be at least 2")
        
        # Validate knowledge config
        if config.knowledge.enable_rag and not config.knowledge.knowledge_base_path:
            errors.append("knowledge_base_path required when RAG is enabled")
        
        valid_domains = ["fashion", "finance", "healthcare", "retail", None]
        if config.knowledge.domain not in valid_domains:
            errors.append(f"knowledge.domain must be one of {valid_domains}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_column_config(column_name: str, column_config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate column-specific configuration
        
        Args:
            column_name: Name of the column
            column_config: Column configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for required fields based on data type
        if 'type' in column_config:
            data_type = column_config['type']
            
            if data_type == 'numeric':
                if 'min' in column_config and 'max' in column_config:
                    if column_config['min'] >= column_config['max']:
                        errors.append(f"{column_name}: min must be less than max")
            
            elif data_type == 'categorical':
                if 'categories' in column_config:
                    if not isinstance(column_config['categories'], list):
                        errors.append(f"{column_name}: categories must be a list")
                    if len(column_config['categories']) == 0:
                        errors.append(f"{column_name}: categories cannot be empty")
        
        return len(errors) == 0, errors


# Default configuration instance
def get_default_config() -> Config:
    """Get the default configuration"""
    return Config()


# Preset configurations
def create_analytics_preset() -> Config:
    """Create analytics-focused preset configuration"""
    config = Config()
    config.generation.num_rows = 10000
    config.numeric.preserve_correlations = True
    config.numeric.method = "correlation"
    config.validation.enable_quality_checks = True
    config.validation.quality_threshold = 0.9
    return config


def create_survey_preset() -> Config:
    """Create survey data preset configuration"""
    config = Config()
    config.generation.num_rows = 5000
    config.text.use_templates = True
    config.text.temperature = 0.8
    config.validation.quality_threshold = 0.75
    return config


def create_healthcare_preset() -> Config:
    """Create healthcare data preset configuration"""
    config = Config()
    config.pii.anonymization_level = "full"
    config.validation.enable_privacy_checks = True
    config.validation.k_anonymity = 10
    config.knowledge.domain = "healthcare"
    config.knowledge.enforce_constraints = True
    return config


def create_finance_preset() -> Config:
    """Create finance data preset configuration"""
    config = Config()
    config.numeric.decimal_places = 2
    config.numeric.range_enforcement = True
    config.validation.quality_threshold = 0.95
    config.knowledge.domain = "finance"
    return config


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a loader
    loader = ConfigLoader()
    
    # Load default config
    config = get_default_config()
    print("Default config:", config.to_dict())
    
    # Validate
    is_valid, errors = ConfigValidator.validate(config)
    print(f"Valid: {is_valid}")
    if errors:
        print("Errors:", errors)
    
    # Create and save a custom config
    custom_config = create_analytics_preset()
    loader.save_config(custom_config, "/tmp/analytics_config.yaml")
    
    # Load and merge configs
    loaded = loader.load_from_file("/tmp/analytics_config.yaml")
    print("\nLoaded config:", loaded.to_dict())
