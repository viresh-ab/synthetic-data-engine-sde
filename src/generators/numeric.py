"""
FIXED Numeric Data Generator

Key Fixes:
1. Preserves integer types (IDs, age, etc.)
2. Enforces positive values for amounts/prices
3. Proper range constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class DistributionParams:
    """Parameters for a statistical distribution"""
    distribution: str
    params: Dict[str, float]
    goodness_of_fit: float = 0.0
    is_integer: bool = False  # NEW: Track if original data was integer
    
    def generate(self, size: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate random values from this distribution"""
        if seed is not None:
            np.random.seed(seed)
        
        if self.distribution == "normal":
            values = np.random.normal(
                loc=self.params['mean'],
                scale=self.params['std'],
                size=size
            )
        elif self.distribution == "uniform":
            values = np.random.uniform(
                low=self.params['min'],
                high=self.params['max'],
                size=size
            )
        elif self.distribution == "exponential":
            values = np.random.exponential(
                scale=self.params['scale'],
                size=size
            )
        elif self.distribution == "lognormal":
            values = np.random.lognormal(
                mean=self.params['mean'],
                sigma=self.params['sigma'],
                size=size
            )
        elif self.distribution == "gamma":
            values = np.random.gamma(
                shape=self.params['shape'],
                scale=self.params['scale'],
                size=size
            )
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
        
        # NEW: Convert to integer if original was integer
        if self.is_integer:
            values = np.round(values).astype(int)
        
        return values


class DistributionFitter:
    """Fits statistical distributions to data"""
    
    def __init__(self):
        self.supported_distributions = [
            'normal', 'uniform', 'exponential', 'lognormal', 'gamma'
        ]
    
    def fit(
        self, 
        data: np.ndarray, 
        distribution: str = 'auto',
        is_integer: bool = False  # NEW parameter
    ) -> DistributionParams:
        """
        Fit a distribution to data
        
        Args:
            data: Numeric data to fit
            distribution: Distribution name or 'auto' for best fit
            is_integer: Whether data should be integer
            
        Returns:
            Distribution parameters
        """
        data = data[~np.isnan(data)]
        
        if len(data) == 0:
            raise ValueError("No valid data to fit")
        
        if distribution == 'auto':
            # Try all distributions and pick best
            best_params = None
            best_gof = -np.inf
            
            for dist in self.supported_distributions:
                try:
                    params = self._fit_distribution(data, dist)
                    if params.goodness_of_fit > best_gof:
                        best_gof = params.goodness_of_fit
                        best_params = params
                except Exception as e:
                    logger.debug(f"Failed to fit {dist}: {e}")
                    continue
            
            if best_params is None:
                # Fallback to normal
                best_params = self._fit_distribution(data, 'normal')
            
            best_params.is_integer = is_integer
            return best_params
        else:
            params = self._fit_distribution(data, distribution)
            params.is_integer = is_integer
            return params
    
    def _fit_distribution(self, data: np.ndarray, distribution: str) -> DistributionParams:
        """Fit a specific distribution"""
        if distribution == 'normal':
            mean = np.mean(data)
            std = np.std(data)
            params = {'mean': mean, 'std': std}
        
        elif distribution == 'uniform':
            min_val = np.min(data)
            max_val = np.max(data)
            params = {'min': min_val, 'max': max_val}
        
        elif distribution == 'exponential':
            scale = np.mean(data)
            params = {'scale': scale}
        
        elif distribution == 'lognormal':
            log_data = np.log(data[data > 0])
            mean = np.mean(log_data)
            sigma = np.std(log_data)
            params = {'mean': mean, 'sigma': sigma}
        
        elif distribution == 'gamma':
            shape, loc, scale = stats.gamma.fit(data)
            params = {'shape': shape, 'scale': scale}
        
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        # Calculate goodness of fit (KS test)
        gof = self._goodness_of_fit(data, distribution, params)
        
        return DistributionParams(
            distribution=distribution,
            params=params,
            goodness_of_fit=gof
        )
    
    def _goodness_of_fit(self, data: np.ndarray, distribution: str, params: Dict) -> float:
        """Calculate goodness of fit using KS test"""
        try:
            if distribution == 'normal':
                ks_stat, _ = stats.kstest(data, 'norm', args=(params['mean'], params['std']))
            elif distribution == 'uniform':
                ks_stat, _ = stats.kstest(data, 'uniform', args=(params['min'], params['max'] - params['min']))
            elif distribution == 'exponential':
                ks_stat, _ = stats.kstest(data, 'expon', args=(0, params['scale']))
            else:
                return 0.0
            
            return 1.0 - ks_stat
        except:
            return 0.0
    
    def generate(self, params: DistributionParams, size: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate values from distribution parameters"""
        return params.generate(size, seed)


class NumericGenerator:
    """
    FIXED: Generate synthetic numeric data with proper type preservation
    """
    
    def __init__(self, config: Any = None):
        self.config = config
        self.fitter = DistributionFitter()
    

    @staticmethod
    def _should_be_integer(series: pd.Series, column_name: str) -> bool:
        """Determine if a numeric column should be generated as integers."""
        if pd.api.types.is_integer_dtype(series):
            return True

        normalized = column_name.lower().replace('_', ' ').replace('-', ' ')
        integer_keywords = ['age', 'sl no', 'serial no', 'serial number', 'slno', 'record id', 'id']
        if any(keyword in normalized for keyword in integer_keywords):
            return True

        clean_series = series.dropna()
        if len(clean_series) == 0:
            return False

        try:
            values = clean_series.astype(float)
            return np.all(np.isclose(values, np.round(values)))
        except Exception:
            return False

    def generate(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        num_rows: int = 1,
        preserve_correlations: bool = False,
        seed: Optional[int] = None,
        schema: Any = None,
        column_names: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate synthetic numeric data with all fixes applied
        
        FIXES:
        - Preserves integer types for IDs, age, etc.
        - Enforces positive values for amounts
        - Proper range constraints
        """
        if reference_data is None:
            raise ValueError("reference_data is required for numeric generation")

        if seed is not None:
            np.random.seed(seed)

        source_data = reference_data
        if column_names:
            available_columns = [col for col in column_names if col in reference_data.columns]
            source_data = reference_data[available_columns]

        synthetic_data = {}

        for col in source_data.columns:
            col_data = source_data[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # FIX 1: Detect if column should be integer
            is_integer = self._should_be_integer(source_data[col], col)
            
            # Fit distribution
            dist_params = self.fitter.fit(
                col_data.values, 
                distribution='auto',
                is_integer=is_integer  # Pass integer flag
            )
            
            # Generate values
            synthetic_values = self.fitter.generate(dist_params, num_rows, seed)
            
            # FIX 2: Enforce positive values for amount/price columns
            if any(keyword in col.lower() for keyword in ['amount', 'price', 'cost', 'value', 'total', 'balance']):
                synthetic_values = np.abs(synthetic_values)
                synthetic_values = np.maximum(synthetic_values, 10.0)  # Min $10
            
            # FIX 3: Apply range constraints
            min_val = col_data.min()
            max_val = col_data.max()
            
            # For IDs, ensure sequential starting from 1
            if 'id' in col.lower() and is_integer:
                synthetic_values = np.arange(1, num_rows + 1)
            else:
                synthetic_values = np.clip(synthetic_values, min_val, max_val)
            
            # FIX 4: Final integer conversion if needed
            if is_integer:
                synthetic_values = np.round(synthetic_values).astype(int)
            
            synthetic_data[col] = synthetic_values
        
        return pd.DataFrame(synthetic_data)


# For backward compatibility
CorrelationEngine = None  # Not needed for basic fix