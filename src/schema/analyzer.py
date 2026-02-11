"""
Schema Analyzer Module

Comprehensive schema analysis including:
- Statistical profiling of columns
- PII pattern detection
- Confidence scoring for type inference
- Data quality metrics
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII that can be detected"""
    EMAIL = "email"
    PHONE = "phone"
    NAME = "name"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    ADDRESS = "address"
    ZIP_CODE = "zip_code"
    IP_ADDRESS = "ip_address"
    URL = "url"
    IDENTIFIER = "identifier"  # UUID, custom IDs
    DATE_OF_BIRTH = "date_of_birth"
    NONE = "none"


@dataclass
class PIIPattern:
    """PII pattern detection result"""
    pii_type: PIIType
    confidence: float  # 0.0 to 1.0
    match_count: int
    total_count: int
    sample_matches: List[str] = field(default_factory=list)
    
    @property
    def match_rate(self) -> float:
        """Percentage of values matching the pattern"""
        return self.match_count / self.total_count if self.total_count > 0 else 0.0
    
    @property
    def is_likely_pii(self) -> bool:
        """Whether this is likely PII based on confidence and match rate"""
        return self.confidence >= 0.7 and self.match_rate >= 0.7


@dataclass
class ConfidenceScore:
    """Confidence score for data type inference"""
    data_type: str
    confidence: float  # 0.0 to 1.0
    reasons: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def add_reason(self, reason: str, weight: float = 0.1):
        """Add a reason and increase confidence"""
        self.reasons.append(reason)
        self.confidence = min(1.0, self.confidence + weight)
    
    def add_evidence(self, key: str, value: Any):
        """Add supporting evidence"""
        self.evidence[key] = value


@dataclass
class ColumnProfile:
    """Comprehensive profile for a single column"""
    name: str
    
    # Basic statistics
    total_count: int = 0
    null_count: int = 0
    unique_count: int = 0
    duplicate_count: int = 0
    
    # Data type information
    inferred_type: Optional[str] = None
    pandas_dtype: Optional[str] = None
    python_type: Optional[str] = None
    
    # Numeric statistics
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    variance: Optional[float] = None
    quantiles: Optional[Dict[str, float]] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # Numeric patterns
    is_integer: bool = False
    is_positive: bool = False
    has_outliers: bool = False
    outlier_count: int = 0
    
    # String statistics
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    median_length: Optional[float] = None
    
    # String patterns
    has_consistent_format: bool = False
    common_patterns: List[str] = field(default_factory=list)
    uppercase_ratio: float = 0.0
    lowercase_ratio: float = 0.0
    numeric_ratio: float = 0.0
    special_char_ratio: float = 0.0
    
    # Categorical information
    is_categorical: bool = False
    categories: Optional[List[Any]] = None
    category_counts: Optional[Dict[Any, int]] = None
    cardinality_ratio: float = 0.0
    
    # Temporal information
    is_temporal: bool = False
    date_format: Optional[str] = None
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    date_range_days: Optional[int] = None
    
    # PII detection
    pii_patterns: List[PIIPattern] = field(default_factory=list)
    contains_pii: bool = False
    pii_type: Optional[PIIType] = None
    
    # Confidence scoring
    type_confidence: Optional[ConfidenceScore] = None
    
    # Sample values
    sample_values: List[Any] = field(default_factory=list)
    most_common: List[Tuple[Any, int]] = field(default_factory=list)
    
    # Data quality
    completeness: float = 0.0  # 1 - (null_count / total_count)
    uniqueness: float = 0.0    # unique_count / total_count
    consistency: float = 0.0   # Based on pattern matching
    
    @property
    def null_percentage(self) -> float:
        """Percentage of null values"""
        return (self.null_count / self.total_count * 100) if self.total_count > 0 else 0.0
    
    @property
    def unique_percentage(self) -> float:
        """Percentage of unique values"""
        return (self.unique_count / self.total_count * 100) if self.total_count > 0 else 0.0


class PIIDetector:
    """
    Detects Personally Identifiable Information (PII) in data
    
    Uses pattern matching and heuristics to identify various PII types
    """
    
    # Regex patterns for PII detection
    PATTERNS = {
        PIIType.EMAIL: r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        PIIType.PHONE: r'^[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,9}$',
        PIIType.SSN: r'^\d{3}-?\d{2}-?\d{4}$',
        PIIType.CREDIT_CARD: r'^(?:\d{4}[-\s]?){3}\d{4}$',
        PIIType.ZIP_CODE: r'^\d{5}(?:-\d{4})?$',
        PIIType.IP_ADDRESS: r'^(?:\d{1,3}\.){3}\d{1,3}$',
        PIIType.URL: r'^https?://[^\s]+$',
        PIIType.IDENTIFIER: r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',  # UUID
    }
    
    # Name patterns
    NAME_PATTERNS = {
        'full_name': r'^[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$',
        'first_name': r'^[A-Z][a-z]{2,}$',
        'last_name': r'^[A-Z][a-z]{2,}$',
    }
    
    # Common first/last names for validation
    COMMON_FIRST_NAMES = {
        'james', 'john', 'robert', 'michael', 'william', 'mary', 'patricia', 'jennifer',
        'linda', 'elizabeth', 'david', 'richard', 'joseph', 'thomas', 'charles'
    }
    
    COMMON_LAST_NAMES = {
        'smith', 'johnson', 'williams', 'brown', 'jones', 'garcia', 'miller', 'davis',
        'rodriguez', 'martinez', 'hernandez', 'lopez', 'gonzalez', 'wilson', 'anderson'
    }
    
    def __init__(self, sample_size: int = 100):
        """
        Initialize PII detector
        
        Args:
            sample_size: Number of values to sample for detection
        """
        self.sample_size = sample_size
    
    def detect(self, series: pd.Series) -> List[PIIPattern]:
        """
        Detect PII patterns in a pandas Series
        
        Args:
            series: Data to analyze
            
        Returns:
            List of detected PII patterns
        """
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return []
        
        # Sample for performance
        if len(clean_series) > self.sample_size:
            sample = clean_series.sample(self.sample_size, random_state=42)
        else:
            sample = clean_series
        
        # Convert to strings
        str_sample = sample.astype(str)
        
        detected_patterns = []
        
        # Check each PII type
        for pii_type, pattern in self.PATTERNS.items():
            result = self._check_pattern(str_sample, pii_type, pattern)
            if result and result.match_rate > 0.1:  # At least 10% match
                detected_patterns.append(result)
        
        # Special handling for names
        name_result = self._detect_names(str_sample)
        if name_result:
            detected_patterns.append(name_result)
        
        # Special handling for addresses
        address_result = self._detect_addresses(str_sample)
        if address_result:
            detected_patterns.append(address_result)
        
        # Sort by confidence
        detected_patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        return detected_patterns
    
    def _check_pattern(
        self, 
        str_sample: pd.Series, 
        pii_type: PIIType, 
        pattern: str
    ) -> Optional[PIIPattern]:
        """Check if values match a specific PII pattern"""
        matches = str_sample.str.match(pattern, na=False)
        match_count = matches.sum()
        total_count = len(str_sample)
        
        if match_count == 0:
            return None
        
        match_rate = match_count / total_count
        
        # Calculate confidence based on match rate
        if match_rate >= 0.95:
            confidence = 0.95
        elif match_rate >= 0.8:
            confidence = 0.85
        elif match_rate >= 0.6:
            confidence = 0.7
        else:
            confidence = match_rate * 0.8
        
        # Get sample matches
        matched_values = str_sample[matches].head(5).tolist()
        
        return PIIPattern(
            pii_type=pii_type,
            confidence=confidence,
            match_count=match_count,
            total_count=total_count,
            sample_matches=matched_values
        )
    
    def _detect_names(self, str_sample: pd.Series) -> Optional[PIIPattern]:
        """Detect person names using multiple heuristics"""
        match_count = 0
        
        for value in str_sample:
            value_str = str(value).strip()
            
            # Check patterns
            if re.match(self.NAME_PATTERNS['full_name'], value_str):
                match_count += 1
                continue
            
            # Check against common names
            words = value_str.lower().split()
            if len(words) >= 2:
                if words[0] in self.COMMON_FIRST_NAMES or words[-1] in self.COMMON_LAST_NAMES:
                    match_count += 1
                    continue
            
            # Check for title prefix (Mr., Mrs., Dr., etc.)
            if re.match(r'^(Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z]', value_str):
                match_count += 1
                continue
        
        if match_count == 0:
            return None
        
        match_rate = match_count / len(str_sample)
        
        if match_rate < 0.3:
            return None
        
        confidence = min(0.9, match_rate + 0.2)
        
        return PIIPattern(
            pii_type=PIIType.NAME,
            confidence=confidence,
            match_count=match_count,
            total_count=len(str_sample),
            sample_matches=str_sample.head(5).tolist()
        )
    
    def _detect_addresses(self, str_sample: pd.Series) -> Optional[PIIPattern]:
        """Detect physical addresses"""
        # Address indicators
        address_keywords = ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'drive', 'dr', 
                           'lane', 'ln', 'boulevard', 'blvd', 'way', 'court', 'ct']
        
        match_count = 0
        
        for value in str_sample:
            value_lower = str(value).lower()
            
            # Check for address keywords
            if any(keyword in value_lower for keyword in address_keywords):
                # Check if it has numbers (street numbers)
                if re.search(r'\d+', value_lower):
                    match_count += 1
        
        if match_count == 0:
            return None
        
        match_rate = match_count / len(str_sample)
        
        if match_rate < 0.3:
            return None
        
        confidence = min(0.85, match_rate + 0.15)
        
        return PIIPattern(
            pii_type=PIIType.ADDRESS,
            confidence=confidence,
            match_count=match_count,
            total_count=len(str_sample),
            sample_matches=str_sample.head(5).tolist()
        )
    
    def get_primary_pii_type(self, patterns: List[PIIPattern]) -> Optional[PIIType]:
        """Get the most likely PII type from detected patterns"""
        if not patterns:
            return None
        
        # Filter to likely PII
        likely_patterns = [p for p in patterns if p.is_likely_pii]
        
        if not likely_patterns:
            return None
        
        # Return highest confidence
        return likely_patterns[0].pii_type


class StatisticalProfiler:
    """
    Performs statistical profiling of columns
    
    Extracts comprehensive statistical metrics for numeric, text, and categorical data
    """
    
    def __init__(self):
        self.outlier_threshold = 3.0  # Standard deviations for outlier detection
    
    def profile_numeric(self, series: pd.Series, profile: ColumnProfile):
        """Profile numeric column"""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return
        
        # Basic statistics
        profile.min_value = float(clean_series.min())
        profile.max_value = float(clean_series.max())
        profile.mean = float(clean_series.mean())
        profile.median = float(clean_series.median())
        profile.std = float(clean_series.std())
        profile.variance = float(clean_series.var())
        
        # Quantiles
        profile.quantiles = {
            'q25': float(clean_series.quantile(0.25)),
            'q50': float(clean_series.quantile(0.50)),
            'q75': float(clean_series.quantile(0.75)),
            'q95': float(clean_series.quantile(0.95)),
            'q99': float(clean_series.quantile(0.99)),
        }
        
        # Distribution shape
        try:
            profile.skewness = float(clean_series.skew())
            profile.kurtosis = float(clean_series.kurtosis())
        except:
            pass
        
        # Check if integer
        profile.is_integer = (clean_series % 1 == 0).all()
        
        # Check if positive
        profile.is_positive = (clean_series > 0).all()
        
        # Outlier detection using IQR method
        Q1 = profile.quantiles['q25']
        Q3 = profile.quantiles['q75']
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (clean_series < lower_bound) | (clean_series > upper_bound)
        profile.outlier_count = outliers.sum()
        profile.has_outliers = profile.outlier_count > 0
    
    def profile_text(self, series: pd.Series, profile: ColumnProfile):
        """Profile text column"""
        clean_series = series.dropna().astype(str)
        
        if len(clean_series) == 0:
            return
        
        # Length statistics
        lengths = clean_series.str.len()
        profile.min_length = int(lengths.min())
        profile.max_length = int(lengths.max())
        profile.avg_length = float(lengths.mean())
        profile.median_length = float(lengths.median())
        
        # Character type ratios
        total_chars = lengths.sum()
        if total_chars > 0:
            profile.uppercase_ratio = clean_series.str.count(r'[A-Z]').sum() / total_chars
            profile.lowercase_ratio = clean_series.str.count(r'[a-z]').sum() / total_chars
            profile.numeric_ratio = clean_series.str.count(r'\d').sum() / total_chars
            profile.special_char_ratio = clean_series.str.count(r'[^a-zA-Z0-9\s]').sum() / total_chars
        
        # Pattern detection
        profile.common_patterns = self._extract_patterns(clean_series)
        profile.has_consistent_format = self._check_format_consistency(clean_series)
    
    def profile_categorical(self, series: pd.Series, profile: ColumnProfile):
        """Profile categorical column"""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return
        
        # Value counts
        value_counts = clean_series.value_counts()
        
        profile.categories = value_counts.index.tolist()
        profile.category_counts = value_counts.to_dict()
        profile.cardinality_ratio = len(value_counts) / len(clean_series)
        
        # Most common values
        profile.most_common = value_counts.head(10).items()
        
        # Determine if truly categorical (low cardinality)
        profile.is_categorical = profile.cardinality_ratio < 0.05 or len(value_counts) < 50
    
    def profile_temporal(self, series: pd.Series, profile: ColumnProfile):
        """Profile temporal column"""
        try:
            dt_series = pd.to_datetime(series.dropna())
            
            if len(dt_series) == 0:
                return
            
            profile.is_temporal = True
            profile.min_date = str(dt_series.min())
            profile.max_date = str(dt_series.max())
            
            date_range = dt_series.max() - dt_series.min()
            profile.date_range_days = date_range.days
            
            # Try to infer format from original string
            if len(series.dropna()) > 0:
                sample = str(series.dropna().iloc[0])
                profile.date_format = self._infer_date_format(sample)
        
        except Exception as e:
            logger.debug(f"Failed to profile as temporal: {e}")
            profile.is_temporal = False
    
    def _extract_patterns(self, series: pd.Series, max_patterns: int = 5) -> List[str]:
        """Extract common string patterns"""
        patterns = []
        
        # Sample values
        sample = series.sample(min(100, len(series)), random_state=42)
        
        # Convert to pattern representation
        pattern_strings = []
        for value in sample:
            pattern = re.sub(r'[a-z]', 'a', str(value))
            pattern = re.sub(r'[A-Z]', 'A', pattern)
            pattern = re.sub(r'\d', '9', pattern)
            pattern_strings.append(pattern)
        
        # Count patterns
        pattern_counts = Counter(pattern_strings)
        
        # Return most common
        for pattern, count in pattern_counts.most_common(max_patterns):
            if count > 1:  # Only patterns that repeat
                patterns.append(pattern)
        
        return patterns
    
    def _check_format_consistency(self, series: pd.Series, threshold: float = 0.8) -> bool:
        """Check if values have consistent format"""
        patterns = self._extract_patterns(series)
        
        if not patterns:
            return False
        
        # Check if most values match the top pattern
        top_pattern = patterns[0]
        match_count = 0
        
        for value in series:
            pattern = re.sub(r'[a-z]', 'a', str(value))
            pattern = re.sub(r'[A-Z]', 'A', pattern)
            pattern = re.sub(r'\d', '9', pattern)
            
            if pattern == top_pattern:
                match_count += 1
        
        return (match_count / len(series)) >= threshold
    
    def _infer_date_format(self, date_string: str) -> str:
        """Infer date format from string"""
        common_formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%Y/%m/%d',
            '%Y-%m-%d %H:%M:%S',
            '%d-%m-%Y',
            '%m-%d-%Y',
            '%Y%m%d',
            '%d.%m.%Y',
            '%b %d, %Y',
            '%B %d, %Y',
        ]
        
        for fmt in common_formats:
            try:
                datetime.strptime(date_string, fmt)
                return fmt
            except:
                continue
        
        return '%Y-%m-%d'  # Default


class ConfidenceScorer:
    """
    Scores confidence in data type inference
    
    Uses multiple signals to determine confidence in type classification
    """
    
    def score_numeric(self, series: pd.Series) -> ConfidenceScore:
        """Score confidence for numeric type"""
        score = ConfidenceScore(data_type='numeric', confidence=0.0)
        
        # Check if pandas thinks it's numeric
        if pd.api.types.is_numeric_dtype(series):
            score.add_reason("Pandas dtype is numeric", weight=0.4)
            score.add_evidence("pandas_dtype", str(series.dtype))
        
        # Try to convert to numeric
        try:
            pd.to_numeric(series.dropna())
            score.add_reason("Successfully converts to numeric", weight=0.3)
        except:
            return score  # Not numeric
        
        # Check for consistent numeric format
        clean_series = series.dropna()
        if len(clean_series) > 0:
            # Check if all values are numeric when converted
            numeric_count = pd.to_numeric(clean_series, errors='coerce').notna().sum()
            numeric_ratio = numeric_count / len(clean_series)
            
            if numeric_ratio >= 0.95:
                score.add_reason(f"{numeric_ratio*100:.1f}% values are numeric", weight=0.2)
            
            score.add_evidence("numeric_ratio", numeric_ratio)
        
        return score
    
    def score_categorical(self, series: pd.Series) -> ConfidenceScore:
        """Score confidence for categorical type"""
        score = ConfidenceScore(data_type='categorical', confidence=0.0)
        
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return score
        
        unique_count = clean_series.nunique()
        total_count = len(clean_series)
        cardinality_ratio = unique_count / total_count
        
        # Low cardinality suggests categorical
        if cardinality_ratio < 0.01:
            score.add_reason("Very low cardinality (<1%)", weight=0.4)
        elif cardinality_ratio < 0.05:
            score.add_reason("Low cardinality (<5%)", weight=0.3)
        elif cardinality_ratio < 0.1:
            score.add_reason("Moderate cardinality (<10%)", weight=0.2)
        
        # Small number of unique values
        if unique_count < 10:
            score.add_reason(f"Only {unique_count} unique values", weight=0.2)
        elif unique_count < 50:
            score.add_reason(f"Only {unique_count} unique values", weight=0.1)
        
        # Check for repeated values
        value_counts = clean_series.value_counts()
        if len(value_counts) > 0:
            top_value_ratio = value_counts.iloc[0] / total_count
            if top_value_ratio > 0.3:
                score.add_reason(f"Top value appears {top_value_ratio*100:.1f}% of time", weight=0.1)
        
        score.add_evidence("cardinality_ratio", cardinality_ratio)
        score.add_evidence("unique_count", unique_count)
        
        return score
    
    def score_text(self, series: pd.Series) -> ConfidenceScore:
        """Score confidence for text type"""
        score = ConfidenceScore(data_type='text', confidence=0.0)
        
        clean_series = series.dropna().astype(str)
        if len(clean_series) == 0:
            return score
        
        # Check average length
        avg_length = clean_series.str.len().mean()
        if avg_length > 20:
            score.add_reason(f"Average length {avg_length:.0f} chars suggests text", weight=0.3)
        
        # Check for variation in length
        length_std = clean_series.str.len().std()
        if length_std > 10:
            score.add_reason("High variation in length", weight=0.2)
        
        # Check for words/spaces
        has_spaces = clean_series.str.contains(' ', na=False).sum() / len(clean_series)
        if has_spaces > 0.5:
            score.add_reason(f"{has_spaces*100:.0f}% values contain spaces", weight=0.2)
        
        # Check for mixed case
        has_mixed_case = (
            clean_series.str.contains('[a-z]', na=False) & 
            clean_series.str.contains('[A-Z]', na=False)
        ).sum() / len(clean_series)
        
        if has_mixed_case > 0.3:
            score.add_reason("Contains mixed case text", weight=0.1)
        
        score.add_evidence("avg_length", avg_length)
        score.add_evidence("has_spaces_ratio", has_spaces)
        
        return score
    
    def score_temporal(self, series: pd.Series) -> ConfidenceScore:
        """Score confidence for temporal type"""
        score = ConfidenceScore(data_type='temporal', confidence=0.0)
        
        # Try to parse as datetime
        try:
            dt_series = pd.to_datetime(series.dropna(), errors='coerce')
            parse_rate = dt_series.notna().sum() / len(series.dropna())
            
            if parse_rate >= 0.95:
                score.add_reason(f"{parse_rate*100:.1f}% values parse as dates", weight=0.5)
            elif parse_rate >= 0.8:
                score.add_reason(f"{parse_rate*100:.1f}% values parse as dates", weight=0.3)
            
            score.add_evidence("parse_rate", parse_rate)
            
            # Check for date-like patterns
            str_series = series.dropna().astype(str)
            date_pattern = r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}'
            has_date_pattern = str_series.str.contains(date_pattern, na=False).sum() / len(str_series)
            
            if has_date_pattern > 0.7:
                score.add_reason("Contains date-like patterns", weight=0.2)
            
        except Exception as e:
            logger.debug(f"Failed to parse as temporal: {e}")
        
        return score
    
    def score_pii(self, pii_patterns: List[PIIPattern]) -> Optional[ConfidenceScore]:
        """Score confidence for PII type"""
        if not pii_patterns:
            return None
        
        # Get highest confidence pattern
        top_pattern = max(pii_patterns, key=lambda x: x.confidence)
        
        if not top_pattern.is_likely_pii:
            return None
        
        score = ConfidenceScore(
            data_type=f'pii_{top_pattern.pii_type.value}',
            confidence=top_pattern.confidence
        )
        
        score.add_reason(
            f"Matches {top_pattern.pii_type.value} pattern at {top_pattern.match_rate*100:.1f}% rate",
            weight=0.0  # Already included in confidence
        )
        
        score.add_evidence("pii_type", top_pattern.pii_type.value)
        score.add_evidence("match_rate", top_pattern.match_rate)
        
        return score


class SchemaAnalyzer:
    """
    Main schema analyzer
    
    Coordinates PII detection, statistical profiling, and confidence scoring
    to build comprehensive column profiles
    """
    
    def __init__(
        self,
        enable_pii_detection: bool = True,
        enable_statistical_profiling: bool = True,
        sample_size: int = 1000
    ):
        """
        Initialize schema analyzer
        
        Args:
            enable_pii_detection: Whether to detect PII
            enable_statistical_profiling: Whether to compute statistics
            sample_size: Number of rows to sample for analysis
        """
        self.enable_pii_detection = enable_pii_detection
        self.enable_statistical_profiling = enable_statistical_profiling
        self.sample_size = sample_size
        
        self.pii_detector = PIIDetector(sample_size=sample_size)
        self.statistical_profiler = StatisticalProfiler()
        self.confidence_scorer = ConfidenceScorer()
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, ColumnProfile]:
        """
        Analyze entire DataFrame
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to profiles
        """
        logger.info(f"Analyzing schema for {len(df.columns)} columns, {len(df)} rows")
        
        profiles = {}
        
        for col in df.columns:
            profile = self.analyze_column(df[col], col)
            profiles[col] = profile
        
        logger.info("Schema analysis complete")
        
        return profiles
    
    def analyze_column(self, series: pd.Series, column_name: str) -> ColumnProfile:
        """
        Analyze a single column
        
        Args:
            series: Column data
            column_name: Name of the column
            
        Returns:
            Column profile
        """
        profile = ColumnProfile(name=column_name)
        
        # Basic statistics
        profile.total_count = len(series)
        profile.null_count = series.isna().sum()
        profile.unique_count = series.nunique()
        profile.duplicate_count = profile.total_count - profile.unique_count
        profile.pandas_dtype = str(series.dtype)
        
        # Data quality metrics
        profile.completeness = 1.0 - (profile.null_count / profile.total_count) if profile.total_count > 0 else 0.0
        profile.uniqueness = profile.unique_count / profile.total_count if profile.total_count > 0 else 0.0
        
        # Sample values
        if len(series.dropna()) > 0:
            profile.sample_values = series.dropna().sample(
                min(5, len(series.dropna())),
                random_state=42
            ).tolist()
        
        # Infer type and compute confidence
        profile.inferred_type, profile.type_confidence = self._infer_type_with_confidence(series)
        
        # PII detection
        if self.enable_pii_detection:
            profile.pii_patterns = self.pii_detector.detect(series)
            profile.pii_type = self.pii_detector.get_primary_pii_type(profile.pii_patterns)
            profile.contains_pii = profile.pii_type is not None
            
            # Override type if PII detected with high confidence
            if profile.contains_pii:
                pii_score = self.confidence_scorer.score_pii(profile.pii_patterns)
                if pii_score and pii_score.confidence > 0.8:
                    if profile.type_confidence is None or pii_score.confidence > profile.type_confidence.confidence:
                        profile.inferred_type = f'pii_{profile.pii_type.value}'
                        profile.type_confidence = pii_score
        
        # Statistical profiling
        if self.enable_statistical_profiling:
            if profile.inferred_type == 'numeric' or pd.api.types.is_numeric_dtype(series):
                self.statistical_profiler.profile_numeric(series, profile)
            
            if profile.inferred_type in ['text', 'categorical']:
                self.statistical_profiler.profile_text(series, profile)
            
            # Always check categorical
            self.statistical_profiler.profile_categorical(series, profile)
            
            # Check temporal
            self.statistical_profiler.profile_temporal(series, profile)
        
        # Consistency score based on various factors
        profile.consistency = self._calculate_consistency_score(profile)
        
        return profile
    
    def _infer_type_with_confidence(self, series: pd.Series) -> Tuple[str, ConfidenceScore]:
        """Infer data type with confidence scoring"""
        candidates = []
        
        # Score each type
        numeric_score = self.confidence_scorer.score_numeric(series)
        if numeric_score.confidence > 0.3:
            candidates.append(numeric_score)
        
        categorical_score = self.confidence_scorer.score_categorical(series)
        if categorical_score.confidence > 0.3:
            candidates.append(categorical_score)
        
        text_score = self.confidence_scorer.score_text(series)
        if text_score.confidence > 0.3:
            candidates.append(text_score)
        
        temporal_score = self.confidence_scorer.score_temporal(series)
        if temporal_score.confidence > 0.3:
            candidates.append(temporal_score)
        
        # Select highest confidence
        if candidates:
            best_score = max(candidates, key=lambda x: x.confidence)
            return best_score.data_type, best_score
        
        # Default to text with low confidence
        default_score = ConfidenceScore(data_type='text', confidence=0.2)
        default_score.add_reason("Default fallback type", weight=0.0)
        return 'text', default_score
    
    def _calculate_consistency_score(self, profile: ColumnProfile) -> float:
        """Calculate consistency score based on various factors"""
        scores = []
        
        # Completeness contributes to consistency
        scores.append(profile.completeness)
        
        # Type confidence contributes
        if profile.type_confidence:
            scores.append(profile.type_confidence.confidence)
        
        # Format consistency for text
        if profile.has_consistent_format:
            scores.append(0.9)
        
        # Low null percentage is consistent
        if profile.null_percentage < 5:
            scores.append(0.9)
        
        if not scores:
            return 0.5
        
        return sum(scores) / len(scores)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 15000, 100),
        'email': [f'user{i}@example.com' for i in range(100)],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson'] * 33 + ['Alice Brown'],
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago'], 100),
        'description': ['Long text description ' * 5 for _ in range(100)],
    })
    
    # Analyze schema
    analyzer = SchemaAnalyzer()
    profiles = analyzer.analyze_dataframe(sample_data)
    
    # Print results
    print("\n" + "="*80)
    print("SCHEMA ANALYSIS RESULTS")
    print("="*80)
    
    for col_name, profile in profiles.items():
        print(f"\nColumn: {col_name}")
        print(f"  Type: {profile.inferred_type}")
        if profile.type_confidence:
            print(f"  Confidence: {profile.type_confidence.confidence:.2f}")
            print(f"  Reasons: {', '.join(profile.type_confidence.reasons[:2])}")
        print(f"  Completeness: {profile.completeness:.2%}")
        print(f"  Uniqueness: {profile.uniqueness:.2%}")
        
        if profile.contains_pii:
            print(f"  ⚠️  PII Detected: {profile.pii_type.value}")
        
        if profile.min_value is not None:
            print(f"  Range: {profile.min_value:.2f} to {profile.max_value:.2f}")
        
        if profile.categories:
            print(f"  Categories: {len(profile.categories)}")
