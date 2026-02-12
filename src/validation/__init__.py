"""
Validation Module

Provides comprehensive validation for synthetic data:
- Quality metrics for numeric, text, PII, and temporal data
- Privacy checks including re-identification risk and k-anonymity
- Statistical similarity measures
- Data integrity validation
"""

from .quality import (
    QualityValidator,
    NumericQualityMetrics,
    TextQualityMetrics,
    PIIQualityMetrics,
    TemporalQualityMetrics,
    QualityReport,
)

from .privacy import (
    PrivacyValidator,
    ReIdentificationRisk,
    KAnonymity,
    UniquenessAnalyzer,
    PrivacyReport,
)

__all__ = [
    # Quality validation
    "QualityValidator",
    "NumericQualityMetrics",
    "TextQualityMetrics",
    "PIIQualityMetrics",
    "TemporalQualityMetrics",
    "QualityReport",
    
    # Privacy validation
    "PrivacyValidator",
    "ReIdentificationRisk",
    "KAnonymity",
    "UniquenessAnalyzer",
    "PrivacyReport",
]
