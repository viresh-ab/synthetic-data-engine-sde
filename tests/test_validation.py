"""
Test Suite for Validation Modules

Tests quality and privacy validation:
- QualityValidator
- PrivacyValidator
- NumericQualityMetrics
- TextQualityMetrics
- PIIQualityMetrics
- TemporalQualityMetrics
- KAnonymity
- ReIdentificationRisk
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.validation import (
    QualityValidator,
    PrivacyValidator,
    NumericQualityMetrics,
    TextQualityMetrics,
    PIIQualityMetrics,
    TemporalQualityMetrics,
    KAnonymity,
    ReIdentificationRisk,
    UniquenessAnalyzer,
)
from src.config import get_default_config


class TestNumericQualityMetrics:
    """Test numeric quality validation"""
    
    @pytest.fixture
    def validator(self):
        return NumericQualityMetrics(threshold=0.8)
    
    @pytest.fixture
    def reference_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            'age': np.random.normal(35, 10, 100),
            'income': np.random.lognormal(10.5, 0.5, 100),
        })
    
    @pytest.fixture
    def good_synthetic(self, reference_data):
        """Synthetic data with good quality"""
        np.random.seed(43)
        return pd.DataFrame({
            'age': np.random.normal(35, 10, 100),
            'income': np.random.lognormal(10.5, 0.5, 100),
        })
    
    @pytest.fixture
    def poor_synthetic(self, reference_data):
        """Synthetic data with poor quality"""
        np.random.seed(44)
        return pd.DataFrame({
            'age': np.random.normal(50, 20, 100),  # Different distribution
            'income': np.random.uniform(0, 100000, 100),  # Wrong distribution
        })
    
    def test_good_quality_validation(self, validator, reference_data, good_synthetic):
        """Test validation of good quality data"""
        metrics = validator.validate(reference_data, good_synthetic)
        
        assert len(metrics) > 0
        # Most metrics should pass
        passed = sum(1 for m in metrics if m.passed)
        assert passed / len(metrics) > 0.7
    
    def test_poor_quality_validation(self, validator, reference_data, poor_synthetic):
        """Test validation of poor quality data"""
        metrics = validator.validate(reference_data, poor_synthetic)
        
        assert len(metrics) > 0
        # Many metrics should fail
        failed = sum(1 for m in metrics if not m.passed)
        assert failed / len(metrics) > 0.3
    
    def test_distribution_similarity(self, validator, reference_data, good_synthetic):
        """Test distribution similarity metrics"""
        metrics = validator.validate(reference_data, good_synthetic)
        
        # Find KS test metrics
        ks_metrics = [m for m in metrics if 'ks_test' in m.name]
        assert len(ks_metrics) > 0
        
        # KS test should show similarity
        for metric in ks_metrics:
            assert metric.value > 0.5
    
    def test_correlation_preservation(self, validator):
        """Test correlation preservation metric"""
        # Create correlated data
        np.random.seed(42)
        reference = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
        })
        reference['y'] = reference['x'] * 0.8 + reference['y'] * 0.2
        
        # Similar correlation
        np.random.seed(43)
        synthetic = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
        })
        synthetic['y'] = synthetic['x'] * 0.8 + synthetic['y'] * 0.2
        
        metrics = validator.validate(reference, synthetic)
        
        corr_metrics = [m for m in metrics if 'correlation' in m.name]
        assert len(corr_metrics) > 0
        assert corr_metrics[0].value > 0.8


class TestTextQualityMetrics:
    """Test text quality validation"""
    
    @pytest.fixture
    def validator(self):
        return TextQualityMetrics(threshold=0.7)
    
    @pytest.fixture
    def reference_data(self):
        return pd.Series([
            "Great product, highly recommended!",
            "Excellent quality and fast shipping.",
            "Not what I expected, disappointed.",
            "Amazing value for the price.",
            "Good but could be better.",
        ] * 20)
    
    @pytest.fixture
    def good_synthetic(self):
        return pd.Series([
            "Wonderful item, would buy again!",
            "Great quality and quick delivery.",
            "Not as expected, let down.",
            "Fantastic value for money.",
            "Decent but room for improvement.",
        ] * 20)
    
    def test_length_distribution(self, validator, reference_data, good_synthetic):
        """Test length distribution validation"""
        metrics = validator.validate(reference_data, good_synthetic, column_name="review")
        
        length_metrics = [m for m in metrics if 'length' in m.name]
        assert len(length_metrics) > 0
        assert length_metrics[0].value > 0.5
    
    def test_vocabulary_overlap(self, validator, reference_data, good_synthetic):
        """Test vocabulary overlap validation"""
        metrics = validator.validate(reference_data, good_synthetic)
        
        vocab_metrics = [m for m in metrics if 'vocabulary' in m.name]
        assert len(vocab_metrics) > 0
        # Should have some overlap
        assert vocab_metrics[0].value > 0.3
    
    def test_uniqueness(self, validator, reference_data, good_synthetic):
        """Test uniqueness validation"""
        metrics = validator.validate(reference_data, good_synthetic)
        
        unique_metrics = [m for m in metrics if 'uniqueness' in m.name]
        assert len(unique_metrics) > 0


class TestPIIQualityMetrics:
    """Test PII quality validation"""
    
    @pytest.fixture
    def validator(self):
        return PIIQualityMetrics(threshold=0.8)
    
    def test_email_format_validation(self, validator):
        """Test email format validation"""
        reference = pd.Series(["john@example.com", "jane@test.org"] * 50)
        synthetic = pd.Series(["bob@company.com", "alice@site.net"] * 50)
        
        metrics = validator.validate(reference, synthetic, pii_type='email', column_name='email')
        
        format_metrics = [m for m in metrics if 'format_validity' in m.name]
        assert len(format_metrics) > 0
        # All should be valid emails
        assert format_metrics[0].value > 0.95
    
    def test_no_data_leakage(self, validator):
        """Test that synthetic doesn't leak reference data"""
        reference = pd.Series(["john@example.com", "jane@test.org"] * 50)
        synthetic = pd.Series(["bob@company.com", "alice@site.net"] * 50)
        
        metrics = validator.validate(reference, synthetic, pii_type='email')
        
        leakage_metrics = [m for m in metrics if 'leakage' in m.name]
        assert len(leakage_metrics) > 0
        # Should have no exact matches
        assert leakage_metrics[0].value == 1.0
    
    def test_diversity_check(self, validator):
        """Test diversity validation"""
        reference = pd.Series(["john@example.com", "jane@test.org"] * 50)
        # Low diversity synthetic
        synthetic = pd.Series(["same@email.com"] * 100)
        
        metrics = validator.validate(reference, synthetic, pii_type='email')
        
        diversity_metrics = [m for m in metrics if 'diversity' in m.name]
        assert len(diversity_metrics) > 0
        # Low diversity should be detected
        assert diversity_metrics[0].value < 0.1


class TestTemporalQualityMetrics:
    """Test temporal quality validation"""
    
    @pytest.fixture
    def validator(self):
        return TemporalQualityMetrics(threshold=0.7)
    
    @pytest.fixture
    def reference_data(self):
        return pd.Series(pd.bdate_range('2023-01-01', '2023-12-31'))
    
    @pytest.fixture
    def good_synthetic(self):
        return pd.Series(pd.bdate_range('2023-02-01', '2023-11-30'))
    
    def test_range_validation(self, validator, reference_data, good_synthetic):
        """Test date range validation"""
        metrics = validator.validate(reference_data, good_synthetic)
        
        range_metrics = [m for m in metrics if 'range' in m.name]
        assert len(range_metrics) > 0
        assert range_metrics[0].value > 0.7
    
    def test_weekday_distribution(self, validator, reference_data, good_synthetic):
        """Test weekday distribution preservation"""
        metrics = validator.validate(reference_data, good_synthetic)
        
        weekday_metrics = [m for m in metrics if 'weekday' in m.name]
        assert len(weekday_metrics) > 0
        # Business days should match
        assert weekday_metrics[0].value > 0.8


class TestQualityValidator:
    """Test main quality validator"""
    
    @pytest.fixture
    def validator(self):
        config = get_default_config()
        return QualityValidator(config)
    
    @pytest.fixture
    def reference_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            'age': np.random.randint(18, 80, 100),
            'name': ['John Doe'] * 100,
            'signup_date': pd.date_range('2023-01-01', periods=100),
        })
    
    @pytest.fixture
    def synthetic_data(self):
        np.random.seed(43)
        return pd.DataFrame({
            'age': np.random.randint(18, 80, 100),
            'name': ['Jane Smith'] * 100,
            'signup_date': pd.date_range('2023-01-15', periods=100),
        })
    
    def test_full_validation(self, validator, reference_data, synthetic_data):
        """Test complete validation workflow"""
        report = validator.validate(reference_data, synthetic_data)
        
        assert report is not None
        assert hasattr(report, 'overall_score')
        assert hasattr(report, 'passed')
        assert 0.0 <= report.overall_score <= 1.0
    
    def test_column_scores(self, validator, reference_data, synthetic_data):
        """Test per-column scoring"""
        report = validator.validate(reference_data, synthetic_data)
        
        assert len(report.column_scores) > 0
        for score in report.column_scores.values():
            assert 0.0 <= score <= 1.0
    
    def test_report_summary(self, validator, reference_data, synthetic_data):
        """Test report summary generation"""
        report = validator.validate(reference_data, synthetic_data)
        
        assert 'total_metrics' in report.summary
        assert 'passed_metrics' in report.summary
        assert 'failed_metrics' in report.summary


class TestKAnonymity:
    """Test k-anonymity validation"""
    
    @pytest.fixture
    def checker(self):
        return KAnonymity(k=5)
    
    @pytest.fixture
    def anonymous_data(self):
        """Data that satisfies k=5"""
        return pd.DataFrame({
            'age': [25, 25, 25, 25, 25, 30, 30, 30, 30, 30] * 10,
            'zipcode': ['12345', '12345', '12345', '12345', '12345',
                       '54321', '54321', '54321', '54321', '54321'] * 10,
        })
    
    @pytest.fixture
    def non_anonymous_data(self):
        """Data that fails k=5"""
        return pd.DataFrame({
            'age': [25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
            'zipcode': ['12345', '12346', '12347', '12348', '12349',
                       '54321', '54322', '54323', '54324', '54325'],
        })
    
    def test_anonymous_data_passes(self, checker, anonymous_data):
        """Test that k-anonymous data passes"""
        is_anonymous, actual_k, details = checker.check(
            anonymous_data,
            quasi_identifiers=['age', 'zipcode']
        )
        
        assert is_anonymous
        assert actual_k >= 5
    
    def test_non_anonymous_data_fails(self, checker, non_anonymous_data):
        """Test that non-anonymous data fails"""
        is_anonymous, actual_k, details = checker.check(
            non_anonymous_data,
            quasi_identifiers=['age', 'zipcode']
        )
        
        assert not is_anonymous
        assert actual_k < 5
    
    def test_quasi_identifier_suggestion(self, checker):
        """Test quasi-identifier auto-detection"""
        data = pd.DataFrame({
            'id': range(100),  # Too unique
            'age': np.random.randint(20, 70, 100),  # Good QI
            'zipcode': np.random.choice(['12345', '54321', '98765'], 100),  # Good QI
            'name': ['John'] * 100,  # Too uniform
        })
        
        suggestions = checker.suggest_quasi_identifiers(data)
        
        assert 'age' in suggestions
        assert 'zipcode' in suggestions
        assert 'id' not in suggestions  # Too unique
        assert 'name' not in suggestions  # Too uniform


class TestReIdentificationRisk:
    """Test re-identification risk assessment"""
    
    @pytest.fixture
    def assessor(self):
        return ReIdentificationRisk()
    
    def test_low_risk_data(self, assessor):
        """Test low re-identification risk"""
        data = pd.DataFrame({
            'age_group': ['20-30', '20-30', '30-40', '30-40'] * 25,
            'gender': ['M', 'F', 'M', 'F'] * 25,
        })
        
        risk_score, risk_level, details = assessor.assess(
            data,
            quasi_identifiers=['age_group', 'gender']
        )
        
        assert risk_score < 0.5
        assert risk_level in ['low', 'medium']
    
    def test_high_risk_data(self, assessor):
        """Test high re-identification risk"""
        data = pd.DataFrame({
            'age': range(100),  # Unique ages
            'zipcode': [f'1234{i}' for i in range(100)],  # Unique zipcodes
        })
        
        risk_score, risk_level, details = assessor.assess(
            data,
            quasi_identifiers=['age', 'zipcode']
        )
        
        assert risk_score > 0.5
        assert risk_level in ['high', 'critical']


class TestUniquenessAnalyzer:
    """Test uniqueness analysis"""
    
    @pytest.fixture
    def analyzer(self):
        return UniquenessAnalyzer(rare_threshold=3)
    
    def test_uniqueness_analysis(self, analyzer):
        """Test basic uniqueness analysis"""
        data = pd.DataFrame({
            'age': [25, 25, 25, 30, 35, 40, 45, 50, 55, 60],
            'zipcode': ['12345'] * 3 + list(range(7)),
        })
        
        results = analyzer.analyze(data, columns=['age', 'zipcode'])
        
        assert 'unique_records' in results
        assert 'rare_records' in results
        assert results['unique_percentage'] >= 0
    
    def test_find_unique_records(self, analyzer):
        """Test finding unique records"""
        data = pd.DataFrame({
            'age': [25, 25, 30, 30, 35],  # 35 is unique
            'zipcode': ['12345', '12345', '54321', '54321', '99999'],
        })
        
        unique_records = analyzer.find_unique_records(data, columns=['age', 'zipcode'])
        
        assert len(unique_records) == 1
        assert unique_records.iloc[0]['age'] == 35


class TestPrivacyValidator:
    """Test main privacy validator"""
    
    @pytest.fixture
    def validator(self):
        config = get_default_config()
        return PrivacyValidator(config)
    
    @pytest.fixture
    def synthetic_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            'age': np.random.randint(20, 70, 100),
            'zipcode': np.random.choice(['12345', '23456', '34567'], 100),
            'gender': np.random.choice(['M', 'F'], 100),
        })
    
    @pytest.fixture
    def reference_data(self):
        np.random.seed(43)
        return pd.DataFrame({
            'age': np.random.randint(20, 70, 100),
            'zipcode': np.random.choice(['12345', '23456', '34567'], 100),
            'gender': np.random.choice(['M', 'F'], 100),
        })
    
    def test_full_validation(self, validator, synthetic_data, reference_data):
        """Test complete privacy validation"""
        report = validator.validate(
            synthetic=synthetic_data,
            reference=reference_data,
            quasi_identifiers=['age', 'zipcode', 'gender']
        )
        
        assert report is not None
        assert report.overall_risk in ['low', 'medium', 'high', 'critical']
    
    def test_k_anonymity_score(self, validator, synthetic_data, reference_data):
        """Test k-anonymity scoring"""
        report = validator.validate(
            synthetic=synthetic_data,
            reference=reference_data,
            quasi_identifiers=['age', 'zipcode', 'gender']
        )
        
        assert report.k_anonymity_score is not None
        assert report.k_anonymity_score >= 0
    
    def test_recommendations(self, validator, synthetic_data, reference_data):
        """Test privacy recommendations generation"""
        report = validator.validate(
            synthetic=synthetic_data,
            reference=reference_data,
            quasi_identifiers=['age', 'zipcode', 'gender']
        )
        
        assert len(report.recommendations) > 0
        assert all(isinstance(rec, str) for rec in report.recommendations)


# Integration tests
class TestValidationIntegration:
    """Integration tests for validation"""
    
    def test_quality_and_privacy_together(self):
        """Test running both quality and privacy validation"""
        np.random.seed(42)
        reference = pd.DataFrame({
            'age': np.random.randint(20, 70, 100),
            'income': np.random.normal(50000, 15000, 100),
            'zipcode': np.random.choice(['12345', '23456'], 100),
        })
        
        np.random.seed(43)
        synthetic = pd.DataFrame({
            'age': np.random.randint(20, 70, 100),
            'income': np.random.normal(50000, 15000, 100),
            'zipcode': np.random.choice(['12345', '23456'], 100),
        })
        
        config = get_default_config()
        
        # Quality validation
        quality_validator = QualityValidator(config)
        quality_report = quality_validator.validate(reference, synthetic)
        
        # Privacy validation
        privacy_validator = PrivacyValidator(config)
        privacy_report = privacy_validator.validate(synthetic, reference)
        
        assert quality_report.overall_score > 0
        assert privacy_report.overall_risk is not None


# Performance tests
class TestValidationPerformance:
    """Performance tests for validation"""
    
    def test_large_dataset_quality(self):
        """Test quality validation on large dataset"""
        np.random.seed(42)
        reference = pd.DataFrame({
            'col1': np.random.randn(10000),
            'col2': np.random.randn(10000),
        })
        
        synthetic = pd.DataFrame({
            'col1': np.random.randn(10000),
            'col2': np.random.randn(10000),
        })
        
        config = get_default_config()
        validator = QualityValidator(config)
        
        import time
        start = time.time()
        report = validator.validate(reference, synthetic)
        elapsed = time.time() - start
        
        assert report is not None
        # Should complete in reasonable time
        assert elapsed < 10.0
    
    def test_large_dataset_privacy(self):
        """Test privacy validation on large dataset"""
        np.random.seed(42)
        data = pd.DataFrame({
            'age': np.random.randint(20, 70, 10000),
            'zipcode': np.random.choice(['12345', '23456', '34567'], 10000),
        })
        
        config = get_default_config()
        validator = PrivacyValidator(config)
        
        import time
        start = time.time()
        report = validator.validate(synthetic=data)
        elapsed = time.time() - start
        
        assert report is not None
        # Should complete in reasonable time
        assert elapsed < 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
