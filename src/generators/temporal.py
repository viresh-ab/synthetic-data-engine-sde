"""
Temporal Data Generator Module

Generates synthetic temporal data (dates, times, datetimes) with:
- Pattern preservation (weekdays, seasonality)
- Business day awareness
- Realistic date ranges
- Format consistency
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from faker import Faker
import logging

from ..orchestrator import GenerationPipeline, PipelineType, DatasetSchema

logger = logging.getLogger(__name__)


class TemporalGenerator(GenerationPipeline):
    """
    Generates synthetic temporal data

    Features:
    - Preserves date ranges from reference
    - Maintains day-of-week distributions
    - Supports business day filtering
    - Handles multiple date formats
    - Preserves seasonal patterns
    """

    def __init__(self, config):
        """
        Initialize temporal generator

        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.pipeline_type = PipelineType.TEMPORAL
        self.preserve_patterns = config.temporal.preserve_patterns
        self.business_days_only = config.temporal.business_days_only
        self.timezone = config.temporal.timezone
        self.faker = Faker()

    def generate(
        self,
        schema: DatasetSchema,
        column_names: List[str],
        num_rows: int,
        reference_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic temporal data

        Args:
            schema: Dataset schema with column metadata
            column_names: List of temporal columns to generate
            num_rows: Number of rows to generate
            reference_data: Optional reference data for pattern learning

        Returns:
            DataFrame with generated temporal columns
        """
        logger.info(
            f"Generating {num_rows} rows for {len(column_names)} temporal columns")

        result = pd.DataFrame()

        for col in column_names:
            meta = schema.get_column(col)

            if meta is None:
                logger.warning(f"No metadata for column {col}, using defaults")
                result[col] = self._generate_random_dates(num_rows)
                continue

            # Get reference data for this column if available
            ref_series = None
            if reference_data is not None and col in reference_data.columns:
                ref_series = reference_data[col]

            # Generate dates
            if self.preserve_patterns and ref_series is not None:
                dates = self._generate_pattern_preserving(
                    num_rows, ref_series, meta)
            else:
                dates = self._generate_from_range(num_rows, meta)

            # Filter to business days if needed
            if self.business_days_only:
                dates = self._ensure_business_days(dates)

            # Convert to appropriate format
            if meta.date_format:
                dates_series = pd.Series(pd.to_datetime(dates))
                result[col] = dates_series.dt.strftime(meta.date_format)
            else:
                result[col] = pd.Series(pd.to_datetime(dates))

        logger.info(f"Generated temporal data: {result.shape}")
        return result

    def _generate_from_range(self, num_rows: int, meta) -> List[datetime]:
        """
        Generate dates within specified range

        Args:
            num_rows: Number of dates to generate
            meta: Column metadata with date range

        Returns:
            List of datetime objects
        """
        # Get date range from metadata
        if meta.min_date and meta.max_date:
            min_date = pd.to_datetime(meta.min_date)
            max_date = pd.to_datetime(meta.max_date)
        else:
            # Default to last 2 years
            max_date = datetime.now()
            min_date = max_date - timedelta(days=730)

        # Generate random dates in range
        date_range = (max_date - min_date).days
        random_days = np.random.randint(0, date_range, num_rows)

        dates = [min_date + timedelta(days=int(d)) for d in random_days]

        return dates

    def _generate_pattern_preserving(
        self,
        num_rows: int,
        reference: pd.Series,
        meta
    ) -> List[datetime]:
        """
        Generate dates preserving patterns from reference

        Args:
            num_rows: Number of dates to generate
            reference: Reference date series
            meta: Column metadata

        Returns:
            List of datetime objects
        """
        # Convert reference to datetime
        ref_dates = pd.Series(pd.to_datetime(reference.dropna()))

        if len(ref_dates) == 0:
            return self._generate_from_range(num_rows, meta)

        # Analyze patterns
        weekday_dist = ref_dates.dt.dayofweek.value_counts(
            normalize=True).sort_index()
        month_dist = ref_dates.dt.month.value_counts(
            normalize=True).sort_index()

        # Get date range
        min_date = ref_dates.min()
        max_date = ref_dates.max()

        # Generate dates preserving weekday distribution
        dates = []
        for _ in range(num_rows):
            # Sample weekday based on distribution
            weekdays = weekday_dist.index.tolist()
            probs = weekday_dist.values
            target_weekday = np.random.choice(weekdays, p=probs)

            # Sample month based on distribution
            months = month_dist.index.tolist()
            month_probs = month_dist.values
            target_month = np.random.choice(months, p=month_probs)

            # Generate random date in range with target month
            attempts = 0
            while attempts < 100:
                date_range = (max_date - min_date).days
                random_days = np.random.randint(0, date_range)
                candidate_date = min_date + timedelta(days=int(random_days))

                # Check if matches target patterns
                if (candidate_date.weekday() == target_weekday and
                        candidate_date.month == target_month):
                    dates.append(candidate_date)
                    break

                attempts += 1

            if attempts >= 100:
                # Fallback: just match weekday
                while True:
                    random_days = np.random.randint(0, date_range)
                    candidate_date = min_date + \
                        timedelta(days=int(random_days))
                    if candidate_date.weekday() == target_weekday:
                        dates.append(candidate_date)
                        break

        return dates

    def _generate_random_dates(
        self,
        num_rows: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[datetime]:
        """
        Generate random dates

        Args:
            num_rows: Number of dates to generate
            start_date: Start of range (default: 2 years ago)
            end_date: End of range (default: today)

        Returns:
            List of datetime objects
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=730)

        dates = [
            self.faker.date_between(start_date=start_date, end_date=end_date)
            for _ in range(num_rows)
        ]

        return dates

    def _ensure_business_days(self, dates: List[datetime]) -> List[datetime]:
        """
        Convert weekend dates to business days

        Args:
            dates: List of dates

        Returns:
            List of dates with weekends converted to weekdays
        """
        business_dates = []

        for date in dates:
            # If weekend, move to next Monday
            if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                days_to_add = 7 - date.weekday()
                date = date + timedelta(days=days_to_add)

            business_dates.append(date)

        return business_dates

    def validate(self, generated_data: pd.DataFrame, schema: DatasetSchema) -> Dict[str, Any]:
        """
        Validate generated temporal data

        Args:
            generated_data: Generated data to validate
            schema: Expected schema

        Returns:
            Validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        for col in generated_data.columns:
            meta = schema.get_column(col)
            if not meta:
                continue

            try:
                dates = pd.Series(pd.to_datetime(generated_data[col]))

                # Check for NaT (Not a Time)
                if dates.isna().any():
                    results["errors"].append(f"{col}: Contains invalid dates")
                    results["valid"] = False

                # Check date range
                if meta.min_date and meta.max_date:
                    min_ref = pd.to_datetime(meta.min_date)
                    max_ref = pd.to_datetime(meta.max_date)

                    # Allow 10% buffer
                    buffer = (max_ref - min_ref) * 0.1

                    if dates.min() < min_ref - buffer:
                        results["warnings"].append(
                            f"{col}: Some dates before reference range"
                        )

                    if dates.max() > max_ref + buffer:
                        results["warnings"].append(
                            f"{col}: Some dates after reference range"
                        )

                # Check for business days if required
                if self.business_days_only:
                    weekend_count = (dates.dt.dayofweek >= 5).sum()
                    if weekend_count > 0:
                        results["errors"].append(
                            f"{col}: Contains {weekend_count} weekend dates"
                        )
                        results["valid"] = False

            except Exception as e:
                results["errors"].append(
                    f"{col}: Failed to parse as dates - {str(e)}")
                results["valid"] = False

        return results


# Example usage
if __name__ == "__main__":
    import logging
    from ..config import get_default_config

    logging.basicConfig(level=logging.INFO)

    # Create reference data
    reference = pd.DataFrame({
        'signup_date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'purchase_date': pd.date_range('2023-06-01', periods=100, freq='D'),
    })

    # Create config and generator
    config = get_default_config()
    generator = TemporalGenerator(config)

    # Create mock schema
    from ..orchestrator import DatasetSchema, ColumnMetadata, DataType
    schema = DatasetSchema()

    for col in reference.columns:
        meta = ColumnMetadata(
            name=col,
            data_type=DataType.TEMPORAL,
            pipeline=PipelineType.TEMPORAL,
            min_date=str(reference[col].min()),
            max_date=str(reference[col].max()),
            date_format='%Y-%m-%d'
        )
        schema.add_column(meta)

    # Generate data
    synthetic = generator.generate(
        schema=schema,
        column_names=list(reference.columns),
        num_rows=50,
        reference_data=reference
    )

    print("\nGenerated temporal data:")
    print(synthetic.head(10))
    print(f"\nDate range:")
    for col in synthetic.columns:
        dates = pd.to_datetime(synthetic[col])
        print(f"{col}: {dates.min()} to {dates.max()}")
