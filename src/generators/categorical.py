"""
Categorical Data Generator Module

Generates synthetic categorical data with:
- Frequency preservation
- Relationship preservation (e.g., city-country)
- Unique category generation
- Pattern matching
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Set
from faker import Faker
import logging

from ..orchestrator import GenerationPipeline, PipelineType, DatasetSchema

logger = logging.getLogger(__name__)


class CategoricalGenerator(GenerationPipeline):
    """
    Generates synthetic categorical data
    
    Features:
    - Preserves frequency distributions
    - Maintains relationships between columns
    - Handles hierarchical data (country -> city)
    - Realistic category generation
    """
    

    DEFAULT_COUNTRY_CITY_MAP: Dict[str, List[str]] = {
        'United States': ['New York', 'Los Angeles', 'Chicago', 'Austin', 'Seattle'],
        'USA': ['New York', 'Los Angeles', 'Chicago', 'Austin', 'Seattle'],
        'India': ['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Chennai'],
        'United Kingdom': ['London', 'Manchester', 'Birmingham', 'Liverpool', 'Leeds'],
        'UK': ['London', 'Manchester', 'Birmingham', 'Liverpool', 'Leeds'],
        'Canada': ['Toronto', 'Vancouver', 'Montreal', 'Calgary', 'Ottawa'],
        'Australia': ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide'],
        'Germany': ['Berlin', 'Munich', 'Hamburg', 'Frankfurt', 'Cologne'],
        'Singapore': ['Singapore'],
    }

    COUNTRY_ALIASES: Dict[str, str] = {
        'us': 'United States',
        'u.s.': 'United States',
        'usa': 'United States',
        'united states of america': 'United States',
        'uk': 'United Kingdom',
        'u.k.': 'United Kingdom',
        'great britain': 'United Kingdom',
    }
    def __init__(self, config):
        """
        Initialize categorical generator
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.pipeline_type = PipelineType.HYBRID  # Categorical uses hybrid pipeline
        self.faker = Faker()
        
        # Relationship mappings
        self.country_city_map: Dict[str, List[str]] = {}
        self._bootstrap_default_country_city_map()
    
    def generate(
        self,
        schema: DatasetSchema,
        column_names: List[str],
        num_rows: int,
        reference_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic categorical data
        
        Args:
            schema: Dataset schema with column metadata
            column_names: List of categorical columns to generate
            num_rows: Number of rows to generate
            reference_data: Optional reference data for pattern learning
            
        Returns:
            DataFrame with generated categorical columns
        """
        logger.info(f"Generating {num_rows} rows for {len(column_names)} categorical columns")
        
        result = pd.DataFrame()
        
        # Identify related columns
        country_cols = [col for col in column_names if 'country' in col.lower()]
        city_cols = [col for col in column_names if 'city' in col.lower()]
        other_cols = [col for col in column_names if col not in country_cols + city_cols]
        
        # Build relationship map from reference data
        if reference_data is not None and country_cols and city_cols:
            self._build_country_city_map(reference_data, country_cols[0], city_cols[0])
        
        # Generate countries first
        for col in country_cols:
            result[col] = self._generate_countries(num_rows, schema.get_column(col), reference_data)
        
        # Generate cities based on countries
        for col in city_cols:
            if country_cols and country_cols[0] in result.columns:
                result[col] = self._generate_cities_for_countries(
                    result[country_cols[0]], 
                    schema.get_column(col),
                    reference_data
                )
            else:
                result[col] = self._generate_cities(num_rows, schema.get_column(col), reference_data)
        
        # Generate other categorical columns
        for col in other_cols:
            result[col] = self._generate_categorical(
                num_rows,
                col,
                schema.get_column(col),
                reference_data
            )
        
        logger.info(f"Generated categorical data: {result.shape}")
        return result
    

    def _normalize_country(self, country: Any) -> str:
        """Normalize country names to canonical form."""
        country_str = str(country).strip()
        lower = country_str.lower()
        if lower in self.COUNTRY_ALIASES:
            return self.COUNTRY_ALIASES[lower]
        return country_str

    def _bootstrap_default_country_city_map(self):
        """Ensure there is always a base country-city mapping available."""
        for country, cities in self.DEFAULT_COUNTRY_CITY_MAP.items():
            normalized = self._normalize_country(country)
            existing = set(self.country_city_map.get(normalized, []))
            existing.update(cities)
            self.country_city_map[normalized] = sorted(existing)

    def _build_country_city_map(
        self,
        reference_data: pd.DataFrame,
        country_col: str,
        city_col: str
    ):
        """
        Build mapping of countries to their cities from reference data
        
        Args:
            reference_data: Reference DataFrame
            country_col: Name of country column
            city_col: Name of city column
        """
        if country_col not in reference_data.columns or city_col not in reference_data.columns:
            return
        
        # Group cities by country
        grouped = reference_data.groupby(country_col)[city_col].apply(list).to_dict()
        
        # Store unique cities for each country
        for country, cities in grouped.items():
            normalized_country = self._normalize_country(country)
            existing = set(self.country_city_map.get(normalized_country, []))
            existing.update(city for city in cities if pd.notna(city))
            self.country_city_map[normalized_country] = sorted(existing)
        
        logger.debug(f"Built country-city map with {len(self.country_city_map)} countries")
    
    def _generate_countries(
        self,
        num_rows: int,
        meta,
        reference_data: Optional[pd.DataFrame]
    ) -> List[str]:
        """
        Generate countries preserving frequency distribution
        
        Args:
            num_rows: Number of countries to generate
            meta: Column metadata
            reference_data: Reference data
            
        Returns:
            List of countries
        """
        if meta and meta.categories and meta.category_frequencies:
            # Use reference distribution
            categories = list(meta.category_frequencies.keys())
            frequencies = list(meta.category_frequencies.values())
            
            # Normalize frequencies
            frequencies = np.array(frequencies)
            frequencies = frequencies / frequencies.sum()
            
            countries = np.random.choice(categories, size=num_rows, p=frequencies)
            return countries.tolist()
        else:
            # Generate random countries
            countries = [self.faker.country() for _ in range(num_rows)]
            return countries
    
    def _generate_cities(
        self,
        num_rows: int,
        meta,
        reference_data: Optional[pd.DataFrame]
    ) -> List[str]:
        """
        Generate cities preserving frequency distribution
        
        Args:
            num_rows: Number of cities to generate
            meta: Column metadata
            reference_data: Reference data
            
        Returns:
            List of cities
        """
        if meta and meta.categories and meta.category_frequencies:
            # Use reference distribution
            categories = list(meta.category_frequencies.keys())
            frequencies = list(meta.category_frequencies.values())
            
            # Normalize frequencies
            frequencies = np.array(frequencies)
            frequencies = frequencies / frequencies.sum()
            
            cities = np.random.choice(categories, size=num_rows, p=frequencies)
            return cities.tolist()
        else:
            # Generate random cities
            cities = [self.faker.city() for _ in range(num_rows)]
            return cities
    
    def _generate_cities_for_countries(
        self,
        countries: pd.Series,
        meta,
        reference_data: Optional[pd.DataFrame]
    ) -> List[str]:
        """
        Generate cities that match their countries
        
        Args:
            countries: Series of countries
            meta: Column metadata
            reference_data: Reference data
            
        Returns:
            List of cities matching countries
        """
        cities = []
        
        for country in countries:
            normalized_country = self._normalize_country(country)

            # Get valid cities for this country
            if normalized_country in self.country_city_map:
                valid_cities = self.country_city_map[normalized_country]
                city = np.random.choice(valid_cities)
            else:
                city = self.faker.city()

            cities.append(city)
        
        return cities
    
    def _generate_categorical(
        self,
        num_rows: int,
        column_name: str,
        meta,
        reference_data: Optional[pd.DataFrame]
    ) -> List[Any]:
        """
        Generate generic categorical data
        
        Args:
            num_rows: Number of values to generate
            column_name: Name of column
            meta: Column metadata
            reference_data: Reference data
            
        Returns:
            List of categorical values
        """
        if meta and meta.categories and meta.category_frequencies:
            # Use reference distribution
            categories = list(meta.category_frequencies.keys())
            frequencies = list(meta.category_frequencies.values())
            
            # Normalize frequencies
            frequencies = np.array(frequencies)
            frequencies = frequencies / frequencies.sum()
            
            values = np.random.choice(categories, size=num_rows, p=frequencies)
            return values.tolist()
        else:
            # Generate random categorical values
            # Default categories based on column name
            if 'status' in column_name.lower():
                categories = ['Active', 'Inactive', 'Pending', 'Completed']
            elif 'type' in column_name.lower():
                categories = ['Type A', 'Type B', 'Type C']
            elif 'category' in column_name.lower():
                categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
            else:
                categories = ['Option 1', 'Option 2', 'Option 3']
            
            values = np.random.choice(categories, size=num_rows)
            return values.tolist()
    
    def validate(self, generated_data: pd.DataFrame, schema: DatasetSchema) -> Dict[str, Any]:
        """
        Validate generated categorical data
        
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
        
        # Check for valid categories
        for col in generated_data.columns:
            meta = schema.get_column(col)
            if not meta or not meta.categories:
                continue
            
            # Check if all values are in expected categories
            valid_categories = set(meta.categories)
            actual_categories = set(generated_data[col].unique())
            
            invalid = actual_categories - valid_categories
            if invalid:
                results["warnings"].append(
                    f"{col}: Contains unexpected categories: {invalid}"
                )
        
        # Validate city-country relationships
        country_cols = [col for col in generated_data.columns if 'country' in col.lower()]
        city_cols = [col for col in generated_data.columns if 'city' in col.lower()]
        
        if country_cols and city_cols and self.country_city_map:
            country_col = country_cols[0]
            city_col = city_cols[0]
            
            # Check for invalid city-country pairs
            invalid_pairs = 0
            for idx, row in generated_data.iterrows():
                country = self._normalize_country(row[country_col])
                city = row[city_col]

                if country in self.country_city_map:
                    if city not in self.country_city_map[country]:
                        invalid_pairs += 1
            
            if invalid_pairs > 0:
                results["warnings"].append(
                    f"Found {invalid_pairs} invalid city-country pairs"
                )
        
        return results


# Example usage
if __name__ == "__main__":
    import logging
    from ..config import get_default_config
    
    logging.basicConfig(level=logging.INFO)
    
    # Create reference data with city-country relationships
    reference = pd.DataFrame({
        'country': ['USA', 'USA', 'UK', 'UK', 'Australia'] * 20,
        'city': ['New York', 'Chicago', 'London', 'Manchester', 'Sydney'] * 20,
        'status': np.random.choice(['Active', 'Inactive'], 100),
    })
    
    # Create config and generator
    config = get_default_config()
    generator = CategoricalGenerator(config)
    
    # Create mock schema
    from ..orchestrator import DatasetSchema, ColumnMetadata, DataType
    schema = DatasetSchema()
    
    for col in reference.columns:
        value_counts = reference[col].value_counts()
        meta = ColumnMetadata(
            name=col,
            data_type=DataType.CATEGORICAL,
            pipeline=PipelineType.HYBRID,
            categories=value_counts.index.tolist(),
            category_frequencies=(value_counts / len(reference)).to_dict()
        )
        schema.add_column(meta)
    
    # Generate data
    synthetic = generator.generate(
        schema=schema,
        column_names=list(reference.columns),
        num_rows=50,
        reference_data=reference
    )
    
    print("\nGenerated categorical data:")
    print(synthetic.head(10))
    print(f"\nValue counts for country:")
    print(synthetic['country'].value_counts())
    print(f"\nSample city-country pairs:")
    print(synthetic[['country', 'city']].head(10))