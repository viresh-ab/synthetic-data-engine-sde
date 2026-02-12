"""
PII (Personally Identifiable Information) Generator

Generates synthetic PII that is realistic but entirely fictitious:
- Names (first, last, full names with titles)
- Email addresses
- Phone numbers (international formats)
- Physical addresses
- Identifiers (SSN, UUID, custom IDs)
"""

import random
import string
import uuid
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NameComponents:
    """Components of a person's name"""
    first_name: str
    last_name: str
    middle_name: Optional[str] = None
    title: Optional[str] = None
    suffix: Optional[str] = None
    
    def full_name(self, include_middle: bool = False, include_title: bool = False) -> str:
        """Generate full name string"""
        parts = []
        
        if include_title and self.title:
            parts.append(self.title)
        
        parts.append(self.first_name)
        
        if include_middle and self.middle_name:
            parts.append(self.middle_name)
        
        parts.append(self.last_name)
        
        if self.suffix:
            parts[-1] = f"{parts[-1]} {self.suffix}"
        
        return " ".join(parts)


class NameGenerator:
    """
    Generate realistic person names
    
    Supports multiple locales and formats
    """
    
    # Name data by locale
    COUNTRY_TO_LOCALE = {
        'united states': 'en_US',
        'usa': 'en_US',
        'us': 'en_US',
        'united kingdom': 'en_GB',
        'uk': 'en_GB',
        'great britain': 'en_GB',
        'india': 'en_IN',
        'canada': 'en_CA',
        'australia': 'en_AU',
    }

    CITY_TO_LOCALE = {
        'mumbai': 'en_IN', 'delhi': 'en_IN', 'new delhi': 'en_IN', 'bengaluru': 'en_IN',
        'bangalore': 'en_IN', 'hyderabad': 'en_IN', 'chennai': 'en_IN', 'kolkata': 'en_IN',
        'pune': 'en_IN', 'ahmedabad': 'en_IN', 'jaipur': 'en_IN', 'lucknow': 'en_IN',
        'patna': 'en_IN', 'kochi': 'en_IN', 'gurgaon': 'en_IN', 'noida': 'en_IN',
    }

    FIRST_NAMES = {
        'en_US': {
            'male': [
                'James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard',
                'Joseph', 'Thomas', 'Charles', 'Christopher', 'Daniel', 'Matthew',
                'Anthony', 'Mark', 'Donald', 'Steven', 'Paul', 'Andrew', 'Joshua',
                'Kenneth', 'Kevin', 'Brian', 'George', 'Timothy', 'Ronald', 'Edward',
                'Jason', 'Jeffrey', 'Ryan', 'Jacob', 'Gary', 'Nicholas', 'Eric',
            ],
            'female': [
                'Mary', 'Patricia', 'Jennifer', 'Linda', 'Barbara', 'Elizabeth',
                'Susan', 'Jessica', 'Sarah', 'Karen', 'Lisa', 'Nancy', 'Betty',
                'Margaret', 'Sandra', 'Ashley', 'Kimberly', 'Emily', 'Donna',
                'Michelle', 'Carol', 'Amanda', 'Dorothy', 'Melissa', 'Deborah',
                'Stephanie', 'Rebecca', 'Sharon', 'Laura', 'Cynthia', 'Kathleen',
            ]
        },
        'en_IN': {
            'male': [
                'Aarav', 'Arjun', 'Rohan', 'Vikram', 'Karan', 'Aditya', 'Rahul',
                'Siddharth', 'Aman', 'Yash', 'Ankit', 'Nikhil', 'Rajat', 'Manish',
            ],
            'female': [
                'Aanya', 'Priya', 'Ananya', 'Ishita', 'Sneha', 'Kavya', 'Diya',
                'Riya', 'Pooja', 'Neha', 'Shreya', 'Meera', 'Tanvi', 'Nisha',
            ]
        },
        'en_GB': {
            'male': ['Oliver', 'George', 'Harry', 'Noah', 'Jack', 'Leo', 'Charlie'],
            'female': ['Olivia', 'Amelia', 'Isla', 'Ava', 'Mia', 'Grace', 'Lily'],
        },
        'en_AU': {
            'male': ['Lachlan', 'Cooper', 'Hudson', 'Will', 'Ethan', 'Lucas'],
            'female': ['Matilda', 'Chloe', 'Zoe', 'Charlotte', 'Sienna', 'Ruby'],
        },
        'en_CA': {
            'male': ['Liam', 'Noah', 'William', 'Benjamin', 'Logan', 'Mason'],
            'female': ['Emma', 'Sophia', 'Avery', 'Mila', 'Ella', 'Hannah'],
        }
    }
    
    LAST_NAMES = {
        'en_US': [
            'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
            'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
            'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
            'Lee', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez',
            'Lewis', 'Robinson', 'Walker', 'Young', 'Allen', 'King', 'Wright',
            'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores', 'Green', 'Adams',
            'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 'Mitchell', 'Carter',
        ],
        'en_IN': [
            'Sharma', 'Patel', 'Singh', 'Gupta', 'Verma', 'Reddy', 'Iyer',
            'Nair', 'Mehta', 'Joshi', 'Chopra', 'Kapoor', 'Bhatia', 'Agarwal',
        ],
        'en_GB': ['Smith', 'Jones', 'Taylor', 'Brown', 'Wilson', 'Evans', 'Thomas'],
        'en_AU': ['Williams', 'Taylor', 'Brown', 'Wilson', 'Martin', 'Thompson'],
        'en_CA': ['Tremblay', 'Roy', 'Gagnon', 'Lee', 'Smith', 'Brown'],
    }
    
    MIDDLE_NAMES = {
        'en_US': [
            'Ann', 'Marie', 'Lee', 'Lynn', 'Rose', 'Grace', 'Jean', 'May',
            'James', 'Michael', 'David', 'Joseph', 'Allen', 'Ray', 'Dean',
        ],
        'en_IN': ['Kumar', 'Devi', 'Raj', 'Lal', 'Prasad'],
        'en_GB': ['James', 'Rose', 'May', 'George'],
        'en_AU': ['Jay', 'Skye', 'Lee', 'Anne'],
        'en_CA': ['Jean', 'Marie', 'Anne', 'Paul'],
    }
    
    TITLES = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.']
    SUFFIXES = ['Jr.', 'Sr.', 'II', 'III', 'IV']
    
    def __init__(self, locale: str = 'en_US', gender_distribution: Optional[Dict[str, float]] = None):
        """
        Initialize name generator
        
        Args:
            locale: Locale for names
            gender_distribution: Distribution of genders (e.g., {'male': 0.5, 'female': 0.5})
        """
        self.locale = locale
        self.gender_distribution = gender_distribution or {'male': 0.5, 'female': 0.5}
    
    def generate(
        self,
        num_names: int = 1,
        gender: Optional[str] = None,
        include_middle: bool = False,
        include_title: bool = False,
        include_suffix: bool = False
    ) -> List[str]:
        """
        Generate full names
        
        Args:
            num_names: Number of names to generate
            gender: Specific gender or None for random
            include_middle: Include middle name
            include_title: Include title (Mr., Mrs., etc.)
            include_suffix: Include suffix (Jr., Sr., etc.)
            
        Returns:
            List of generated names
        """
        names = []
        
        for _ in range(num_names):
            components = self._generate_components(
                gender=gender,
                include_middle=include_middle,
                include_title=include_title,
                include_suffix=include_suffix
            )
            
            full_name = components.full_name(
                include_middle=include_middle,
                include_title=include_title
            )
            
            names.append(full_name)
        
        return names
    
    def _generate_components(
        self,
        gender: Optional[str] = None,
        include_middle: bool = False,
        include_title: bool = False,
        include_suffix: bool = False
    ) -> NameComponents:
        """Generate name components"""
        # Determine gender
        if gender is None:
            gender = np.random.choice(
                list(self.gender_distribution.keys()),
                p=list(self.gender_distribution.values())
            )
        
        # Get names from data
        first_names = self.FIRST_NAMES.get(self.locale, {}).get(gender, ['John', 'Jane'])
        last_names = self.LAST_NAMES.get(self.locale, ['Smith'])
        middle_names = self.MIDDLE_NAMES.get(self.locale, ['M'])
        
        components = NameComponents(
            first_name=random.choice(first_names),
            last_name=random.choice(last_names),
        )
        
        if include_middle:
            components.middle_name = random.choice(middle_names)
        
        if include_title:
            if gender == 'male':
                components.title = random.choice(['Mr.', 'Dr.', 'Prof.'])
            else:
                components.title = random.choice(['Ms.', 'Mrs.', 'Dr.', 'Prof.'])
        
        if include_suffix and random.random() < 0.1:  # 10% chance
            components.suffix = random.choice(self.SUFFIXES)
        
        return components
    
    def generate_first_names(self, num_names: int = 1, gender: Optional[str] = None) -> List[str]:
        """Generate only first names"""
        names = []
        
        for _ in range(num_names):
            if gender is None:
                g = np.random.choice(
                    list(self.gender_distribution.keys()),
                    p=list(self.gender_distribution.values())
                )
            else:
                g = gender
            
            first_names = self.FIRST_NAMES.get(self.locale, {}).get(g, ['John'])
            names.append(random.choice(first_names))
        
        return names
    
    def generate_last_names(self, num_names: int = 1) -> List[str]:
        """Generate only last names"""
        last_names = self.LAST_NAMES.get(self.locale, ['Smith'])
        return [random.choice(last_names) for _ in range(num_names)]

    @classmethod
    def locale_from_country(cls, country: Optional[str], default_locale: str = 'en_US') -> str:
        """Map country to locale for region-aware name generation."""
        if not country:
            return default_locale

        normalized = str(country).strip().lower()
        return cls.COUNTRY_TO_LOCALE.get(normalized, default_locale)

    @classmethod
    def locale_from_city(cls, city: Optional[str], default_locale: str = 'en_US') -> str:
        """Map city to locale for region-aware name generation."""
        if not city:
            return default_locale

        normalized = str(city).strip().lower()
        return cls.CITY_TO_LOCALE.get(normalized, default_locale)


class EmailGenerator:
    """
    Generate realistic email addresses
    
    Formats:
    - firstname.lastname@domain.com
    - firstnamelastname@domain.com
    - firstname123@domain.com
    - f.lastname@domain.com
    """
    
    DOMAINS = [
        'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
        'icloud.com', 'aol.com', 'live.com', 'msn.com',
         ]
    
    def __init__(self, domains: Optional[List[str]] = None):
        """
        Initialize email generator
        
        Args:
            domains: Custom domain list (uses defaults if None)
        """
        self.domains = domains or self.DOMAINS

    @staticmethod
    def _sanitize_name(value: str) -> str:
        """Normalize names for email local-part generation."""
        cleaned = ''.join(ch for ch in str(value).lower() if ch.isalpha())
        return cleaned or 'user'
    
    def generate(
        self,
        num_emails: int = 1,
        first_names: Optional[List[str]] = None,
        last_names: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate email addresses
        
        Args:
            num_emails: Number of emails to generate
            first_names: Optional list of first names to use
            last_names: Optional list of last names to use
            
        Returns:
            List of email addresses
        """
        emails = []
        
        for i in range(num_emails):
            # Get names
            if first_names and i < len(first_names):
                first = self._sanitize_name(first_names[i])
            else:
                first = f"user{i}"

            if last_names and i < len(last_names):
                last = self._sanitize_name(last_names[i])
            else:
                last = f"name{i}"
            
            # Choose format
            format_type = random.choice(['dot', 'concat', 'number', 'initial'])
            domain = random.choice(self.domains)
            
            if format_type == 'dot':
                email = f"{first.lower()}.{last.lower()}@{domain}"
            elif format_type == 'concat':
                email = f"{first.lower()}{last.lower()}@{domain}"
            elif format_type == 'number':
                num = random.randint(1, 999)
                email = f"{first.lower()}{num}@{domain}"
            else:  # initial
                email = f"{first[0].lower()}.{last.lower()}@{domain}"
            
            emails.append(email)
        
        return emails


class PhoneGenerator:
    """
    Generate phone numbers in various formats
    
    Supports multiple country formats
    """
    
    def __init__(self, country: str = 'US'):
        """
        Initialize phone generator
        
        Args:
            country: Country code for phone format
        """
        self.country = country
    
    def generate(self, num_phones: int = 1, format_style: str = 'standard') -> List[str]:
        """
        Generate phone numbers
        
        Args:
            num_phones: Number to generate
            format_style: 'standard', 'dashes', 'dots', 'spaces', 'parentheses'
            
        Returns:
            List of phone numbers
        """
        phones = []
        
        for _ in range(num_phones):
            if self.country == 'US':
                phone = self._generate_us_phone(format_style)
            else:
                phone = self._generate_generic_phone()
            
            phones.append(phone)
        
        return phones
    
    def _generate_us_phone(self, format_style: str) -> str:
        """Generate US phone number"""
        area_code = random.randint(200, 999)
        exchange = random.randint(200, 999)
        number = random.randint(1000, 9999)
        
        if format_style == 'standard':
            return f"{area_code}{exchange}{number}"
        elif format_style == 'dashes':
            return f"{area_code}-{exchange}-{number}"
        elif format_style == 'dots':
            return f"{area_code}.{exchange}.{number}"
        elif format_style == 'spaces':
            return f"{area_code} {exchange} {number}"
        elif format_style == 'parentheses':
            return f"({area_code}) {exchange}-{number}"
        else:
            return f"{area_code}-{exchange}-{number}"
    
    def _generate_generic_phone(self) -> str:
        """Generate generic international phone"""
        country_code = random.randint(1, 99)
        number = ''.join([str(random.randint(0, 9)) for _ in range(9)])
        return f"+{country_code} {number}"


class AddressGenerator:
    """
    Generate physical addresses
    
    Components: street number, street name, city, state, zip
    """
    
    STREET_NAMES = [
        'Main', 'Oak', 'Maple', 'Cedar', 'Elm', 'Washington', 'Lake',
        'Hill', 'Park', 'Sunset', 'Pine', 'River', 'Forest', 'Broadway',
        'Church', 'Spring', 'Madison', 'Jackson', 'Lincoln', 'Franklin'
    ]
    
    STREET_TYPES = [
        'Street', 'Avenue', 'Boulevard', 'Drive', 'Lane', 'Road',
        'Way', 'Court', 'Place', 'Circle', 'Terrace'
    ]
    
    CITIES = [
        'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
        'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
        'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte',
        'San Francisco', 'Indianapolis', 'Seattle', 'Denver', 'Boston'
    ]
    
    STATES = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
        'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
        'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
        'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
        'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
        'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
        'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
        'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
        'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
        'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
        'WI': 'Wisconsin', 'WY': 'Wyoming'
    }
    
    def generate(
        self,
        num_addresses: int = 1,
        format_style: str = 'full'
    ) -> List[str]:
        """
        Generate addresses
        
        Args:
            num_addresses: Number to generate
            format_style: 'full', 'street_only', 'city_state_zip'
            
        Returns:
            List of addresses
        """
        addresses = []
        
        for _ in range(num_addresses):
            street_num = random.randint(1, 9999)
            street_name = random.choice(self.STREET_NAMES)
            street_type = random.choice(self.STREET_TYPES)
            city = random.choice(self.CITIES)
            state = random.choice(list(self.STATES.keys()))
            zip_code = f"{random.randint(10000, 99999)}"
            
            street = f"{street_num} {street_name} {street_type}"
            
            if format_style == 'street_only':
                address = street
            elif format_style == 'city_state_zip':
                address = f"{city}, {state} {zip_code}"
            else:  # full
                address = f"{street}, {city}, {state} {zip_code}"
            
            addresses.append(address)
        
        return addresses


class IdentifierGenerator:
    """
    Generate various identifier formats
    
    - UUIDs
    - SSNs
    - Custom IDs
    """
    
    def generate_uuid(self, num_ids: int = 1) -> List[str]:
        """Generate UUIDs"""
        return [str(uuid.uuid4()) for _ in range(num_ids)]
    
    def generate_ssn(self, num_ids: int = 1) -> List[str]:
        """Generate SSN-like identifiers (not real SSNs)"""
        ssns = []
        
        for _ in range(num_ids):
            # Avoid actual SSN patterns
            part1 = random.randint(100, 899)  # Avoid 900-999
            part2 = random.randint(10, 99)
            part3 = random.randint(1000, 9999)
            ssn = f"{part1}-{part2}-{part3}"
            ssns.append(ssn)
        
        return ssns
    
    def generate_custom_id(
        self,
        num_ids: int = 1,
        prefix: str = "ID",
        length: int = 8,
        include_letters: bool = True
    ) -> List[str]:
        """
        Generate custom formatted IDs
        
        Args:
            num_ids: Number to generate
            prefix: ID prefix
            length: Length of random part
            include_letters: Include letters in random part
            
        Returns:
            List of custom IDs
        """
        ids = []
        
        for _ in range(num_ids):
            if include_letters:
                chars = string.ascii_uppercase + string.digits
            else:
                chars = string.digits
            
            random_part = ''.join(random.choices(chars, k=length))
            custom_id = f"{prefix}{random_part}"
            ids.append(custom_id)
        
        return ids


class PIIGenerator:
    """
    Main PII generator that coordinates all PII types
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize PII generator
        
        Args:
            config: Configuration object with PII settings
        """
        self.config = config
        
        # Get settings from config
        if config and hasattr(config, 'pii'):
            locale = config.pii.locale
            gender_dist = config.pii.gender_distribution
        else:
            locale = 'en_US'
            gender_dist = {'male': 0.5, 'female': 0.5}
        
        # Initialize sub-generators
        self.name_generator = NameGenerator(locale=locale, gender_distribution=gender_dist)
        self.email_generator = EmailGenerator()
        self.phone_generator = PhoneGenerator()
        self.address_generator = AddressGenerator()
        self.id_generator = IdentifierGenerator()
    
    def generate_names(self, num_rows: int, **kwargs) -> pd.Series:
        """Generate names"""
        names = self.name_generator.generate(num_rows, **kwargs)
        return pd.Series(names)

    def generate_names_by_country(
        self,
        countries: pd.Series,
        include_middle: bool = False,
        include_title: bool = False,
        include_suffix: bool = False,
    ) -> pd.Series:
        """Generate region-appropriate names from country values."""
        names: List[str] = []

        for country in countries:
            locale = NameGenerator.locale_from_country(country, default_locale=self.name_generator.locale)
            regional_generator = NameGenerator(locale=locale, gender_distribution=self.name_generator.gender_distribution)
            generated = regional_generator.generate(
                num_names=1,
                include_middle=include_middle,
                include_title=include_title,
                include_suffix=include_suffix,
            )[0]
            names.append(generated)

        return pd.Series(names)
    

    def _infer_locale_from_reference(self, reference_data: Optional[pd.DataFrame]) -> Optional[str]:
        """Infer locale from reference country/city columns when explicit country is absent."""
        if reference_data is None or reference_data.empty:
            return None

        country_column = next((col for col in reference_data.columns if 'country' in col.lower()), None)
        if country_column:
            countries = reference_data[country_column].dropna().astype(str)
            if len(countries) > 0:
                locale_votes = [
                    NameGenerator.locale_from_country(country, default_locale=self.name_generator.locale)
                    for country in countries
                ]
                if locale_votes:
                    return pd.Series(locale_votes).mode().iloc[0]

        city_column = next((col for col in reference_data.columns if 'city' in col.lower()), None)
        if city_column:
            cities = reference_data[city_column].dropna().astype(str)
            if len(cities) > 0:
                locale_votes = [
                    NameGenerator.locale_from_city(city, default_locale=self.name_generator.locale)
                    for city in cities
                ]
                if locale_votes:
                    return pd.Series(locale_votes).mode().iloc[0]

        return None

    def generate_emails(self, num_rows: int, names: Optional[pd.Series] = None) -> pd.Series:
        """Generate emails, optionally based on names"""
        if names is not None:
            # Extract first and last names from full names
            first_names = []
            last_names = []
            
            for name in names:
                parts = str(name).split()
                if len(parts) >= 2:
                    first_names.append(parts[0])
                    last_names.append(parts[-1])
                else:
                    first_names.append(parts[0] if parts else f"user{len(first_names)}")
                    last_names.append(f"name{len(last_names)}")
            
            emails = self.email_generator.generate(
                num_rows,
                first_names=first_names,
                last_names=last_names
            )
        else:
            emails = self.email_generator.generate(num_rows)
        
        return pd.Series(emails)
    
    def generate_phones(self, num_rows: int, format_style: str = 'dashes') -> pd.Series:
        """Generate phone numbers"""
        phones = self.phone_generator.generate(num_rows, format_style=format_style)
        return pd.Series(phones)
    
    def generate_addresses(self, num_rows: int, format_style: str = 'full') -> pd.Series:
        """Generate addresses"""
        addresses = self.address_generator.generate(num_rows, format_style=format_style)
        return pd.Series(addresses)
    
    def generate_ids(self, num_rows: int, id_type: str = 'uuid') -> pd.Series:
        """Generate identifiers"""
        if id_type == 'uuid':
            ids = self.id_generator.generate_uuid(num_rows)
        elif id_type == 'ssn':
            ids = self.id_generator.generate_ssn(num_rows)
        else:
            ids = self.id_generator.generate_custom_id(num_rows)
        
        return pd.Series(ids)
    

    def _split_name(self, full_name: str) -> Tuple[str, str]:
        """Split a full name into first and last names."""
        parts = str(full_name).split()
        if not parts:
            return 'User', 'Name'
        if len(parts) == 1:
            return parts[0], 'Name'
        return parts[0], parts[-1]

    def generate(
        self,
        schema,
        column_names: List[str],
        num_rows: int,
        reference_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Pipeline-compatible generation for PII columns."""
        result = pd.DataFrame(index=range(num_rows))

        countries = None
        inferred_locale = self._infer_locale_from_reference(reference_data)
        if reference_data is not None:
            country_column = next((col for col in reference_data.columns if 'country' in col.lower()), None)
            if country_column:
                source_countries = reference_data[country_column].dropna()
                if len(source_countries) > 0:
                    countries = pd.Series(np.random.choice(source_countries, size=num_rows))

        # Generate names first because email depends on names.
        has_full_name = any(col.lower() == 'name' or 'full_name' in col.lower() for col in column_names)
        first_name_col = next((col for col in column_names if 'first_name' in col.lower()), None)
        last_name_col = next((col for col in column_names if 'last_name' in col.lower()), None)

        full_names = None

        def _generate_names_with_fallback() -> pd.Series:
            if countries is not None:
                return self.generate_names_by_country(countries)
            if inferred_locale:
                regional_generator = NameGenerator(
                    locale=inferred_locale,
                    gender_distribution=self.name_generator.gender_distribution,
                )
                return pd.Series(regional_generator.generate(num_rows))
            return self.generate_names(num_rows)

        if has_full_name:
            full_names = _generate_names_with_fallback()

            for col in column_names:
                if col.lower() == 'name' or 'full_name' in col.lower():
                    result[col] = full_names

        if first_name_col or last_name_col:
            if full_names is None:
                full_names = _generate_names_with_fallback()

            split_names = full_names.apply(self._split_name)
            if first_name_col:
                result[first_name_col] = split_names.apply(lambda x: x[0])
            if last_name_col:
                result[last_name_col] = split_names.apply(lambda x: x[1])

        # Generate emails using available names.
        email_cols = [col for col in column_names if 'email' in col.lower()]
        if email_cols:
            if first_name_col and last_name_col and first_name_col in result and last_name_col in result:
                derived_names = result[first_name_col] + ' ' + result[last_name_col]
            elif full_names is not None:
                derived_names = full_names
            else:
                derived_names = self.generate_names(num_rows)

            emails = self.generate_emails(num_rows, names=derived_names)
            for col in email_cols:
                result[col] = emails

        # Other PII columns
        for col in column_names:
            col_lower = col.lower()
            if col in result.columns:
                continue
            if 'phone' in col_lower or 'mobile' in col_lower:
                result[col] = self.generate_phones(num_rows)
            elif 'address' in col_lower:
                result[col] = self.generate_addresses(num_rows)
            elif 'id' in col_lower or 'identifier' in col_lower:
                result[col] = self.generate_ids(num_rows, id_type='custom')
            elif 'name' in col_lower:
                result[col] = self.generate_names(num_rows)

        return result

    def generate_complete_profile(self, num_rows: int) -> pd.DataFrame:
        """Generate complete PII profiles with all fields"""
        names = self.generate_names(num_rows)
        emails = self.generate_emails(num_rows, names=names)
        phones = self.generate_phones(num_rows)
        addresses = self.generate_addresses(num_rows)
        ids = self.generate_ids(num_rows, id_type='custom')
        
        return pd.DataFrame({
            'id': ids,
            'name': names,
            'email': emails,
            'phone': phones,
            'address': addresses
        })


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create PII generator
    generator = PIIGenerator()
    
    # Generate complete profiles
    profiles = generator.generate_complete_profile(num_rows=10)
    
    print("Generated PII Profiles:")
    print("="*80)
    print(profiles.to_string(index=False))
    
    print("\n" + "="*80)
    print("Individual Component Examples:")
    print("="*80)
    
    print("\nNames with titles:")
    names = generator.name_generator.generate(5, include_title=True, include_middle=True)
    for name in names:
        print(f"  {name}")
    
    print("\nPhone numbers (different formats):")
    for style in ['dashes', 'dots', 'parentheses']:
        phones = generator.phone_generator.generate(2, format_style=style)
        print(f"  {style}: {phones}")
