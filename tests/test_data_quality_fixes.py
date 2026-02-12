import numpy as np
import pandas as pd

from src.generators.numeric import NumericGenerator
from src.generators.pii import PIIGenerator
from src.generators.categorical import CategoricalGenerator
from src.generators.text import TextGenerator
from src.generators.temporal import TemporalGenerator
from src.config import get_default_config
from src.orchestrator import DataOrchestrator, DatasetSchema, ColumnMetadata, DataType, PipelineType


def test_numeric_age_and_serial_number_are_integers():
    reference = pd.DataFrame({
        'Age': [21.0, 35.0, 42.0, 29.0],
        'Sl No': [1.0, 2.0, 3.0, 4.0],
    })

    generator = NumericGenerator()
    synthetic = generator.generate(reference_data=reference, num_rows=20, seed=7)

    assert pd.api.types.is_integer_dtype(synthetic['Age'])
    assert pd.api.types.is_integer_dtype(synthetic['Sl No'])


def test_email_ids_are_derived_from_names():
    pii = PIIGenerator()
    names = pd.Series(['Aarav Sharma', 'Olivia Smith', 'Priya Patel'])

    emails = pii.generate_emails(3, names=names)

    for name, email in zip(names, emails):
        first, last = name.split()
        local = email.split('@')[0].lower()
        assert first.lower() in local or last.lower() in local


def test_name_generator_loads_json_name_files_when_present():
    pii = PIIGenerator()

    assert len(pii.name_generator.FIRST_NAMES['en_US']['male']) >= 200
    assert len(pii.name_generator.FIRST_NAMES['en_US']['female']) >= 200
    assert len(pii.name_generator.LAST_NAMES['en_US']) >= 200

    # India JSON uses state-based surnames and generator flattens them for en_IN.
    assert 'Bhosale' in pii.name_generator.LAST_NAMES['en_IN']


def test_region_based_name_generation_for_india():
    pii = PIIGenerator()
    countries = pd.Series(['India'] * 30)

    names = pii.generate_names_by_country(countries)
    india_last_names = set(pii.name_generator.LAST_NAMES['en_IN'])

    matched = sum(name.split()[-1] in india_last_names for name in names)
    assert matched / len(names) >= 0.8


def test_city_country_consistency_uses_country_mapping():
    config = get_default_config()
    gen = CategoricalGenerator(config)

    reference = pd.DataFrame({
        'country': ['India', 'United States', 'United Kingdom'],
        'city': ['Mumbai', 'Chicago', 'London'],
    })

    schema = DatasetSchema()
    for col in ['country', 'city']:
        value_counts = reference[col].value_counts()
        schema.add_column(
            ColumnMetadata(
                name=col,
                data_type=DataType.CATEGORICAL,
                pipeline=PipelineType.HYBRID,
                categories=value_counts.index.tolist(),
                category_frequencies=(value_counts / len(reference)).to_dict(),
            )
        )

    synthetic = gen.generate(
        schema=schema,
        column_names=['country', 'city'],
        num_rows=40,
        reference_data=reference,
    )

    for _, row in synthetic.iterrows():
        country = gen._normalize_country(row['country'])
        assert row['city'] in gen.country_city_map[country]


def test_orchestrator_preserves_integrity_across_fields():
    config = get_default_config()
    config.generation.num_rows = 60
    config.generation.seed = 42

    reference = pd.DataFrame({
        'record_id': [1, 2, 3, 4, 5, 6],
        'first_name': ['Aarav', 'Priya', 'Rohan', 'Ananya', 'Arjun', 'Meera'],
        'last_name': ['Sharma', 'Patel', 'Singh', 'Gupta', 'Reddy', 'Iyer'],
        'age': [22, 34, 41, 28, 37, 45],
        'city': ['Mumbai', 'Delhi', 'Bengaluru', 'Mumbai', 'Delhi', 'Bengaluru'],
        'country': ['India', 'India', 'India', 'India', 'India', 'India'],
        'email': ['aarav.sharma@test.com', 'priya.patel@test.com', 'rohan.singh@test.com', 'ananya.gupta@test.com', 'arjun.reddy@test.com', 'meera.iyer@test.com'],
    })

    orchestrator = DataOrchestrator(config)
    orchestrator.register_pipeline(PipelineType.NUMERIC, NumericGenerator(config))
    orchestrator.register_pipeline(PipelineType.TEXT, TextGenerator(config))
    orchestrator.register_pipeline(PipelineType.PII, PIIGenerator(config))
    orchestrator.register_pipeline(PipelineType.TEMPORAL, TemporalGenerator(config))
    orchestrator.register_pipeline(PipelineType.HYBRID, CategoricalGenerator(config))

    schema = orchestrator.analyze_schema(reference)
    result = orchestrator.generate(reference_data=reference, schema=schema, num_rows=60)
    synthetic = result.data

    assert pd.api.types.is_integer_dtype(synthetic['age'])
    assert pd.api.types.is_integer_dtype(synthetic['record_id'])

    for _, row in synthetic.iterrows():
        assert row['city'] in CategoricalGenerator(config).DEFAULT_COUNTRY_CITY_MAP['India']
        local = row['email'].split('@')[0].lower()
        assert row['first_name'].lower() in local or row['last_name'].lower() in local


def test_temporal_generation_handles_datetimeindex_without_dt_error():
    config = get_default_config()
    generator = TemporalGenerator(config)

    reference = pd.DataFrame({
        'signup_date': pd.date_range('2024-01-01', periods=20, freq='D')
    })

    schema = DatasetSchema()
    schema.add_column(
        ColumnMetadata(
            name='signup_date',
            data_type=DataType.TEMPORAL,
            pipeline=PipelineType.TEMPORAL,
            min_date=str(reference['signup_date'].min()),
            max_date=str(reference['signup_date'].max()),
            date_format='%Y-%m-%d'
        )
    )

    synthetic = generator.generate(
        schema=schema,
        column_names=['signup_date'],
        num_rows=25,
        reference_data=reference
    )

    assert len(synthetic) == 25
    assert synthetic['signup_date'].notna().all()


def test_gender_matches_generated_first_name():
    config = get_default_config()
    config.generation.seed = 123

    reference = pd.DataFrame({
        'first_name': ['Aarav', 'Priya', 'Rohan', 'Ananya'],
        'last_name': ['Sharma', 'Patel', 'Singh', 'Gupta'],
        'gender': ['Male', 'Female', 'Male', 'Female'],
        'email': ['a@test.com', 'b@test.com', 'c@test.com', 'd@test.com'],
    })

    orchestrator = DataOrchestrator(config)
    orchestrator.register_pipeline(PipelineType.PII, PIIGenerator(config))

    schema = orchestrator.analyze_schema(reference)
    result = orchestrator.generate(reference_data=reference, schema=schema, num_rows=40)
    synthetic = result.data

    male_names = {'aarav', 'rohan', 'arjun', 'rahul'}
    female_names = {'priya', 'ananya', 'aanya', 'meera'}

    for _, row in synthetic.iterrows():
        first = str(row['first_name']).lower()
        if first in male_names:
            assert str(row['gender']).lower() == 'male'
        if first in female_names:
            assert str(row['gender']).lower() == 'female'


def test_output_column_sequence_matches_input_sequence():
    config = get_default_config()
    config.generation.seed = 7

    reference = pd.DataFrame({
        'country': ['India', 'India', 'India'],
        'city': ['Mumbai', 'Delhi', 'Bengaluru'],
        'first_name': ['Aarav', 'Priya', 'Rohan'],
        'last_name': ['Sharma', 'Patel', 'Singh'],
        'gender': ['Male', 'Female', 'Male'],
        'age': [25, 30, 35],
        'record_id': [1, 2, 3],
        'email': ['a@test.com', 'b@test.com', 'c@test.com'],
    })

    orchestrator = DataOrchestrator(config)
    orchestrator.register_pipeline(PipelineType.NUMERIC, NumericGenerator(config))
    orchestrator.register_pipeline(PipelineType.PII, PIIGenerator(config))
    orchestrator.register_pipeline(PipelineType.HYBRID, CategoricalGenerator(config))

    schema = orchestrator.analyze_schema(reference)
    result = orchestrator.generate(reference_data=reference, schema=schema, num_rows=20)

    assert list(result.data.columns)[:len(reference.columns)] == list(reference.columns)


def test_indian_city_without_country_generates_indian_names():
    config = get_default_config()
    config.generation.seed = 19

    reference = pd.DataFrame({
        'Respondent_ID': [1, 2, 3, 4],
        'Name': ['Aarav Sharma', 'Priya Patel', 'Rohan Gupta', 'Ananya Iyer'],
        'City': ['Mumbai', 'Delhi', 'Jaipur', 'Noida'],
        'Gender': ['Male', 'Female', 'Male', 'Female'],
    })

    orchestrator = DataOrchestrator(config)
    orchestrator.register_pipeline(PipelineType.NUMERIC, NumericGenerator(config))
    orchestrator.register_pipeline(PipelineType.PII, PIIGenerator(config))
    orchestrator.register_pipeline(PipelineType.HYBRID, CategoricalGenerator(config))

    schema = orchestrator.analyze_schema(reference)
    result = orchestrator.generate(reference_data=reference, schema=schema, num_rows=50)
    synthetic = result.data

    indian_last_names = set(PIIGenerator().name_generator.LAST_NAMES['en_IN'])
    city_surnames = set()
    for surnames in DataOrchestrator.INDIAN_CITY_SURNAME_MAP.values():
        city_surnames.update(surnames)

    valid_surnames = indian_last_names | city_surnames
    matched = sum(str(name).split()[-1] in valid_surnames for name in synthetic['Name'])
    assert matched / len(synthetic) >= 0.8


def test_indian_city_surname_alignment():
    config = get_default_config()
    config.generation.seed = 31

    reference = pd.DataFrame({
        'Name': ['Aarav Sharma', 'Priya Patel', 'Rohan Gupta', 'Ananya Iyer'],
        'City': ['Kolkata', 'Pune', 'Chennai', 'Kochi'],
        'Gender': ['Male', 'Female', 'Male', 'Female'],
    })

    orchestrator = DataOrchestrator(config)
    orchestrator.register_pipeline(PipelineType.PII, PIIGenerator(config))
    orchestrator.register_pipeline(PipelineType.HYBRID, CategoricalGenerator(config))

    schema = orchestrator.analyze_schema(reference)
    result = orchestrator.generate(reference_data=reference, schema=schema, num_rows=80)
    synthetic = result.data

    city_surname_map = DataOrchestrator.INDIAN_CITY_SURNAME_MAP
    checks = 0
    matches = 0
    for _, row in synthetic.iterrows():
        city = str(row['City']).strip().lower()
        pool = city_surname_map.get(city)
        if not pool:
            continue
        checks += 1
        surname = str(row['Name']).split()[-1]
        if surname in pool:
            matches += 1

    assert checks > 0
    assert matches / checks >= 0.8


def test_name_column_gender_and_city_surname_alignment():
    config = get_default_config()
    config.generation.seed = 17

    reference = pd.DataFrame({
        'Respondent_ID': [1, 2, 3, 4],
        'Name': ['Aarav Sharma', 'Priya Nair', 'Rohan Banerjee', 'Ananya Verma'],
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'City': ['Bangalore', 'Chandigarh', 'Nagpur', 'Indore'],
    })

    orchestrator = DataOrchestrator(config)
    orchestrator.register_pipeline(PipelineType.NUMERIC, NumericGenerator(config))
    orchestrator.register_pipeline(PipelineType.PII, PIIGenerator(config))
    orchestrator.register_pipeline(PipelineType.HYBRID, CategoricalGenerator(config))

    schema = orchestrator.analyze_schema(reference)
    result = orchestrator.generate(reference_data=reference, schema=schema, num_rows=80)
    synthetic = result.data

    male_names = set()
    female_names = set()
    for locale_data in PIIGenerator().name_generator.FIRST_NAMES.values():
        male_names.update(n.lower() for n in locale_data.get('male', []))
        female_names.update(n.lower() for n in locale_data.get('female', []))

    gender_checks = 0
    gender_matches = 0
    surname_checks = 0
    surname_matches = 0
    for _, row in synthetic.iterrows():
        first = str(row['Name']).split()[0].strip().lower()
        gender = str(row['Gender']).strip().lower()
        city = str(row['City']).strip().lower()
        surname = str(row['Name']).split()[-1].strip()

        if first in male_names and first not in female_names:
            gender_checks += 1
            gender_matches += int(gender == 'male')
        elif first in female_names and first not in male_names:
            gender_checks += 1
            gender_matches += int(gender == 'female')

        pool = DataOrchestrator.INDIAN_CITY_SURNAME_MAP.get(city)
        if pool:
            surname_checks += 1
            surname_matches += int(surname in pool)

    assert gender_checks > 0
    assert gender_matches / gender_checks >= 0.9
    assert surname_checks > 0
    assert surname_matches / surname_checks >= 0.9
